"""
upload_jbedu.py — jbedu 데이터셋을 Railway PostgreSQL + Volume으로 업로드

사용법:
    # 1. 환경변수 설정 (Railway 대시보드 → PostgreSQL → Connect 탭에서 복사)
    export DATABASE_URL="postgresql://..."
    export JBEDU_VIDEO_ROOT="/path/to/railway/volume/videos"   # Volume 마운트 경로
    export JBEDU_DATASET_PATH="/path/to/jbedu_dataset"         # 로컬 데이터셋

    # 2. 실행
    python upload_jbedu.py                  # 전체 업로드
    python upload_jbedu.py --section 단어   # 특정 섹션만
    python upload_jbedu.py --dry-run        # DB 업로드만, 영상 복사 생략
    python upload_jbedu.py --skip-video     # 키포인트+메타만 (영상 제외)
    python upload_jbedu.py --skip-keypoints # 영상+메타만 (키포인트 제외)

중단 후 재실행하면 이미 업로드된 항목은 건너뜁니다 (uuid PRIMARY KEY 기준).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import asyncpg
import asyncio

DATASET_PATH = Path(os.getenv("JBEDU_DATASET_PATH",
    "/Users/spinai_dev/Desktop/Sogang/26 Spring/Capstone/AI-Embedding"
    "/dataset/jbedu-scraper/output/jbedu_dataset"))
VIDEO_ROOT   = Path(os.getenv("JBEDU_VIDEO_ROOT", "/data/videos"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

SECTIONS = ["단어", "문장", "지명", "회화수어"]


def _load_metadata(meta_path: Path) -> dict:
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _collect_entries(dataset_path: Path, section_filter: str | None = None) -> list[dict]:
    entries = []
    for section in SECTIONS:
        if section_filter and section != section_filter:
            continue
        sec_path = dataset_path / section
        if not sec_path.exists():
            continue
        for cat_dir in sorted(sec_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            meta_dir = cat_dir / "metadata"
            kp_dir   = cat_dir / "keypoints"
            vid_dir  = cat_dir / "videos"
            if not meta_dir.exists():
                continue
            for meta_file in sorted(meta_dir.glob("*.json")):
                uuid = meta_file.stem
                meta = _load_metadata(meta_file)
                entries.append({
                    "uuid":           uuid,
                    "section":        section,
                    "category":       category,
                    "meta":           meta,
                    "kp_path":        kp_dir  / f"{uuid}.npz",
                    "video_path_src": vid_dir / f"{uuid}.mp4",
                    "video_path_rel": f"{section}/{category}/{uuid}.mp4",
                })
    return entries


async def _run_migration(pool: asyncpg.Pool) -> None:
    sql_path = Path(__file__).parent / "migrations" / "001_jbedu.sql"
    sql = sql_path.read_text(encoding="utf-8")
    await pool.execute(sql)
    print("[migration] 001_jbedu.sql 완료")


async def _get_existing_uuids(pool: asyncpg.Pool) -> set[str]:
    rows = await pool.fetch("SELECT uuid FROM jbedu_entries")
    return {r["uuid"] for r in rows}


async def _upload_entry(
    pool: asyncpg.Pool,
    entry: dict,
    skip_video: bool,
    skip_keypoints: bool,
    dry_run: bool,
) -> tuple[str, str]:
    """
    Returns (uuid, status) where status in:
      "inserted" / "skipped" / "error:<msg>"
    """
    uuid    = entry["uuid"]
    meta    = entry["meta"]
    section = entry["section"]
    cat     = entry["category"]

    # 키포인트 읽기
    kp_bytes: bytes | None = None
    if not skip_keypoints and entry["kp_path"].exists():
        kp_bytes = entry["kp_path"].read_bytes()

    # 영상 복사 (Volume 경로로)
    has_video = False
    video_rel = entry["video_path_rel"]
    if not skip_video and not dry_run and entry["video_path_src"].exists():
        dest = VIDEO_ROOT / video_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(entry["video_path_src"], dest)
        has_video = True
    elif entry["video_path_src"].exists():
        has_video = True  # dry_run: 파일 존재 여부만 기록

    try:
        await pool.execute(
            """
            INSERT INTO jbedu_entries
                (uuid, cid, section, category, korean_word, english_word,
                 description, video_duration, video_path, has_video, has_keypoints,
                 scraped_at, keypoints)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,
                    $12::timestamptz, $13)
            ON CONFLICT (uuid) DO NOTHING
            """,
            uuid,
            meta.get("cid"),
            section,
            cat,
            meta.get("korean_word", ""),
            meta.get("english_word"),
            meta.get("description"),
            meta.get("video_duration"),
            video_rel if has_video else None,
            has_video,
            kp_bytes is not None,
            meta.get("scraped_at"),
            kp_bytes,
        )
        return uuid, "inserted"
    except Exception as e:
        return uuid, f"error:{e}"


async def main(args: argparse.Namespace) -> None:
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL 환경변수가 설정되지 않았습니다.")
        print("  export DATABASE_URL='postgresql://user:pass@host:port/db'")
        sys.exit(1)

    print(f"[config] 데이터셋: {DATASET_PATH}")
    print(f"[config] 영상 대상: {VIDEO_ROOT}")
    print(f"[config] dry_run={args.dry_run}, skip_video={args.skip_video}, skip_keypoints={args.skip_keypoints}")
    if args.section:
        print(f"[config] 섹션 필터: {args.section}")

    # 항목 수집
    print("\n[collect] 항목 수집 중...")
    entries = _collect_entries(DATASET_PATH, args.section)
    print(f"[collect] 총 {len(entries)}개 항목")

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    try:
        # 마이그레이션
        await _run_migration(pool)

        # 기존 항목 파악 (재시작 시 건너뛰기)
        existing = await _get_existing_uuids(pool)
        new_entries = [e for e in entries if e["uuid"] not in existing]
        print(f"[upload] 기존 {len(existing)}개 건너뜀, 신규 {len(new_entries)}개 업로드 예정\n")

        inserted = skipped = errors = 0
        t0 = time.time()

        for i, entry in enumerate(new_entries, 1):
            uuid, status = await _upload_entry(
                pool, entry,
                skip_video=args.skip_video or args.dry_run,
                skip_keypoints=args.skip_keypoints,
                dry_run=args.dry_run,
            )

            if status == "inserted":
                inserted += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"  [!] {uuid}: {status}")

            # 진행 상황 출력
            if i % 50 == 0 or i == len(new_entries):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta  = (len(new_entries) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(new_entries)}] "
                    f"inserted={inserted} errors={errors} "
                    f"속도={rate:.1f}건/s ETA={eta:.0f}s"
                )

    finally:
        await pool.close()

    print(f"\n[완료] inserted={inserted}, skipped={skipped}, errors={errors}")
    if errors:
        print(f"[경고] {errors}개 항목 업로드 실패. 재실행하면 성공한 항목은 건너뜁니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jbedu 데이터셋 Railway 업로드")
    parser.add_argument("--section",          help="특정 섹션만 (단어/문장/지명/회화수어)")
    parser.add_argument("--dry-run",          action="store_true", help="DB만, 영상 복사 생략")
    parser.add_argument("--skip-video",       action="store_true", help="영상 복사 건너뜀")
    parser.add_argument("--skip-keypoints",   action="store_true", help="키포인트 DB 저장 건너뜀")
    args = parser.parse_args()

    asyncio.run(main(args))
