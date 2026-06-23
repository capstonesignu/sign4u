"""
Upload jbedu dataset to Railway PostgreSQL.
- metadata + keypoints (BYTEA) → PostgreSQL
- videos: skipped (Railway Volume is not directly accessible from local)

Usage:
    DATABASE_URL="postgresql://..." python upload_jbedu.py [--dry-run] [--section 단어]
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import asyncpg

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset/jbedu-scraper/output/jbedu_dataset"
MIGRATION_SQL = Path(__file__).parent / "migrations/001_jbedu.sql"
SECTIONS = ["단어", "문장", "지명", "회화수어"]
CONCURRENCY = 8


def _collect_entries(section_filter=None):
    entries = []
    sections = [section_filter] if section_filter else SECTIONS
    for section in sections:
        section_dir = DATASET_ROOT / section
        if not section_dir.exists():
            print(f"[warn] 섹션 없음: {section_dir}")
            continue
        for category_dir in sorted(section_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            meta_dir = category_dir / "metadata"
            kp_dir = category_dir / "keypoints"
            vid_dir = category_dir / "videos"
            if not meta_dir.exists():
                continue
            for meta_file in sorted(meta_dir.glob("*.json")):
                uuid = meta_file.stem
                kp_path = kp_dir / f"{uuid}.npz"
                vid_path = vid_dir / f"{uuid}.mp4"
                entries.append({
                    "uuid": uuid,
                    "section": section,
                    "category": category,
                    "meta_path": meta_file,
                    "kp_path": kp_path if kp_path.exists() else None,
                    "has_video_local": vid_path.exists(),
                })
    return entries


async def _run_migration(pool):
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(sql)
    print("[migration] 완료")


async def _get_existing_uuids(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT uuid FROM jbedu_entries")
    return {r["uuid"] for r in rows}


async def _upload_entry(pool, entry, dry_run=False):
    meta = json.loads(entry["meta_path"].read_text(encoding="utf-8"))
    kp_bytes = entry["kp_path"].read_bytes() if (entry["kp_path"] and not entry.get("no_keypoints")) else None

    if dry_run:
        kp_size = f"{len(kp_bytes)//1024}KB" if kp_bytes else "없음"
        print(f"  [dry] {entry['section']}/{entry['category']}/{meta.get('korean_word','?')} kp={kp_size}")
        return True

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO jbedu_entries (
                uuid, cid, section, category,
                korean_word, english_word, description,
                video_duration, video_path,
                has_video, has_keypoints, scraped_at, keypoints
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (uuid) DO NOTHING
            """,
            meta["uuid"],
            meta.get("cid"),
            entry["section"],
            entry["category"],
            meta.get("korean_word", ""),
            meta.get("english_word"),
            meta.get("description"),
            meta.get("video_duration"),
            None,  # video_path: 아직 Volume 미업로드
            False,  # has_video
            kp_bytes is not None,
            datetime.fromisoformat(meta["scraped_at"]) if meta.get("scraped_at") else None,
            kp_bytes,
        )
    return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="실제 업로드 안 함")
    parser.add_argument("--section", help="특정 섹션만 (예: 단어)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="이미 업로드된 항목 건너뜀 (기본값)")
    parser.add_argument("--no-keypoints", action="store_true",
                        help="keypoints 제외하고 메타데이터만 업로드")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL 환경변수가 필요합니다")
        sys.exit(1)

    print(f"[info] 데이터셋 경로: {DATASET_ROOT}")
    entries = _collect_entries(args.section)
    print(f"[info] 발견된 항목: {len(entries)}개")

    if args.dry_run:
        print("[dry-run] 처음 10개만 출력:")
        for e in entries[:10]:
            meta = json.loads(e["meta_path"].read_text())
            kp_size = f"{e['kp_path'].stat().st_size//1024}KB" if e["kp_path"] else "없음"
            print(f"  {e['section']}/{e['category']}/{meta.get('korean_word','?')} kp={kp_size}")
        print(f"\n총 {len(entries)}개, keypoints={sum(1 for e in entries if e['kp_path'])}개 보유")
        return

    pool = await asyncpg.create_pool(db_url, min_size=2, max_size=CONCURRENCY)

    await _run_migration(pool)

    existing = await _get_existing_uuids(pool)
    print(f"[info] 기존 업로드: {len(existing)}개")

    todo = [e for e in entries if e["uuid"] not in existing]
    print(f"[info] 업로드 예정: {len(todo)}개")

    if not todo:
        print("[done] 모두 이미 업로드됨")
        await pool.close()
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    errors = 0

    async def upload_with_sem(entry):
        nonlocal done, errors
        async with sem:
            try:
                entry["no_keypoints"] = args.no_keypoints
                await _upload_entry(pool, entry)
                done += 1
                if done % 100 == 0:
                    print(f"  [{done}/{len(todo)}] 업로드 중...")
            except Exception as e:
                errors += 1
                print(f"  [err] {entry['uuid']}: {e}")

    await asyncio.gather(*[upload_with_sem(e) for e in todo])

    print(f"\n[done] 완료: {done}개 성공, {errors}개 실패")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
