"""
Upload jbedu videos to Railway Volume via data-server upload endpoint.

Usage:
    python upload_videos.py \
        --base-url https://ui-server-production-4dff.up.railway.app \
        --api-key  <DATA_API_KEY> \
        [--section 단어] [--dry-run] [--workers 4]
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

import requests

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset/jbedu-scraper/output/jbedu_dataset"
SECTIONS = ["단어", "문장", "지명", "회화수어"]


def _collect_videos(section_filter=None):
    videos = []
    sections = [section_filter] if section_filter else SECTIONS
    for section in sections:
        section_dir = DATASET_ROOT / section
        if not section_dir.exists():
            continue
        for category_dir in sorted(section_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            vid_dir = category_dir / "videos"
            if not vid_dir.exists():
                continue
            for vid_file in sorted(vid_dir.glob("*.mp4")):
                videos.append({"uuid": vid_file.stem, "path": vid_file})
    return videos


def upload_one(item, base_url, api_key, session):
    uuid = item["uuid"]
    path = item["path"]
    url = f"{base_url}/data/admin/upload/{uuid}"
    try:
        with open(path, "rb") as f:
            resp = session.post(
                url,
                data=f,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(path.stat().st_size),
                },
                timeout=120,
                stream=False,
            )
        if resp.status_code == 200:
            size_mb = resp.json().get("size_bytes", 0) / 1024 / 1024
            return "ok", f"{uuid[:8]} ({size_mb:.1f}MB)"
        else:
            return "err", f"{uuid[:8]} HTTP {resp.status_code}: {resp.text[:80]}"
    except Exception as e:
        return "err", f"{uuid[:8]} {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="express-server public URL")
    parser.add_argument("--api-key", required=True, help="DATA_API_KEY")
    parser.add_argument("--section", help="특정 섹션만 (예: 단어)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="동시 업로드 수 (기본 4)")
    args = parser.parse_args()

    videos = _collect_videos(args.section)
    print(f"[info] 비디오 {len(videos)}개 발견")

    if args.dry_run:
        total_mb = sum(v["path"].stat().st_size for v in videos) / 1024 / 1024
        print(f"[dry-run] 총 {total_mb:.0f}MB, 섹션: {args.section or '전체'}")
        for v in videos[:5]:
            mb = v["path"].stat().st_size / 1024 / 1024
            print(f"  {v['uuid'][:8]}  {mb:.1f}MB  {v['path'].parent.parent.name}")
        return

    session = requests.Session()
    done = 0
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(upload_one, v, args.base_url, args.api_key, session): v
            for v in videos
        }
        for fut in concurrent.futures.as_completed(futs):
            status, msg = fut.result()
            done += 1
            if status == "err":
                errors += 1
                print(f"  [err] {msg}")
            if done % 50 == 0 or done == len(videos):
                print(f"  [{done}/{len(videos)}] 완료 (오류 {errors}개)")

    print(f"\n[done] 성공 {done - errors}개 / 실패 {errors}개")


if __name__ == "__main__":
    main()
