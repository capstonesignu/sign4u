"""
qc_recordings.py — jbedu-recordings 품질 검수 (Phase 4 데이터 보강 전제 도구).

각 단어 폴더의 녹화를 인코더로 임베딩 → 단어 내부(intra-class) 코사인 분석으로:
  1. 손상/오라벨 녹화 자동 적발 — 같은 단어 다른 샘플들과 평균 cos < thr 인 개별 파일
     (잘린 영상 / 손 미검출 / 다른 동작이 섞임. Phase 1 에서 간호사·부작용·약이 이 케이스)
  2. 참조 부족 단어 — 녹화 수 < min_refs (보강 우선순위)

CPU 기본(--device cpu) → 진행 중 학습(MPS)과 비경합.
출력: 콘솔 리포트 + diagnostics/qc_recordings.json (UI/작업자가 처리할 목록).

데이터 규칙: dataset/recorded-video/ 는 평가전용 — 본 도구는 기본적으로
            dataset/jbedu-recordings/ (DB 소스)만 본다.

Usage:
    python qc_recordings.py
    python qc_recordings.py --dir dataset/jbedu-recordings --min-refs 3 --bad-thr 0.4
    python qc_recordings.py --encoder result_cs_collapse/encoder_cross_stream_best.pt
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

import upload_to_hf as U   # load_npz / LandmarkCrossStreamEncoder (전처리 동일)


def load_encoder(path: str, device):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    enc = U.LandmarkCrossStreamEncoder(d_model=256, nhead=8, num_layers=2,
                                       dim_feedforward=512, dropout=0.0, norm_first=True).to(device)
    enc.load_state_dict(state, strict=True); enc.eval()
    return enc


@torch.no_grad()
def embed_word(encoder, npz_paths, device):
    """단어 폴더의 각 npz → (파일명, 256-d 임베딩) 리스트. 로딩 실패 파일은 스킵."""
    out = []
    for p in npz_paths:
        seg = U.load_npz(p)            # (64,136) 또는 None
        if seg is None:
            continue
        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        e = F.normalize(encoder(x)[0], dim=0).cpu().numpy().astype(np.float32)
        out.append((p.name, e))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",     default="dataset/jbedu-recordings")
    p.add_argument("--encoder", default="result_cs_collapse/encoder_cross_stream_best.pt")
    p.add_argument("--min-refs", type=int,   default=3, help="이 미만이면 참조부족(보강대상)")
    p.add_argument("--bad-thr",  type=float, default=0.4,
                   help="같은 단어 형제들과 평균 cos 가 이 미만이면 손상 의심")
    p.add_argument("--device",  default="cpu", choices=["cpu", "mps"])
    p.add_argument("--out",     default="diagnostics/qc_recordings.json")
    args = p.parse_args()

    device = torch.device(args.device)
    encoder = load_encoder(args.encoder, device)
    rec_dir = Path(args.dir)
    print(f"[qc] dir={rec_dir}  encoder={Path(args.encoder).name}  device={device}\n")

    bad_files, low_ref_words = [], []
    n_words = n_files = 0

    for wd in sorted(rec_dir.iterdir()):
        if not wd.is_dir() or wd.name.startswith("."):
            continue
        npzs = sorted(wd.glob("*.npz"))
        embs = embed_word(encoder, npzs, device)
        n_words += 1; n_files += len(embs)
        n = len(embs)

        if n < args.min_refs:
            low_ref_words.append({"word": wd.name, "count": n})

        # 손상 검출: 각 샘플의 "나머지 형제 평균과의 cos". 형제 1개뿐이면 판정 불가.
        if n >= 2:
            M = np.stack([e for _, e in embs])           # (n,256)
            S = M @ M.T                                   # (n,n) cos
            for i, (fname, _) in enumerate(embs):
                others = np.delete(S[i], i)
                mean_to_sib = float(others.mean())
                if mean_to_sib < args.bad_thr:
                    bad_files.append({
                        "word": wd.name, "file": fname,
                        "mean_cos_to_siblings": round(mean_to_sib, 3),
                        "n_siblings": n - 1,
                    })

    bad_files.sort(key=lambda x: x["mean_cos_to_siblings"])
    low_ref_words.sort(key=lambda x: x["count"])

    print(f"[scan] {n_words}단어 · {n_files}녹화\n")
    print(f"── 손상/오라벨 의심 (형제 평균 cos < {args.bad_thr}): {len(bad_files)}건 ──")
    for b in bad_files:
        print(f"  {b['word']:<10} {b['file']:<42} cos={b['mean_cos_to_siblings']:+.3f} "
              f"(형제 {b['n_siblings']})")
    if not bad_files:
        print("  (없음)")

    print(f"\n── 참조 부족 (< {args.min_refs}개): {len(low_ref_words)}단어 ──")
    by_count = {}
    for w in low_ref_words:
        by_count.setdefault(w["count"], []).append(w["word"])
    for c in sorted(by_count):
        ws = by_count[c]
        print(f"  {c}개 ({len(ws)}단어): {', '.join(ws[:20])}{' ...' if len(ws) > 20 else ''}")

    need = sum(max(0, args.min_refs - w["count"]) for w in low_ref_words)
    print(f"\n[요약] 손상 의심 {len(bad_files)}건 + 참조부족 {len(low_ref_words)}단어 "
          f"→ min_refs={args.min_refs} 달성에 신규 ~{need}녹화 필요")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"corrupt_suspect": bad_files, "low_reference": low_ref_words,
         "min_refs": args.min_refs, "bad_thr": args.bad_thr,
         "n_words": n_words, "n_files": n_files, "new_recordings_needed": need},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
