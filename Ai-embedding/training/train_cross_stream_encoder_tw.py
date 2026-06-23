"""
train_cross_stream_encoder_tw.py — Phase 1: Time Warp augmentation 추가.

변경점 (vs train_cross_stream_encoder.py):
  - time_warp(rate_range=(0.5, 2.0))  추가 — 느린 수어(100+ frames) 커버
  - spatial_noise 추가
  - augment_seq()가 flip + warp + noise를 한 번에 처리
  - best model 기준: AI Hub holdout Top-3 → recorded-video LOO Top-3
    (실사용 도메인 기준으로 저장)

Usage:
    python train_cross_stream_encoder_tw.py
    python train_cross_stream_encoder_tw.py --epochs 2 --steps 10  # smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 model/preprocess/services

from model import LandmarkCrossStreamEncoder, VelocityLandmarkCrossStreamEncoder, add_velocity
from preprocess import feature_dim_for, temporal_interpolate, TARGET_LENGTH, _load_npz_raw
from services.preprocessing import convert_stream
from collapse_utils import effective_rank   # dimensional collapse 모니터링 (B1)


@torch.no_grad()
def _measure_rank(encoder, segs, device, use_velocity=False, max_samples=600):
    """recorded/val segs 일부 → effective rank (collapse 추적). eval 임베딩과 동일 경로."""
    encoder.eval()
    embs, n = [], 0
    for _w, ss in segs.items():
        for s in ss:
            sv = add_velocity(s) if use_velocity else s
            x  = torch.from_numpy(sv).unsqueeze(0).to(device)
            embs.append(F.normalize(encoder(x)[0], dim=0).cpu())
            n += 1
            if n >= max_samples:
                break
        if n >= max_samples:
            break
    return effective_rank(torch.stack(embs))

PRESET         = "B"
AIHUB_DIR      = "dataset/word-keypoints/1.Training"
WORD_MAP_PATH  = "word_mapping.json"
DIR_WORDS_PATH = "directional_words.json"
OUT_DIR        = "result_cross_stream_tw"
RECORDED_DIR   = "dataset/recorded-video"
N_HOLDOUT      = 500
SEED           = 42
CKPT_NAME      = "encoder_cross_stream_best.pt"

_WORD_FILE_RE = re.compile(r"NIA_SL_(WORD\d+)_")


# ── Augmentation ──────────────────────────────────────────────────────────────

def flip_horizontal_B(seq: np.ndarray) -> np.ndarray:
    """Preset B (T, 136) 좌우 반전."""
    s = seq.copy()
    s[:, 0::2] *= -1
    for l_start, r_start in [(2, 4), (6, 8), (10, 12)]:
        tmp = s[:, l_start:l_start + 2].copy()
        s[:, l_start:l_start + 2] = s[:, r_start:r_start + 2]
        s[:, r_start:r_start + 2] = tmp
    tmp = s[:, 14:56].copy()
    s[:, 14:56] = s[:, 56:98]
    s[:, 56:98] = tmp
    return s


def time_warp(seq: np.ndarray,
              rate_range: Tuple[float, float] = (0.5, 2.0)) -> np.ndarray:
    """시간축 속도 변환 — 고정 T 유지.

    rate < 1: 느리게(프레임 늘임) → 느린 수어 시뮬레이션
    rate > 1: 빠르게(프레임 압축) → 빠른 수어 시뮬레이션

    원리: 현재 T 프레임 → new_len으로 리샘플 → 다시 T로 복원.
    """
    T, D = seq.shape
    rate = random.uniform(rate_range[0], rate_range[1])
    new_len = max(4, int(round(T * rate)))
    src_idx = np.linspace(0, T - 1, new_len)
    resampled = seq[np.round(src_idx).astype(int)]
    back_idx = np.linspace(0, new_len - 1, T)
    return resampled[np.round(back_idx).astype(int)].astype(seq.dtype)


def augment_seq(seq: np.ndarray,
                word: str,
                directional: Set[str],
                flip_prob: float = 0.5,
                warp_prob: float = 0.8,
                warp_range: Tuple[float, float] = (0.5, 2.0),
                noise_std: float = 0.005) -> np.ndarray:
    """flip + time_warp + spatial noise를 순서대로 적용."""
    out = seq.copy()
    if word not in directional and random.random() < flip_prob:
        out = flip_horizontal_B(out)
    if random.random() < warp_prob:
        out = time_warp(out, warp_range)
    if noise_std > 0:
        out = (out + np.random.normal(0.0, noise_std, out.shape)).astype(out.dtype)
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_segs(word_map: dict, aihub_dir: str,
                  target_length: int = TARGET_LENGTH) -> Dict[str, List[np.ndarray]]:
    segs: Dict[str, List[np.ndarray]] = defaultdict(list)
    paths = glob.glob(os.path.join(aihub_dir, "**", "*.npz"), recursive=True)
    print(f"[data] scanning {len(paths)} npz files...")
    for path in paths:
        m = _WORD_FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        kor = word_map.get(m.group(1))
        if not kor:
            continue
        try:
            d      = np.load(path, allow_pickle=True)
            stream = _load_npz_raw(path, PRESET)
            if "start_sec" in d and "end_sec" in d:
                fps = float(d["fps"])
                T   = stream.shape[0]
                sf  = max(0, round(float(d["start_sec"]) * fps))
                ef  = min(T - 1, round(float(d["end_sec"]) * fps))
                crop = stream[sf:ef + 1] if ef > sf + 1 else stream
            else:
                crop = stream
            if len(crop) < 3:
                continue
            fixed = temporal_interpolate(crop, target_length).astype(np.float32)
            segs[kor].append(fixed)
        except Exception:
            continue

    result = {w: v for w, v in segs.items() if len(v) >= 2}
    raw_lens = [s.shape[0] for v in result.values() for s in v]
    print(f"[data] {len(result)} words, {sum(len(v) for v in result.values())} segs  "
          f"target_length={target_length}")
    return result


def split_holdout(
    all_words: List[str], n_holdout: int = 500, seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    words = sorted(all_words)
    rng.shuffle(words)
    return words[:n_holdout], words[n_holdout:]


def _pick2(segs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(segs) >= 2:
        i, j = random.sample(range(len(segs)), 2)
    else:
        i = j = 0
    return segs[i], segs[j]


# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temp: float) -> torch.Tensor:
    N = emb_a.shape[0]
    a = F.normalize(emb_a, dim=1)
    b = F.normalize(emb_b, dim=1)
    z = torch.cat([a, b], dim=0)
    sim = torch.mm(z, z.T) / temp
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(sim.device)
    return F.cross_entropy(sim, labels)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_nshot_openvocab(
    encoder: LandmarkCrossStreamEncoder,
    holdout_segs: Dict[str, List[np.ndarray]],
    device: torch.device,
    n_shot: int = 3,
    top_k_list: Tuple[int, ...] = (3, 5),
    n_rounds: int = 5,
    use_velocity: bool = False,
) -> Dict[str, float]:
    encoder.eval()
    words = [w for w, v in holdout_segs.items() if len(v) >= n_shot + 1]
    if not words:
        return {f"top{k}": 0.0 for k in top_k_list}

    all_seqs: List[np.ndarray] = []
    ranges = []
    for w in words:
        s = len(all_seqs)
        segs = [add_velocity(seg) if use_velocity else seg for seg in holdout_segs[w]]
        all_seqs.extend(segs)
        ranges.append((w, s, len(all_seqs)))

    all_emb_list = []
    for start in range(0, len(all_seqs), 256):
        batch = np.stack(all_seqs[start:start + 256])
        x = torch.from_numpy(batch).to(device)
        e = encoder(x)
        all_emb_list.append(F.normalize(e, dim=1).cpu().numpy())
    all_emb = np.concatenate(all_emb_list, axis=0).astype(np.float32)

    word_embs: Dict[str, np.ndarray] = {}
    for w, s, e in ranges:
        word_embs[w] = all_emb[s:e]

    max_k = max(top_k_list)
    round_accs: Dict[int, List[float]] = {k: [] for k in top_k_list}

    for _ in range(n_rounds):
        db_embs, db_labels, q_embs, q_labels = [], [], [], []
        for w in words:
            n = len(word_embs[w])
            idxs = list(range(n))
            random.shuffle(idxs)
            for i in idxs[:n_shot]:
                db_embs.append(word_embs[w][i]); db_labels.append(w)
            for i in idxs[n_shot:]:
                q_embs.append(word_embs[w][i]); q_labels.append(w)
        if not db_embs or not q_embs:
            continue

        db_mat = np.array(db_embs, dtype=np.float32)
        faiss.normalize_L2(db_mat)
        fidx = faiss.IndexFlatIP(db_mat.shape[1])
        fidx.add(db_mat)

        q_mat = np.array(q_embs, dtype=np.float32)
        faiss.normalize_L2(q_mat)
        k_search = min(max_k, len(db_labels))
        _, I = fidx.search(q_mat, k_search)

        for k in top_k_list:
            actual_k = min(k, k_search)
            hits = sum(
                q_labels[qi] in [db_labels[ii] for ii in I[qi, :actual_k]]
                for qi in range(len(q_labels))
            )
            round_accs[k].append(hits / len(q_labels))

    return {
        f"top{k}": float(np.mean(round_accs[k])) if round_accs[k] else 0.0
        for k in top_k_list
    }


# ── recorded-video eval ───────────────────────────────────────────────────────

def load_recorded_segs(recorded_dir: str,
                       target_length: int = TARGET_LENGTH,
                       fps_norm: bool = False,
                       target_fps: float = 30.0) -> Dict[str, List[np.ndarray]]:
    """recorded-video/{word}/*.npz → convert_stream 전처리.

    fps_norm=True: 녹화 fps 메타데이터 기준으로 target_fps에 맞게 먼저 리샘플한 뒤
                   target_length로 보간. 17fps→30fps 도메인 정규화에 사용.
    """
    segs: Dict[str, List[np.ndarray]] = {}
    rec = Path(recorded_dir)
    for wd in sorted(rec.iterdir()):
        if not wd.is_dir():
            continue
        samples = []
        for npz in sorted(wd.glob("*.npz")):
            try:
                data = np.load(str(npz))
                T    = data["pose"].shape[0]
                kps  = np.concatenate([
                    data["pose"].reshape(T, -1),
                    data["left_hand"].reshape(T, -1),
                    data["right_hand"].reshape(T, -1),
                    data["face"].reshape(T, -1),
                ], axis=-1)
                seg = convert_stream(kps, target_preset=PRESET)
                if fps_norm and "fps" in data:
                    src_fps  = float(data["fps"])
                    norm_len = max(8, int(round(len(seg) * target_fps / src_fps)))
                    seg = temporal_interpolate(seg, norm_len)
                seg = temporal_interpolate(seg, target_length)
                samples.append(seg.astype(np.float32))
            except Exception:
                pass
        if samples:
            segs[wd.name] = samples
    return segs


@torch.no_grad()
def eval_recorded_loo(
    encoder: LandmarkCrossStreamEncoder,
    recorded_segs: Dict[str, List[np.ndarray]],
    device: torch.device,
    use_velocity: bool = False,
) -> Dict[str, float]:
    """recorded-video leave-one-out Top-1 / Top-3."""
    encoder.eval()

    word_embs: Dict[str, List[np.ndarray]] = {}
    for word, segs in recorded_segs.items():
        embs = []
        for seg in segs:
            s = add_velocity(seg) if use_velocity else seg
            x   = torch.from_numpy(s).unsqueeze(0).to(device)
            emb = encoder(x)
            embs.append(F.normalize(emb[0], dim=0).cpu().numpy().astype(np.float32))
        word_embs[word] = embs

    top1_hits, top3_hits, total = 0, 0, 0
    for qw, embs in word_embs.items():
        for i, qemb in enumerate(embs):
            db_v, db_l = [], []
            for w, wembs in word_embs.items():
                for j, e in enumerate(wembs):
                    if w == qw and j == i:
                        continue
                    db_v.append(e); db_l.append(w)
            if not db_v:
                continue
            mat = np.stack(db_v).astype(np.float32)
            faiss.normalize_L2(mat)
            fidx = faiss.IndexFlatIP(mat.shape[1]); fidx.add(mat)
            q = qemb.reshape(1, -1).copy(); faiss.normalize_L2(q)
            _, I = fidx.search(q, min(3, len(db_l)))
            top3w = [db_l[ii] for ii in I[0]]
            if top3w[0] == qw:  top1_hits += 1
            if qw in top3w:     top3_hits += 1
            total += 1

    return {
        "top1": top1_hits / total if total else 0.0,
        "top3": top3_hits / total if total else 0.0,
        "total": total,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(args.output, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    warp_range = (args.warp_min, args.warp_max)

    word_map    = json.load(open(WORD_MAP_PATH, encoding="utf-8"))
    directional: Set[str] = set()
    if os.path.exists(DIR_WORDS_PATH):
        directional = set(json.load(open(DIR_WORDS_PATH, encoding="utf-8")))
    print(f"[aug] directional words excluded from flip: {len(directional)}")
    print(f"[aug] time_warp  prob={args.warp_prob}  range={warp_range}")
    print(f"[aug] flip_prob={args.flip_prob}  noise_std={args.noise_std}")

    print("[0] recorded-video 로딩 (eval 기준)...")
    recorded_segs = load_recorded_segs(
        args.recorded_dir, target_length=args.target_length,
        fps_norm=args.fps_norm, target_fps=args.target_fps,
    )
    rec_lens = [s.shape[0] for v in recorded_segs.values() for s in v]
    print(f"     {len(recorded_segs)} words, "
          f"{sum(len(v) for v in recorded_segs.values())} samples  "
          f"fps_norm={args.fps_norm}")

    print("[1] AI Hub 고립 단어 로딩...")
    all_segs = load_all_segs(word_map, args.aihub_dir,
                             target_length=args.target_length)

    all_words = sorted(all_segs.keys())
    holdout_words, train_words = split_holdout(all_words, n_holdout=N_HOLDOUT, seed=SEED)
    print(f"[split] train={len(train_words)}, holdout={len(holdout_words)}")

    rng_inst = random.Random(SEED + 1)
    train_segs: Dict[str, List[np.ndarray]] = {}
    val_segs:   Dict[str, List[np.ndarray]] = {}
    for w in train_words:
        if w not in all_segs: continue
        insts = all_segs[w][:]
        rng_inst.shuffle(insts)
        n_val = max(1, len(insts) // 5)
        val_segs[w]   = insts[:n_val]
        train_segs[w] = insts[n_val:]
        if len(train_segs[w]) < 2:
            train_segs[w] = insts; val_segs[w] = insts[:1]

    holdout_segs = {w: all_segs[w] for w in holdout_words if w in all_segs}
    print(f"[inst split] train={sum(len(v) for v in train_segs.values())} insts, "
          f"val={sum(len(v) for v in val_segs.values())} insts")
    print(f"[holdout] {len(holdout_segs)} words")

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"[device] {device}")

    ModelCls = VelocityLandmarkCrossStreamEncoder if getattr(args, "use_velocity", False) \
               else LandmarkCrossStreamEncoder
    encoder = ModelCls(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        norm_first=True,
    ).to(device)

    if getattr(args, "equal_init", False):
        with torch.no_grad():
            encoder.stream_logits.copy_(torch.zeros(4))

    n_params = sum(p.numel() for p in encoder.parameters())
    init_w = F.softmax(encoder.stream_logits, dim=0).detach().cpu().numpy()
    model_name = "VelocityLandmarkCrossStreamEncoder" if getattr(args, "use_velocity", False) \
                 else "LandmarkCrossStreamEncoder"
    print(f"[model] {model_name} | d_model={args.d_model} | "
          f"layers/stream={args.num_layers} | params={n_params:,}")
    print(f"[weights] pose={init_w[0]:.3f}  left={init_w[1]:.3f}  "
          f"right={init_w[2]:.3f}  face={init_w[3]:.3f}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps
    )

    # best model: recorded-video LOO Top-3 기준
    best_rec_top3  = -1.0
    best_val_loss  = float("inf")
    patience_count = 0
    overfit_count  = 0
    history        = []
    train_word_list = list(train_segs.keys())
    val_word_list   = list(val_segs.keys())

    # target word oversampling: 51 recorded-video words appear 2× in sampling pool
    _target_words = set(recorded_segs.keys())
    _oversample = getattr(args, "oversample_target", 1)
    if _oversample > 1:
        _extra = [w for w in train_word_list if w in _target_words] * (_oversample - 1)
        train_word_list_weighted = train_word_list + _extra
        print(f"[oversample] {len(_extra)} extra target words added "
              f"(pool {len(train_word_list)} → {len(train_word_list_weighted)})")
    else:
        train_word_list_weighted = train_word_list

    print(f"\n[train] {args.epochs} ep × {args.steps} steps  "
          f"lr={args.lr}  temp={args.temperature}  batch={args.ww_batch}")
    print("=" * 70)

    for ep in range(1, args.epochs + 1):
        encoder.train()
        ep_loss = 0.0
        t0 = time.time()

        for _ in range(args.steps):
            optimizer.zero_grad()
            ww_words = random.sample(train_word_list_weighted, min(args.ww_batch, len(train_word_list_weighted)))
            aa, bb = [], []
            for w in ww_words:
                a, b = _pick2(train_segs[w])
                a_aug = augment_seq(a, w, directional,
                                    args.flip_prob, args.warp_prob, warp_range,
                                    args.noise_std)
                b_aug = augment_seq(b, w, directional,
                                    args.flip_prob, args.warp_prob, warp_range,
                                    args.noise_std)
                if getattr(args, "use_velocity", False):
                    a_aug = add_velocity(a_aug)
                    b_aug = add_velocity(b_aug)
                aa.append(a_aug); bb.append(b_aug)

            ta = torch.from_numpy(np.stack(aa)).float().to(device)
            tb = torch.from_numpy(np.stack(bb)).float().to(device)
            loss = nt_xent_loss(encoder(ta), encoder(tb), args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()

        # ── Val loss ──────────────────────────────────────────────────────
        encoder.eval()
        with torch.no_grad():
            vl_acc = 0.0
            for _ in range(args.val_steps):
                ww = random.sample(val_word_list, min(args.ww_batch, len(val_word_list)))
                aa_v, bb_v = [], []
                for w in ww:
                    a, b = _pick2(val_segs[w])
                    if getattr(args, "use_velocity", False):
                        a = add_velocity(a); b = add_velocity(b)
                    aa_v.append(a); bb_v.append(b)
                ta = torch.from_numpy(np.stack(aa_v)).float().to(device)
                tb = torch.from_numpy(np.stack(bb_v)).float().to(device)
                vl_acc += nt_xent_loss(encoder(ta), encoder(tb), args.temperature).item()
            val_loss = vl_acc / args.val_steps

        elapsed  = time.time() - t0
        avg_loss = ep_loss / args.steps

        if val_loss < best_val_loss:
            best_val_loss = val_loss; overfit_count = 0
        else:
            overfit_count += 1
        overfit_flag = (f"  [↑val {overfit_count}/{args.overfit_patience}]"
                        if overfit_count > 0 else "")

        if ep % 10 == 0:
            w = F.softmax(encoder.stream_logits, dim=0).detach().cpu().numpy()
            weight_str = (f"  [w] pose={w[0]:.3f} L={w[1]:.3f} "
                          f"R={w[2]:.3f} face={w[3]:.3f}")
        else:
            weight_str = ""

        print(
            f"Ep {ep:03d}/{args.epochs}  "
            f"train={avg_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s"
            f"{overfit_flag}{weight_str}",
            flush=True,
        )

        if overfit_count >= args.overfit_patience:
            print(f"[overfit stop] {args.overfit_patience} consecutive epochs")
            break

        # ── val_every: recorded LOO + AI Hub open-vocab ───────────────────
        if ep % args.val_every == 0:
            # 1. recorded-video LOO (primary metric)
            ext = eval_recorded_loo(encoder, recorded_segs, device,
                                    use_velocity=getattr(args, "use_velocity", False))
            marker = ""
            if ext["top3"] > best_rec_top3:
                best_rec_top3 = ext["top3"]
                best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                torch.save(best_state, os.path.join(args.output, CKPT_NAME))
                patience_count = 0
                marker = " ★"
            else:
                patience_count += 1
            print(f"  [recorded LOO]  Top-1: {ext['top1']:.1%}  "
                  f"Top-3: {ext['top3']:.1%}  (n={ext['total']}){marker}",
                  flush=True)

            # dimensional collapse 추적 (B1)
            rk = _measure_rank(encoder, recorded_segs, device,
                               use_velocity=getattr(args, "use_velocity", False))
            print(f"  [rank] PR={rk['participation_ratio']:.1f}/{rk['dim']}  "
                  f"entropy_rank={rk['entropy_rank']:.1f}  "
                  f"top1_var={rk['top1_var_frac']:.1%}", flush=True)

            # 2. AI Hub open-vocab (참고용)
            print(f"  [open-vocab]  R={args.rounds} rounds, {len(holdout_segs)} words")
            ep_3shot = {}
            for n_shot in (3, 5):
                accs = eval_nshot_openvocab(
                    encoder, holdout_segs, device,
                    n_shot=n_shot, top_k_list=(3, 5), n_rounds=args.rounds,
                    use_velocity=getattr(args, "use_velocity", False),
                )
                ep_3shot[n_shot] = accs
                print(f"  {n_shot}-shot | Top-3: {accs['top3']:.4f}  "
                      f"Top-5: {accs['top5']:.4f}", flush=True)

            history.append({
                "ep": ep,
                "train_loss": round(avg_loss, 4),
                "val_loss":   round(val_loss, 4),
                "rec_top1":   round(ext["top1"], 4),
                "rec_top3":   round(ext["top3"], 4),
                "3shot_top3": round(ep_3shot[3]["top3"], 4),
                "5shot_top3": round(ep_3shot[5]["top3"], 4),
                "pr": round(rk["participation_ratio"], 2),
                "entropy_rank": round(rk["entropy_rank"], 2),
                "stream_weights": F.softmax(
                    encoder.stream_logits, dim=0
                ).detach().cpu().tolist(),
            })
            json.dump(history,
                      open(os.path.join(args.output, "history.json"), "w"), indent=2)

            if patience_count >= args.patience:
                print(f"[early stop] patience={args.patience} at ep {ep}")
                break

    print("=" * 70)
    print(f"[done] best recorded LOO Top-3 = {best_rec_top3:.4f}")

    # 메타데이터 포함 저장
    final_state = torch.load(os.path.join(args.output, CKPT_NAME), map_location="cpu",
                             weights_only=False)
    torch.save({
        "encoder_state_dict": final_state if "encoder_state_dict" not in final_state
                              else final_state["encoder_state_dict"],
        "model_type": "landmark_cross_stream",
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "input_dim": 136,
    }, os.path.join(args.output, CKPT_NAME))

    json.dump({
        "model_type": "landmark_cross_stream",
        "preset": PRESET,
        "n_holdout": N_HOLDOUT,
        "best_rec_top3": best_rec_top3,
        "d_model": args.d_model,
        "num_layers_per_stream": args.num_layers,
        "target_length": args.target_length,
        "fps_norm": args.fps_norm,
        "aug": {
            "warp_prob": args.warp_prob,
            "warp_range": [args.warp_min, args.warp_max],
            "noise_std": args.noise_std,
            "flip_prob": args.flip_prob,
        },
    }, open(os.path.join(args.output, "results.json"), "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aihub-dir",    default=AIHUB_DIR)
    p.add_argument("--output",       default=OUT_DIR)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--steps",        type=int,   default=300)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--temperature",  type=float, default=0.07)
    p.add_argument("--ww-batch",     type=int,   default=64)
    p.add_argument("--val-every",    type=int,   default=5)
    p.add_argument("--patience",     type=int,   default=5,
                   help="rec LOO Top-3 미개선 평가 횟수 기준 early stop")
    p.add_argument("--rounds",       type=int,   default=5)
    p.add_argument("--val-steps",    type=int,   default=30)
    p.add_argument("--overfit-patience", type=int, default=100,
                   help="val loss 연속 상승 허용 (사실상 비활성화)")
    p.add_argument("--recorded-dir", default=RECORDED_DIR)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--dim-ff",       type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    # ── Time Warp augmentation ──
    p.add_argument("--warp-prob",    type=float, default=0.8,
                   help="Time Warp 적용 확률 (per sample)")
    p.add_argument("--warp-min",     type=float, default=0.5,
                   help="최소 속도 배율 (0.5 = 2배 느리게)")
    p.add_argument("--warp-max",     type=float, default=2.0,
                   help="최대 속도 배율 (2.0 = 2배 빠르게)")
    p.add_argument("--noise-std",    type=float, default=0.005,
                   help="Spatial Gaussian noise std")
    p.add_argument("--flip-prob",    type=float, default=0.5)
    # ── Data / normalization ──
    p.add_argument("--target-length", type=int,   default=TARGET_LENGTH,
                   help="temporal_interpolate 목표 길이 (기본=64)")
    p.add_argument("--equal-init",         action="store_true",
                   help="초기 stream_logits을 [0,0,0,0]으로 설정 (균등 0.25)")
    p.add_argument("--oversample-target",  type=int, default=1,
                   help="recorded-video 51개 단어 오버샘플링 배율 (1=없음, 2=2배)")
    p.add_argument("--use-velocity",       action="store_true",
                   help="입력에 velocity(프레임 차분) 피처 추가 → 272dim")
    p.add_argument("--fps-norm",      action="store_true",
                   help="recorded-video를 target-fps로 먼저 리샘플 후 target-length 보간")
    p.add_argument("--target-fps",    type=float, default=30.0,
                   help="fps-norm 목표 fps (기본=30.0)")
    args = p.parse_args()
    train(args)
