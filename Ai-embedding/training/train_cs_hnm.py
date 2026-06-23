"""
train_cs_hnm.py — CrossStreamEncoder + Hard Negative Mining.

Phase 1 대비 변경:
  - Hard Negative Mining: 10 epoch마다 전체 임베딩 재계산 → FAISS 인덱스
  - 배치 구성: random 절반 + hard-negative 절반
  - AI Hub 15fps + target=64 + fps_norm
  - 더 많은 steps (500/ep) + patience=8

얼굴 부위 / 동작 동사 혼동 해소 목표.

Usage:
    python train_cs_hnm.py
    python train_cs_hnm.py --epochs 2 --steps 10   # smoke test
"""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 model/preprocess/services
from model import LandmarkCrossStreamEncoder
from preprocess import temporal_interpolate, TARGET_LENGTH, _load_npz_raw
from services.preprocessing import convert_stream

PRESET         = "B"
AIHUB_DIR      = "dataset/word-keypoints/1.Training"
WORD_MAP_PATH  = "word_mapping.json"
DIR_WORDS_PATH = "directional_words.json"
OUT_DIR        = "result_cs_hnm"
RECORDED_DIR   = "dataset/recorded-video"
N_HOLDOUT      = 500
SEED           = 42
CKPT_NAME      = "encoder_cross_stream_best.pt"
_WORD_FILE_RE  = re.compile(r"NIA_SL_(WORD\d+)_")


# ── Augmentation ──────────────────────────────────────────────────────────────

def flip_B(seq):
    s = seq.copy()
    s[:, 0::2] *= -1
    for l, r in [(2,4),(6,8),(10,12)]:
        tmp = s[:, l:l+2].copy(); s[:, l:l+2]=s[:, r:r+2]; s[:, r:r+2]=tmp
    tmp = s[:, 14:56].copy(); s[:, 14:56]=s[:, 56:98]; s[:, 56:98]=tmp
    return s

def time_warp(seq, rate_range=(0.5, 2.0)):
    T, D = seq.shape
    rate = random.uniform(*rate_range)
    new_len = max(4, int(round(T * rate)))
    idx = np.linspace(0, T-1, new_len)
    r   = seq[np.round(idx).astype(int)]
    b   = np.linspace(0, new_len-1, T)
    return r[np.round(b).astype(int)].astype(seq.dtype)

def augment(seq, word, directional, flip_prob=0.5, warp_prob=0.8,
            warp_range=(0.5,2.0), noise_std=0.005):
    out = seq.copy()
    if word not in directional and random.random() < flip_prob:
        out = flip_B(out)
    if random.random() < warp_prob:
        out = time_warp(out, warp_range)
    if noise_std > 0:
        out = (out + np.random.normal(0, noise_std, out.shape)).astype(out.dtype)
    return out


# ── Data ──────────────────────────────────────────────────────────────────────

def load_all_segs(word_map, aihub_dir, target_length=TARGET_LENGTH):
    segs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for path in glob.glob(os.path.join(aihub_dir, "**", "*.npz"), recursive=True):
        m = _WORD_FILE_RE.search(os.path.basename(path))
        if not m: continue
        kor = word_map.get(m.group(1))
        if not kor: continue
        try:
            d = np.load(path, allow_pickle=True)
            stream = _load_npz_raw(path, PRESET)
            if "start_sec" in d and "end_sec" in d:
                fps=float(d["fps"]); T=stream.shape[0]
                sf=max(0,round(float(d["start_sec"])*fps))
                ef=min(T-1,round(float(d["end_sec"])*fps))
                crop = stream[sf:ef+1] if ef>sf+1 else stream
            else:
                crop = stream
            if len(crop) < 3: continue
            segs[kor].append(temporal_interpolate(crop, target_length).astype(np.float32))
        except: pass
    result = {w:v for w,v in segs.items() if len(v)>=2}
    print(f"[data] {len(result)} words, {sum(len(v) for v in result.values())} segs")
    return result

def load_recorded(recorded_dir, target_length=TARGET_LENGTH, fps_norm=False,
                  target_fps=15.0):
    segs = {}
    for wd in sorted(Path(recorded_dir).iterdir()):
        if not wd.is_dir(): continue
        samples = []
        for npz in sorted(wd.glob("*.npz")):
            try:
                data = np.load(str(npz))
                T = data["pose"].shape[0]
                kps = np.concatenate([data["pose"].reshape(T,-1), data["left_hand"].reshape(T,-1),
                                      data["right_hand"].reshape(T,-1), data["face"].reshape(T,-1)], axis=-1)
                seg = convert_stream(kps, target_preset=PRESET)
                if fps_norm and "fps" in data:
                    src_fps = float(data["fps"])
                    nl = max(8, int(round(len(seg)*target_fps/src_fps)))
                    seg = temporal_interpolate(seg, nl)
                samples.append(temporal_interpolate(seg, target_length).astype(np.float32))
            except: pass
        if samples: segs[wd.name] = samples
    return segs


# ── NT-Xent ───────────────────────────────────────────────────────────────────

def nt_xent(a, b, temp):
    N = a.shape[0]
    z = torch.cat([F.normalize(a,dim=1), F.normalize(b,dim=1)], dim=0)
    sim = torch.mm(z, z.T) / temp
    sim.fill_diagonal_(-1e9)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(sim.device)
    return F.cross_entropy(sim, labels)


# ── Hard Negative Mining ───────────────────────────────────────────────────────

@torch.no_grad()
def build_hnm_index(encoder, all_segs, word_list, device,
                    target_length=TARGET_LENGTH):
    """전체 단어 임베딩 → FAISS → 단어별 top-K hard negative 목록."""
    encoder.eval()
    word_embs: Dict[str, np.ndarray] = {}
    for w in word_list:
        segs = all_segs.get(w, [])
        if not segs: continue
        batch = np.stack([segs[i % len(segs)] for i in range(min(4, len(segs)))]).astype(np.float32)
        x = torch.from_numpy(batch).to(device)
        e = F.normalize(encoder(x), dim=1).cpu().numpy()
        word_embs[w] = e.mean(axis=0).astype(np.float32)

    words = list(word_embs.keys())
    mat   = np.stack([word_embs[w] for w in words]).astype(np.float32)
    faiss.normalize_L2(mat)
    idx = faiss.IndexFlatIP(mat.shape[1]); idx.add(mat)
    _, I = idx.search(mat, min(21, len(words)))   # top-20 neighbors

    hard_negs: Dict[str, List[str]] = {}
    for i, w in enumerate(words):
        hard_negs[w] = [words[j] for j in I[i, 1:21]]   # skip self
    return hard_negs


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loo(encoder, recorded, device):
    encoder.eval()
    word_embs = {}
    for word, segs in recorded.items():
        embs = []
        for s in segs:
            x = torch.from_numpy(s).unsqueeze(0).to(device)
            embs.append(F.normalize(encoder(x)[0], dim=0).cpu().numpy().astype(np.float32))
        word_embs[word] = embs

    top1=top3=total=0
    for qw, embs in word_embs.items():
        for i, qe in enumerate(embs):
            db_v, db_l = [], []
            for w, wembs in word_embs.items():
                for j, e in enumerate(wembs):
                    if w==qw and j==i: continue
                    db_v.append(e); db_l.append(w)
            if not db_v: continue
            mat = np.stack(db_v).astype(np.float32); faiss.normalize_L2(mat)
            fidx = faiss.IndexFlatIP(mat.shape[1]); fidx.add(mat)
            q = qe.reshape(1,-1).copy(); faiss.normalize_L2(q)
            _, I = fidx.search(q, min(3, len(db_l)))
            top3w = [db_l[ii] for ii in I[0]]
            if top3w[0]==qw: top1+=1
            if qw in top3w: top3+=1
            total+=1
    return {"top1":top1/total if total else 0., "top3":top3/total if total else 0., "total":total}


# ── Training ──────────────────────────────────────────────────────────────────

def split_holdout(words, n=500, seed=42):
    rng=random.Random(seed); ws=sorted(words); rng.shuffle(ws)
    return ws[:n], ws[n:]

def _pick2(segs):
    if len(segs)>=2:
        i,j=random.sample(range(len(segs)),2)
    else:
        i=j=0
    return segs[i], segs[j]

def train(args):
    os.makedirs(args.output, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    word_map = json.load(open(WORD_MAP_PATH, encoding="utf-8"))
    directional: Set[str] = set()
    if os.path.exists(DIR_WORDS_PATH):
        directional = set(json.load(open(DIR_WORDS_PATH, encoding="utf-8")))

    print("[0] recorded-video 로딩...")
    recorded = load_recorded(args.recorded_dir, target_length=args.target_length,
                              fps_norm=args.fps_norm, target_fps=args.target_fps)
    print(f"     {len(recorded)} words, {sum(len(v) for v in recorded.values())} samples")

    print("[1] AI Hub 로딩...")
    all_segs = load_all_segs(word_map, args.aihub_dir, target_length=args.target_length)
    holdout_words, train_words = split_holdout(sorted(all_segs.keys()), N_HOLDOUT)

    rng_inst = random.Random(SEED+1)
    train_segs: Dict[str, List[np.ndarray]] = {}
    val_segs:   Dict[str, List[np.ndarray]] = {}
    for w in train_words:
        if w not in all_segs: continue
        insts = all_segs[w][:]
        rng_inst.shuffle(insts)
        n_val = max(1, len(insts)//5)
        val_segs[w]   = insts[:n_val]
        train_segs[w] = insts[n_val:] if len(insts[n_val:])>=2 else insts
    holdout_segs = {w: all_segs[w] for w in holdout_words if w in all_segs}
    print(f"[split] train={sum(len(v) for v in train_segs.values())} segs")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[device] {device}")

    encoder = LandmarkCrossStreamEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        dropout=args.dropout, norm_first=True,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[model] d_model={args.d_model} L={args.num_layers} params={n_params:,}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs*args.steps)

    best_rec_top3 = -1.0
    patience_count = 0
    history = []
    train_wl = list(train_segs.keys())
    val_wl   = list(val_segs.keys())
    hard_negs: Dict[str, List[str]] = {}   # word → hard negative words
    warp_range = (args.warp_min, args.warp_max)

    print(f"\n[train] {args.epochs}ep × {args.steps}steps  "
          f"lr={args.lr}  temp={args.temperature}  hnm_every={args.hnm_every}")
    print("=" * 70)

    for ep in range(1, args.epochs+1):
        # ── HNM index refresh ────────────────────────────────────────────────
        if ep == 1 or (ep % args.hnm_every == 0):
            print(f"  [HNM] rebuilding index at ep {ep}...", flush=True)
            hard_negs = build_hnm_index(encoder, train_segs, train_wl, device,
                                         target_length=args.target_length)
            encoder.train()

        encoder.train()
        ep_loss = 0.0
        t0 = time.time()

        for _ in range(args.steps):
            optimizer.zero_grad()
            n_random = args.ww_batch // 2
            n_hard   = args.ww_batch - n_random

            # random words
            rand_words = random.sample(train_wl, min(n_random, len(train_wl)))
            # hard negatives: pick words that are hard negatives of random anchors
            hard_words = []
            if hard_negs:
                for w in rand_words[:n_hard]:
                    neg_pool = hard_negs.get(w, [])
                    if neg_pool:
                        hw = random.choice(neg_pool[:10])
                        if hw in train_segs: hard_words.append(hw)
            # combine & deduplicate
            ww_words = list(dict.fromkeys(rand_words + hard_words))[:args.ww_batch]

            aa, bb = [], []
            for w in ww_words:
                a, b = _pick2(train_segs[w])
                aa.append(augment(a, w, directional, args.flip_prob,
                                  args.warp_prob, warp_range, args.noise_std))
                bb.append(augment(b, w, directional, args.flip_prob,
                                  args.warp_prob, warp_range, args.noise_std))

            ta = torch.from_numpy(np.stack(aa)).float().to(device)
            tb = torch.from_numpy(np.stack(bb)).float().to(device)
            loss = nt_xent(encoder(ta), encoder(tb), args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item()

        # val loss
        encoder.eval()
        with torch.no_grad():
            vl = 0.
            for _ in range(args.val_steps):
                ww = random.sample(val_wl, min(args.ww_batch, len(val_wl)))
                av, bv = [], []
                for w in ww:
                    a, b = _pick2(val_segs[w]); av.append(a); bv.append(b)
                ta = torch.from_numpy(np.stack(av)).float().to(device)
                tb = torch.from_numpy(np.stack(bv)).float().to(device)
                vl += nt_xent(encoder(ta), encoder(tb), args.temperature).item()
            val_loss = vl / args.val_steps

        elapsed = time.time() - t0
        w_str = ""
        if ep % 10 == 0:
            sw = F.softmax(encoder.stream_logits, dim=0).detach().cpu().numpy()
            w_str = f"  [w] pose={sw[0]:.3f} L={sw[1]:.3f} R={sw[2]:.3f} f={sw[3]:.3f}"
        print(f"Ep {ep:03d}/{args.epochs}  "
              f"train={ep_loss/args.steps:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s{w_str}", flush=True)

        if ep % args.val_every == 0:
            ext = eval_loo(encoder, recorded, device)
            marker = ""
            if ext["top3"] > best_rec_top3:
                best_rec_top3 = ext["top3"]
                torch.save({k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                           os.path.join(args.output, CKPT_NAME))
                patience_count = 0; marker = " ★"
            else:
                patience_count += 1
            print(f"  [LOO] Top-1:{ext['top1']:.1%}  Top-3:{ext['top3']:.1%}"
                  f"  (n={ext['total']}){marker}", flush=True)
            history.append({"ep":ep, "train":round(ep_loss/args.steps,4),
                            "val":round(val_loss,4), "rec_top3":round(ext["top3"],4)})
            json.dump(history, open(os.path.join(args.output,"history.json"),"w"), indent=2)
            if patience_count >= args.patience:
                print(f"[early stop] patience={args.patience} at ep {ep}"); break

    print("=" * 70)
    print(f"[done] best LOO Top-3 = {best_rec_top3:.4f}")
    json.dump({"best_rec_top3": best_rec_top3},
              open(os.path.join(args.output, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aihub-dir",    default=AIHUB_DIR)
    p.add_argument("--output",       default=OUT_DIR)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--steps",        type=int,   default=500)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--temperature",  type=float, default=0.07)
    p.add_argument("--ww-batch",     type=int,   default=64)
    p.add_argument("--val-every",    type=int,   default=5)
    p.add_argument("--patience",     type=int,   default=8)
    p.add_argument("--val-steps",    type=int,   default=30)
    p.add_argument("--hnm-every",    type=int,   default=5,
                   help="HNM 인덱스 재구축 주기 (epoch)")
    p.add_argument("--recorded-dir", default=RECORDED_DIR)
    p.add_argument("--target-length",type=int,   default=TARGET_LENGTH)
    p.add_argument("--fps-norm",     action="store_true")
    p.add_argument("--target-fps",   type=float, default=15.0)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--dim-ff",       type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--wd",           type=float, default=1e-4)
    p.add_argument("--warp-prob",    type=float, default=0.8)
    p.add_argument("--warp-min",     type=float, default=0.5)
    p.add_argument("--warp-max",     type=float, default=2.0)
    p.add_argument("--noise-std",    type=float, default=0.005)
    p.add_argument("--flip-prob",    type=float, default=0.5)
    args = p.parse_args()
    train(args)
