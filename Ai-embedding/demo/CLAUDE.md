# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This `demo/` directory is the **live web demo** for the KSL translation system: a browser camera UI that streams MediaPipe keypoints to a FastAPI inference server and shows recognized signs. The embedding-training subproject is `../` (see `../CLAUDE.md`); the full pipeline overview is `../../CLAUDE.md`. This file covers only the demo's two servers and the JS↔Python contract between them.

Two processes:
- **express-server** (Node, port 3000) — serves static frontend, proxies `/api/*` to FastAPI, serves reference sign videos.
- **fastapi-server** (Python, port **8001**) — loads the encoder + FAISS index + segmenter, runs inference.

## Running the demo

```bash
# 1. FastAPI inference server (port from .env, currently 8001)
cd demo/fastapi-server
pip install -r requirements.txt
python main.py                      # uvicorn with reload=True

# 2. Express frontend + proxy (separate terminal)
cd demo/express-server
npm install
npm start                           # http://localhost:3000
```

Open `http://localhost:3000` in a browser with camera access. The FAISS index must already exist (see below) or FastAPI startup raises `FileNotFoundError`.

### README is stale — trust `.env` + `config.py`, not `README.md`

`README.md` documents an older design (port 8000, `build_index.py`, `demo_kg.faiss`, `EMBEDDING_MODEL_PATH`, single-word `/api/recognize`). The **current** system is configured entirely through `fastapi-server/.env` consumed by `config.py`. As checked in, `.env` runs:
- `PORT=8001` (express `server.js` defaults `FASTAPI_URL` to `http://localhost:8001` to match — keep these in sync)
- `EXP026_ENCODER_PATH` → cross-stream encoder under `../../result_cross_stream_encoder/`
- `EXP026_FAISS_PATH` / `_LABELS_PATH` → `index/cross_stream_recorded.faiss(.labels.json)`

When changing the model or index, edit `.env`, not code defaults.

## Inference architecture

```
Browser (MediaPipe Holistic, Tasks API)
  │  extract 70 landmarks × 3 axes = 210 dim/frame  (mediapipe.js)
  │  preprocess to preset B = 136 dim/frame          (preprocessing.js)
  ▼
Express :3000  ──proxy /api/*──►  FastAPI :8001
                                    │
                                    ├─ EmbeddingServiceExp026  (services/embedding.py)
                                    │     interpolate segment → 64 frames → encoder → L2-norm 256-d
                                    ├─ VectorDBService          (services/vectordb.py)  FAISS inner-product top-k, dedup-by-word
                                    └─ segmenter (BiLSTM or velocity-pause)
```

### The 210→136 contract (most important thing to get right)

The frontend extracts **70 landmarks × 3 axes = 210 dims/frame**, ordered `[pose 9, left_hand 21, right_hand 21, face 19]` (`mediapipe.js`: `POSE_INDICES`, `FACE_INDICES` pick these subsets from MediaPipe's 33/478). Pose carries 9 landmarks but only **7 are used** as features (hips dropped).

`preprocessing.js` (browser) and `services/preprocessing.py:convert_stream` (server) implement the **same** preset-B pipeline and must stay byte-compatible:
1. interpolate all-zero hand frames,
2. per-landmark pose interpolation (browser MediaPipe drops wrists etc.; training data rarely does),
3. shoulder-midpoint centering + shoulder-width scaling,
4. preset B = xy axes only, pose 7 + hands 21×2 + face 19 → **136 dim**.

The server accepts **either** shape: `(N, 210)` raw (server runs `convert_stream`) **or** `(N, 136)` already-preprocessed (browser did it). Both `/api/recognize` and `/api/recognize/sentence` branch on `kp.shape[1]`. If you change preprocessing, change **both** files or recognition silently degrades.

Feature presets (`preprocessing.py:FEATURE_PRESETS`): A=210 (3 axes), B=136 (xy), C=102 (xy, no face). Demo uses B.

### Encoder auto-detection

`EmbeddingServiceExp026.__init__` inspects the checkpoint to pick the architecture — there's no config flag. It reads `ckpt["model_type"]` and probes state-dict keys to choose among `LandmarkCrossStreamEncoder` / `LandmarkStreamEncoder` (input_dim 136, `_is_stream=True`, called as `encoder(x)`) vs `VelocityGatedSignEncoder` / `SignEncoder` (input_dim from `input_proj.weight`, called with `src_key_padding_mask=None`). `num_layers`, `d_model`, `input_dim` are inferred from the state dict when not stored in the ckpt. All these encoder classes are imported from the **training repo** `../../model.py` (`sys.path` is extended to `PROJECT_ROOT = demo/../../`). The demo depends on `../../model.py`, `../../preprocess.py` (for `temporal_interpolate`, `feature_dim_for`), and `../../seg_model.py` (`BiLSTMSegmenter`) — changes to those upstream files can break demo loading.

`embed_varlen` always interpolates any segment to **TARGET_LENGTH=64** before encoding, because the FAISS index was built at 64 frames. Query and DB frame-length must match.

### Two recognition endpoints + segmentation

- `POST /api/recognize` — single segment, whole sequence = one word, returns top-10.
- `POST /api/recognize/sentence` — the path the demo actually uses (`app.js` always calls this). Request: `{keypoints, word_mode, boundaries}`.
  - `word_mode=true` → skip segmentation, treat whole clip as one word.
  - `boundaries` non-empty → use **client-detected** word boundaries (frame indices). `app.js` detects word segments by hand-visibility gaps and sends boundary frame indices — this sidesteps server/client velocity-scale mismatch and is the normal flow.
  - else → server-side `_velocity_segment`: L2 velocity over the first **98 dims** (pose+hands, face skipped as noise), 3-frame smoothing, a pause ≥ `VELOCITY_PAUSE_FRAMES` low-velocity frames marks a boundary at its midpoint. Thresholds in `config.py` (`VELOCITY_THRESHOLD`, `VELOCITY_PAUSE_FRAMES`).

The BiLSTM `SegmenterService` (`SEGMENTER_MODEL_PATH`) is loaded if present but the current recognize router uses velocity/client boundaries; missing segmenter model is non-fatal (startup catches `FileNotFoundError`).

### FAISS search semantics

`VectorDBService.search` over-fetches `top_k*10`, then keeps the **best score per word** (multiple vectors per word are common — each recorded sample + its horizontal flip), so top-k means k distinct words. Index files live in `index/*.faiss` with a sibling `*.labels.json` (parallel list of word strings, index-aligned to FAISS row). There are ~20 prebuilt indices for different encoders/experiments; the active pair is whichever `.env` points at.

## Building a FAISS index

Each `build_*.py` pairs a specific encoder + dataset → an index. They share the pattern: load encoder from `../../result_*/...pt`, walk a dataset dir, embed each sample (often original + horizontal flip via `flip_horizontal_B`), write `index/<name>.faiss` + `.labels.json`.

- `build_recorded_index.py` — from `../../dataset/recorded-video/<label>/*.npz` using the 30fps-fixed baseline encoder (the live-recorded-words index).
- `build_cam_index.py`, `build_finetuned_index.py`, `build_exp026_index.py`, `build_index.py` — other encoder/dataset combinations.

After building, point `.env` `EXP026_FAISS_PATH`/`_LABELS_PATH` at the new files and restart FastAPI. **The encoder used to build the index must match `EXP026_ENCODER_PATH`** — mismatched embedding spaces give garbage similarity. (See `../memory/...` index-naming note: include the strategy/encoder name in the filename to avoid collisions.)

Horizontal flip (`flip_horizontal_B`) is the **only** sanctioned augmentation here — it negates x and swaps left/right hand+pose+elbow blocks. This is for left/right-handed signer coverage in the DB; do **not** confuse it with the parent project's "No Mirror augmentation" rule for *training* (flip changes meaning there). Here it's a DB-population trick keyed to the recorded-words demo.

## Recording new DB words

The demo can capture new signs from the browser:
- `POST /api/record/save` (`record.js`) — buffers raw 210-dim frames, returns an `.npz` download (split into pose/left_hand/right_hand/face), also saved to `../../dataset/recordings/`.
- `POST /api/db/add` (`db_record.js`) — the DB-recording tab with video+keypoint sync/crop UI; saves cropped raw keypoints as `.npz` to `../../dataset/recorded-video/<label>/`. **Note: it does not live-add to FAISS** — it only persists the npz. Rebuild the index (`build_recorded_index.py`) to make new words searchable. `VectorDBService.add`/`save` exist but aren't wired to this endpoint.
- `GET /api/db/words`, `/api/db/info` — DB stats (FAISS vector counts + recorded-npz counts per word).

## Frontend layout

Plain JS, no build step. `index.html` loads scripts in order with `?v=N` cache-busting (bump `v` when editing JS so browsers reload). Key files:
- `mediapipe.js` — HolisticLandmarker (Tasks API), landmark subset selection, `extractKeypoints` → 210-dim.
- `preprocessing.js` — 210→136 preset-B conversion mirroring `services/preprocessing.py`.
- `app.js` — capture loop, auto-trigger (hand dwells in a screen zone to start/stop) vs manual Record/Stop, hand-visibility word segmentation, calls `/api/recognize/sentence`.
- `interpolation.js` — `temporalInterpolate` mirroring `preprocess.py` nearest-neighbor resample.
- `ui.js` — renders sentence / top-1 / top-5 / top-10 result panels.
- `reference.js` — searchable word list + plays reference videos served from `../../dataset/WORD-video` (express `/ref/words`, `/ref-videos`).
- `db_record.js` — DB-recording tab with MediaRecorder video↔keypoint sync.

Express also serves `../../sentence_mapping.json` (`/ref/sentences`) and `../../word_mapping.json` (WORD-id → Korean).

## CPU-only / macOS notes

`main.py` forces CPU: sets `CUDA_VISIBLE_DEVICES=""`, `PYTORCH_MPS_DISABLE=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1` **before** importing torch — matching the parent project's torch/faiss OpenMP-clash workaround. Keep these at the top of `main.py`; FAISS here is CPU (`faiss-cpu`).
