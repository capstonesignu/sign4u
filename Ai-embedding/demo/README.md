# KSL Sign Language Recognition Demo

## Quick Start

### 1. Build FAISS Index (one-time)

```bash
cd demo/fastapi-server
pip install -r requirements.txt
python build_index.py
```

This loads the trained model and aihub dataset, embeds all samples, and saves the FAISS index to `demo/fastapi-server/index/demo_kg.faiss`.

### 2. Start FastAPI Server

```bash
cd demo/fastapi-server
cp .env.example .env   # edit if needed
python main.py
# -> http://localhost:8000
```

### 3. Start Express Server

```bash
cd demo/express-server
npm install
npm start
# -> http://localhost:3000
```

### 4. Open Demo

Open http://localhost:3000 in a browser with camera access.

1. Click **Record** and perform a sign
2. Click **Stop** when done
3. View Top-1 / Top-5 / Top-10 results

## Configuration

Edit `demo/fastapi-server/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_PATH` | `result_baseline/.../best_model.pt` | Trained model weights |
| `FEATURE_PRESET` | `B` | A=210dim/3axes, B=140dim/2axes, C=102dim/2axes |
| `SEQUENCE_LENGTH` | `128` | Temporal interpolation target frames |
| `FAISS_INDEX_PATH` | `./index/demo_kg.faiss` | Pre-built FAISS index |
| `DATASET_PATH` | `../../dataset/aihub` | AI Hub npz dataset root |
| `WORD_MAPPING_PATH` | `../../word_mapping.json` | WORD ID -> Korean mapping |

## Architecture

```
Browser (MediaPipe) -> Express (:3000) -> FastAPI (:8000)
                       static files       model inference
                       API proxy          FAISS search
```

Frontend extracts 70 landmarks x 3 axes = 210 dim per frame.
Server converts to model's expected format (e.g., preset B = 140 dim).
