# DDoS Detection

ML classifier that detects DDoS attacks in network traffic using Random Forest trained on CICIDS2017.

## Project structure

```
ddos_detection.ipynb  - training notebook
api/main.py           - FastAPI server  
src/
  config.py           - paths and hyperparameters
  preprocessing.py    - data cleaning and scaling
  model.py            - training and evaluation
static/index.html     - web UI
models/               - saved model files (after training)
```

## Setup

```bash
pip install -r requirements.txt
```

Or with uv:
```bash
uv sync
```

## Training

Run `ddos_detection.ipynb` to train the model. It loads the CICIDS2017 data, trains a Random Forest with 80/20 split, and saves the model to `models/`.

## Running

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the web UI or http://localhost:8000/docs for API docs.

## API

| Route | Method | Description |
|-------|--------|-------------|
| `/predict` | POST | Classify a single flow |
| `/predict/batch` | POST | Classify multiple flows (max 1000) |
| `/api/test-samples` | GET | Get random test samples |
| `/model/info` | GET | Model metrics |
| `/health` | GET | Server status |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Flow Duration": 100000, "Total Fwd Packets": 10, ...}}'
```

The model expects 82 features. Missing ones are filled with zeros.

## Performance

On CICIDS2017 test set: ~99.9% accuracy, F1, recall.

This reflects the dataset - it's synthetic with clear separation between classes. Real-world performance will vary.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) Friday afternoon DDoS capture.