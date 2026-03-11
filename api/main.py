"""
API server for the DDoS detection model.

This exposes our trained classifier via REST endpoints so other services
can query it in real-time. Runs on FastAPI for async support.

Main routes:
    POST /predict       - single flow classification
    POST /predict/batch - bulk classification (max 1000)
    GET  /health        - liveness check
    GET  /model/info    - model metadata & metrics
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import DDoSDetector
from src.preprocessing import DataPreprocessor

# spin up the app
app = FastAPI(
    title="DDoS Detection API",
    description="""
    REST API for classifying network flows as benign or DDoS.
    
    Uses a Random Forest trained on the CICIDS2017 Friday afternoon capture.
    
    **What you get:**
    - Single-sample and batch prediction
    - Probability scores for both classes
    - Model info endpoint with training stats
    
    **Note:** We optimized for recall on attacks - better to flag a
    few false positives than miss real attacks.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# allow requests from anywhere (fine for dev, lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# these get loaded at startup - None until then
model: Optional[DDoSDetector] = None
preprocessor: Optional[DataPreprocessor] = None

# Mount static files
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# -- request/response schemas --
class NetworkFlowFeatures(BaseModel):
    """All the flow-level features we can accept. Most have defaults so you only need the important ones."""

    # core features (required)
    flow_duration: float = Field(
        ...,
        alias="Flow Duration",
        description="how long the flow lasted (microseconds)",
    )
    total_fwd_packets: float = Field(
        ..., alias="Total Fwd Packets", description="packets going client -> server"
    )
    total_backward_packets: float = Field(
        ...,
        alias="Total Backward Packets",
        description="packets coming back server -> client",
    )
    total_length_fwd_packets: float = Field(
        ...,
        alias="Total Length of Fwd Packets",
        description="total bytes in forward packets",
    )
    total_length_bwd_packets: float = Field(
        ...,
        alias="Total Length of Bwd Packets",
        description="total bytes in backward packets",
    )

    # optional stuff - zeros are fine if you don't have these
    fwd_packet_length_max: float = Field(0, alias="Fwd Packet Length Max")
    fwd_packet_length_min: float = Field(0, alias="Fwd Packet Length Min")
    fwd_packet_length_mean: float = Field(0, alias="Fwd Packet Length Mean")
    fwd_packet_length_std: float = Field(0, alias="Fwd Packet Length Std")
    bwd_packet_length_max: float = Field(0, alias="Bwd Packet Length Max")
    bwd_packet_length_min: float = Field(0, alias="Bwd Packet Length Min")
    bwd_packet_length_mean: float = Field(0, alias="Bwd Packet Length Mean")
    bwd_packet_length_std: float = Field(0, alias="Bwd Packet Length Std")
    flow_bytes_per_s: float = Field(0, alias="Flow Bytes/s")
    flow_packets_per_s: float = Field(0, alias="Flow Packets/s")
    flow_iat_mean: float = Field(0, alias="Flow IAT Mean")
    flow_iat_std: float = Field(0, alias="Flow IAT Std")
    flow_iat_max: float = Field(0, alias="Flow IAT Max")
    flow_iat_min: float = Field(0, alias="Flow IAT Min")
    fwd_iat_total: float = Field(0, alias="Fwd IAT Total")
    fwd_iat_mean: float = Field(0, alias="Fwd IAT Mean")
    fwd_iat_std: float = Field(0, alias="Fwd IAT Std")
    fwd_iat_max: float = Field(0, alias="Fwd IAT Max")
    fwd_iat_min: float = Field(0, alias="Fwd IAT Min")
    bwd_iat_total: float = Field(0, alias="Bwd IAT Total")
    bwd_iat_mean: float = Field(0, alias="Bwd IAT Mean")
    bwd_iat_std: float = Field(0, alias="Bwd IAT Std")
    bwd_iat_max: float = Field(0, alias="Bwd IAT Max")
    bwd_iat_min: float = Field(0, alias="Bwd IAT Min")

    class Config:
        populate_by_name = True


class PredictionInput(BaseModel):
    """Wrapper for a single prediction request. Just pass your features as a dict."""

    features: Dict[str, float] = Field(
        ..., description="Dictionary of feature names to values"
    )


class PredictionResponse(BaseModel):
    """What you get back after classifying a flow."""

    prediction: int = Field(..., description="0 for BENIGN, 1 for DDoS")
    label: str = Field(..., description="Human-readable label")
    probability_benign: float = Field(..., description="Probability of BENIGN class")
    probability_ddos: float = Field(..., description="Probability of DDoS class")
    confidence: float = Field(..., description="Confidence of the prediction")
    is_attack: bool = Field(
        ..., description="Whether the traffic is classified as an attack"
    )


class BatchPredictionInput(BaseModel):
    """For when you want to classify a bunch of flows at once."""

    samples: List[Dict[str, float]] = Field(
        ..., description="List of feature dictionaries"
    )


class BatchPredictionResponse(BaseModel):
    """Batch results with summary counts."""

    predictions: List[PredictionResponse]
    total_samples: int
    attack_count: int
    benign_count: int


class ModelInfo(BaseModel):
    """Metadata about the loaded model."""

    model_type: str
    is_loaded: bool
    feature_count: int
    training_metrics: Dict[str, float]
    feature_names: List[str]


class HealthResponse(BaseModel):
    """Simple health status."""

    status: str
    model_loaded: bool
    preprocessor_loaded: bool


@app.on_event("startup")
async def load_model():
    """Try to load the saved model when the server starts."""
    global model, preprocessor

    models_dir = PROJECT_ROOT / "models"

    try:
        model = DDoSDetector.load(models_dir / "ddos_model.pkl")
        preprocessor = DataPreprocessor.load(
            models_dir / "scaler.pkl",
            models_dir / "feature_names.pkl",
            models_dir / "preprocessing_config.pkl",
        )
        print("✅ Model and preprocessor loaded successfully!")
    except FileNotFoundError as e:
        print(f"⚠️ Model files not found: {e}")
        print("Run the notebook first to train and save the model.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Serves the web UI if it exists, otherwise just returns API info."""
    static_dir = PROJECT_ROOT / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "name": "DDoS Attack Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api", tags=["Root"])
async def api_info():
    """Basic info about this API."""
    return {
        "name": "DDoS Attack Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Quick check to see if things are running ok."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Returns model type, feature count, and training metrics."""
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running the notebook.",
        )

    return ModelInfo(
        model_type=model.model_type,
        is_loaded=True,
        feature_count=len(preprocessor.feature_names),
        training_metrics=model.training_metrics,
        feature_names=preprocessor.feature_names,
    )


@app.get("/model/features", tags=["Model"])
async def get_feature_names():
    """Lists all the features the model expects (useful for debugging)."""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "feature_names": preprocessor.feature_names,
        "count": len(preprocessor.feature_names),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Classify one network flow.

    Pass in a dict of features - you don't need all of them,
    we'll fill in zeros for anything missing.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    try:
        # Fill missing features with 0
        features = {
            name: input_data.features.get(name, 0)
            for name in preprocessor.feature_names
        }

        # Transform and predict
        X = preprocessor.transform_single(features)
        result = model.predict_single(X)

        return PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            probability_benign=result["probability_benign"],
            probability_ddos=result["probability_ddos"],
            confidence=result["confidence"],
            is_attack=result["prediction"] == 1,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Classify multiple flows in one call.

    Capped at 1000 samples per request - if you need more,
    split it up.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    if len(input_data.samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")

    if len(input_data.samples) > 1000:
        raise HTTPException(
            status_code=400, detail="Maximum batch size is 1000 samples"
        )

    try:
        predictions = []
        attack_count = 0

        for sample in input_data.samples:
            # Fill missing features
            features = {
                name: sample.get(name, 0) for name in preprocessor.feature_names
            }

            X = preprocessor.transform_single(features)
            result = model.predict_single(X)

            pred = PredictionResponse(
                prediction=result["prediction"],
                label=result["label"],
                probability_benign=result["probability_benign"],
                probability_ddos=result["probability_ddos"],
                confidence=result["confidence"],
                is_attack=result["prediction"] == 1,
            )
            predictions.append(pred)

            if result["prediction"] == 1:
                attack_count += 1

        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            attack_count=attack_count,
            benign_count=len(predictions) - attack_count,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict/raw", tags=["Prediction"])
async def predict_raw(features: List[float]):
    """
    Direct inference with pre-scaled features.

    This skips the preprocessing step - only use this if you're
    doing the scaling yourself. Features must be in the exact order
    from /model/features.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    expected_count = len(preprocessor.feature_names)
    if len(features) != expected_count:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_count} features, got {len(features)}",
        )

    try:
        X = np.array(features).reshape(1, -1)
        result = model.predict_single(X)

        return {
            "prediction": result["prediction"],
            "label": result["label"],
            "probability_benign": result["probability_benign"],
            "probability_ddos": result["probability_ddos"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# handy for testing
@app.get("/example", tags=["Examples"])
async def get_example_request():
    """Generates a sample request body you can copy-paste into /predict."""
    if preprocessor is None:
        return {
            "message": "Model not loaded. Example uses placeholder features.",
            "example_request": {
                "features": {
                    "Flow Duration": 100000,
                    "Total Fwd Packets": 10,
                    "Total Backward Packets": 5,
                    "Total Length of Fwd Packets": 1000,
                    "Total Length of Bwd Packets": 500,
                }
            },
        }

    # Create example with all features
    example_features = {name: 0.0 for name in preprocessor.feature_names}
    example_features["Flow Duration"] = 100000
    example_features["Total Fwd Packets"] = 10
    example_features["Total Backward Packets"] = 5

    return {
        "example_request": {"features": example_features},
        "all_features": preprocessor.feature_names,
    }


@app.get("/api/test-samples", tags=["Examples"])
async def get_test_samples(
    count: int = Query(
        default=5, ge=1, le=20, description="Number of test samples to return"
    ),
    include_benign: bool = Query(default=True, description="Include benign samples"),
    include_ddos: bool = Query(default=True, description="Include DDoS samples"),
):
    """
    Generate random test samples from the actual test dataset.

    Returns real samples that the model hasn't been trained on,
    with all 82 features included. Perfect for copy-paste testing in the UI.
    """
    if preprocessor is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    # Load test data from saved file or regenerate from CSV
    csv_path = PROJECT_ROOT / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found. Cannot generate test samples.",
        )

    try:
        # Load data
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Clean up labels
        df[" Label"] = (
            df[" Label"].str.strip()
            if " Label" in df.columns
            else df["Label"].str.strip()
        )
        label_col = " Label" if " Label" in df.columns else "Label"

        # Filter by requested labels
        labels_to_include = []
        if include_benign:
            labels_to_include.append("BENIGN")
        if include_ddos:
            labels_to_include.extend(["DDoS"])

        if not labels_to_include:
            raise HTTPException(
                status_code=400, detail="Must include at least one class"
            )

        df_filtered = df[df[label_col].isin(labels_to_include)]

        # Sample random rows
        n_samples = min(count, len(df_filtered))
        samples_df = df_filtered.sample(
            n=n_samples, random_state=None
        )  # None = truly random

        # Build response with features that match model's expected features
        test_samples = []
        for idx, (_, row) in enumerate(samples_df.iterrows()):
            actual_label = row[label_col]

            # Get only the features the model expects
            features = {}
            for feature_name in preprocessor.feature_names:
                if feature_name in row.index:
                    value = row[feature_name]
                    # Handle inf/nan
                    if pd.isna(value) or np.isinf(value):
                        value = 0.0
                    features[feature_name] = float(value)
                else:
                    features[feature_name] = 0.0

            test_samples.append(
                {
                    "sample_id": idx + 1,
                    "expected_label": actual_label,
                    "features": features,
                }
            )

        return {
            "count": len(test_samples),
            "feature_count": len(preprocessor.feature_names),
            "samples": test_samples,
            "usage_hint": "Copy the 'features' object from any sample and paste it in the JSON input field",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating test samples: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
