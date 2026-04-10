# =============================================================
# LAB 6 — backend-iris/main.py
# API FastAPI con health check robusto y carga asíncrona del modelo
# =============================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import threading
import os
from typing import Optional

# ── Inicialización de la aplicación ──────────────────────────
app = FastAPI(
    title="Iris Prediction API",
    description="Backend MLOps LAB 6 — modelo Iris cargado desde Hugging Face Hub",
    version="1.0.0",
)

# ── Estado global del modelo ─────────────────────────────────
# Se utiliza un diccionario de estado para evitar variables globales mutables.
model_state = {
    "loaded": False,
    "model": None,
    "version": None,
    "start_time": time.time(),
    "load_error": None,
}

# ── Carga asíncrona del modelo ───────────────────────────────
def load_model_background():
    """
    Carga el modelo desde Hugging Face Hub en un hilo secundario.
    De esta forma el proceso de uvicorn arranca inmediatamente y el
    health check puede responder HTTP 503 mientras el modelo se carga,
    en vez de bloquear el arranque del contenedor indefinidamente.
    """
    try:
        from huggingface_hub import hf_hub_download
        import joblib

        hf_token = os.getenv("HF_TOKEN") # Only needed if the repo is private
        repo_id  = os.getenv("HF_REPO_ID", "brjapon/iris-dt")
        filename = os.getenv("HF_MODEL_FILE", "iris_dt.joblib")

        print(f"[INFO] Descargando modelo '{filename}' desde '{repo_id}'...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
        )

        model = joblib.load(model_path)

        # Escritura atómica del estado — Python GIL garantiza que las
        # asignaciones de dict son visibles de inmediato en otros hilos.
        model_state["model"]   = model
        model_state["version"] = os.getenv("MODEL_VERSION", "v1.0-base")
        model_state["loaded"]  = True
        print("[INFO] Modelo cargado correctamente.")

    except Exception as exc:
        model_state["load_error"] = str(exc)
        print(f"[ERROR] Fallo al cargar el modelo: {exc}")


# Lanzar la carga en background al iniciar la aplicación
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()


# ── Schemas de entrada / salida ──────────────────────────────
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionOutput(BaseModel):
    prediction: int
    species: str
    confidence: Optional[float] = None


SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check robusto:
    - HTTP 200  → modelo cargado y listo para inferencia.
    - HTTP 503  → modelo aún cargándose (el contenedor arrancó pero el
                  modelo no está en memoria todavía).

    Docker HEALTHCHECK llama a este endpoint para decidir si el contenedor
    está "healthy". El frontend sólo arrancará cuando este endpoint devuelva 200.
    """
    uptime = round(time.time() - model_state["start_time"], 1)

    if model_state["loaded"]:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": model_state["version"],
            "uptime_seconds": uptime,
        }

    # Si hubo un error irrecuperable en la carga, lo reportamos
    if model_state["load_error"]:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "model_loaded": False,
                "message": f"Error al cargar el modelo: {model_state['load_error']}",
                "uptime_seconds": uptime,
            },
        )

    # Modelo aún cargándose normalmente
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "model_loaded": False,
            "message": "Cargando modelo desde Hugging Face Hub...",
            "uptime_seconds": uptime,
        },
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    """
    Realiza una predicción de especie de iris.
    Devuelve HTTP 503 si el modelo todavía no está listo.
    """
    if not model_state["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no está disponible. Reintenta en unos segundos.",
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = int(model_state["model"].predict(features)[0])

    # Confianza (probabilidad máxima) si el modelo la soporta
    confidence = None
    if hasattr(model_state["model"], "predict_proba"):
        proba = model_state["model"].predict_proba(features)[0]
        confidence = round(float(max(proba)), 4)

    return PredictionOutput(
        prediction=prediction,
        species=SPECIES_MAP.get(prediction, "unknown"),
        confidence=confidence,
    )


@app.get("/")
def root():
    return {"message": "Iris Prediction API — LAB 6 MLOps & AI BI · Loyola"}
