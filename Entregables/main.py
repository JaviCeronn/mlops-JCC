"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
Extends app-iris with:
  - POST /train      → reentrenamiento incremental con nuevas muestras
  - GET  /model/info → versión activa, métricas, historial
  - POST /predict    → inferencia (igual que app-iris, con versión activa)
  - GET  /health     → estado del servicio
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class IrisSample(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sépalo (cm)")
    sepal_width: float  = Field(..., example=3.5, description="Anchura del sépalo (cm)")
    petal_length: float = Field(..., example=1.4, description="Longitud del pétalo (cm)")
    petal_width: float  = Field(..., example=0.2, description="Anchura del pétalo (cm)")


class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0,
                       description="0=setosa, 1=versicolor, 2=virginica")


class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(
        ..., min_items=5,
        description="Nuevas muestras etiquetadas para reentrenamiento (mínimo 5)"
    )
    retrain_from_scratch: bool = Field(
        False,
        description="Si True, ignora datos anteriores y entrena solo con las muestras enviadas"
    )
    policy: str = Field(
        "any_improvement",
        description=(
            "Política de activación del modelo. "
            "'any_improvement': activa si accuracy >= anterior (comportamiento por defecto). "
            "'min_delta': activa solo si mejora al menos `min_delta` (p.ej. 0.02 = 2%). "
            "'per_class_f1': activa si la F1 de una clase específica mejora."
        )
    )
    policy_params: dict = Field(
        default_factory=dict,
        description=(
            "Parámetros de la política seleccionada. "
            "min_delta → {'min_delta': 0.02}. "
            "per_class_f1 → {'target_class': 2, 'min_f1_delta': 0.0}."
        )
    )


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    message: str
    policy_used: str
    activation_reason: str


class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]


# ---------------------------------------------------------------------------
# Políticas de activación (patrón Strategy)
# ---------------------------------------------------------------------------

class ActivationPolicy:
    """Clase base abstracta para las políticas de activación del modelo."""

    def should_activate(self, eval_result: dict, previous_meta: Optional[dict]) -> tuple:
        """
        Decide si el nuevo modelo debe activarse.

        Args:
            eval_result   : métricas del nuevo modelo
                            {"accuracy": float, "per_class_f1": {int: float}}
            previous_meta : último registro del historial (None si no hay modelo previo)

        Returns:
            (activar: bool, razon: str)
        """
        raise NotImplementedError


class AnyImprovementPolicy(ActivationPolicy):
    """
    Activa el modelo si su accuracy >= al anterior.
    Equivale al gate original. Ideal cuando el coste de error es bajo
    (p. ej. recomendación de películas).
    """

    def should_activate(self, eval_result: dict, previous_meta: Optional[dict]) -> tuple:
        if previous_meta is None:
            return True, "Primer modelo; no existe referencia previa."

        acc_new  = eval_result["accuracy"]
        acc_prev = previous_meta["accuracy"]

        if acc_new >= acc_prev:
            return True,  f"any_improvement: {acc_new:.4f} >= {acc_prev:.4f}"
        return False, f"any_improvement: {acc_new:.4f} < {acc_prev:.4f}; modelo rechazado."


class MinDeltaPolicy(ActivationPolicy):
    """
    Activa el modelo solo si mejora al menos `min_delta` puntos sobre el anterior.
    Útil cuando la estabilidad es tan importante como la precisión
    (p. ej. diagnóstico médico).

    Parámetros:
        min_delta (float): mejora mínima requerida (default 0.02 = 2 %).
    """

    def __init__(self, min_delta: float = 0.02):
        if min_delta < 0:
            raise ValueError("min_delta debe ser >= 0")
        self.min_delta = min_delta

    def should_activate(self, eval_result: dict, previous_meta: Optional[dict]) -> tuple:
        if previous_meta is None:
            return True, "Primer modelo; no existe referencia previa."

        acc_new   = eval_result["accuracy"]
        acc_prev  = previous_meta["accuracy"]
        threshold = round(acc_prev + self.min_delta, 6)

        if acc_new >= threshold:
            return True,  (f"min_delta(Δ={self.min_delta}): "
                           f"{acc_new:.4f} >= {threshold:.4f}")
        return False, (f"min_delta(Δ={self.min_delta}): "
                       f"{acc_new:.4f} < {threshold:.4f}; mejora insuficiente.")


class PerClassF1Policy(ActivationPolicy):
    """
    Activa el modelo si mejora la F1-score de una clase específica,
    independientemente del accuracy global.
    Esencial cuando los errores en una clase concreta son muy costosos
    (p. ej. detección de fraude bancario).

    Parámetros:
        target_class  (int)   : índice de la clase crítica (0, 1 o 2 en Iris).
        min_f1_delta  (float) : mejora mínima de F1 requerida (default 0.0).
    """

    def __init__(self, target_class: int, min_f1_delta: float = 0.0):
        if target_class < 0:
            raise ValueError("target_class debe ser >= 0")
        if min_f1_delta < 0:
            raise ValueError("min_f1_delta debe ser >= 0")
        self.target_class = target_class
        self.min_f1_delta = min_f1_delta

    def should_activate(self, eval_result: dict, previous_meta: Optional[dict]) -> tuple:
        if previous_meta is None:
            return True, "Primer modelo; no existe referencia previa."

        pcf1 = eval_result.get("per_class_f1")
        if pcf1 is None:
            raise ValueError("La política per_class_f1 requiere métricas de F1 por clase.")

        f1_new  = pcf1.get(self.target_class)
        if f1_new is None:
            raise ValueError(f"No existe F1 para la clase {self.target_class} en los resultados.")

        prev_pcf1 = previous_meta.get("per_class_f1", {})
        f1_prev   = prev_pcf1.get(str(self.target_class))  # el historial serializa como str

        if f1_prev is None:
            return True, (f"per_class_f1(clase={self.target_class}): "
                          "sin F1 de referencia previa para esta clase.")

        threshold = round(f1_prev + self.min_f1_delta, 6)

        if f1_new >= threshold:
            return True,  (f"per_class_f1(clase={self.target_class}, Δ={self.min_f1_delta}): "
                           f"F1 {f1_new:.4f} >= {threshold:.4f}")
        return False, (f"per_class_f1(clase={self.target_class}, Δ={self.min_f1_delta}): "
                       f"F1 {f1_new:.4f} < {threshold:.4f}; clase crítica no mejora suficiente.")


def build_policy(policy_name: str, policy_params: dict) -> ActivationPolicy:
    """Factory: instancia la política correcta a partir del nombre y sus parámetros."""
    if policy_name == "any_improvement":
        return AnyImprovementPolicy()

    if policy_name == "min_delta":
        min_delta = float(policy_params.get("min_delta", 0.02))
        return MinDeltaPolicy(min_delta=min_delta)

    if policy_name == "per_class_f1":
        if "target_class" not in policy_params:
            raise ValueError("per_class_f1 requiere 'target_class' en policy_params.")
        target_class  = int(policy_params["target_class"])
        min_f1_delta  = float(policy_params.get("min_f1_delta", 0.0))
        return PerClassF1Policy(target_class=target_class, min_f1_delta=min_f1_delta)

    raise ValueError(f"Política desconocida: '{policy_name}'. "
                     "Valores válidos: 'any_improvement', 'min_delta', 'per_class_f1'.")


# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_active_model_meta() -> Optional[dict]:
    history = load_history()
    return history[-1] if history else None


# ---------------------------------------------------------------------------
# Bootstrap: si no existe modelo, lo entrenamos con el dataset original
# ---------------------------------------------------------------------------

def bootstrap_model():
    """Entrena un modelo base con el dataset Iris completo al arrancar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)"
    }]
    save_history(history)
    print(f"[bootstrap] Modelo base creado → versión={version}, accuracy={accuracy:.4f}")


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Continuous Training API",
    description=(
        "Extensión MLOps de app-iris. Sirve predicciones y permite reentrenar "
        "el modelo con nuevas muestras etiquetadas, registrando el historial de versiones."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()
    else:
        meta = get_active_model_meta()
        if meta:
            print(f"[startup] Modelo activo cargado → versión={meta['version']}, "
                  f"accuracy={meta['accuracy']}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    meta = get_active_model_meta()
    return {
        "status": "ok",
        "active_model_version": meta["version"] if meta else "none",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(sample: IrisSample):
    """
    Realiza una predicción con el modelo activo.
    Devuelve la clase predicha, su nombre y la versión del modelo usado.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Llama primero a /train.")

    clf = joblib.load(MODEL_PATH)
    X = np.array([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    ]])
    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_version=meta["version"] if meta else "unknown"
    )


@app.post("/train", response_model=TrainResponse, tags=["Entrenamiento"])
def train(request: TrainRequest):
    """
    Reentrena el modelo con las nuevas muestras enviadas.

    - Si `retrain_from_scratch=False` (por defecto), las nuevas muestras se añaden
      al dataset de entrenamiento anterior (si existe) y se reentrena sobre el total.
    - Si `retrain_from_scratch=True`, solo se usan las muestras enviadas.
    - El nuevo modelo **reemplaza al activo solo si su accuracy ≥ accuracy anterior**.
    - Cada entrenamiento queda registrado en el historial aunque no se active.
    """

    # 1. Preparar nuevas muestras
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
                       for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    # 2. Recuperar accuracy del modelo activo
    history = load_history()
    previous_accuracy = history[-1]["accuracy"] if history else None

    # 3. Construir dataset de entrenamiento
    data_file = MODELS_DIR / "accumulated_data.joblib"

    if not request.retrain_from_scratch and data_file.exists():
        saved = joblib.load(data_file)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras nuevas, {len(saved['X'])} anteriores)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    # 4. Necesitamos al menos 2 clases para entrenar
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=422,
            detail="El dataset de entrenamiento debe contener al menos 2 clases distintas."
        )

    # 5. Entrenar nuevo modelo
    clf_new = LogisticRegression(max_iter=300, random_state=42)

    # Evaluación: si hay suficientes datos, usamos split; si no, evaluamos en train
    n_classes = len(np.unique(y_train))
    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        clf_new.fit(X_tr, y_tr)
        y_pred       = clf_new.predict(X_val)
        accuracy_new = float(accuracy_score(y_val, y_pred))
        f1_raw       = f1_score(y_val, y_pred, average=None, labels=list(range(n_classes)),
                                zero_division=0)
        eval_note = f"validación con {len(X_val)} muestras"
    else:
        clf_new.fit(X_train, y_train)
        y_pred       = clf_new.predict(X_train)
        accuracy_new = float(accuracy_score(y_train, y_pred))
        f1_raw       = f1_score(y_train, y_pred, average=None, labels=list(range(n_classes)),
                                zero_division=0)
        eval_note = "evaluación en train (dataset pequeño, < 20 muestras)"

    accuracy_new     = round(accuracy_new, 4)
    per_class_f1_map = {i: round(float(f1_raw[i]), 4) for i in range(len(f1_raw))}

    eval_result = {
        "accuracy":      accuracy_new,
        "per_class_f1":  per_class_f1_map,
    }

    # 6. Decidir si activar el nuevo modelo según la política elegida
    try:
        policy_obj = build_policy(request.policy, request.policy_params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    previous_meta = history[-1] if history else None
    model_updated, activation_reason = policy_obj.should_activate(eval_result, previous_meta)

    version = f"v{len(history) + 1}.0-{uuid.uuid4().hex[:6]}"
    status  = "activado" if model_updated else "rechazado"

    if model_updated:
        joblib.dump(clf_new, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, data_file)
        message = (
            f"Nuevo modelo activado. [{request.policy}] {activation_reason}"
        )
    else:
        message = (
            f"Modelo NO activado. [{request.policy}] {activation_reason} "
            "El modelo activo se mantiene sin cambios."
        )

    # 7. Registrar en historial (incluye política y razón de activación/rechazo)
    history.append({
        "version":            version,
        "trained_at":         datetime.utcnow().isoformat() + "Z",
        "accuracy":           accuracy_new,
        "per_class_f1":       {str(k): v for k, v in per_class_f1_map.items()},
        "n_training_samples": len(X_train),
        "algorithm":          "LogisticRegression",
        "source":             source,
        "eval_note":          eval_note,
        "policy_used":        request.policy,
        "policy_params":      request.policy_params,
        "activation_reason":  activation_reason,
        "status":             status,
        "activated":          model_updated,
    })
    save_history(history)

    return TrainResponse(
        status=status,
        model_version=version,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        model_updated=model_updated,
        message=message,
        policy_used=request.policy,
        activation_reason=activation_reason,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """
    Devuelve información del modelo activo y el historial completo de entrenamientos.
    """
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No hay ningún modelo entrenado aún.")

    active = history[-1]
    # El modelo activo es el último con activated=True (o el primero si es bootstrap)
    active_entries = [h for h in history if h.get("activated", True)]
    active = active_entries[-1] if active_entries else history[-1]

    return ModelInfo(
        active_version=active["version"],
        trained_at=active["trained_at"],
        accuracy=active["accuracy"],
        n_training_samples=active["n_training_samples"],
        algorithm=active["algorithm"],
        history=history
    )


@app.delete("/model/history", tags=["Modelo"])
def reset_history():
    """
    [CUIDADO] Elimina el historial y el modelo activo. Fuerza bootstrap en el próximo arranque.
    Útil para pruebas y demostración.
    """
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    data_file = MODELS_DIR / "accumulated_data.joblib"
    if data_file.exists():
        data_file.unlink()
    bootstrap_model()
    return {"status": "ok", "message": "Historial eliminado y modelo base restaurado."}
