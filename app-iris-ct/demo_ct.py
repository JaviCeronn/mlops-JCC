#!/usr/bin/env python3
"""
demo_ct.py  –  Script de demostración del flujo Continuous Training
====================================================================
Simula un ciclo MLOps completo:
  1. Consulta el modelo base (bootstrap)
  2. Hace predicciones
  3. Envía nuevas muestras correctamente etiquetadas → modelo mejora
  4. Envía muestras con ruido/errores → modelo empeora, NO se activa
  5. Muestra el historial final de versiones

Uso:
    python demo_ct.py [--host http://localhost:8000]
"""

import argparse
import json
import sys
import time

import requests

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Muestras de ejemplo
# ---------------------------------------------------------------------------

# 20 muestras bien etiquetadas (mezcla de las 3 clases)
GOOD_SAMPLES = [
    # setosa (clase 0)
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "label": 0},
    {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2, "label": 0},
    {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2, "label": 0},
    {"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4, "label": 0},
    {"sepal_length": 4.6, "sepal_width": 3.4, "petal_length": 1.4, "petal_width": 0.3, "label": 0},
    {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2, "label": 0},
    {"sepal_length": 4.4, "sepal_width": 2.9, "petal_length": 1.4, "petal_width": 0.2, "label": 0},
    # versicolor (clase 1)
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "label": 1},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5, "label": 1},
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5, "label": 1},
    {"sepal_length": 5.5, "sepal_width": 2.3, "petal_length": 4.0, "petal_width": 1.3, "label": 1},
    {"sepal_length": 6.5, "sepal_width": 2.8, "petal_length": 4.6, "petal_width": 1.5, "label": 1},
    {"sepal_length": 5.7, "sepal_width": 2.8, "petal_length": 4.5, "petal_width": 1.3, "label": 1},
    # virginica (clase 2)
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5, "label": 2},
    {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9, "label": 2},
    {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1, "label": 2},
    {"sepal_length": 6.3, "sepal_width": 2.9, "petal_length": 5.6, "petal_width": 1.8, "label": 2},
    {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2, "label": 2},
    {"sepal_length": 7.6, "sepal_width": 3.0, "petal_length": 6.6, "petal_width": 2.1, "label": 2},
    {"sepal_length": 4.9, "sepal_width": 2.5, "petal_length": 4.5, "petal_width": 1.7, "label": 2},
]

# 10 muestras con etiquetas incorrectas (simula ruido en los datos nuevos)
NOISY_SAMPLES = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "label": 2},  # setosa → mal
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "label": 1},  # setosa → mal
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "label": 0},  # versic → mal
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5, "label": 2},  # versic → mal
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5, "label": 1},  # virg → mal
    {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9, "label": 0},  # virg → mal
    {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2, "label": 1},  # setosa → mal
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5, "label": 2},  # versic → mal
    {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1, "label": 0},  # virg → mal
    {"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4, "label": 2},  # setosa → mal
]

PREDICT_SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title=""):
    print("\n" + "─" * 60)
    if title:
        print(f"  {title}")
        print("─" * 60)


def ok(msg):   print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def info(msg): print(f"  ℹ️   {msg}")
def err(msg):  print(f"  ❌  {msg}")


def get(host, path):
    r = requests.get(f"{host}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def post(host, path, body):
    r = requests.post(f"{host}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Pasos del demo
# ---------------------------------------------------------------------------

def step_health(host):
    sep("PASO 0 – Health check")
    data = get(host, "/health")
    ok(f"Servicio activo. Modelo activo: {data['active_model_version']}")


def step_model_info(host, label="Estado actual del modelo"):
    sep(label)
    data = get(host, "/model/info")
    ok(f"Versión activa : {data['active_version']}")
    ok(f"Entrenado el   : {data['trained_at']}")
    ok(f"Accuracy       : {data['accuracy']:.4f}")
    ok(f"Nº muestras    : {data['n_training_samples']}")
    ok(f"Algoritmo      : {data['algorithm']}")
    print()
    info(f"Historial de versiones ({len(data['history'])} entradas):")
    for entry in data["history"]:
        activated_icon = "🟢" if entry.get("activated", True) else "🔴"
        print(f"    {activated_icon} {entry['version']}  |  acc={entry['accuracy']:.4f}  "
              f"|  {entry['trained_at'][:19]}  |  {entry.get('status','–')}")


def step_predict(host):
    sep("PASO 1 – Predicción con el modelo base")
    result = post(host, "/predict", PREDICT_SAMPLE)
    ok(f"Predicción: clase {result['prediction']} ({result['class_name']}), "
       f"versión modelo: {result['model_version']}")


def step_train_good(host):
    sep("PASO 2 – Reentrenamiento con muestras CORRECTAS")
    info(f"Enviando {len(GOOD_SAMPLES)} muestras bien etiquetadas...")
    result = post(host, "/train", {"samples": GOOD_SAMPLES, "retrain_from_scratch": False})

    if result["model_updated"]:
        ok(f"Nuevo modelo ACTIVADO → versión: {result['model_version']}")
        prev_acc = f"{result['accuracy_previous']:.4f}" if result.get('accuracy_previous') is not None else 'N/A'
        ok(f"Accuracy nuevo: {result['accuracy_new']:.4f}  |  "
           f"Anterior: {prev_acc}")
    else:
        warn(f"Modelo NO activado. {result['message']}")
    info(result["message"])


def step_train_noisy(host):
    sep("PASO 3 – Reentrenamiento con muestras RUIDOSAS (etiquetas incorrectas)")
    info(f"Enviando {len(NOISY_SAMPLES)} muestras con etiquetas erróneas...")
    result = post(host, "/train", {"samples": NOISY_SAMPLES, "retrain_from_scratch": True})

    if result["model_updated"]:
        warn(f"Modelo activado igualmente (accuracy similar o mayor). "
             f"Versión: {result['model_version']}")
    else:
        ok(f"Modelo RECHAZADO como esperado. Accuracy {result['accuracy_new']:.4f} "
           f"< anterior {result['accuracy_previous']:.4f}")
    info(result["message"])


def step_predict_after(host):
    sep("PASO 4 – Predicción con el modelo actualizado")
    result = post(host, "/predict", PREDICT_SAMPLE)
    ok(f"Predicción: clase {result['prediction']} ({result['class_name']}), "
       f"versión modelo: {result['model_version']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo Continuous Training – Iris API")
    parser.add_argument("--host", default="http://localhost:8000",
                        help="URL base del servidor (default: http://localhost:8000)")
    parser.add_argument("--reset", action="store_true",
                        help="Resetea el historial antes de ejecutar el demo")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  🌸  DEMO: Iris Continuous Training API")
    print("═" * 60)
    print(f"  Host: {args.host}")

    try:
        if args.reset:
            sep("RESET – Restaurando modelo base")
            requests.delete(f"{args.host}/model/history", timeout=10)
            ok("Historial eliminado. Modelo base restaurado.")
            time.sleep(0.5)

        step_health(host=args.host)
        step_model_info(host=args.host, label="PASO 0b – Info modelo base")
        step_predict(host=args.host)
        step_train_good(host=args.host)
        step_train_noisy(host=args.host)
        step_predict_after(host=args.host)
        step_model_info(host=args.host, label="RESUMEN FINAL – Historial de versiones")

        sep()
        ok("Demo completado. Abre http://localhost:8000/docs para explorar la API.")

    except requests.exceptions.ConnectionError:
        err(f"No se puede conectar al servidor en {args.host}")
        err("Asegúrate de que el servidor está arrancado:")
        print("    uvicorn main:app --reload")
        print("  o con Docker:")
        print("    docker run -d -p 8000:80 iris-ct")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        err(f"Error HTTP: {e}")
        err(f"Respuesta: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
