import pandas as pd
import numpy as np
import requests
import json
import time
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------
API_URL      = "http://127.0.0.1:5001/invocations"
X_TEST_PATH  = "X_test.csv"
Y_TEST_PATH  = "y_test.csv"
BATCH_SIZE   = 50
OUTPUT_MD    = "reporte_test_metricas.md"

print("=" * 65)
print("EVALUACION SOBRE TEST SET - VIA API DE MLFLOW")
print("=" * 65)

# --------------------------------------------------
# 1. CARGA DEL TEST SET
# --------------------------------------------------
for path in [X_TEST_PATH, Y_TEST_PATH]:
    if not os.path.exists(path):
        print(f"\n[ERROR] No se encuentra: {path}")
        print("[INFO] Ejecuta primero: python 02_experimentos_completos.py")
        sys.exit(1)

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print(f"\n[INFO] Test set cargado:")
print(f"  Muestras:      {X_test.shape[0]}")
print(f"  Features:      {X_test.shape[1]}")
print(f"  Positivos:     {int(y_test.sum())} ({y_test.mean():.1%})")
print(f"  Negativos:     {int(len(y_test) - y_test.sum())} ({1 - y_test.mean():.1%})")

# --------------------------------------------------
# 2. VERIFICAR CONEXION CON LA API
# --------------------------------------------------
def check_api(url, max_retries=5, wait_sec=5):
    """Verifica disponibilidad de la API de MLflow."""
    print(f"\n[INFO] Verificando API en: {url}")
    for attempt in range(1, max_retries + 1):
        try:
            test_payload = {
                "dataframe_split": {
                    "columns": X_test.columns.tolist(),
                    "data": [X_test.iloc[0].tolist()]
                }
            }
            r = requests.post(url, json=test_payload,
                              headers={"Content-Type": "application/json"},
                              timeout=10)
            if r.status_code == 200:
                print(f"[OK] API disponible. Prediccion de prueba: {r.json()}")
                return True
            else:
                print(f"[INTENTO {attempt}] HTTP {r.status_code}: {r.text[:100]}")
        except requests.exceptions.ConnectionError:
            print(f"[INTENTO {attempt}/{max_retries}] Conexion rechazada. "
                  f"Esperando {wait_sec}s...")
        time.sleep(wait_sec)
    return False


if not check_api(API_URL):
    print("\n" + "=" * 65)
    print("  API NO DISPONIBLE")
    print("=" * 65)

    # Leer modelo recomendado si existe
    model_cmd = "models:/diabetes_<MODELO>/1"
    if os.path.exists("best_model_info.json"):
        with open("best_model_info.json") as f:
            info = json.load(f)
        model_cmd = f"models:/{info['model_name']}/{info['version']}"

    print(f"""
Para servir el modelo, ejecuta en una terminal separada:

  mlflow models serve -m "{model_cmd}" --host 0.0.0.0 --port 5001 --no-conda

Una vez iniciado, vuelve a ejecutar este script.
""")
    sys.exit(1)

# --------------------------------------------------
# 3. PREDICCIONES EN BATCHES A TRAVES DE LA API
# --------------------------------------------------
n_samples = X_test.shape[0]
n_batches = int(np.ceil(n_samples / BATCH_SIZE))
all_predictions = []

print(f"\n[INFO] Enviando {n_samples} muestras en {n_batches} batches de {BATCH_SIZE}...")
print(f"[INFO] Endpoint: {API_URL}")

start_time = time.time()

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end   = min(start + BATCH_SIZE, n_samples)
    batch = X_test.iloc[start:end]

    payload = {
        "dataframe_split": {
            "columns": batch.columns.tolist(),
            "data": batch.values.tolist()
        }
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code != 200:
            print(f"[ERROR] Batch {batch_idx+1}: HTTP {response.status_code}")
            print(response.text[:300])
            sys.exit(1)

        result = response.json()
        if isinstance(result, dict):
            preds = result.get("predictions", result.get("data", []))
        else:
            preds = result

        all_predictions.extend(preds)
        print(f"  Batch {batch_idx+1}/{n_batches} -> {end - start} muestras [OK]")

    except Exception as e:
        print(f"[ERROR] Batch {batch_idx+1}: {e}")
        sys.exit(1)

elapsed = time.time() - start_time
print(f"\n[OK] {n_samples} predicciones obtenidas en {elapsed:.2f}s "
      f"({n_samples / elapsed:.1f} pred/s)")

# --------------------------------------------------
# 4. CALCULO DE METRICAS REALES
# --------------------------------------------------
y_pred = np.array(all_predictions)
y_true = y_test.values

print("\n" + "=" * 65)
print("METRICAS REALES SOBRE TEST SET")
print("=" * 65)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv         = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"""
  Accuracy:       {acc:.4f}  ({acc:.1%})
  Precision:      {prec:.4f}  ({prec:.1%})
  Recall:         {rec:.4f}  ({rec:.1%})  <-- METRICA PRIORITARIA (medica)
  F1 Score:       {f1:.4f}  ({f1:.1%})
  Specificity:    {specificity:.4f}  ({specificity:.1%})
  NPV:            {npv:.4f}  ({npv:.1%})

  MATRIZ DE CONFUSION:
              Pred: 0   Pred: 1
  Real: 0:    TN={tn:>4}   FP={fp:>4}
  Real: 1:    FN={fn:>4}   TP={tp:>4}
""")

print("[INTERPRETACION MEDICA]")
total_positivos = tp + fn
total_negativos = tn + fp
print(f"  De {total_positivos} pacientes con diabetes real:")
print(f"    -> {tp} ({tp/total_positivos:.1%}) detectados correctamente (TP)")
print(f"    -> {fn} ({fn/total_positivos:.1%}) NO detectados - FALSOS NEGATIVOS (riesgo alto)")
print(f"  De {total_negativos} pacientes sanos:")
print(f"    -> {tn} ({tn/total_negativos:.1%}) descartados correctamente (TN)")
print(f"    -> {fp} ({fp/total_negativos:.1%}) falsos positivos (ansiedad, mas tests)")

print("\n[REPORTE COMPLETO sklearn]")
print(classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes"]))

# --------------------------------------------------
# 5. GUARDAR PREDICCIONES
# --------------------------------------------------
pred_df = X_test.copy()
pred_df["y_real"]       = y_true
pred_df["y_prediccion"] = y_pred
pred_df["correcto"]     = (pred_df["y_real"] == pred_df["y_prediccion"]).astype(int)
pred_df.to_csv("predicciones_test.csv", index=False)
print(f"[INFO] Predicciones guardadas en predicciones_test.csv")
