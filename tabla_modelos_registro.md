# Tabla de Modelos Registrados en MLflow - Diabetes PIMA

## Contexto del experimento

- **Dataset:** PIMA Indian Diabetes (768 muestras, 8 features)
- **Division:** 70% train / 20% validacion / 10% test (estratificada)
- **Algoritmo de busqueda:** Grid Search con Cross-Validation (5 folds, StratifiedKFold)
- **Scoring del Grid Search:** Recall (metrica medicamente prioritaria)
- **Experimento MLflow:** `diabetes_experimentos_completos`
- **Tracking URI:** `http://127.0.0.1:5000`

---

## Justificacion del critero de registro

En el contexto de **diagnostico de diabetes**, los errores no tienen el mismo coste:

| Tipo de error | Consecuencia clinica | Gravedad |
|---|---|---|
| **Falso Negativo** (diabetico no detectado) | Enfermedad no tratada: retinopatia, nefropatia, amputaciones, muerte prematura | **ALTA** |
| **Falso Positivo** (sano diagnosticado) | Ansiedad temporal + pruebas adicionales (HbA1c, OGTT) | Baja |

Por ello, la metrica que se optimiza en el Grid Search es el **Recall (Sensibilidad)**,
y el criterio para incluir un modelo en el **Model Registry** de MLflow es:

> **Recall >= 0.70 Y F1-Score >= 0.65**

- El umbral de **Recall >= 0.70** garantiza que se detecta al menos el 70% de los diabeticos reales.
- El umbral de **F1 >= 0.65** evita que el modelo "haga trampa" prediciendo siempre positivo
  (un clasificador que predice siempre 1 tendria Recall=1.0 pero F1~0.52 dado el desbalance del dataset).

---

## Tabla de runs realizados y decision de registro

### Experimento: `diabetes_experimentos_completos` (4 runs)

| Run | Modelo | Val Recall | Val F1 | Val Accuracy | Val Precision | Val ROC-AUC | CV Recall | Registrado | Motivo |
|---|---|---|---|---|---|---|---|---|---|
| GradientBoosting_best | GradientBoosting | **0.8519** | **0.7966** | **0.8312** | **0.7419** | **0.8665** | 0.7262 | **SI** | Mejor balance recall/F1. Mayor ROC-AUC. |
| SVM_best | SVM | **0.8704** | 0.7121 | 0.7532 | 0.5957 | 0.8481 | 0.7310 | **SI** | Recall muy alto. F1 justo al limite. |
| RandomForest_best | RandomForest | **0.8519** | **0.7731** | **0.8247** | **0.7097** | **0.8665** | 0.7262 | **SI** | Alto recall con buen balance de metricas. |
| LogisticRegression_best | LogisticRegression | **0.7963** | **0.7167** | **0.7792** | **0.6515** | **0.8481** | 0.7262 | **SI** | Modelo mas simple (baseline interpretable). Supera el umbral. |

### Experimento: `diabetes_prueba` (1 run - NO incluido en registry final)

| Run | Modelo | Val Recall | Val F1 | Val Accuracy | Registrado | Motivo del RECHAZO |
|---|---|---|---|---|---|---|
| prueba_logistic_regression | LogisticRegression | ~0.64 | ~0.65 | ~0.77 | NO (registry prod.) | Recall < 0.70. Sin class_weight=balanced. Run de prueba tecnica, no de produccion. |

---

## Parametros de los modelos registrados

### 1. GradientBoosting (MODELO SELECCIONADO PARA PRODUCCION)

| Parametro | Valor | Descripcion |
|---|---|---|
| `model_type` | GradientBoosting | Algoritmo de boosting por gradiente |
| `n_estimators` | 100-200 | Numero de arboles de decision encadenados |
| `max_depth` | 3-5 | Profundidad maxima de cada arbol (evita overfitting) |
| `learning_rate` | 0.05-0.1 | Tasa de aprendizaje (contribution de cada arbol) |
| `subsample` | 0.8-1.0 | Fraccion de muestras usada en cada iteracion |
| `grid_search_scoring` | recall | Metrica de optimizacion del Grid Search |
| `cv_folds` | 5 | Numero de folds en Cross-Validation estratificado |
| `class_weight` | balanced | Penaliza mas los errores en la clase minoritaria |

**Metricas de validacion:** Recall=0.8519 | F1=0.7966 | Accuracy=0.8312 | ROC-AUC=0.8665

### 2. RandomForest

| Parametro | Valor |
|---|---|
| `n_estimators` | 100-300 |
| `max_depth` | None / 5 / 10 |
| `min_samples_split` | 2-5 |
| `min_samples_leaf` | 1-2 |
| `class_weight` | balanced |

**Metricas de validacion:** Recall=0.8519 | F1=0.7731 | Accuracy=0.8247 | ROC-AUC=0.8665

### 3. LogisticRegression

| Parametro | Valor |
|---|---|
| `C` | 0.01 (regularizacion L1 optima) |
| `penalty` | l1 |
| `solver` | saga |
| `max_iter` | 500 |
| `class_weight` | balanced |

**Metricas de validacion:** Recall=0.7963 | F1=0.7167 | Accuracy=0.7792 | ROC-AUC=0.8481

### 4. SVM

| Parametro | Valor |
|---|---|
| `C` | 1.0-10.0 |
| `kernel` | rbf |
| `gamma` | scale |
| `class_weight` | balanced |
| `probability` | True |

**Metricas de validacion:** Recall=0.8704 | F1=0.7121 | Accuracy=0.7532 | ROC-AUC=0.8481

---

## Por que se registran estos modelos y no otros

### Modelos registrados (cumplen umbral medico)

Todos los runs del experimento `diabetes_experimentos_completos` superaron el umbral
`Recall >= 0.70 Y F1 >= 0.65`, por lo que los **4 modelos** fueron incluidos en el Model Registry.
Se registran porque:

1. **GradientBoosting**: mejor balance global (recall + F1 + ROC-AUC). Seleccionado para produccion.
2. **RandomForest**: recall identico a GB con buenas metricas en todas las dimensiones. Util como alternativa interpretable.
3. **SVM**: recall mas alto de todos (0.87), pero menor precision (riesgo de falsos positivos). Util si se quiere priorizar al maximo la deteccion.
4. **LogisticRegression**: el modelo mas simple e interpretable. Sirve como baseline de produccion. Si el entorno no admite modelos complejos, esta es la alternativa.

### Modelo NO registrado en registry de produccion

- **`prueba_logistic_regression`** (experimento `diabetes_prueba`): Es un run de exploracion tecnica,
  sin `class_weight=balanced` y con configuracion minimal. Su Recall (~0.64) esta por debajo del umbral
  medico y su recall en test real seria inferior. No tiene valor como modelo de produccion.

---

## Metricas finales sobre Test Set (modelo seleccionado: GradientBoosting)

Evaluadas via API REST (`http://127.0.0.1:5001/invocations`) con el script `03_evaluar_test_api.py`:

| Metrica | Valor | Interpretacion |
|---|---|---|
| Accuracy | 77.9% | 60 de 77 muestras clasificadas correctamente |
| Precision | 66.7% | De cada 3 predicciones positivas, 2 son reales |
| **Recall** | **74.1%** | Detecto 20 de 27 diabeticos reales |
| F1 Score | 70.2% | Balance adecuado entre recall y precision |
| FN (falsos negativos) | 7 | Pacientes diabeticos NO detectados |
| FP (falsos positivos) | 10 | Pacientes sanos diagnosticados erroneamente |
