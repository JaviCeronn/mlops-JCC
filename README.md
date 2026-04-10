# MLOps Capstone Project — Iris Prediction Stack

Repositorio de partida para el **LAB 6: Despliegue con Docker** de la asignatura MLOps.

El proyecto implementa un sistema de clasificación de flores Iris con dos servicios:

- **Backend** — API FastAPI que descarga el modelo desde Hugging Face Hub y sirve predicciones.
- **Frontend** — Interfaz Gradio para interacción del usuario.

Tu tarea es **containerizar** ambos servicios y orquestarlos con Docker Compose.

---

## Estructura del repositorio

```
MLOps-Capstone-Project/
├── backend-iris/
│   ├── main.py               ← API FastAPI (punto de partida — modificar)
│   ├── Dockerfile            ← Dockerfile del backend (punto de partida — modificar)
│   └── requirements.txt      ← Dependencias del backend
├── frontend-iris/
│   ├── gradio_app.py         ← Interfaz Gradio (punto de partida — modificar)
│   └── requirements.txt      ← Dependencias del frontend
└── docker-compose.yml        ← a crear
```

---

## Entregables del laboratorio

| Fichero                      | Descripción                                               |
| ---------------------------- | ---------------------------------------------------------- |
| `backend-iris/Dockerfile`  | Imagen del backend con `HEALTHCHECK` incluido            |
| `frontend-iris/Dockerfile` | Imagen del frontend                                        |
| `docker-compose.yml`       | Orquestación del stack con `condition: service_healthy` |
| `verify_stack.sh`          | Script de verificación end-to-end                         |

Consulta el enunciado completo en `LAB6_Docker_Deployment.md` para los requisitos detallados de cada fichero.

---

## API del backend (referencia)

### `GET /health`

Devuelve el estado del servicio. Debe retornar HTTP 200 **solo cuando el modelo esté cargado en memoria**.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0-base",
  "uptime_seconds": 12.4
}
```

### `POST /predict`

**Request:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "prediction": 0,
  "species": "setosa",
  "confidence": 1.0
}
```

---

## Variables de entorno

| Variable        | Obligatoria   | Descripción                                                    |
| --------------- | ------------- | --------------------------------------------------------------- |
| `HF_TOKEN`    | Sí           | Token de Hugging Face Hub para descargar el modelo              |
| `BACKEND_URL` | No (frontend) | URL del backend; en Docker Compose usar `http://backend:8000` |

Crea un fichero `.env` en la raíz con tu token antes de arrancar el stack:

```bash
HF_TOKEN=hf_tu_token_aqui
```

> `.env` está en `.gitignore` — nunca subas tu token a Git.

---

## Referencias

- [Enunciado del laboratorio](../STATEMENT/LAB6_Docker_Deployment.md)
- [Solución de referencia](../STATEMENT/SOLUTION/code/)
- [Docker Documentation — Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose — Getting started](https://docs.docker.com/compose/gettingstarted/)
- [FastAPI — Deployment with Docker](https://fastapi.tiangolo.com/deployment/docker/)

---

# ANEXO: Mapa de ficheros

Guía para orientar el proceso de edición de código (análisis de gap):

| Fichero                            | Repo de partida     | Solución          | Estado del gap       |
| ---------------------------------- | ------------------- | ------------------ | -------------------- |
| `backend-iris/Dockerfile`        | Existe (incompleto) | Existe (completo) | **Reescribir**      |
| `backend-iris/main.py`           | Existe (incompleto) | Existe (completo)  | **Reescribir** |
| `backend-iris/requirements.txt`  | **Existe**    | Existe             | **OK**         |
| `frontend-iris/gradio_app.py`    | Existe              | Existe             | **OK**         |
| `frontend-iris/Dockerfile`       | **No existe** | Existe             | **Crear**      |
| `frontend-iris/requirements.txt` | **Existe**    | Existe             | **OK**         |
| `docker-compose.yml`             | **Existe**    | Existe             | **OK**         |
| `verify_stack.sh`                | **Existe**    | Existe             | **OK**         |
| `.env` / `.env.example`        | **No existe** | Existe             | **Crear**      |