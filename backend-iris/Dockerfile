# Imagen base oficial de Python, versión slim para reducir tamaño
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar e instalar dependencias ANTES que el código fuente
# Así la capa de pip se cachea aunque cambie el código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar curl ya que python:3.11-slim no lo trae nativo para el healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar el código de la aplicación
COPY . .

# Chequeo de estado real (verificado por docker y docker-compose)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Puerto que expone el contenedor (documentación, no mapeo real)
EXPOSE 8000

# Bind en 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
