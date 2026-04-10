#!/bin/bash
# =============================================================
# LAB 6 — verify_stack.sh
# Script de verificación end-to-end del stack iris
#
# Uso:
#   chmod +x verify_stack.sh
#   ./verify_stack.sh
#
# Salidas posibles:
#   STACK OK     → health check superado + predicción válida
#   STACK FAILED → timeout o respuesta inesperada
#
# Compatibilidad: Linux, macOS, Windows Git Bash
# Nota: no usa ficheros temporales para evitar conflictos de
# rutas entre Bash y Python3 en entornos Windows.
# =============================================================

set -euo pipefail

# ── Configuración ─────────────────────────────────────────────
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:7860}"
TIMEOUT=60       # segundos máximos de espera al backend
ELAPSED=0
INTERVAL=2       # segundos entre reintentos

# Colores para la salida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'     # Sin color

# ── Funciones auxiliares ──────────────────────────────────────
log_info()    { echo -e "${YELLOW}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

fail() {
    log_error "$*"
    echo ""
    echo "══════════════════════════════════════"
    echo -e "${RED}  STACK FAILED${NC}"
    echo "══════════════════════════════════════"
    exit 1
}

# ── Fase 1: Esperar a que el backend esté listo ───────────────
echo ""
echo "══════════════════════════════════════"
echo "  Verificación del Stack — LAB 6"
echo "  Backend : $BACKEND_URL"
echo "  Frontend: $FRONTEND_URL"
echo "══════════════════════════════════════"
echo ""

log_info "Esperando a que el backend esté listo (timeout: ${TIMEOUT}s)..."

BACKEND_READY=false
while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
    # Primera llamada: obtener el HTTP status code
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 3 \
        "$BACKEND_URL/health" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        # Segunda llamada: obtener el cuerpo y parsearlo via stdin
        # (evita ficheros temporales para compatibilidad Windows/Git Bash)
        BODY=$(curl -s --connect-timeout 3 "$BACKEND_URL/health" 2>/dev/null || echo "{}")

        # Extracción segura sin requerir python instalado en el host local
        STATUS=$(echo "$BODY" | grep -o '"status"\s*:\s*"[^"]*"' | cut -d'"' -f4 || echo "")

        if [ "$STATUS" = "ok" ]; then
            BACKEND_READY=true
            MODEL_VERSION=$(echo "$BODY" | grep -o '"model_version"\s*:\s*"[^"]*"' | cut -d'"' -f4 || echo "unknown")
            log_ok "Backend listo en ${ELAPSED}s — modelo: ${MODEL_VERSION}"
            break
        fi
    fi

    log_info "Backend no listo aún (HTTP ${HTTP_CODE}, ${ELAPSED}/${TIMEOUT}s). Reintentando..."
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$BACKEND_READY" = "false" ]; then
    fail "El backend no respondió con status='ok' en ${TIMEOUT} segundos."
fi

# ── Fase 2: Predicción de prueba ─────────────────────────────
log_info "Enviando predicción de prueba a POST $BACKEND_URL/predict..."

PREDICT_PAYLOAD='{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --connect-timeout 5 \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$PREDICT_PAYLOAD" \
    "$BACKEND_URL/predict" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" != "200" ]; then
    fail "POST /predict devolvió HTTP ${HTTP_CODE}."
fi

PREDICT_BODY=$(curl -s --connect-timeout 5 \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$PREDICT_PAYLOAD" \
    "$BACKEND_URL/predict" 2>/dev/null || echo "{}")

HAS_PREDICTION=$(echo "$PREDICT_BODY" | grep -q '"prediction"' && echo "yes" || echo "no")

if [ "$HAS_PREDICTION" != "yes" ]; then
    fail "La respuesta de /predict no contiene el campo 'prediction'. Respuesta: $PREDICT_BODY"
fi

SPECIES=$(echo "$PREDICT_BODY" | grep -o '"species"\s*:\s*"[^"]*"' | cut -d'"' -f4 || echo "unknown")

PREDICTION=$(echo "$PREDICT_BODY" | grep -o '"prediction"\s*:\s*[0-9]*' | grep -o '[0-9]*' || echo "?")

log_ok "Predicción recibida → clase=${PREDICTION}, especie=${SPECIES}"
echo ""
echo "  Payload enviado:    $PREDICT_PAYLOAD"
echo "  Respuesta completa: $PREDICT_BODY"
echo ""

# ── Fase 3: Verificar frontend (opcional, sin fallar si no está) ──
log_info "Comprobando disponibilidad del frontend en $FRONTEND_URL..."
FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --connect-timeout 5 \
    "$FRONTEND_URL" 2>/dev/null || echo "000")

if [ "$FRONTEND_CODE" = "200" ]; then
    log_ok "Frontend disponible (HTTP 200)."
else
    log_info "Frontend respondió HTTP ${FRONTEND_CODE} (puede ser normal si usa WebSockets)."
fi

# ── Resultado final ───────────────────────────────────────────
echo ""
echo "══════════════════════════════════════"
echo -e "${GREEN}  STACK OK${NC}"
echo "══════════════════════════════════════"
echo ""
exit 0
