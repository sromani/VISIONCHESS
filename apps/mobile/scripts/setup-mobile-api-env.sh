#!/bin/bash
# Writes apps/mobile/.env with VITE_API_URL pointing at this Mac (same Wi‑Fi as iPhone).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"
PORT="${API_PORT:-8001}"

pick_lan_ip() {
  for iface in en0 en1 wlan0; do
    local ip
    ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
    if [[ -n "$ip" ]]; then
      echo "$ip"
      return 0
    fi
  done
  return 1
}

IP="$(pick_lan_ip || true)"
if [[ -z "$IP" ]]; then
  echo "No se encontró IP de red local (en0/en1). Edita $ENV_FILE a mano." >&2
  exit 1
fi

API_URL="http://${IP}:${PORT}/api/v1"

cat > "$ENV_FILE" <<EOF
# Generado por scripts/setup-mobile-api-env.sh — iPhone en la misma Wi‑Fi que esta Mac
VITE_API_URL=${API_URL}
EOF

echo "Wrote $ENV_FILE"
echo "  VITE_API_URL=${API_URL}"
echo ""
echo "En la Mac, inicia la API accesible en la red:"
echo "  cd apps/api && uvicorn main:app --reload --host 0.0.0.0 --port ${PORT}"
echo ""
echo "Luego recompila la app móvil:"
echo "  cd apps/mobile && npm run cap:sync"
