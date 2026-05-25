#!/bin/bash
# Always open the CocoaPods workspace (never App.xcodeproj).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$ROOT/ios/App/App.xcworkspace"

if [[ ! -d "$WORKSPACE" ]]; then
  echo "No se encontró $WORKSPACE — ejecuta primero: npm run cap:sync" >&2
  exit 1
fi

echo "Abriendo App.xcworkspace (requerido para el módulo Capacitor)..."
open "$WORKSPACE"
