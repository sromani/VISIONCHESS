#!/usr/bin/env bash
# Capacitor / Xcode need `pod` on PATH. User-installed gems often live outside IDE PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUBY_MINOR="$(ruby -e 'print RUBY_VERSION[/\d+\.\d+/]' 2>/dev/null || echo "2.6")"
GEM_BIN="$HOME/.gem/ruby/${RUBY_MINOR}.0/bin"

mkdir -p "$HOME/.local/bin"
if [[ -x "$GEM_BIN/pod" ]]; then
  ln -sf "$GEM_BIN/pod" "$HOME/.local/bin/pod"
fi

export PATH="$HOME/.local/bin:$GEM_BIN:/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v pod >/dev/null 2>&1; then
  echo "[error] CocoaPods is not installed or not on PATH." >&2
  echo "" >&2
  echo "Install one of:" >&2
  echo "  sudo gem install cocoapods" >&2
  echo "  brew install cocoapods   # after: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"" >&2
  echo "" >&2
  echo "Then add to ~/.zshrc (or ~/.bash_profile):" >&2
  echo "  export PATH=\"\$HOME/.local/bin:\$HOME/.gem/ruby/\$(ruby -e 'print RUBY_VERSION[/\\d+\\.\\d+/]').0/bin:\$PATH\"" >&2
  echo "" >&2
  echo "Docs: https://capacitorjs.com/docs/getting-started/environment-setup#ios" >&2
  exit 1
fi

cd "$ROOT"
exec npx cap "$@"
