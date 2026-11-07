#!/bin/bash
# Build a double-clickable macOS launcher for scripts/app.sh; the paths in use now are baked in: rebuild after moving.
set -euo pipefail
# Stop: "<app>/Contents/MacOS/TensorFoldStudio" stop; log: ~/Library/Logs/TensorFoldStudio.log
[ "$(uname -s)" = Darwin ] || { echo "macOS only" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
[ -x "$PREFIX/venv/bin/python" ] || { echo "run oneshot-setup.sh first: no environment at $PREFIX" >&2; exit 1; }
APP="${APP_DIR:-$HOME/Applications}/Wasserman Slate AI Studio.app"
PORT="${PORT:-7870}"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Wasserman Slate AI Studio</string>
  <key>CFBundleDisplayName</key><string>Wasserman Slate AI Studio</string>
  <key>CFBundleIdentifier</key><string>studio.tensorfold.launcher</string>
  <key>CFBundleExecutable</key><string>TensorFoldStudio</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$(cat "$HERE/VERSION")</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
</dict></plist>
PLIST
{
  echo '#!/bin/bash'
  echo "# written by scripts/make-app.sh on $(date +%F)"
  printf 'export PACK=%q PREFIX=%q PORT=%q\n' "$HERE" "$PREFIX" "$PORT"
  for name in QWEN_MODEL_DIR H3_MODEL_DIR FASTH3_DIR STUDIO_HOME HOST; do  # only what differs from the defaults matters
    [ -z "${!name:-}" ] || printf 'export %s=%q\n' "$name" "${!name}"
  done
  cat <<'LAUNCH'
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
URL="http://127.0.0.1:$PORT"; LOG="$HOME/Library/Logs/TensorFoldStudio.log"
say() { osascript -e "display dialog \"$1\" with title \"Wasserman Slate AI Studio\" buttons {\"OK\"} default button 1" >/dev/null 2>&1 || echo "$1" >&2; }
up() { curl -s -m 2 "$URL/api/state" >/dev/null 2>&1; }
if [ "${1:-}" = stop ]; then lsof -ti "tcp:$PORT" | xargs kill 2>/dev/null; exit 0; fi
if ! up; then
  [ -f "$PACK/scripts/app.sh" ] || { say "The studio's folder is not at $PACK any more. Run scripts/make-app.sh again from where it is now."; exit 1; }
  mkdir -p "$(dirname "$LOG")"
  nohup bash "$PACK/scripts/app.sh" >> "$LOG" 2>&1 &
  for _ in $(seq 60); do up && break; sleep 0.5; done
  up || { say "Wasserman Slate AI Studio did not start. See $LOG"; exit 1; }
fi
open "$URL"
LAUNCH
} > "$APP/Contents/MacOS/TensorFoldStudio"
chmod +x "$APP/Contents/MacOS/TensorFoldStudio"
# the icon: drawn here, so the repository carries no binary for it; the app works without one
ICONSET="$(mktemp -d)/AppIcon.iconset"; mkdir -p "$ICONSET"
if "$PREFIX/venv/bin/python" - "$ICONSET/base.png" <<'PY' 2>/dev/null
import sys
from PIL import Image, ImageDraw
n = 1024
grad = Image.new("RGB", (n, n))
px = grad.load()
for y in range(n):
    for x in range(n):
        t = (x + y) / (2 * n - 2)
        px[x, y] = (int(103 + 40 * t), int(227 - 102 * t), 255)
mask = Image.new("L", (n, n), 0)
ImageDraw.Draw(mask).rounded_rectangle((90, 90, n - 90, n - 90), radius=210, fill=255)
icon = Image.new("RGBA", (n, n), (0, 0, 0, 0))
icon.paste(grad, (0, 0), mask)
draw = ImageDraw.Draw(icon)
draw.rounded_rectangle((330, 330, n - 330, n - 330), radius=70, fill=(9, 10, 17, 255))
draw.polygon([(470, 430), (470, 594), (600, 512)], fill=(103, 227, 255, 255))
icon.save(sys.argv[1])
PY
then
  for size in 16 32 128 256 512; do
    sips -z $size $size "$ICONSET/base.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$ICONSET/base.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
  done
  rm "$ICONSET/base.png"
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns" 2>/dev/null || true
fi
if [ "${DESKTOP:-0}" = 1 ]; then ln -sfn "$APP" "$HOME/Desktop/Wasserman Slate AI Studio"; echo "Shortcut on the Desktop: Wasserman Slate AI Studio"; fi
echo "Built $APP"
echo "Double-click it (or: open \"$APP\"). It serves http://127.0.0.1:$PORT from $HERE."
