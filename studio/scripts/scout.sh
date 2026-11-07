#!/bin/bash
# Scout pictures of one prompt from one load + sheet.jpg. Usage: bash scripts/scout.sh "a prompt" [count] [folder]
set -euo pipefail
# default 1280x736 is the frame a QHD clip opens on; seeds run from SEED (default 1) upwards
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local/opt/tensorfold-studio}"
PROMPT="${1:-}"; COUNT="${2:-4}"; DIR="${3:-outputs/scout}"
FIRST="${SEED:-1}"
SEEDS=$(seq -s, "$FIRST" $((FIRST + COUNT - 1)))
export WIDTH="${WIDTH:-1280}" HEIGHT="${HEIGHT:-736}"
mkdir -p "$DIR"
EXTRA="${EXTRA:-} --seeds ${SEEDS%,}" bash "$HERE/image.sh" "$PROMPT" "$DIR/scout.png"
"$PREFIX/venv/bin/python" - "$DIR" "${SEEDS%,}" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw
folder, seeds = Path(sys.argv[1]), sys.argv[2].split(",")
images = [(seed, Image.open(folder / f"scout_s{seed}.png").convert("RGB")) for seed in seeds]
columns = 2 if len(images) <= 4 else 3
width = 640
height = images[0][1].height * width // images[0][1].width
rows = -(-len(images) // columns)
sheet = Image.new("RGB", (columns * width, rows * height), "black")
for index, (seed, image) in enumerate(images):
    tile = image.resize((width, height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 110, 26), fill="black")
    draw.text((8, 6), f"seed {seed}", fill="white")
    sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
sheet.save(folder / "sheet.jpg", quality=90)
print(f"[studio] {len(images)} scouts in {folder}; contact sheet {folder / 'sheet.jpg'}")
PY
