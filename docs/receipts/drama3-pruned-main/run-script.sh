#!/bin/sh
# Drama 3 FP8 and both pruned NVFP4 checkpoints from main, on one Spark. Run inside a GPU window the Desk opens
# (it pauses and resumes the box's own services; this script pauses nothing):
#   sh run-script.sh TREE FP8 PRUNED PRUNED_5900 VOICES VOICE NVCC OUT > OUT.log 2>&1
# TREE is a checkout of the commit under test; VOICE a reference-backed catalog voice in VOICES.
set -eu
[ "$#" -eq 8 ] || { echo 'usage: run-script.sh TREE FP8 PRUNED PRUNED_5900 VOICES VOICE NVCC OUT' >&2; exit 2; }
tree=$1 fp8=$2 pruned=$3 pruned5900=$4 voices=$5 voice=$6 nvcc=$7 out=$8
cd "$tree"
zig build -Dgpu=spark -Dnvcc="$nvcc"
echo "head $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for model in "$fp8" "$pruned" "$pruned5900"; do
    sum=$(find "$model" -name '*.safetensors' -o -name codec.pth | sort | xargs cat | sha256sum | cut -d' ' -f1)
    echo "input $model: $sum"
done
cd tools/zig
python3 drama3_checkpoints_check.py --bin "$tree/zig-out/native/bin/tensorfold-native" --fp8 "$fp8" \
    --pruned "$pruned" --pruned-5900 "$pruned5900" --voices "$voices" --voice "$voice" --out "$out"
