#!/usr/bin/env bash
# Idol's GB10 qualification on CUDA, one command on a DGX Spark (or an sm_120 card); --help for the usage.
set -u

usage() {
    cat <<'EOF'
tools/zig/idol_gb10_qualify.sh [--build-only | --skip-build | --dry-run] WEIGHTS NOTES [OUT]

WEIGHTS: the speridlabs/idol snapshot (config.json, denoiser, decoder, decoder_gs and birefnet .safetensors).
NOTES:   the case root copied from the Mac (docs/recipes/idol-cuda.md, "GB10 qualification", lists its files).
OUT:     where every log and summary.md go (default ./idol-gb10-DATE). WEIGHTS, NOTES and OUT may come from the
         environment instead.
Environment: VAE_DIR (FLUX.2-klein-4B's vae/, default WEIGHTS/flux2-vae), DINO_DIR (timm's DINOv3 ViT-H+/16,
default WEIGHTS/dino-timm-base), NVCC (/usr/local/cuda-13.0/bin/nvcc), CUOBJDUMP (beside NVCC), SM (121; 120 for an
RTX 50), LIMIT_SMS (80,86,89,120,121), ZIG (zig), PYTHON (python3, with numpy for the parity step),
TENSORFOLD_LIBTURBOJPEG (libjpeg-turbo 3.1.4.1, for the JPEG case), SKIP_LIMITS=1 and SKIP_SASS=1 to leave out
the all-SM limits build or the SASS identity check, SASS_BASE (the git revision with #1767's idol_attention.cuh).

Each step's output goes to OUT/logs/STEP.log and its result to OUT/steps.tsv; a failed step does not stop the run.
--build-only checks the GPU and runs the build steps alone; --skip-build reuses zig-out/; --dry-run prints the
steps. The exit status is 0 only if every step passed.
EOF
}

mode=full
while [ $# -gt 0 ]; do
    case "$1" in
        --build-only) mode=build ;;
        --skip-build) mode=nobuild ;;
        --dry-run) mode=dry ;;
        -h | --help) usage; exit 0 ;;
        --*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
    shift
done
WEIGHTS=${1:-${WEIGHTS:-}}
NOTES=${2:-${NOTES:-}}
OUT=${3:-${OUT:-idol-gb10-$(date +%Y%m%d-%H%M%S)}}
if [ -z "$WEIGHTS" ] || [ -z "$NOTES" ]; then
    usage >&2
    exit 2
fi
VAE_DIR=${VAE_DIR:-$WEIGHTS/flux2-vae}
DINO_DIR=${DINO_DIR:-$WEIGHTS/dino-timm-base}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
CUOBJDUMP=${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}
SM=${SM:-121}
LIMIT_SMS=${LIMIT_SMS:-80,86,89,120,121}
ZIG=${ZIG:-zig}
PYTHON=${PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$REPO/zig-out/bin
mkdir -p "$OUT/logs" "$OUT/runs"
OUT=$(cd "$OUT" && pwd)
STEPS=$OUT/steps.tsv
: > "$STEPS"
failed=0
gpu_ok=0

# step NAME COMMAND...: runs it in REPO with its output in logs/NAME.log, records pass/FAIL and the seconds.
step() {
    local name=$1 log=$OUT/logs/$1.log start rc
    shift
    if [ "$mode" = dry ]; then
        local cmd="$*"
        echo "[$name] ${cmd:0:240}"
        return 0
    fi
    echo "== $name: $*" | tee "$log"
    start=$(date +%s)
    (cd "$REPO" && "$@") >> "$log" 2>&1
    rc=$?
    record "$name" "$rc" $(($(date +%s) - start))
    return "$rc"
}

# record NAME STATUS SECONDS [NOTE]: STATUS 0 is pass, "skip" a step not run, anything else a failure.
record() {
    local result
    case "$2" in
        0) result=pass ;;
        skip) result=skip ;;
        *) result="FAIL (exit $2)"; failed=$((failed + 1)) ;;
    esac
    printf '%s\t%s\t%s\t%s\n' "$1" "$result" "$3" "${4:-}" >> "$STEPS"
    echo "-- $1: $result in $3 s"
}

skip() {
    [ "$mode" = dry ] && { echo "[$1] skipped: $2"; return 0; }
    echo "$2" > "$OUT/logs/$1.log"
    record "$1" skip 0 "$2"
}

# The GNU time wrapper for a timed run, when the box has it.
timed() {
    if [ -x /usr/bin/time ] && /usr/bin/time -v true > /dev/null 2>&1; then /usr/bin/time -v "$@"; else "$@"; fi
}

# sampled NAME COMMAND...: the command, with nvidia-smi's memory and MemAvailable every 0.5 s in runs/NAME.mem.csv.
sampled() {
    local csv=$OUT/runs/$1.mem.csv pid rc
    shift
    echo "time,gpu_used_mib,apps_used_mib,mem_available_kb" > "$csv"
    (
        while :; do
            printf '%s,%s,%s,%s\n' "$(date +%s.%N)" \
                "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1)" \
                "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2> /dev/null | awk '{s += $1} END {print s + 0}')" \
                "$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2> /dev/null)"
            sleep 0.5
        done
    ) >> "$csv" &
    pid=$!
    timed "$@"
    rc=$?
    kill "$pid" 2> /dev/null
    wait "$pid" 2> /dev/null
    return "$rc"
}

# peaks CSV: the largest GPU and compute-app MiB sampled, and the drop in MemAvailable (GB) from the first sample.
peaks() {
    awk -F, 'NR > 1 { if ($2 + 0 > g) g = $2 + 0; if ($3 + 0 > p) p = $3 + 0;
        if (first == "") first = $4; if (min == "" || $4 + 0 < min) min = $4 + 0 }
        END { printf "nvidia-smi peak %s MiB (apps %s MiB), MemAvailable drop %.2f GB", g + 0, p + 0, (first - min) / 1e6 }' "$1"
}

# value FILE KEY: the first number or boolean after "KEY": in a JSON file (the tools' JSON, read as text).
value() {
    [ -f "$1" ] || { echo "-"; return; }
    grep -o "\"$2\": [-0-9.eE+a-z]*" "$1" | head -1 | sed 's/.*: //'
}

# copycase SRC DST FILE...: DST gets links to SRC's input files only, so the tools' outputs never write into NOTES.
copycase() {
    local src=$1 dst=$2 f
    shift 2
    [ "$mode" = dry ] && return 0
    rm -rf "$dst"
    mkdir -p "$dst"
    for f in "$@"; do [ -e "$src/$f" ] && ln -s "$src/$f" "$dst/$f"; done
    [ -d "$src" ]
}

echo "Idol GB10 qualification: mode $mode, WEIGHTS $WEIGHTS, NOTES $NOTES, OUT $OUT"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(uname -a)"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2> /dev/null) ($(git -C "$REPO" status --short 2> /dev/null | wc -l | tr -d ' ') changed files)"
    echo "zig: $("$ZIG" version 2> /dev/null)"
    echo "nvcc: $("$NVCC" --version 2> /dev/null | tail -1)"
    echo "gpu: $(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2> /dev/null)"
} > "$OUT/environment.txt"

# 1. The GPU: sm_121 (GB10) or sm_120, as every Idol component admits.
gpucheck() {
    local cap
    command -v nvidia-smi > /dev/null || { echo "no nvidia-smi: no NVIDIA driver on this machine"; return 1; }
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader || return 1
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$cap" in
        12.1 | 12.0) echo "sm_${cap/./} admitted" ;;
        *) echo "compute capability '$cap': Idol admits sm_121 and sm_120 alone"; return 1 ;;
    esac
}
step gpu-check gpucheck && gpu_ok=1

# 2. The build: fatbins for SM, the CUDA tools and tf-cuda-test; the SASS identity; the all-SM contract images.
sass() {
    local base=${SASS_BASE:-} work=$OUT/runs/sass sm same=0 total=0 flags
    [ -n "$base" ] || base="$(git log -n1 --format=%H -- zig/kernels/cuda/idol_attention.cuh)^"
    echo "idol.cu with idol_attention.cuh at $base (#1767's) against this tree's"
    rm -rf "$work"
    mkdir -p "$work/before"
    cp -R zig/kernels/cuda "$work/before/cuda"
    git show "$base:zig/kernels/cuda/idol_attention.cuh" > "$work/before/cuda/idol_attention.cuh" || return 1
    flags="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
        -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 --fmad=false --ftz=false \
        --prec-div=true --prec-sqrt=true"
    for sm in 80 86 89 120 121; do
        total=$((total + 1))
        # shellcheck disable=SC2086 # the flags are words
        "$NVCC" -cubin $flags -I"$work/before/cuda" -arch=sm_$sm -o "$work/before_$sm.cubin" "$work/before/cuda/idol.cu" &&
            "$NVCC" -cubin $flags -Izig/kernels/cuda -arch=sm_$sm -o "$work/after_$sm.cubin" zig/kernels/cuda/idol.cu || return 1
        "$CUOBJDUMP" -sass "$work/before_$sm.cubin" | grep -v "Fatbin\|code for sm_\|^$" > "$work/before_$sm.sass"
        "$CUOBJDUMP" -sass "$work/after_$sm.cubin" | grep -v "Fatbin\|code for sm_\|^$" > "$work/after_$sm.sass"
        if [ -s "$work/after_$sm.sass" ] && cmp -s "$work/before_$sm.sass" "$work/after_$sm.sass"; then
            same=$((same + 1))
            echo "sm_$sm: tf_idol_attention's SASS identical ($(wc -l < "$work/after_$sm.sass") lines)"
        else
            echo "sm_$sm: SASS DIFFERENT"
        fi
    done
    echo "SASS identity: $same of $total SMs identical"
    [ "$same" -eq "$total" ]
}
if [ "$mode" != nobuild ]; then
    step build "$ZIG" build -Dnvcc="$NVCC" -Dsm="$SM" fatbins install --summary line
    if [ "${SKIP_SASS:-0}" = 1 ]; then skip sass-identity "SKIP_SASS=1"; else step sass-identity sass; fi
    if [ "${SKIP_LIMITS:-0}" = 1 ]; then
        skip limits-build "SKIP_LIMITS=1"
    else
        step limits-build "$ZIG" build -Dnvcc="$NVCC" -Dsm="$LIMIT_SMS" fatbins kernel-contract-fatbins \
            --prefix "$OUT/limits" --summary line
    fi
fi

gpu_steps() {
    local c photo
    # 3. The kernel contract: every Idol component's set, the full contract on this GPU, gate 5 for every SM.
    for c in idol-contract idol-dino-contract idol-vae-contract idol-decoder-contract idol-birefnet-contract kernel-contract; do
        step "$c" "$BIN/tf-cuda-test" "$c"
    done
    if [ "${SKIP_LIMITS:-0}" = 1 ]; then
        skip kernel-contract-limits "SKIP_LIMITS=1"
    else
        step kernel-contract-limits "$BIN/tf-cuda-test" kernel-contract-limits "$OUT/limits/fatbin"
    fi

    # 4. Each component's tool on its Mac case, in a directory of links to the case's inputs.
    copycase "$NOTES/case50" "$OUT/runs/case50" case.safetensors exact.safetensors reference.json \
        blocks0.safetensors blocks0_f32.safetensors
    step denoiser-case50 sampled denoiser-case50 "$BIN/tf-idol" "$WEIGHTS" "$OUT/runs/case50"
    step denoiser-blocks0 "$BIN/tf-idol" "$WEIGHTS" "$OUT/runs/case50" blocks 0
    for c in decode262k decode32k; do
        copycase "$NOTES/$c" "$OUT/runs/$c" decode.safetensors decode.json reference.ply
        step "decoders-$c" sampled "decoders-$c" "$BIN/tf-idol-decode" "$WEIGHTS" "$OUT/runs/$c"
    done
    step decoders-time262k sampled decoders-time262k "$BIN/tf-idol-decode" "$WEIGHTS" "$OUT/runs/decode262k" \
        time 262144 "$OUT/runs/decode-time262k.ply"
    copycase "$NOTES/vae/case" "$OUT/runs/vae" vae.safetensors vae.json vae_stages.safetensors
    step vae-case sampled vae-case "$BIN/tf-idol-vae" "$VAE_DIR" "$OUT/runs/vae"
    copycase "$NOTES/dino/case" "$OUT/runs/dino" dino.safetensors dino.json dino_blocks.safetensors \
        dino_blocks_f32.safetensors
    step dino-case sampled dino-case "$BIN/tf-idol-dino" "$DINO_DIR" "$OUT/runs/dino" 10
    for c in avocado:reference-avocado-input.png webp:food-avocado.webp opaque:avocado-opaque.png \
        opaque-real:avocado-opaque.png rgba:avocado-rgba.png jpeg:avocado-opaque.jpg; do
        photo=${c#*:}
        c=${c%%:*}
        if [ "$c" = jpeg ] && [ -z "${TENSORFOLD_LIBTURBOJPEG:-}" ]; then
            skip birefnet-jpeg "TENSORFOLD_LIBTURBOJPEG is not set (libjpeg-turbo 3.1.4.1)"
            continue
        fi
        copycase "$NOTES/birefnet/cases/$c" "$OUT/runs/birefnet-$c" birefnet.safetensors birefnet.json \
            birefnet_stages.safetensors
        step "birefnet-$c" sampled "birefnet-$c" "$BIN/tf-idol-birefnet" "$WEIGHTS" "$OUT/runs/birefnet-$c" \
            "$NOTES/birefnet/photos/$photo"
    done

    # 5. End to end (seed 42, 50 steps, 262,144 Gaussians): the picture, a photo, then the SDK's noise injected.
    local gen="--seed 42 --steps 50 --gaussians 262144"
    # shellcheck disable=SC2086 # the options are words
    step e2e-picture sampled e2e-picture "$BIN/tf-idol-generate" "$WEIGHTS" "$VAE_DIR" "$DINO_DIR" \
        "$NOTES/vae/case/vae.safetensors" "$OUT/runs/e2e-picture" $gen
    # shellcheck disable=SC2086
    step e2e-photo sampled e2e-photo "$BIN/tf-idol-generate" "$WEIGHTS" "$VAE_DIR" "$DINO_DIR" \
        "$NOTES/birefnet/photos/avocado-opaque.png" "$OUT/runs/e2e-photo" $gen
    # shellcheck disable=SC2086
    step e2e-parity sampled e2e-parity "$BIN/tf-idol-generate" "$WEIGHTS" "$VAE_DIR" "$DINO_DIR" \
        "$NOTES/vae/case/vae.safetensors" "$OUT/runs/e2e-parity" $gen --noise "$NOTES/e2e/sdk/sdk-record.safetensors"
    step e2e-compare "$PYTHON" -c "$compare_py" "$OUT/runs" "$NOTES"
}

# The parity numbers: sample cosines and the sampled nearest-neighbour distance of compare.py, with numpy alone.
compare_py='
import json, struct, sys
import numpy as np
runs, notes = sys.argv[1], sys.argv[2]
def tensor(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; head = json.loads(f.read(n)); t = head[name]
        assert t["dtype"] == "F32"; f.seek(8 + n + t["data_offsets"][0])
        return np.frombuffer(f.read(t["data_offsets"][1] - t["data_offsets"][0]), "<f4")
def ply(path):
    data = open(path, "rb").read(); end = data.index(b"end_header\n") + 11
    names = [l.split()[-1] for l in data[:end].decode().splitlines() if l.startswith("property")]
    v = np.frombuffer(data[end:], "<f4").reshape(-1, len(names)); return v[:, :3].astype("float64")
def nn(a, b):
    return np.sqrt(np.stack([((b - p) ** 2).sum(1).min() for p in a]))
def cos(a, b):
    a, b = a.astype("float64").ravel(), b.astype("float64").ravel(); return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
x = ply(notes + "/e2e/sdk/sdk.ply"); y = ply(runs + "/e2e-parity/generate.ply"); g = np.random.default_rng(42)
xs, ys = x[g.choice(len(x), 4096, replace=False)], y[g.choice(len(y), 4096, replace=False)]
mean = float((nn(xs, ys).mean() + nn(ys, xs).mean()) / 2)
base = x[g.choice(len(x), 4096, replace=False)]; x0 = x[np.random.default_rng(42).choice(len(x), 4096, replace=False)]
sample = lambda d: np.fromfile(d + "/sample.f32", "<f4")[:131072]
r = {"sample_cosine_vs_sdk": cos(sample(runs + "/e2e-parity"), tensor(notes + "/e2e/sdk/sdk-record.safetensors", "sample")),
     "sample_cosine_vs_metal_sdknoise": cos(sample(runs + "/e2e-parity"), sample(notes + "/e2e/sdknoise")),
     "sampled_nn_mean_vs_sdk": mean, "ref_vs_ref_baseline": float((nn(x0, base).mean() + nn(base, x0).mean()) / 2),
     "picture_sample_cosine_vs_metal": cos(sample(runs + "/e2e-picture"), sample(notes + "/e2e/cold")),
     "photo_sample_cosine_vs_metal": cos(sample(runs + "/e2e-photo"), sample(notes + "/birefnet/e2e/ours-opaque-final"))}
print(json.dumps(r, indent=1)); open(runs + "/e2e-compare.json", "w").write(json.dumps(r, indent=1))
ok = r["sample_cosine_vs_sdk"] >= 0.998 and r["sampled_nn_mean_vs_sdk"] <= 0.0100
print("parity:", "pass" if ok else "FAIL", "(cosine >= 0.998 and sampled NN mean <= 0.0100; Metal: 0.99835, 0.00922)")
sys.exit(0 if ok else 1)
'

if [ "$mode" = full ] || [ "$mode" = nobuild ] || [ "$mode" = dry ]; then
    if [ "$gpu_ok" = 1 ] || [ "$mode" = dry ]; then
        gpu_steps
    else
        skip gpu-steps "no admitted GPU (see logs/gpu-check.log): the contracts, cases and end-to-end runs did not run"
        failed=$((failed + 1))
    fi
fi
[ "$mode" = dry ] && exit 0

# The summary: each step, then the timings and peaks beside the Metal reference (M5 Ultra), then the parity.
{
    echo "# Idol GB10 qualification"
    echo
    sed 's/^/    /' "$OUT/environment.txt"
    echo
    echo "| step | result | seconds | note |"
    echo "| --- | --- | --- | --- |"
    awk -F'\t' '{ printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4 }' "$STEPS"
    echo
    echo "## Contract lines"
    echo
    for f in "$OUT/logs/"*contract*.log "$OUT/logs/sass-identity.log"; do
        [ -f "$f" ] && grep -h "SUMMARY\|SASS identity\|limit problems\|FAIL" "$f" | head -3 | sed 's/^/    /'
    done
    echo
    echo "## Timings and peaks (CUDA here; Metal on the M5 Ultra from NOTES)"
    echo
    j=$OUT/runs
    echo "- denoiser case50: denoise_s $(value "$j/case50/tf-idol.json" denoise_s) (Metal $(value "$NOTES/case50/tf-idol.json" denoise_s)), worst_row_fp32 $(value "$j/case50/tf-idol.json" worst_row_fp32) (Metal 0.04738)"
    echo "- VAE: median_s $(value "$j/vae/tf-idol-vae.json" median_s) (Metal $(value "$NOTES/vae/case/tf-idol-vae.json" median_s))"
    echo "- DINOv3: encode_s $(value "$j/dino/tf-idol-dino.json" encode_s) (Metal $(value "$NOTES/dino/case/tf-idol-dino.json" encode_s))"
    echo "- decoders 262k: octree_s $(value "$j/decode262k/tf-idol-decode.json" octree_s), gaussian_s $(value "$j/decode262k/tf-idol-decode.json" gaussian_s) (Metal $(value "$NOTES/decode262k/tf-idol-decode.json" octree_s), $(value "$NOTES/decode262k/tf-idol-decode.json" gaussian_s)); time run: $(grep -ho 'cuda_peak_bytes[^,]*' "$OUT/logs/decoders-time262k.log" 2> /dev/null | tail -1)"
    echo "- BiRefNet avocado: median_s $(value "$j/birefnet-avocado/tf-idol-birefnet.json" median_s) (Metal $(value "$NOTES/birefnet/cases/avocado/tf-idol-birefnet.json" median_s))"
    for e in picture photo parity; do
        echo "- e2e $e: total_s $(value "$j/e2e-$e/tf-idol-generate.json" total_s), denoise_s $(value "$j/e2e-$e/tf-idol-generate.json" denoise_s), cuda_peak_allocated_gb $(value "$j/e2e-$e/tf-idol-generate.json" cuda_peak_allocated_gb); $(grep -h 'Maximum resident' "$OUT/logs/e2e-$e.log" 2> /dev/null | sed 's/^[[:space:]]*//')"
        [ -f "$j/e2e-$e.mem.csv" ] && echo "  $(peaks "$j/e2e-$e.mem.csv")"
    done
    echo "- Metal reference (seed 42, 50 steps, 262,144 Gaussians): picture 29.1 s, 2.71 GB peak footprint (2.40 GB"
    echo "  allocated); opaque photo 30.07 s, 2.77 GB; the SDK on MPS about 100.8 s, 12.88 GB."
    echo
    echo "## Parity with the SDK's noise (Metal: sample cosine 0.99835, sampled NN 0.00922, baseline 0.00896)"
    echo
    sed 's/^/    /' "$j/e2e-compare.json" 2> /dev/null || echo "    (no e2e-compare.json)"
    echo
    echo "Failed steps: $failed. Logs: logs/; outputs and memory samples: runs/."
} > "$OUT/summary.md"
echo "summary: $OUT/summary.md; $failed failed step(s)"
[ "$failed" -eq 0 ]
