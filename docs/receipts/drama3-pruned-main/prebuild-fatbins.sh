#!/bin/sh
# Builds the Spark's (sm_121) kernel images for this commit into $MODELS/tensorfold-fatbins-<sha>, with SHA256SUMS,
# where run-script.sh takes them. For a machine with the CUDA toolkit but no C++ compiler for nvcc (nor network to
# install one): run it from a clean checkout of the commit, with MODELS=<the models dir> and IMAGE=<a container image
# with this repo's Zig toolchain>. The container runs as root with network so apt can add g++. On a machine with nvcc
# and a C++ compiler, skip this script and build with -Dnvcc as the guide's Build step does.
set -eu
models=${MODELS:?the models folder}
image=${IMAGE:?a container image with the repo Zig toolchain}
[ -z "$(git status --porcelain)" ] || { echo "the checkout has changes; prebuild a clean commit" >&2; exit 1; }
sha=$(git rev-parse HEAD)
dir=$models/tensorfold-fatbins-$sha
[ ! -e "$dir" ] || { echo "$dir exists" >&2; exit 1; }
mkdir "$dir"
docker run --rm --user 0 --entrypoint /bin/sh -e OWNER="$(id -u):$(id -g)" -v /usr/local/cuda-13.0:/usr/local/cuda:ro \
    -v "$PWD:/src:ro" -v "$dir:/out" -w /src "$image" -euc '
trap "chown -R $OWNER /out" EXIT
apt-get update -qq && apt-get install -y -qq g++ >/dev/null
work=$(mktemp -d)
ZIG_GLOBAL_CACHE_DIR=$work ZIG_LOCAL_CACHE_DIR=$work zig build fatbins -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc --prefix "$work/zig-out"
cp "$work"/zig-out/fatbin/*.fatbin /out
cd /out && sha256sum *.fatbin > SHA256SUMS' || { status=$?; rm -rf "$dir"; exit $status; }
echo "$dir: $(sha256sum < "$dir/SHA256SUMS" | cut -d" " -f1)"
