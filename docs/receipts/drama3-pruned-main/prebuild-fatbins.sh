#!/bin/sh
# Builds the Spark's (sm_121) kernel images for this commit into $PILOT/tensorfold-fatbins-<sha>, with SHA256SUMS, where
# run-script.sh takes them (the GPU job image has no C++ compiler for nvcc, nor network to get one). The Desk runs it on
# BOTTOM through `bottom shell`, from a clean checkout of the job's commit, with PILOT=~/Documents/drama3-pilot-20261014
# and IMAGE=<the job image, macbots:...>: the container runs as root with network to install g++.
set -eu
pilot=${PILOT:?the drama3-pilot folder}
image=${IMAGE:?the GPU job image}
[ -z "$(git status --porcelain)" ] || { echo "the checkout has changes; prebuild a clean commit" >&2; exit 1; }
sha=$(git rev-parse HEAD)
dir=$pilot/tensorfold-fatbins-$sha
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
