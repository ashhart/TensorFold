# TensorFold serving image for DeepSeek-V4.1-Flash on two DGX Sparks: the NGC PyTorch base the dev container
# runs, the Python dependencies baked in, and a snapshot of the source tree (build/tensorfold, staged by `make image`).
# CUDA extensions and Triton kernels build into the cache volume (/root/.cache: `make prebuild`), not here: they need
# the GPU.
FROM nvcr.io/nvidia/pytorch:26.07-py3

# a pinned image: `tensorfold update` does not apply, `make image` does
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 TENSORFOLD_NO_UPDATE_CHECK=1
# JIT caches in the cache volume (/root/.cache, CACHE_DIR), not in the container layer that every `make up` / restart
# recreates: Triton's kernels (default ~/.triton/cache) and the driver's PTX JIT cache (default ~/.nv/ComputeCache).
# `make prebuild` and every later container then share them; a kernel compiles once per source and specialization.
ENV TRITON_CACHE_DIR=/root/.cache/triton CUDA_CACHE_PATH=/root/.cache/nv-compute CUDA_CACHE_MAXSIZE=4294967296

# the base's torch must survive: fail the build if a dependency replaced it
RUN python -c "import torch; print(torch.__version__)" > /tmp/torch.before \
 && pip install --no-cache-dir "xgrammar==0.2.8" "transformers==5.18.0" "tokenizers==0.23.1" "safetensors==0.8.0" \
        "jinja2>=3.1" "huggingface-hub>=0.24" "numpy>=1.26" \
 && python -c "import torch; print(torch.__version__)" | cmp - /tmp/torch.before \
 && python -c "import xgrammar, transformers"

COPY tensorfold /opt/tensorfold
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/tensorfold \
 && python -m tensorfold --version
WORKDIR /opt/tensorfold

ARG TF_REVISION=unknown
LABEL org.opencontainers.image.title="tensorfold-dsv41" org.opencontainers.image.revision="${TF_REVISION}"
# the source snapshot's hash: keys per-image caches (the decode cost curves of multi.calibrate)
ENV TF_REVISION=${TF_REVISION}
