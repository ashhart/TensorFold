"""CUDA verification preserves serial bits with row-invariant kernels and fp32 partials gathered and summed in rank order."""

import os

LATENT = os.environ.get("TF_GLM_LATENT", "1") != "0"   # DSA caches its 512-wide latent unless TF_GLM_LATENT=0
