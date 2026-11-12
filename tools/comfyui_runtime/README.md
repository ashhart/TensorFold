# TensorFold runtime installer for ComfyUI

This standalone Python CLI installs the latest stable TensorFold native CLI and builds the MOSS worker required by this integration. It uses only Python's standard library. ComfyUI nodes read the installed manifest and never check for updates or download code.

## Install

The first MOSS-only PR is scoped to experimental BF16 generation on Linux aarch64 with NVIDIA GB10, SM121. Use Python 3.10 or later, Git, Zig **0.17.0**, and a complete CUDA 13 toolchain with matching `nvcc`, NVVM, CRT and CCCL headers.

SM120 is refused by name. `install` and `update` read every GPU's compute capability with `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` before any download or build, and `run` reads it again before the worker starts. Any GPU other than 12.1 stops them with `MOSS ComfyUI generation is qualified on GB10 (sm_121) only; this GPU is sm_120` (naming the capability found). A capability that cannot be read, because `nvidia-smi` is missing, the driver lacks the field, it lists no GPU or it times out, is refused rather than assumed. The worker is compiled for sm_121 alone, with no sm_120 image or PTX, so its kernels cannot load on an SM120 GPU even when it is started directly. No SM120 workflow, numerical comparison or live kernel-contract check has run. Before this check, Linux x86_64 passed software tests and an actual native build on a CPU-only host; a host without an sm_121 GPU is now refused at install. Windows and macOS media installation fail before network access or builds.

Generation also requires the NVIDIA driver and CUDA 13 cuBLASLt runtime libraries. The dynamic loader must find `libcuda.so.1` and `libcublasLt.so.13` (or `libcublasLt.so`) on its search path. An nvcc-only installation does not provide these generation prerequisites.

```sh
python installer/tensorfold_runtime.py install \
  --comfy-root /path/to/ComfyUI \
  --backend cuda \
  --zig /path/to/zig \
  --nvcc /usr/local/cuda/bin/nvcc
```

The command obtains the latest stable release from `ashhart/TensorFold`, matches the exact Linux archive and companion checksum, verifies the archive and every internally listed file, and builds the reviewed media revision below. Use ComfyUI's default `models` directory with this installer. It installs only under `ComfyUI/models/tensorfold`; no global packages, shell startup files or services are changed. Builds use two concurrent jobs and allow up to two hours for each native build command. The patched `moss-image-cuda` target builds only the three denoiser kernel images. This setup does not build unrelated model families.

The standalone CLI version and media source are independent. The latest CLI archive does not contain the MOSS backend. Updating the CLI never silently changes the media source revision.

## Compatibility pins

| Component | Pin |
|---|---|
| TensorFold media source | `1ddac962008367ec2c50e10202c910ecf430b5eb` |
| Official source URL | `https://codeload.github.com/ashhart/TensorFold/tar.gz/1ddac962008367ec2c50e10202c910ecf430b5eb` |
| Source archive SHA-256 | `013d687c5b72718220ac8e137252492d72be85d27ff9cec37525d714c52dc2ba` |
| ComfyUI compatibility patch SHA-256 | `e6156ed11263a2d32586245b1702a6bc47ac0049e65bf1724afad69242eb617f` |
| Zig | `0.17.0`, checked against the source `.zig-version` |
| Compiled CUDA targets | `121` only (`-Dsm=121`); live generation validated on GB10 SM121 only |

The bundled `tf_installer/moss-comfy.patch` is verified before application. It changes numerical materialization, reduction and cuBLASLt planning to match ComfyUI, adds a MOSS-only attention entrypoint, and adds the narrow build target. It is applied to the extracted build copy, including when that copy is nested inside a ComfyUI Git repository. The supplied checkout is never edited. `runtime.json` records both the source revision and patch digest.

The source archive pin was taken from the exact official URL above. An optional `--source-archive /path/to/source.tar.gz` uses the same checksum verification. An optional `--source-checkout /path/to/TensorFold` requires the exact commit and a clean Git working tree, then builds an archive of tracked Git objects in a separate directory. Existing source checkouts are never built in place.

## Run

```sh
python installer/tensorfold_runtime.py run --comfy-root /path/to/ComfyUI -- TRANSFORMER_DIR CASE OUT_DIR
```

`run` refuses any GPU but GB10 (sm_121), verifies the active manifest and the worker's SHA-256, then replaces itself with the installed `tf-moss-image`, passing the arguments after `--` unchanged. The worker never starts, and so never loads weights, on a refused GPU.

## Updates and status

```sh
python installer/tensorfold_runtime.py update --comfy-root /path/to/ComfyUI --nvcc /usr/local/cuda/bin/nvcc
python installer/tensorfold_runtime.py status --comfy-root /path/to/ComfyUI
```

`update` resolves the latest stable standalone engine automatically. If the installed engine, media revision and compatibility patch are current, it keeps the existing installation. An old manifest without the compatibility digest rebuilds MOSS before activation, including when the CLI version is unchanged. When only the standalone engine changes, it reuses the verified MOSS worker without requiring Zig or nvcc and without rebuilding them. Downgrades are refused. A changed media revision requires an explicitly updated integration release and its compatibility pin. If an existing installation fails integrity checks, `update` stops; `install` builds a verified replacement while preserving the old manifest until activation.

`status` is offline: it checks the manifest, artifact SHA-256 values and required executable permissions without executing the engine, loading the library, querying a GPU or making network requests. If executable permissions are missing, `update` stops and `install` builds a replacement, including when the latest engine version is already installed.

Each successful install occupies a new directory under `models/tensorfold/installs`. The installer checks the native CLI version and MOSS invocation contract before atomically replacing `runtime.json`. A failed download, hash verification, build or pre-activation probe leaves the previously active manifest intact. Failed-run logs remain under `models/tensorfold/logs/failed-*`, and the CLI prints their location. At most 16 logs of up to 1 MiB each are retained per failed run; larger logs keep their beginning and end. Existing install directories remain available while running ComfyUI processes finish using them. Stop and restart ComfyUI after an update to load the new worker. No automatic cleanup deletes older runtimes.

The checksum files for the latest CLI are fetched from the same official release over HTTPS; they provide integrity and release consistency, not an independent publisher signature. Media source additionally has the fixed checksum above. Archive traversal, links, special files, duplicate members, extra unhashed files and excessive sizes are refused.

## Verification

```sh
python -m unittest discover -s installer/tests -v
```

These tests exercise archive attacks, release metadata, exact source/toolchain pins, GPU admission at install and run, subprocess failures, deadlines and descendant cleanup, offline status, tampering and failed-build/activation preservation. They do not establish image correctness, GPU compatibility or performance. The installer makes no claim that its probes qualify real generation.

## Prepare the official MOSS checkpoint

The pinned native loader expects separate Q/K/V weights. ComfyUI's published BF16 MOSS checkpoint combines them. Convert that file once before selecting it in TensorFold MOSS Image:

```sh
python tools/comfyui_runtime/prepare_moss.py moss_image_base_bf16.safetensors moss_image_base_tensorfold_bf16.safetensors
```

This offline command writes a new safetensors header and copies the original tensor payload unchanged. It uses bounded memory, needs about 10 GB of extra disk space, and refuses to overwrite an existing file. Keep the original checkpoint for standard ComfyUI comparisons. Existing unfused BF16 checkpoints need no conversion; INT8, FP8 and other quantized files are unsupported.
