# Native runtime files

`bin/tensorfold-native --version` and `--help` need no model, driver, Python package or repository checkout.
The macOS binary targets the Apple M1 CPU instruction set and macOS 13 or later. Metal is supplied by macOS.
The Linux binaries target baseline x86_64 or aarch64 with glibc 2.28 or later. The NVIDIA driver supplies `libcuda.so.1`.

At this source revision, Nemotron's Metal kernels and draft vocabulary are embedded in the executable.
It does not load an adjacent Nemotron coop metallib. `lib/` holds `libtfgrammar` (`.dylib` on macOS, `.so` on Linux,
built against the same OS floor as the binary), which structured output and required tool calls load at their first
request; without it, those requests get HTTP 400 and everything else serves. It is xgrammar v0.2.8 (Apache-2.0) with
dlpack (Apache-2.0) and picojson (BSD-2-Clause); their notices are in `LICENSES/` and `THIRD_PARTY_NOTICES.md`.
CUDA `.cu` fatbins are embedded at build time from the qualified directory supplied to the release build.
JPEG pictures need libjpeg-turbo 3.1.4.1's `libturbojpeg`, which the archives do not include: the server loads it at the
first JPEG from the file `TENSORFOLD_LIBTURBOJPEG` names, `lib/`, beside the binary, Homebrew's `jpeg-turbo` on macOS or
the system's library path, and without that release refuses JPEG images by name while PNG images still serve.

Inference also needs a model directory or an existing Hugging Face cache, including config, tokenizer, template
and weight files. No model download occurs at startup. Flash Next Metal loads its checkpoint directly and needs no
captured directory; `TF_FLASHNEXT_DUMP` selects an optional older recording. CUDA Nemotron runs on the kernels built into the binary. A captured Triton set is optional: `TENSORFOLD_CUDA_KERNELS`
names one (a folder with `aot.json` and `cubins/`), `native` forces the built-in kernels, and a set found at
`share/tensorfold/cuda/sm<capability>/` relative to the executable is used when present.
These model-specific capture assets are not created by the release build. An optional capture root can be
packaged with `-Ddist-cuda-aot=DIR`. The release owner must qualify it for the intended GPU.

Flash Next speed-up settings also name the separately built MCDMA dynamic library and peer settings.
That library is not part of these archives. The server can read an API key file when requested.
None of those paths requires an installed Python package, though capture generation currently uses Python tools.
Archives marked `host-only` are CPU verification artifacts and cannot serve CUDA inference.


## Optional shared libraries

The release includes the native executable, built-in CUDA kernels and `libtfgrammar`. A driver-only NVIDIA host does not
supply cuBLASLt or NCCL. A GLM CUDA pair needs NCCL on both ranks; adding `--vision` also needs cuBLASLt.

| Feature | Library | Library filename (Linux; macOS where applicable) | Where it comes from |
| --- | --- | --- | --- |
| CUDA vision towers (`--vision`), cuBLASLt projections | cuBLASLt | `libcublasLt.so.13` (`cublasLt64_13.dll` on Windows) | NVIDIA CUDA 13 cuBLAS runtime; not the display driver |
| Drama3 CUDA prompt and codec matrix operations | cuBLAS | `libcublas.so.13` | NVIDIA CUDA 13 cuBLAS runtime |
| CUDA tensor-parallel ranks, including the GLM pair | NCCL | `libnccl.so.2` | NVIDIA NCCL 2 runtime compatible with CUDA 13 |
| RoCE/RDMA gathers | verbs | `libibverbs.so.1`; `/usr/lib/librdma.dylib` | rdma-core and adapter provider on Linux; the macOS RDMA runtime |
| Video inputs | TensorFold video | `libtfvideo.so`; `libtfvideo.dylib` | Optional `zig build video -Dffmpeg=PREFIX` runtime built against FFmpeg 9.0.2, installed in the server's `lib` directory |
| JPEG inputs | libjpeg-turbo 3.1.4.1 | `libturbojpeg.so.0`; `libturbojpeg.0.dylib` | Separately installed pinned decoder; `TENSORFOLD_LIBTURBOJPEG` can name its file |
| WebP inputs | libwebp 1.6.0 | `libwebpdecoder.so.3`; `libwebpdecoder.3.dylib` | Separately installed pinned decoder; `TENSORFOLD_LIBWEBP` can name its file |
| Structured output and required tool calls | TensorFold grammar | `libtfgrammar.so`; `libtfgrammar.dylib` | The matching release archive's `lib` directory, or `zig build grammar` |
| HIP tensor-parallel ranks | RCCL | `librccl.so.1` | ROCm's matching RCCL runtime |
| MCDMA mailbox/Metal speed-up communication | MCDMA RPC | `libmcdma-rpc.so`; `libmcdma-rpc.dylib` | Separately built MCDMA RPC helper; `MCDMA_RPC_LIBRARY` can name its file |
| MCDMA peer fabric | MCDMA fabric | `libmcdma-fabric.so`; `libmcdma-fabric.dylib` | Separately built MCDMA fabric helper; name the library's full path in the peer configuration file |

`capabilities --json` includes `optional_libraries`, with each dependency's feature, soname, `found` flag and installation
source. Discovery keeps every found library mapped until process exit, without calling its APIs or creating a CUDA context itself.
A library can register process or thread exit handlers during loading; those handlers must retain their code.
`found` means the loader found a file. It does not establish whether available libraries
meet the feature's required functions, decoder version and hardware checks.
Codec environment overrides take precedence, including a missing override. Capabilities does not read your peer
configuration, so a library configured only there may be reported missing.
No private candidate paths are emitted in the JSON. An unneeded missing optional dependency does not stop capabilities
or plain text serving; a requested feature refuses with the needed library and installation instructions.

On Linux, add the containing directory to `LD_LIBRARY_PATH` when an installed runtime is outside the loader's normal
search path. Use the library's full path in the peer configuration file when its option asks for one.
