# TensorFold on Windows (native)

`tensorfold-native.exe` runs on Windows with an NVIDIA GPU through the same CUDA engine the Linux build uses. Only the
CUDA backend is built on Windows; Metal is macOS's and HIP is not built. Windows is supported from source only.
The PR reports RTX 4090 (`x86_64-windows`, `sm_89`) and NVIDIA RTX Spark N1X (`aarch64-windows`, `sm_121`) runs.
Those device receipts have not been reproduced on the maintained tree.

## What you need

- An NVIDIA GPU and a current display driver (it ships `nvcuda.dll`; no CUDA toolkit is needed to run).
- To build: [Zig 0.17.0](https://ziglang.org/download/) (the version in `.zig-version`), the CUDA Toolkit
  (13.x tested) for `nvcc`, and MSVC C++ Build Tools with the C++ workload (`cl.exe`, nvcc's host compiler).

## Build

From a "x64 Native Tools" prompt (or after running `vcvars64.bat`, or `vcvarsall.bat arm64_amd64` on Windows ARM64), so
`cl.exe` is on `PATH`:

```bat
zig build native install -Dsm=89 -Dstrip=true ^
  -Dnvcc="%CUDA_PATH%\bin\nvcc.exe"
```

`-Dsm` takes the GPU's compute capability without the dot, comma separated for several: 86 (RTX 30), 89 (RTX 40),
120 (RTX 50), 121 (GB10 / RTX Spark N1X). The server is `zig-out\native\bin\tensorfold-native.exe`;
`zig-out\bin\tf-cuda-test.exe` runs the GPU suites (`tf-cuda-test info`, `smoke`, `glue`, `sample`, `quant-ref`,
`affine8`, `modelopt`, `exlr8`).

`zig build test` runs the host tests without a GPU.

## Run

```bat
tensorfold-native pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold-native models
tensorfold-native serve <model path> --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

The server listens on `http://127.0.0.1:8080/v1`. Ctrl+C (or closing the console) stops it cleanly.

## Models on CUDA

Nemotron 3.5 Lightning serves as on Linux. The Qwen3.5 hybrids have a CUDA engine of their own in this port
(`zig/src/families/qwen3_5/cuda*.zig`, kernels `zig/kernels/cuda/qwen35*.cu`), the Metal kernels' arithmetic with
the same rounding points, so a window of up to 16 rows reproduces one-token steps bit for bit:

```bat
tensorfold-native serve <Qwen3.5-2B-MLX-4bit> --name qwen2b --context 32768
tensorfold-native serve <Qwen3.8-27B-MLX-4bit> --name qwen27b --context 8192 --parallel 4 ^
  --drafter <Qwen3.8-27B-DFlash2>
```

- Qwen3.5-2B: about 340 tokens/s for one request on an RTX 4090, 1,500 across eight.
- Qwen3.8-27B (16 GiB): 40 tokens/s plain, at the card's memory bandwidth; 240 across eight requests.
- `--drafter` (4-bit only) loads z-lab's DFlash2 draft model: the target writes five layers' residuals into a ring
  for it, and it drafts a tree of up to 12 lanes a round that the round loop's planner shapes from what the target
  took. A lone greedy request decodes at 80 to 160 tokens/s by how predictable the text is (prose at the low end,
  code at the high), sampled ones at 50 to 70, and four at once reach about 160 aggregate; replies stay identical to
  plain decoding. Loading takes about 80 s: the drafter's weights are quantized to 4 bits on the host and every
  shape is run once, as the driver stalls a process's first use of each kernel and buffer.
- A verify window's rows are not snapshotted: each stream logs its rows' recurrence updates and a keep replays the
  accepted path, so the 27B records 32 rows a forward where 8 fit before.
- `TF_TIMING=1` prints each verify, keep, absorb and hold's milliseconds and the drafted shape (chain or lanes);
  `TF_NO_TREES=1` drafts chains only.
- No prompt cache (`--prompt-cache-gib`), `--learn` or `--slide` for these two models.

### Mixed-width checkpoints

The Qwen3.5 engine also loads MLX checkpoints whose modules carry their own widths (the config's `quantization`
table names a width per module, as mlx-lm's mixed recipes and the OptiQ and oQ conversions write), 2 to 8 bits in
groups of 64. Each module stays at its own width: 4-bit modules on the 4-bit tiles as before, 2, 3, 5 and 6-bit ones
on `affine8.cu`'s tiles of that width, and 8-bit ones on its byte tiles (a 7-bit module would take the 8-bit tiles).
Every width runs the 4-bit lane GEMV's arithmetic and hands its MMAs the same bf16 operands, so a code gives the
same bits whatever width it is stored at, and nothing is rounded. An embedding of a width other than 4 or 8 is
widened to the next at load. Windows still reproduce one-token steps bit for bit, and a uniform 4-bit checkpoint
loads and runs exactly as before.

- Tested on an RTX 4080 SUPER (16 GB): `mlx-community/Qwen3.5-4B-OptiQ-4bit` (4 and 8 bits, a tied 8-bit
  embedding) at about 130 tokens/s for one request and 400 across four, and `mlx-community/Qwen3.5-9B-OptiQ-4bit`
  (an untied 8-bit head) at about 78 and 260; replies alone, at four streams and with `"draft": false` equal.
- `TensorFold/Qwen3.8-27B-oQ2` (mostly 2-bit, 136 modules at 5 bits) takes 10.3 GiB, so the 27B fits a 16 GB card:
  one stream at a 32k window, about 45 to 64 tokens/s with drafts. The card itself calls oQ2 an aggressive
  low-bit release, and its replies show it.
- Prompt chunks take a GEMM over the same tiles (`affine_prefill.cu`) with the 4-bit prompt GEMM's arithmetic (each weight rounded once to
  bf16, one fp32 chain over K), so a code of up to 4 bits gives the bits the 4-bit prompt GEMM gives it. On the
  9B 8-bit it lifts prompts from about 1,500 to 1,900 tokens/s; the 27B oQ2's prompts (about 770 tokens/s) were
  already bound by the card's bf16 tensor rate with fp32 accumulation, not by weight reads.
- `tf-cuda-test affine8` checks each narrow width's products against its codes widened onto the 8-bit tiles, bit
  for bit, the products against float64, every row count and split-K path byte for byte, and the prompt GEMM
  against the 4-bit and 8-bit tiles' prompt GEMMs.

## How the port works

- `zig/src/platform/sys.zig` (`@import("sys")`, wired into every module by `zig/build/sys.zig`) is the operating
  system as the engine calls it. On Linux and macOS it is `std.c`, `std.posix` and `std.DynLib` unchanged; on Windows
  `win_c.zig` and `win_posix.zig` give the same names over the UCRT, Win32 and Winsock.
- Files are opened with every share mode, so rename-over and unlink-while-open behave as on POSIX. Checkpoint reads
  skip the page cache (`FILE_FLAG_NO_BUFFERING`, as `O_DIRECT` on Linux). `/tmp` is the user's temporary folder and
  `HOME` falls back to `USERPROFILE`.
- The Hugging Face cache links snapshot files to blobs. Windows allows symlinks only with Developer Mode or elevation,
  so `pull` falls back to hard links (no extra space).
- Checkpoint reads use threads (io_uring is Linux's).
- Integrated unified-memory GPUs on Windows (such as RTX Spark N1X `sm_121`) admit against the WDDM CUDA device pool
  (`cuMemGetInfo_v2`) while keeping the unified 10% reserve and 4 GiB default floor. A fresh device receipt must check
  the WDDM budget, allocation deltas and equality cells; source admission alone does not qualify the device.
- The API key file is checked for type and size; Windows keeps access in ACLs, not mode bits, so the "unreadable by
  other users" check is the folder's ACL rather than `chmod 600`.

## Differences from Linux

- No io_uring, `ulimit` or Linux `/proc/meminfo` probe, and no `SIGHUP` (the key file is still reread when it changes).
- `TF_HEAT_*` thermal pacing reads Linux's `/sys/class/thermal` and does nothing on Windows.
