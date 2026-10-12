# TensorFold on Windows (native)

An unofficial port: `tensorfold-native.exe` runs on Windows x86_64 with an NVIDIA GPU through the same CUDA engine
the Linux build uses. Only the CUDA backend is ported; Metal is macOS's and HIP is not built.

## What you need

- An NVIDIA GPU and a current display driver (it ships `nvcuda.dll`; no CUDA toolkit is needed to run).
- To build: [Zig 0.17.0](https://ziglang.org/download/) (the version in `.zig-version`), the CUDA Toolkit
  (13.x tested) for `nvcc`, and Visual Studio Build Tools with the C++ workload (`cl.exe`, nvcc's host compiler).

## Build

From a "x64 Native Tools" prompt (or after running `vcvars64.bat`), so `cl.exe` is on `PATH`:

```bat
zig build native install -Dsm=89 -Dstrip=true ^
  -Dnvcc="%CUDA_PATH%\bin\nvcc.exe"
```

`-Dsm` takes the GPU's compute capability without the dot, comma separated for several: 86 (RTX 30), 89 (RTX 40),
120 (RTX 50). The server is `zig-out\native\bin\tensorfold-native.exe`; `zig-out\bin\tf-cuda-test.exe` runs the GPU
suites (`tf-cuda-test info`, `smoke`, `glue`, `sample`, `quant-ref`, `modelopt`, `exlr8`).

`zig build test` runs the host tests without a GPU.

## Run

```bat
tensorfold-native pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold-native models
tensorfold-native serve <model path> --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

The server listens on `http://127.0.0.1:8888/v1`. Ctrl+C (or closing the console) stops it cleanly.

## How the port works

- `zig/src/platform/sys.zig` (`@import("sys")`, wired into every module by `zig/build/sys.zig`) is the operating
  system as the engine calls it. On Linux and macOS it is `std.c`, `std.posix` and `std.DynLib` unchanged; on Windows
  `win_c.zig` and `win_posix.zig` give the same names over the UCRT, Win32 and Winsock.
- Files are opened with every share mode, so rename-over and unlink-while-open behave as on POSIX. `O_DIRECT` maps to
  unbuffered reads. `/tmp` is the user's temporary folder and `HOME` falls back to `USERPROFILE`.
- The Hugging Face cache links snapshot files to blobs. Windows allows symlinks only with Developer Mode or elevation,
  so `pull` falls back to hard links (no extra space).
- Checkpoint reads use threads (io_uring is Linux's).
- The API key file is checked for type and size; Windows keeps access in ACLs, not mode bits, so the "unreadable by
  other users" check is the folder's ACL rather than `chmod 600`.

## Differences from Linux

- No io_uring, `ulimit` or GB10 carveout, and no `SIGHUP` (the key file is still reread when it changes).
- `TF_HEAT_*` thermal pacing reads Linux's `/sys/class/thermal` and does nothing on Windows.
