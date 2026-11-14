# Is this machine supported? `tensorfold device`

Run `tensorfold device` to see what this machine is, which TensorFold backend runs on it, which checkpoint formats it
can serve and how much memory model weights may take. Bare `tensorfold`, `tensorfold help` and `tensorfold --help`
start with a two-line summary of the same, for example on a Mac Studio:

```text
This machine: Apple M3 Ultra, 256 GiB unified memory, macOS 27.2.1.
Supported: Metal. Formats: MLX 2/3/4/5/6/8-bit, NVFP4, MXFP4, FP8, EXLR8, BF16. Details: tensorfold device
```

or, where nothing runs, the reason, such as `Not supported: H100 80GB HBM3 is a Hopper GPU (sm_90): TensorFold has no
Hopper kernels now`.

## What it reads

The scan is quick and touches no GPU: `sysctl` values on macOS (chip, cores, RAM, `iogpu.wired_limit_mb`), `/proc` and
`/sys` on Linux (CPU, RAM, the PCI display devices, the NVIDIA driver's information files and AMD's gfx targets), and
DXGI's adapter list and the registry on Windows. An NVIDIA card's compute capability then comes from its name, so the
summary says `(likely)`.

`tensorfold device` also asks the GPU's driver in a CUDA build (names, compute capabilities, memory, integrated or not,
with no context made) and keeps that answer. Under WSL 2 the GPU shows only to the driver, so run `tensorfold device`
there before trusting the summary.

## What it decides

| Machine | With this binary |
| --- | --- |
| Apple silicon M1 to M5 | Metal. Llama needs M5 tensor units; before the M5 Gemma 4 runs on its row path without its drafters. |
| Intel Mac | Nothing: the Metal engines need Apple silicon. |
| GB10 (DGX Spark class), sm_121 | CUDA on unified memory. MiniMax-M3.1-Flash and `--slide` learning run here only. |
| RTX 5090 or RTX PRO 6000 Blackwell, sm_120 | CUDA, with the FP4 tensor-core paths. |
| RTX 3090 (sm_86), RTX 4090 (sm_89) | CUDA. NVFP4 and FP8 weights run by decoding to BF16; no NVFP4 or FP8 model run is recorded there yet. |
| Hopper (sm_90), data-center Blackwell (sm_100, sm_103), anything below sm_80 | Nothing: the build has no kernels for them now. |
| AMD or Intel GPUs | HIP or XPU builds; a CUDA build names the build that would run. |

A CUDA build covers the SMs its kernels were built for (the release: sm_80, sm_86, sm_120 and sm_121, with an sm_86
image also running on sm_89). The A100 (sm_80) has release images and kernel checks but no served model on record.
Formats are the union of what this binary's families read, less those of a family that cannot run on this machine;
`tensorfold capabilities --json` lists them per family.

The memory for model weights follows each backend's own admission: on Metal, 70% of RAM (at most the GPU working set)
less a 2 GiB reserve; on CUDA, device memory (system RAM on a GB10) less the CUDA reserve that
`TENSORFOLD_MEMORY_RESERVE_GIB` can change. `tensorfold device` lists the checkpoints in the Hugging Face cache that
this build serves and whether each fits.

## The profile

The scan is saved to `~/.tensorfold/device.json` (Windows: `%USERPROFILE%\.tensorfold\device.json`) with a schema
number, the scan time, the binary's version, a fingerprint of the hardware the OS lists, the machine and what it
supports. It holds no serial number, host name, user name or path. Later runs read it and scan again only when the
fingerprint or the binary's version changes, or with `tensorfold device --rescan`; `--json` prints it with the models.

`tensorfold serve` reads a current profile and stops early, before loading, only for what this machine certainly cannot
run: a GPU whose compute capability the driver reported and this build's recorded kernels do not cover, an Intel Mac,
or a family this device cannot run (Llama before the M5, MiniMax-M3.1-Flash off a GB10). With no profile, a stale one or
a verdict from a name, serve behaves as before.
`TENSORFOLD_DEVICE_CHECK=0` skips this check, for a profile that names the machine wrongly.

`zig build test-device` runs the host tests: each OS's recorded readings, the support table, the profile's round trip
and rescan rules, and the summary's exact words.
