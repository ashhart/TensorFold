# Building native release archives

The release owner chooses `MAJOR.MINOR.PATCH`. No new release number is set here.
Use the pinned Zig in `.zig-version`, a Mac with the Xcode macOS SDK, and qualified CUDA fatbins:

```sh
zig build dist -Dversion=MAJOR.MINOR.PATCH -Ddist-fatbins=/absolute/path/to/fatbins -j2
zig build dist-smoke -Dversion=MAJOR.MINOR.PATCH -Ddist-fatbins=/absolute/path/to/fatbins -j2
```

`dist` builds all three native servers and writes archives and companion SHA-256 files to `zig-out/dist/`.
The macOS target is fixed to `aarch64-macos.13.0` with `apple_m1`, regardless of the builder's CPU or `-Dcpu`.
Both Linux targets use the CUDA backend and baseline CPU features, with glibc 2.28 as the minimum.
The backend runs with release optimization; the HTTP server retains its safety checks.
Distribution binaries omit debug symbols so the archives do not carry builder paths in debug metadata.
`-Ddist-macos-sdk=DIR` overrides the SDK found by `xcrun`.
`dist-smoke` runs only the archive matching the builder's OS and architecture.

```text
tensorfold-MAJOR.MINOR.PATCH-macos-arm64.tar.gz
tensorfold-MAJOR.MINOR.PATCH-macos-arm64.tar.gz.sha256
tensorfold-MAJOR.MINOR.PATCH-linux-x86_64.tar.gz
tensorfold-MAJOR.MINOR.PATCH-linux-x86_64.tar.gz.sha256
tensorfold-MAJOR.MINOR.PATCH-linux-aarch64.tar.gz
tensorfold-MAJOR.MINOR.PATCH-linux-aarch64.tar.gz.sha256
```

Each archive has one named top-level directory containing `bin/tensorfold-native`, `lib/`, `LICENSE`, `NOTICE`,
`VERSION`, `RUNTIME.md`, `LICENSES/`, `THIRD_PARTY_NOTICES.md` and `SHA256SUMS`. Metal sources and the Nemotron draft vocabulary are embedded at this
revision, so no adjacent metallib is needed. The packaging does not include test programs, source, build caches,
Python packages or model weights. There is no Python dependency in the packaging or smoke scripts.

## CUDA box inputs

The CUDA box produces one flat directory, supplied as `-Ddist-fatbins=DIR`. All of these files are required:

```text
DIR/
  gdn.fatbin
  probe.fatbin
  qmm_group.fatbin
  qmm_prefill.fatbin
  experts.fatbin
  experts_prefill.fatbin
  experts_pack.fatbin
  prefill_attention.fatbin
  scan_rows.fatbin
  nemotron_ops.fatbin
  torch_argmax.fatbin
  torch_topk.fatbin
  torch_pointwise.fatbin
  torch_indexing.fatbin
  torch_movement.fatbin
  torch_nemotron_constants.fatbin
```

Build this input on the CUDA box with the same pinned Zig and qualified nvcc:

```sh
zig build fatbins -Dtarget=x86_64-linux-gnu -Dnvcc=/path/to/nvcc -Dsm=121
# Input for the Mac release builder is the resulting zig-out/fatbin/ directory.
```

`-Dsm` chooses the GPU SASS targets, independently of the Linux host CPU. The existing build defaults to SM121.
Use another list only after qualifying those GPUs. The same fatbins are embedded into the x86_64 and aarch64
servers; they contain GPU machine code, not host CPU code. Missing input files fail the build.
The release build does not install or download nvcc, create cubins, or run a GPU.

The captured Triton set is a separate optional input, supplied as `-Ddist-cuda-aot=DIR`:

```text
DIR/
  sm121/
    aot.json
    cubins/
      <captured-hash>.cubin
```

It is installed under `share/tensorfold/cuda/` inside each Linux archive, matching the executable-relative lookup.
Additional qualified `sm<capability>/` directories can share the same capture root.
The release owner must obtain and qualify these assets; packaging a file is not inference qualification.
Without this input, users must set `TENSORFOLD_CUDA_KERNELS` to their qualified capture directory.
Flash Next's Metal dump and the separately built MCDMA SDK remain external inputs described in `RUNTIME.md`.

## CPU verification before CUDA artifacts exist

```sh
zig build dist-smoke -Dversion=0.6.5 -Ddist-host-only=true -j2
```

Here `0.6.5` is the existing manifest version used only as a test fixture, not a proposed release.
This mode emits the normal macOS fixture archive and Linux archives whose names end in `-host-only`.
Each Linux archive also contains `HOST-ONLY` stating that it cannot run CUDA inference.
The release `dist` step refuses absent fatbins unless this verification mode is explicitly enabled,
and refuses combining verification mode with fatbins. It never emits a normal Linux inference archive without fatbins.

On each supported Linux machine, run the shared smoke script against that machine's archive:

```sh
sh tools/release/smoke.sh /path/to/tensorfold-MAJOR.MINOR.PATCH-linux-x86_64.tar.gz MAJOR.MINOR.PATCH
sh tools/release/smoke.sh /path/to/tensorfold-MAJOR.MINOR.PATCH-linux-aarch64.tar.gz MAJOR.MINOR.PATCH
```

Run only the command matching the machine. The script verifies the archive and every shipped file, extracts into
a fresh directory and executes version and help from a different empty working directory with a clean environment.
It needs neither the checkout nor a model, Python, the NVIDIA driver or a GPU. It retains its temporary receipt.
The script itself can be copied anywhere and run without the checkout.
On macOS it also checks the unpacked binary with `nm -m` and requires `memcpy` to import from libSystem.
All NOTICE-referenced license files are bundled and hashed.

Each packaging invocation uses a fresh unique stage, so rerunning a version cannot retain captures from a previous
larger input set. Stages stay in the build cache for inspection. The CPU regression runs full capture, smaller capture
and no capture against the same version/platform/output, checking exact archive members and hashes:

```sh
zig build test-dist-package
```

This development test uses Python's standard library; the release packager and smoke script remain POSIX shell.

## Homebrew draft

`homebrew/tensorfold.rb.in` installs the macOS archive under `libexec` and exposes `tensorfold-native` and `tensorfold`.
Replace `@VERSION@` and `@MACOS_ARM64_SHA256@` with the approved release version and companion checksum.
The formula declares Apple Silicon and macOS Ventura requirements and has CPU-only version/help checks.
The tap has not been changed, and the formula still needs review and an actual release URL before publication.
