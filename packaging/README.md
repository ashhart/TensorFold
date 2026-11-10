# Building native release archives

The release owner chooses `MAJOR.MINOR.PATCH`. No new release number is set here.
Use the pinned Zig in `.zig-version`, a Mac with the Xcode macOS SDK, and qualified CUDA fatbins:

```sh
zig build dist -Dversion=MAJOR.MINOR.PATCH -Ddist-fatbins=/absolute/path/to/fatbins -j2
zig build dist-smoke -Dversion=MAJOR.MINOR.PATCH -Ddist-fatbins=/absolute/path/to/fatbins -j2
```

`dist` keeps the three Unix archives as its default and writes archives and companion SHA-256 files to `zig-out/dist/`.
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
Python packages or model weights. The release builder uses Python 3.11 or newer for archive and privacy checks.
The engine and Unix smoke script do not require Python.

## Windows x86_64

Select Windows explicitly; this uses the same CUDA `distServer` and native serving entry point as the Unix release:

```sh
zig build dist -Ddist-platform=windows-x86_64 -Dversion=MAJOR.MINOR.PATCH \
  -Ddist-fatbins=/path/to/qualified-fatbins -Ddist-windows-runtime=/path/to/reviewed-overlay -j2
```

No macOS SDK is needed for this selection. `-Ddist-platform=all` includes Windows and the three Unix targets;
an individual Unix label also works. The default Unix commands and archive filenames are unchanged.
Windows writes `tensorfold-MAJOR.MINOR.PATCH-windows-x86_64.zip` and its `.zip.sha256` companion.
Its named root contains `bin/tensorfold.exe`, `lib/`, the usual notices, `SOURCE.json`, runtime inventory and `SHA256SUMS`.
Keep these directories together when moving the installation.

`zig build native` on Windows installs the serving executable at `zig-out/native/bin/tensorfold.exe`.
The build's separate `zig-out/bin/tensorfold.exe` remains the checkpoint utility.
Only the native installation and packaged filename change: `--version` still prints `tensorfold-native VERSION`,
which preserves the existing desktop parser. No inference implementation or server banner is changed.

The Windows overlay is an explicit, reviewed input; the packager does not collect DLLs from the builder's PATH.
It contains only `WINDOWS-RUNTIMES.json`, `WINDOWS-RUNTIME-NOTICES.txt`, inventoried DLLs under `bin/` or `lib/`,
and their nonempty `LICENSES/windows-*.txt` or `.md` files. Each DLL needs its SHA-256, version, public source URL
and license filenames in the schema-1 inventory:

```json
{"schema":1,"target":"windows-x86_64","libraries":[{
  "path":"lib/tfgrammar.dll","sha256":"<64 lowercase hexadecimal characters>",
  "version":"<reviewed upstream version>","source":"https://example.invalid/reviewed-runtime",
  "licenses":["LICENSES/windows-runtime-license.txt"]
}]}
```

Include the matching `tfgrammar.dll` and its xgrammar, dlpack and picojson notices for structured output and tools.
Additional CUDA or codec DLLs need their own inventory entries and required redistributable notices.
The packager rejects unlisted files, symlinks, non-AMD64 DLLs, missing licenses and system/driver DLLs such as `nvcuda.dll`.
This inventory records a release owner's reviewed inputs; it cannot establish their license, privacy or feature qualification.
No weights, Python runtime, installer, user cache or driver is collected or bundled automatically.

`SOURCE.json` is generated after linking and binds the executable hash to the Git commit/tree, Zig version and
fatbin input hashes. Production packaging refuses a dirty checkout. It records build identity, not GPU qualification.
`-Ddist-host-only=true` produces a separately named Windows host-only ZIP with `HOST-ONLY`; it cannot serve inference.
It is suitable for cross-compile and package checks while qualified CUDA inputs are unavailable.

On Windows with PowerShell 7, run the standalone smoke script against the matching archive:

```powershell
pwsh -NoProfile -File tools/release/smoke-windows.ps1 .\tensorfold-MAJOR.MINOR.PATCH-windows-x86_64.zip MAJOR.MINOR.PATCH
```

It checks the archive checksum and member paths, extracts into a fresh directory, and checks every file hash before
probing native version, serve help and schema-1 CUDA capabilities from an empty cwd and clean child environment. It checks that the
packaged grammar DLL loads. It opens no listener, loads no checkpoint and runs no GPU inference.
Host-only checks require the explicit `-AllowHostOnly` switch. CPU fixture coverage is available on any PowerShell host:

```sh
pwsh -NoProfile -File tools/release/check_windows_smoke.ps1
```

## CUDA box inputs

The CUDA box produces one flat directory, supplied as `-Ddist-fatbins=DIR`, with
`zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,120,121`: Ampere (sm_80 and sm_86), RTX 50-series and
RTX PRO Blackwell (sm_120), and GB10 (sm_121). `dist` refuses a file that lacks an image for any of the four. All of
these files are required:

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
`tools/zig/triton_aot_pack.py` writes this layout from a capture's manifest or Recorder launch dump.
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
