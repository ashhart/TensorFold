# Windows 5090 preview producer

This recipe builds a Windows preview from one clean TensorFold commit on a CPU runner.
The archive contains bin/tensorfold.exe, a fresh bridge DLL, a source-built grammar DLL,
the unchanged official cuBLASLt runtime, licenses and source provenance.
It does not establish GPU correctness, tool-call reliability, cache equality or performance on that commit.
The previous cache candidate's small highest-thinking warm-code decode regression remains unresolved.

The workflow runs on a push to windows/5090-runtime-base or an authorized manual dispatch.
It has read-only repository permissions, no inference client, no release step and no connection to a development PC.
Artifact upload runs only after the exact ZIP and build evidence pass the automated review and final byte checks.
The maintainer must inspect the full source snapshot, history, identities and intended public vendor attribution before pushing it.

## Build inputs and behavior

The source directory is the six-file production include closure of TensorFold's tuned bridge.
source-manifest.json pins its bytes, seven v1 C exports and eight device transforms.
The ABI keeps the 128-byte Create, 160-byte Run and 64-byte Requirements structures.
Compatibility FP8 modes remain in that unchanged source, while this preview's intended profile uses BF16 and NVFP4.
native_quant_reference.cuh and the standalone experimental harnesses are not build inputs.

inputs.lock.json pins the official CUDA 13.5.0 redistributable metadata and Windows archives:
NVCC/CRT/cudart/libnvvm 13.5.37, CCCL 13.3.5.0.6, cuBLASLt 13.10.0.20 and cuobjdump 13.5.37.
Archive size/digest checks precede extraction; extracted files and merged inputs are inventoried.
The producer records and rechecks the selected MSVC and Windows SDK files, ordered search paths,
actual compiler commands, compiler logs and all output hashes.

The bridge preserves -std=c++17 -O3 --fmad=false --ftz=false and sm_120a.
This fresh port explicitly adds --cudart=static and /MT.
CUDA 13.5 defaults to hybrid cudart on Windows, so these are deliberate linkage differences from the historical producer.
They do not change the bridge source or device arithmetic flags.
There is no claim that the new DLL reproduces the historical DLL's digest.

The native build uses the existing Zig graph and complete source CUDA catalog:
x86_64-windows-gnu, safe, stripped, SM120 and qwen-prefill-max-rows=8192.
The graph retains each kernel's existing flags and architecture-specific SM120a selection.
It does not enable native-fp-decode.
Version 1.2.3 is the historical qualification fixture, not a new release or tag assignment.

runtime-inputs.lock.json pins xgrammar v0.2.8's annotated tag and exact commit,
its exact dlpack gitlink, complete public archive file digests and gitleaks 8.30.1.
The source checkouts must equal the pinned public archives byte for byte.
The existing zig build grammar graph compiles the library and TensorFold shim.
cpptrace is disabled and googletest is not built; their submodules are not fetched.
The Windows library installs into native/lib/tfgrammar.dll.

## Required private digest context

The repository secret WINDOWS_PREVIEW_PRIVATE_CONTEXT contains only a versioned digest list:

    {"schema":1,"normalization":"casefold-alnum-v1","sha256":["<64 lowercase hexadecimal characters>"]}

Generate each digest locally by case-folding an identifier, retaining only alphanumeric characters,
UTF-8 encoding it and applying SHA256.
Include single tokens and complete identifiers that matter to the local privacy review.
The scanner checks tokens and adjacent two/three-token forms, including separator variants.
Do not transfer plaintext private names, employer identifiers, device names or addresses to GitHub.
The workflow passes this nonplaintext configuration to the runner, never prints it and excludes it from all artifacts.
A missing, empty or malformed context fails the gate.

Generic home-directory, private-network, email, private-key and credential-URL rules supplement the digest context.
The secret scanner uses its pinned executable, explicit default rules, an empty ignore directory and no baseline.
The gate checks its exit code, report and exact scanned byte count, with target size skipping disabled.
A successful scan is evidence for the specified inputs, not a guarantee that no private information exists.

## Artifact review before upload

The gate verifies the source commit/tree and clean checkout, native executable hash,
complete catalog-derived fatbin set and each fatbin's exact embedding in the executable.
It checks the bridge output against the fresh producer receipt and requires its complete eight-transform payload.
It separately validates PE imports, exports, debug data, resources, certificates, overlay bounds,
CUDA registration wrappers and every accepted CUDA payload section.
An unbundled non-OS dependency is refused.

CUDA ELF/PTX images and compressed ELF sections are decoded and scanned.
The official cuBLASLt runtime also has 409 Zstandard-compressed CUDA ELFs in its data section.
Its CASK section interleaves CUDA containers with proprietary records:
those records retain their exact official SDK provenance and are scanned for readable content.
Their proprietary machine/data content is not represented as available source.

The 17 email-shaped coincidences in this exact vendor's encoded bytes are listed by offset and match digest.
Each requires proof that it is wholly inside a completely decoded/scanned Zstandard frame.
A new match at a different offset, a changed vendor digest or an undecodable frame fails.
This does not exempt a vendor path or an entire binary from scanning.

ASCII and UTF16 spans, names, decoded images, legal inputs and public build evidence are inspected.
Base64 is recursively inspected within bounds; a recognized encoded archive/binary requiring another decoder is refused.
Extraction limits, unknown payload kinds, unreviewed sections, changed inputs and failed tools fail the gate.
Private decoder reports, raw findings, diagnostic logs, readable corpora and context configuration are never uploaded.
A failed compiler stage can print only a normalized diagnostic tail that passes both privacy and secret scans.

The shared Windows packager preserves bin/tensorfold.exe and the native CLI banner.
The reviewed archive then receives a model-free Windows CLI/grammar-loading smoke check.
A final check revalidates the source snapshot and precise publication file hashes immediately before upload.

## Runtime and license boundaries

The archive retains the exact cuBLASLt DLL from its pinned official public NVIDIA archive.
The bridge uses static cudart and MSVC CRT linkage; its fresh PE imports must confirm that no extra CRT DLL is needed.
Windows OS DLLs are external system components.
A compatible NVIDIA R620-or-newer driver supplies nvcuda.dll and the driver's CUDA runtime provider.
cuBLASLt has an NVRTC dynamic-loader candidate; its optional JIT/provider paths remain unqualified on this fresh build.
A loader string alone is not treated as a mandatory PE import.

NVIDIA license files, TensorFold attribution, xgrammar/dlpack/picojson notices and Zig/LLVM/MinGW runtime notices are retained.
Original license text and its exact source binding remain available in the provenance.
Public attribution emails are recognized only by exact values from verified shipped legal inputs.
Private-context matches remain failures even when a legal input has an intended public email.

The first hosted build records the actual MSVC/SDK/Zig input bytes and rolling runner image.
NVCC-generated internal symbol names can vary between builds.
This recipe provides source/toolchain/output provenance and a replay contract, not a bit-reproducibility guarantee.
The bridge producer's --expected-host-manifest option can require an exact previously recorded MSVC/SDK snapshot.

Primary references:
[CUDA compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/),
[CUDA binary tools](https://docs.nvidia.com/cuda/cuda-binary-utilities/),
[official redistributables](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json),
[xgrammar source](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa).
