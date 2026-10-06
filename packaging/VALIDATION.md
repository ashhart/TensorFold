# Packaging checks, 6 October 2026

The base is `db281878d`. All commands used the pinned Zig 0.17.0 and a task-local global cache.
The existing manifest version `0.6.5` was used only as a fixture. No release number was chosen.

| Check | Result |
| --- | --- |
| `zig build dist-smoke -Dversion=0.6.5 -Ddist-host-only=true -j2` | All three archives built; macOS version, help and serve help passed from an empty cwd with a clean environment. |
| `zig build test -Dcpu=apple_m1 -j2` | 45 build steps succeeded, 221 host tests passed. |
| `zig test build.zig --test-filter 'release versions'` | Two tests passed, including strict version acceptance and refusal cases. |
| Archive contents and hashes | All three fixture archives had the expected layout, matching internal and companion SHA-256 files and the expected Mach-O or ELF CPU architecture. |
| macOS deployment target | Mach-O declares macOS 13.0; compiler arguments select `apple_m1`. |
| Builder paths | Stripped binaries contained no builder home path bytes in the checked artifacts. |
| Release refusals | Missing version, absent fatbins, a prerelease version, conflicting verification options, a missing fatbin and an empty fatbin were all rejected. |
| CUDA production build branch | Both Linux architectures built with deliberately synthetic fatbin inputs in a separate ignored cache prefix. Embedding and the optional executable-relative `sm121/aot.json` and `cubins/` layout were checked. These inputs are not valid CUDA code and this is not CUDA qualification. |
| Direct server harness | `zig/tests/server/build.sh` built the fake and engine-less servers; the native CLI version, help and capabilities version matched the manifest. |
| Homebrew draft and shell scripts | Ruby syntax and shell syntax checks passed. The tap was not changed. |

The default output contains the macOS fixture archive and the two explicitly labelled Linux host-only archives,
with their companion checksum files. Linux version/help execution was not attempted on macOS.
The shared smoke script is ready for each matching Linux host. No GPU, model, network, install or download run occurred.
Actual CUDA fatbins and Triton captures still need release-time generation and inference qualification on a CUDA box.
The Homebrew template still needs the approved version, real archive checksum and tap review.

Runtime dependency inspection at this pin found no direct Python-package or repository-source read by the native server.
Model/config/tokenizer/template/weight files, Flash Next capture assets, CUDA Triton captures, speed-up settings and
the separately built MCDMA library remain runtime inputs; `RUNTIME.md` records their lookup paths.
The embedded-source Nemotron Metal engine at this pin does not load an adjacent coop metallib.

## Packaging corrections, 6 October 2026

Each invocation now stages in a fresh `mktemp` directory. `zig build test-dist-package` passed full capture to smaller
capture to no capture with the same version, platform and archive output, without retaining an old cubin or SM directory.
The regression against the original `f4ae084b4` script failed at the stale-cubin assertion, using an argument adapter only.
`LICENSES/` and `THIRD_PARTY_NOTICES.md` are now shipped, included in SHA256SUMS and included in the Zig package paths.
The macOS `dist-smoke` check runs `nm -m` on the unpacked release binary and requires libSystem's memcpy import.
The updated fixture dist-smoke passed, all three archives built and their legal files and hashes were checked.
No GPU, model, download, install, tap change or push occurred.
