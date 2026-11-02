# Desktop platform qualification

The renderer, agent runtime, conversations, approval policy and engine service are shared across
platforms. This branch adds native Linux engine archives and a Windows build/installer lane.
It remains a development candidate. The bundled baseline is TensorFold 1.2.0, release commit
`a388876f41e19f863ccd82e99e028bd8e9486802`. Published beta 10 still contains engine 1.1.4.

| Platform | Package | Engine 1.2.0 | Qualification |
| --- | --- | --- | --- |
| Apple silicon Mac | Development app bundle | Pinned release archive | Build, helper self-test and isolated packaged startup pass on macOS |
| Linux x64 | AppImage and Debian package | Pinned CUDA release archive | Local Ubuntu 24.04 under QEMU TCG: packaged startup, sandboxed renderer, Assistant and inbox bridge checks pass; AppImage approval survives reboot |
| Linux ARM64 | AppImage and Debian package | Pinned CUDA release archive | Local Ubuntu 24.04 VM: Debian installation, sandboxed graphical launch, AppImage approval and removal, packaged smoke, updater activation and rollback pass |
| Windows x64 | Local NSIS installer and application ZIP | Built from the exact 1.2.0 source commit | Local cross-build includes all 103 CUDA images and matching grammar DLL; Windows execution and installer qualification await a Windows VM |

Earlier Linux package checks passed in [CI run 40490220594](https://github.com/ashhart/TensorFold/actions/runs/40490220594).
The AppImage sandbox changes were built and checked locally on 30 October 2026. The 1.2.0 engine
base and refreshed GUI packages are checked locally on 1 November 2026. No CI run was triggered.
Packaged startup checks use a temporary profile and loopback endpoint, require health 200 and a
nonempty UI response, and require unauthenticated snapshot access to return 401. They clean up the
owned listener and profile. These checks do not exercise a graphical desktop or GPU inference.

## Build and check

Use Node 22 and npm 10.9.8. From `app`, install the locked dependencies with lifecycle scripts disabled,
then run the reviewed Electron/node-pty setup:

```sh
npm ci --ignore-scripts
npm run deps:trusted-scripts
npm run typecheck
```

On Linux x64 or ARM64:

```sh
npm run dist:linux
npm run smoke:linux:packaged -- --appimage
```

On Windows x64, the `Desktop platform packages` workflow first builds the engine from commit
`a388876f41e19f863ccd82e99e028bd8e9486802` with pinned Zig 0.17.0 and CUDA 13.2.0. It checks the
source and executable version before staging the binary and public notices. After that stage:

```sh
npm run dist:windows
npm run smoke:windows:packaged
```

Build the matching xgrammar library from commit `97787376faee5ed8466cfad57c99855e4ce2f6aa`.
Run the Windows stage from `app` with the engine checkout and xgrammar checkout as its two arguments.
It requires all 103 embedded CUDA images and a successful native capabilities probe, including the grammar library.
The pinned 1.2.0 loader uses `lib/libtfgrammar.so` on Windows too; that file contains the Windows PE DLL.
The CI definition builds that grammar library from the pinned xgrammar checkout before staging.
No CI run was triggered for the local updater and parity changes.

The Windows stage writes a source commit and executable SHA-256 to `BUILD-PROVENANCE.json`.
The installer preserves application data on uninstall. All platform packages retain the shared
Electron fuse policy. The Windows and Linux configurations exclude the Apple-only computer helper.

## Remaining parity and release gates

- Linux now supports staged writable commands and direct MCP isolation when bubblewrap is permitted
  by the OS. Host publication refuses changed Git controls and concurrent edits. Windows has a native
  LPAC helper and private runtime/workspace copies, with actual Windows qualification still pending.
  See [command and MCP isolation](process-isolation.md) for the contract and limits.
- Apple automation, computer control and VM integration
  need platform equivalents where applicable. They are not established by producing an installer.
- Windows grammar-library packaging includes the matching PE DLL and its 12 required exports.
  Native loading and structured-output behavior still need Windows qualification.
- Linux and Windows GPU inference, model load/cancel/recovery, hardware admission and performance
  require receipts from those devices. Standard hosted CI runners provide build/startup evidence.
- Native graphical window behavior, permissions, accessibility and clean-machine installation need
  platform checks beyond the headless loopback smoke tests.
- Public distribution still needs platform signing/trust and platform-specific update publication.
  The CI artifacts are development builds. Do not replace existing immutable beta 10 assets with them.

These gaps prevent a claim of full feature parity with the Mac app. Shared-source tests and browser
journeys cover common UI behavior; they do not establish the missing OS integrations.

## Independent engine updates

The Models details pane and Models settings now expose signed runtime installation, explicit activation
and rollback to the bundled engine. See [Engine runtime updates](engine-runtime-updates.md) for the trust
contract and separate build-host/signing-host workflow. The public runtime catalog has not been deployed.
Local acceptance covers real Mac model reload and replies, Linux install/select/rollback, and browser controls.

## AppImage sandbox integration

The AppImage now has a reviewed launcher and explicit, per-copy AppArmor integration for
restricted Ubuntu hosts. See [AppImage sandbox setup](appimage-sandbox.md) for prerequisites,
administrator approval, immutable installation, removal and qualification limits. The launcher
keeps Chromium sandboxing enabled and does not change global namespace restrictions.

## Engine 1.2.0 baseline

The Mac and both Linux archive pins include the release SHA-256 and source commit. Windows uses
the same source commit, with all 103 CUDA images and the matching grammar library. The Windows
CI definition reads the shared engine pin. Updating the baseline does not publish the independent
runtime catalog, replace a user-selected runtime or change the public desktop release.

Benchmark v2 receipts retain the same schema and frozen suite; saved 1.1.6 results remain readable.
The local 1.2.0 checks establish packaging, CLI and GUI compatibility. GPU model inference and
performance require separate device receipts.

The 1.2.0 image decoder requires libjpeg-turbo 3.1.4.1 for JPEG and libwebp 1.6.0 for WebP.
Those optional libraries are not part of the published engine archive. A missing or different
library is reported by `capabilities --json`; decoder availability is not implied by a successful
text-engine build or startup.
