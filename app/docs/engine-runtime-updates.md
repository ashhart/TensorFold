# Engine runtime updates

The desktop app can install and select an engine independently of an app release. Open **Models → Details → Engine updates** or the Models settings pane. Checking and downloading are explicit actions. Downloading never changes the engine serving a conversation.

A runtime switch requires idle local work, no model downloads, Living Weights off and no shared API session. The existing engine supervisor checks request leases inside its serialized queue. It reloads an idle model with the selected runtime and restores the previous runtime when activation or loading fails. **Use bundled engine** restores the engine shipped in the app. Downloaded model weights and conversations stay outside runtime installation directories.

## Trust and compatibility

The fixed catalog endpoint is `https://tensorfold.dev/studio/runtimes.json`. This change does not publish that endpoint or any release assets. An unavailable endpoint reports an inline error and leaves the current runtime usable.

The catalog uses the app's embedded Ed25519 public root and a runtime-specific signature domain. Renderer arguments contain only verified package IDs. They cannot supply a URL, public key, local executable or environment override. Only the owner can check, install, cancel or switch runtimes. Status is available to operators.

The JSON envelope has `payload` and `signature`, both canonical base64. The signature covers the UTF-8 bytes `TensorFold runtime catalog v1` followed by a newline and the decoded payload. The payload contains schema 1, publication and expiration timestamps, and packages. New catalogs must be fresh and cannot precede a previously accepted publication time. An installed package retains its signed catalog and remains usable offline after that catalog expires.

Each package names an immutable GitHub release archive, its complete byte count and SHA-256, OS, architecture, backend, engine version, minimum app version and protocol. Protocol 1 is the current capabilities and CLI contract. The runtime probe checks schema 1, exact engine version, the declared backend and the matching grammar library. A protocol change needs an app compatibility review. A compatible engine release needs a new signed runtime package and catalog, without rebasing the app.

The archive is gzip-compressed tar. Every regular file is in the signed inventory with its path, size, SHA-256 and executable bit. The inventory includes the executable, grammar library, GPU assets where needed, licenses and notices. Extraction rejects unexpected files, links, duplicate or unsafe paths and size mismatches. Compressed and expanded sizes are bounded. Archives are validated before extraction and installed through a private staging directory and atomic rename. Installed files are rehashed on startup and before selection.

A pending selection records the previous committed runtime. A crash during activation recovers that selection. A corrupt or unavailable installed engine falls back to the bundle. Invalid installations never become executable. Installed versions remain available for manual rollback.

Benchmarks validate the fixed community-v2 suite hash, receipt schema, settings and sample arithmetic. A compatible stable 1.x engine can emit that protocol without adding its version to an app allowlist. Saved historical v1 receipts remain readable; an unknown suite or schema is rejected.

## Preparing a runtime release

Run `node scripts/make-runtime-package.mjs` on the native build host with `--source`, `--output`, `--id`, `--version`, `--platform`, `--arch`, `--backend`, `--min-app` and `--url`. Optional `--notes` reads release notes from a file. Keep output outside the source runtime directory. The source layout contains `bin/tensorfold-native` or the Windows executable, its matching library directory, `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md`.

The tool inventories and probes the real engine and writes an archive plus unsigned package metadata. Transfer those two files from each qualified native build to the trusted signing host. Build hosts and CI do not need the update-signing private key.

On that host, run `node scripts/sign-runtime-catalog.mjs` with one `--package` metadata file for each platform, the protected `--signing-key` and a new `--output` catalog file. The signer independently checks every complete archive hash and checks every declared file inventory in private scratch space, without executing foreign binaries. It verifies the key against the embedded production public root and signs the combined catalog. Native execution and device qualification receipts still need release review.

For a one-host local release, the package tool also accepts `--signing-key`. Both tools create files locally, upload nothing and refuse to overwrite catalog output. Preserve earlier immutable archive bytes.

## Verification scope

Automated tests cover signature and catalog validation, portable paths, wrong platforms, links, duplicates, missing files, tampering, cancellation, offline restore, activation failure, crash recovery and busy request admission. The native Mac acceptance test also installs a real engine archive, reloads a pinned checkpoint, produces a reply, rolls back and produces another reply. Browser journeys exercise the actual controls and minimum window layout. A shared-code test is not a native Windows or Linux qualification receipt.
