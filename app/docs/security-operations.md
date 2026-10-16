# TensorFold Security Operations Checklist

This checklist covers controls that cannot be proven by application code alone.
Release owners should keep it current before distributing TensorFold outside a local
developer checkout.

## Release Trust

- Production macOS artifacts must be code signed and notarized.
- `package.json` must keep `build.mac.hardenedRuntime` set to `true`.
- Run `npm run pack:mac` / `npm run dist:mac` for releases. They require `CSC_NAME` and
  `APPLE_TEAM_ID` to exactly match the production-ready identity and Team ID pinned in the strict
  `build/release-trust.json` source, plus either a pre-stored notarytool keychain profile in
  `APPLE_KEYCHAIN_PROFILE` (and optional `APPLE_KEYCHAIN`) or
  `APPLE_API_KEY`/`APPLE_API_KEY_ID`/`APPLE_API_ISSUER`. Raw `APPLE_ID` /
  `APPLE_APP_SPECIFIC_PASSWORD` mode is refused because command failures and process listings can
  expose the password; create the keychain profile outside the build and CI log stream.
- The release wrapper must pass codesign verification and a notarized Developer ID Gatekeeper
  assessment for the unpacked app and the app independently mounted/extracted from every DMG/ZIP.
  Every contained app must match the baseline bundle metadata and CDHash exactly. DMGs are mounted
  read-only and detached on success or failure; ZIP extraction directories are always removed.
  Undefined, generic, caller-selected, Apple Development, and ad-hoc identities are failures.
- `npm run pack:mac:dev` is the only unsigned packaging path and its output must
  never be distributed.
- Run `npm run smoke:mac:packaged` after packaging instead of hand-writing an Electron launch.
  The harness accepts no profile override, passes only an allowlisted OS environment plus its own
  mode-0700 temporary `VONTRA_HOME`, suppresses bootstrap output, requires loopback health/UI/auth
  evidence, terminates and awaits the process, verifies the listener closed, and removes only the
  inode/device/owner-matched directory it created. Never use `HOME=...` as profile isolation.
  A configured `VONTRA_HOME` is applied before `app.whenReady()`: Electron `appData`, `userData`,
  `sessionData` (including Chromium cache), temporary files, logs, and crash dumps are created as
  private subdirectories beneath it, and real app-data legacy migration is a no-op.
- `build/entitlements.mac.plist` must not enable unsigned executable memory or
  disabled library validation without a documented exception.
- Update feeds must include a SHA-256 checksum and an Ed25519 signature made by the key whose
  public half is embedded in the application. Runtime environment variables cannot replace that
  trust root or disable verification.
- `build/release-trust.json` pins the release signer and public update-verification key.
  Both readiness flags must be true, and release signing keys stay outside the repository.
  Canonical strict JSON parsing rejects duplicate keys and comments; credentials cannot bypass
  the trust checks. Development packaging remains available for local testing.
- `release:manifest` requires `--private-key`; `release:manifest:dev` is an
  explicitly unsigned fixture path and must never feed production clients. The manifest command
  derives the supplied private key's public SPKI and fails unless it exactly matches the embedded
  trust root.
- Keep the update signing private key outside the repo, rotate it on suspected
  exposure, and record rotations in release notes. A trust-root rotation requires a
  reviewed app release that embeds the replacement public key; never distribute the
  private key with the app or place it in runtime environment variables.
- Update artifacts are downloaded by the main process into the private managed `updates` directory,
  not the browser or the shared OS temp directory. Every redirect is
  independently HTTPS/DNS validated and socket-pinned; the stream has a 1 GiB cap, lands in a
  unique atomic partial file, and is opened only after its actual SHA-256 matches the value
  bound by the signed manifest. The final file path, inode/device identity, size, and hash are
  revalidated immediately before opening. Partial or mismatched artifacts are deleted.
  The owner-only local-memory purge also securely removes this managed directory so verified
  installers do not accumulate indefinitely.

## WhatsApp Linked-Device Credentials

- The experimental QR bridge uses a host-injected Baileys `AuthenticationState` backed by one
  app-key protected JSON envelope per source. Credentials and Signal keys are never written via
  Baileys' plaintext multi-file sample store.
- On first use, recognized and bounded legacy multi-file state is migrated into the protected
  envelope, verified readable, then each plaintext file is overwritten and unlinked. Unsafe,
  malformed, or unrecognized legacy state fails closed and pairing remains disabled.
- Removing a bridge source securely shreds its protected state. Local-memory purge stops all
  bridge sockets, drains credential writes, shreds every linked-device state without following
  symlinks, and marks retained source settings unpaired. Provider and official WhatsApp Cloud
  credentials remain configured.
- Readiness warns when a paired linked-device credential is older than 90 days (or its age is
  unknown). Label, enablement, and connection-status edits do not refresh that age.

## Identity And Access

- The trusted local desktop GUI starts as `owner` so its operator can save settings and perform
  explicit administrative actions. Headless/server launches default to `operator`. An explicit
  `VONTRA_SECURITY_ROLE` or `VONTRA_ROLE` overrides either launch default; use `owner` for a
  headless process only during a deliberate administrative session.
- Owner-only actions include settings writes, approval-policy revocation, MCP
  server configuration changes, remote token reveal/regeneration, and local
  memory purge.
- Process roles are authorization boundaries for remote/headless principals; they are not an
  XSS boundary inside an owner desktop renderer. Token reveal/rotation, device-pairing issuance,
  local-memory purge, destructive checkpoint/workspace restores, and removal of a WhatsApp
  linked-device account therefore require a separate native main-process confirmation. Continue
  to treat renderer CSP, Trusted Types, sender validation, and dependency hygiene as primary
  controls—other owner IPC remains reachable if the trusted renderer itself is compromised.
- Project, room, ticket, and thread removal is recoverable archival rather than immediate erasure;
  relationship-aware restore validates the whole batch before mutation. Recreateable configuration
  deletion and ordinary settings saves remain inside the trusted-owner-renderer boundary: native
  confirmation is a deliberately narrow defense for the sensitive actions above, not a universal
  XSS boundary for every owner IPC operation.
- Review owner sessions during incident response and after plugin/MCP changes.

## Headless Orchestrator Bootstrap

- `npm run start:headless` force-enables the control plane and binds loopback by
  default. A startup that cannot listen exits non-zero instead of idling headlessly.
- The first successful bootstrap persists a 256-bit host token and prints that
  token plus one expiring browser pairing link exactly once. Treat terminal and
  supervisor output from that first launch as secret material.
- The built-in server is loopback-only. `VONTRA_HEADLESS_BIND_HOST` accepts only `127.0.0.1`; any
  other value fails startup. Access it through the printed SSH tunnel, or keep TensorFold on loopback
  behind a separate TLS reverse proxy you control. Do not bind TensorFold itself to `0.0.0.0`.
- `VONTRA_HEADLESS_BOOTSTRAP=1` is a deliberate recovery operation: it rotates
  the host token, clears paired sessions, and emits a new one-time pairing link.

## Dependency Reproducibility

- TensorFold uses the exact npm version in `packageManager` and `package-lock.json`
  is the only supported lock file. Install with `npm ci`, never pnpm/yarn/bun, then run
  `npm run deps:trusted-scripts` to execute only the allowlisted Electron/node-pty native
  setup that project-level `ignore-scripts=true` deliberately skipped.
- The install guard rejects mixed-manager metadata and a self-referential
  `node_modules/node_modules` symlink. Recreate a contaminated tree with `npm ci`.
  Rerun `npm run deps:trusted-scripts` after recreating it.
- `npm run install:verify-clean` performs a real clean install, the explicit
  Electron binary installation/node-pty trusted rebuild, and a native import smoke check in a temporary
  directory without deleting or rewriting the active `node_modules`.

## MCP And Third-Party Tools

- Prefer Docker Gateway launch mode for third-party MCP servers.
- Direct MCP launch requires a supported macOS sandbox. If no sandbox is
  available, TensorFold fails closed unless `VONTRA_ALLOW_UNSANDBOXED_MCP=1` is set
  for deliberate local development.
- Require signed MCP manifests for auto-connect or trusted servers; set
  `VONTRA_REQUIRE_SIGNED_MCP=1` in stricter environments.
- Keep MCP environment variables allowlisted and avoid forwarding provider API
  keys unless the server explicitly needs them.

## Data Protection

- Cloud model routes must not receive connector, chat, document, or project
  memory retrieval results by default.
- Use local providers for workflows that need email, Slack, calendar, browser,
  document, or durable memory retrieval context.
- Run git hygiene before commits and triage all unsuppressed critical, high, and
  medium findings.

### SSH Execution Boundaries

- HTTP-model threads can use an SSH workspace through the connected remote filesystem/command
  backend. A missing or disconnected SSH backend fails closed; TensorFold never substitutes a local
  directory that happens to have the same path.
- Host-local supervisors and processes—rooms, ticket workers, missions, loops, local PR worktrees,
  project terminals, and delegated CLI engines—are not available for SSH workspaces. These paths
  fail closed until they have an explicit remote implementation.
- Repository instruction and skill discovery/preview is unavailable for SSH workspaces until the
  scanner is backend-aware. This is surfaced as unavailable rather than reading a local lookalike.
- SSH runs may register bounded inline artifacts. Path-backed artifact registration is refused;
  legacy remote/unknown path records are metadata-only in preview, and redact/export fails closed
  because the local host cannot prove or safely read their bytes.

### Workspace Mutation Concurrency Limits

- Local add/move destinations use an atomic no-clobber link and replacement files are staged in
  the verified parent before an atomic rename. Edits also revalidate the target inode, metadata,
  and SHA-256 immediately before rename. This is an optimistic concurrency preflight, not a true
  filesystem compare-and-swap: portable Node does not expose an `openat`/`renameat` conditional
  replacement primitive, so a non-cooperating writer can still replace the target in the tiny
  check-to-rename gap. Keep work in version control, review diffs, and do not run competing writers
  against the same files when lossless coordination is required.
- SSH writes have the same limitation more broadly: SFTP v3 provides no server-atomic digest CAS
  or open-no-follow operation, so containment and expected-hash checks are preflight defenses.
  Treat the remote account's OS permissions and a server-side jail as the security boundary, and
  avoid concurrent remote writers during agent mutations.
- Git turn checkpoints restore only paths represented by the captured Git snapshot. New paths that
  appeared after capture are preserved and surfaced as warnings; Git-ignored paths are not covered
  by checkpoint rollback.

### Local Secret Encryption

- The default `profile-key` mode AES-256-GCM encrypts protected state with an app-managed key at
  `~/.vontra/.secretkey`; TensorFold does not use Electron `safeStorage` or the macOS Keychain. The key
  and profile files are owner-only, which protects a ciphertext file copied by itself from casual
  disclosure. A complete copy of `~/.vontra`, however, includes the decryption key and must be
  treated as sensitive plaintext-equivalent material. This mode is not a control against code
  already running as the same OS user.
- Managed deployments should generate exactly 32 random bytes in their secret manager and inject
  the canonical base64 encoding at runtime as `VONTRA_SECRET_KEY_BASE64`. Also set
  `VONTRA_REQUIRE_EXTERNAL_SECRET_KEY=1`; a missing key then fails closed instead of silently
  creating or falling back to `.secretkey`. A configured value that is not canonical base64 for
  exactly 32 bytes also fails closed.
- Hold the managed key independently from TensorFold. Do not place it in the repository, a project
  `.env`, `VONTRA_HOME`, app settings, profile backups, build or packaging environments, command
  output, or logs. TensorFold captures the value at startup and removes it from `process.env` before
  terminals, model CLIs, Git hooks, connector helpers, or other children can inherit it. Supply it
  from the runtime supervisor or secret manager on every launch.
- A fresh profile started with both managed variables does not create `.secretkey`. Protected data
  written under that external key cannot be decrypted from a copied profile alone. This claim is
  limited to encrypted data: ordinary non-secret metadata may remain readable, and access to the
  running process, OS account, or independently held key remains sufficient to read protected data.
- Transitioning an existing profile is different. The old `.secretkey` remains a read-only
  decryption fallback and readable settings/state is rewrapped as it is migrated, but historical or
  on-demand artifacts may still use the old key. Security controls therefore continue to report the
  copied profile as decryptable while `.secretkey` exists. Retain and protect that key until a
  deliberate migration and recovery validation has covered every required artifact; only then
  archive or remove it from the profile. There is no blanket migration-complete claim merely because
  the app started successfully.
- Losing the external key makes data written under it unrecoverable. Keep a separately encrypted,
  access-controlled recovery copy and test its restoration. Replacing the environment value is not
  an automatic key rotation: preserve the prior key until a planned migration has re-encrypted and
  verified all retained data.
- When a whole protected file cannot be decoded, TensorFold preserves a recovery copy, quarantines the
  live path, and refuses subsequent writes that could replace it with fallback/default data.
  Readiness reports the affected and backup paths. Restore the matching key/file, or use an explicit
  purge only when permanent loss of that quarantined state is understood and accepted.
- Pre-refactor `safeStorage` ciphertext is not decrypted by the current build because TensorFold does
  not access the macOS Keychain. Each affected source file is backed up, unresolved ciphertext is
  preserved through unrelated settings/MCP/webhook saves, and Readiness reports the affected count
  and an available recovery path. Re-enter each credential to replace its preserved ciphertext,
  verify the new value works, and retain every recovery backup until that verification is complete.

## Logging And Monitoring

- `security-events.jsonl` is local and hash chained; it is not a substitute for
  centralized monitoring in managed deployments.
- Set `VONTRA_STRICT_AUDIT=1` when audit write failures should block sensitive
  workflows.
- Preserve audit logs, run exports, and relevant connector events during
  incident response.
- For team or managed deployments, ship signed audit events to an external log
  store with retention aligned to the data classification below.

## Backup And Disaster Recovery

- Define RPO and RTO before relying on TensorFold for long-running missions or
  regulated work.
- Back up protected state, project memory, incidents, and run exports through an
  encrypted channel controlled by the user or organization.
- Test restore at least once per release cycle and record the restore date,
  operator, source backup, and result.
- Keep backups out of the source repo and out of update artifacts.

## Incident Response

- Maintain an owner for security triage, release signing, and user notification.
- For suspected credential exposure: revoke the credential, rotate it, run git
  hygiene against working tree and history, and inspect run exports for spread.
- For suspected MCP/plugin compromise: disable the server, preserve logs, revoke
  related approval rules, rotate forwarded environment variables, and require a
  signed manifest before reconnecting.
- For suspected update compromise: disable the feed, rotate update signing keys,
  publish a signed advisory through an independent channel, and block downloads
  until a verified manifest is available.

## Compliance Evidence

- Keep a data inventory for threads, connector events, project memory, webhooks,
  incidents, run exports, and audit logs.
- Define retention and deletion rules for each data class.
- Maintain a vendor/subprocessor register for model providers, update hosting,
  connector services, and remote orchestrator infrastructure.
- For SOC 2 / ISO 27001 alignment, map evidence to access reviews, change
  management, logging, incident response, backup/restore, and vendor-risk
  controls.
- For GDPR, document lawful basis, export/delete workflows, subprocessors, and
  cross-border transfer safeguards before processing personal data.
- HIPAA and PCI workloads require separate scoping before use; do not assume
  TensorFold is in scope-ready without a BAA/PCI control assessment.
