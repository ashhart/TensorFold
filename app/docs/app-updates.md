# App Updates

TensorFold checks signed releases and downloads verified updates inside the app.
Production-signed macOS installations can install and restart after the user finishes active work;
development builds and installer-only releases retain the verified manual installer path.

## Trust And Feed

- The sole production feed is hardcoded as `https://tensorfold.dev/studio/update.json`. The legacy
  `settings.updateFeedUrl`, `VONTRA_UPDATE_FEED_URL`, and `SYNTHENA_UPDATE_FEED_URL` values cannot
  repoint it.
- The main process checks that feed shortly after startup and every six hours when automatic checks
  are enabled. The renderer can request a manual check but cannot supply a feed or public key.
- Every accepted release needs a SHA-256 checksum and an Ed25519 signature over TensorFold's canonical
  manifest payload. Verification uses the public trust root compiled into the application; runtime
  environment variables cannot replace or disable it.
- `build/release-trust.json` is the single strict-JSON source imported by the runtime and read by
  release preflight. Canonical serialization is enforced, so duplicate keys, comments, reordered
  fields, and TypeScript comment/string decoys cannot make packaging validate values different from
  those compiled into the application.
- The pinned update public key is marked production-ready. The release helper checks that the
  protected private key derives that exact public key before signing a manifest; private key
  material stays outside the repository and runtime environment.
  The former orphaned development key remains blocked. Production packaging rejects its immutable SPKI fingerprint even if its mutable ID/readiness
  labels are changed.
- The same file pins the exact macOS `Developer ID Application: Legal Name (TEAMID)` identity and
  Team ID, both marked production-ready. A matching Developer ID certificate is installed on the
  release Mac; signing still needs notarization credentials and successful artifact verification
  before distribution.

The supported release helper computes the checksum from the artifact bytes, confirms that the
supplied Ed25519 private key derives the embedded public SPKI, and signs the canonical payload:

```bash
npm run release:manifest -- \
  --version 1.0.1 \
  --artifact release/TensorFold-1.0.1-arm64.dmg \
  --download-url https://example.com/tensorfold/TensorFold-1.0.1-arm64.dmg \
  --update-zip "release/TensorFold-1.0.1-arm64.zip" \
  --update-zip-url https://example.com/tensorfold/TensorFold-1.0.1-arm64.zip \
  --notes CHANGELOG.md \
  --out release/update.json \
  --private-key /protected/release/update-signing-ed25519.pem
```

`release:manifest:dev` creates a visibly unsigned local fixture only. The application rejects that
fixture, so it must never be published to the production feed.

The optional ZIP arguments add `automaticUpdate` metadata inside the signed release payload.
Production signing verifies the exact packaged ZIP, its contained app and engine, the pinned
Developer ID authority and notarization against the built bundle before signing that ZIP's hash.
The ZIP architecture must match the DMG; a manual-only manifest can omit both arguments.

## Production Packaging

`npm run pack:mac` and `npm run dist:mac` fail before invoking the packager unless the checked-in
update root and macOS signer are both production-ready. Release mode requires `CSC_NAME` and
`APPLE_TEAM_ID` to exactly equal the repository pin; an arbitrary valid Developer ID organization
cannot redefine the expected publisher through environment variables. Notarization must use a
protected `notarytool` keychain profile or App Store Connect API-key files.

After packaging, TensorFold verifies the unpacked app's Developer ID authority, Team Identifier,
Gatekeeper notarization assessment, bundle identifier, version, and CDHash. A DMG is validated and
mounted read-only in a private temporary directory; a ZIP is extracted into a separate private
temporary directory. Each contained app is independently verified and must match the unpacked
app's CDHash and bundle metadata exactly. Mounts are detached (with a force-detach fallback) and
temporary directories are removed on both success and verification failure; an un-detachable mount
is left intact rather than recursively deleted.

## Verified Download

Once a newer signed manifest is accepted, the main process:

1. Revalidates and socket-pins every public HTTPS redirect hop.
2. Streams at most 1 GiB into a unique owner-only partial file under TensorFold's managed `updates`
   directory while computing SHA-256.
3. Deletes the download unless its bytes match the checksum bound by the signature.
4. Atomically finalizes the file, then immediately rechecks its managed path, regular-file status,
   device/inode identity, size, and hash before calling `shell.openPath` on that local artifact.

The UI distinguishes manifest-signature verification, artifact verification, and verification
failure. It never sends the browser directly to a manifest-provided download URL.

## Install and restart

Automatic installation uses [Electron's built-in macOS updater](https://www.electronjs.org/docs/latest/api/auto-updater),
which stages a ZIP through Squirrel.Mac and verifies that the replacement app matches the running
app's code-signing requirement.
The running app must first pass strict code-signature verification against the pinned production
Developer ID authority and Team ID, use a writable app installation, and match the ZIP architecture.
Apps running from a disk image, App Translocation, development builds and browser demos cannot
install automatically.
The originally published beta.1 predates this installation flow, so its first upgrade uses the
verified DMG/manual path; automatic installation is available in later releases containing this code.
General custom `VONTRA_HOME` profiles also use the manual installer, because Squirrel's native
LaunchServices relaunch does not preserve custom environment variables.

Downloading never quits the app or gives Squirrel a package while work is active.
The main process owns download progress, cancellation and the verified-ready state, so closing a
dialog does not lose progress.
A private ready receipt can survive relaunch, but never grants trust by itself: a fresh signed
release check must bind the same version, ZIP URL and SHA-256, and the managed artifact is hashed
again before its ready state is restored.

On an explicit Install and restart click, the host rejects active runs, approvals, jobs and
terminals, closes the workspace admission gate and durably saves app state.
Only then does it start a random-path-authenticated loopback feed serving the verified ZIP to
Squirrel; the public feed and download still use the existing HTTPS signature/hash/DNS checks.
The local feed checks Host headers, file identity and SHA-256 before serving the artifact.
After Squirrel reports `update-downloaded`, the host checks idle state and preservation again,
closes the private feed and calls `quitAndInstall` while retaining the admission gate.
Failed checks leave the current app open and report the failure.
As documented by Electron, a package already staged by Squirrel can also install on a later normal
quit; staging is therefore deferred until idle, preserved installation was explicitly requested.

Real release validation needs two successive production-signed builds in a disposable profile:
install the earlier build into a writable Applications folder, publish the later signed manifest
and ZIP, download it from the canonical feed, confirm active work/approvals prevent restart,
finish work, install and restart, then verify the later version, pinned signature and preserved
threads/settings/projects on relaunch.
Mock tests validate the signature binding, local feed and restart gate, but cannot prove native
Squirrel replacement/relaunch or notarization on an unsigned development bundle.

The native smoke command copies the earlier signed app into an owner-held mode-0700
`/private/tmp/vontra-packaged-smoke-*` profile and writes a sibling mode-0600 descriptor before
launching it; descriptors are never placed in distributable artifacts.
Recovery accepts only the exact fixed schema, actual bundle ancestry, exact private sibling profile
and loopback-only test port.
A recognized fixture with missing or invalid recovery refuses startup instead of opening real
default user data; ordinary Applications bundles never load this test descriptor.
The command requires both the earlier fixture and later target to contain that bootstrap guard,
and the target must pass its pinned Developer ID and notarized Gatekeeper checks.
Before a successful signed feed check it says updates have not been checked, rather than claiming
the installed version is current; installation requires the explicit Install and restart action or
the verified installer fallback when automatic installation is unavailable.

## Deployment Prerequisite

The renamed production feed is `https://tensorfold.dev/studio/update.json`; its availability has
not been verified as part of the application rename. Before distribution, release owners must
verify public DNS and HTTPS for that host and publish the signed manifest plus referenced artifacts
at the canonical URL, or make a reviewed source change to a real organization-controlled feed. This is an
external deployment blocker; the application must not regain a user- or environment-configurable
feed as a workaround. Packaging does not pretend that a local signature check provisions external
DNS or hosting, so release/deployment automation must retain this separate availability gate.
