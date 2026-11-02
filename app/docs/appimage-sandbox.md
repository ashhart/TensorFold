# AppImage sandbox setup

The AppImage keeps Chromium sandboxing enabled. It does not fall back to a sandbox-disable flag.
On hosts that allow user namespaces, the portable app starts directly.

Ubuntu 24.04 can restrict user namespaces through AppArmor. TensorFold offers a one-time,
administrator-approved integration on these hosts. Setup copies the exact image, verifies its
SHA-256 and extracts a root-owned, read-only application under
`/opt/tensorfold-appimage/<digest>/app`. A profile under `/etc/apparmor.d/` allows user namespaces
for that copy's executable. It contains no user-directory or temporary-mount wildcard.
The original portable file redirects to the approved copy on later launches.

Setup requires the AppArmor utilities and `squashfs-tools`. Ubuntu 24.04 users can install them
through the system package manager. Both architectures use electron-builder's checksum-pinned
static AppImage runtime. It avoids the legacy ARM64 runtime's unversioned `libz.so` dependency.
Extraction mode is also available:

```sh
chmod +x TensorFold.AppImage
./TensorFold.AppImage --install-sandbox-support
APPIMAGE_EXTRACT_AND_RUN=1 ./TensorFold.AppImage
```

Desktop setup uses the system PolicyKit authorization dialog when available. An interactive
terminal can use sudo. Cancelling either approval leaves the app stopped. Headless callers are
given instructions rather than an automatic privilege request. The app never runs as root.
A host with user namespaces disabled in its kernel needs administrator review; setup does not
change global kernel, AppArmor or namespace settings.

The native executable and its payload remain root-owned and unwritable by ordinary users.
Neither a user-writable extraction cache nor a filename wildcard receives the namespace rule.
Copying and extraction use standard system tools, and the AppImage is not executed as root.
Failed setup rolls back its newly created copy and profile. Existing installations with unexpected
ownership or modified policies are refused for administrator review.

A new AppImage digest needs its own approval. Existing copies are retained. Independent engine
updates continue to use the protected per-user runtime store. User accounts, conversations,
models and drafts are not copied into the system installation.

To remove integration for the portable file's exact digest:

```sh
./TensorFold.AppImage --remove-sandbox-support
```

This requests administrator approval and removes only that copy and its policy. User data stays
in its existing profile.

## Building

Linux build hosts need `squashfs-tools`. The runtime toolset is pinned to `1.0.3`, with its archive
SHA-256 checked by electron-builder. `package-linux.mjs --appimage` replaces the generated
launcher with the reviewed launcher and embeds setup support before accepting the final artifact.
It preserves the AppImage runtime and application payload and checks the reconstructed archive.
The desktop entry does not add a sandbox-disable argument. The local finalizer can also be run
against an explicitly selected build artifact:

```sh
node scripts/finalize-linux-appimage.mjs release/linux/TensorFold-VERSION-linux-x86_64.AppImage
```

The regression checks cover cancellation, normal argument forwarding, restricted-host refusal,
policy scope, malformed archives and failed-build cleanup. Native qualification must also check
the original portable path before setup, the approved copy after setup, the renderer's seccomp
state, removal and an unrelated namespace-denied process. GPU and live-account checks are separate.

The policy follows the per-application approach described in the
[Ubuntu 24.04 security notes](https://documentation.ubuntu.com/release-notes/24.04/#unprivileged-user-namespace-restrictions)
and [Chromium's AppArmor guidance](https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md).
