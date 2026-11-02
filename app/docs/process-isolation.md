# Command and MCP isolation

This local development candidate implements Linux bubblewrap and Windows x64 LPAC backends.
Linux kernel checks run in owned Ubuntu VMs. The Windows helper compiles and packages locally;
its actual Windows API, filesystem and process checks remain pending a Windows machine.
An installer or a successful cross-build does not qualify that backend.

## Workspace writes

Writable Linux and Windows commands work in a private project copy. Direct MCP servers keep a
private copy for their session. Before publishing changes, the host checks Git metadata, portable
filenames, regular files, links and concurrent host edits. It prepares new files separately, backs
up replaced files and rolls back a failed publication while preserving newer user edits.
Concurrent edits or unsafe output stop publication. Recovery backups are reported if newer edits
prevent rollback. Cancelled, timed-out or failed isolation launches do not publish their staging tree.

Git metadata is read-only to isolated children. New or changed Git controls, including nested
repositories, cannot be published. Linked worktrees require a validated backlink and get read-only
metadata views. Arbitrary Git mutations are outside this command contract. New directories are
supported; changing an existing directory's permissions is refused.

Copies are bounded to 200,000 entries, 80 directory levels and 2 GiB of regular file data.
Files stream through bounded buffers. External runtime copies have a separate 100,000-entry,
2 GiB, 32-root limit. Oversized projects fail closed. These limits are not general memory or CPU quotas.

Linux preserves validated internal symlinks and links into declared read-only SDK roots.
Windows input links must resolve inside the selected tree and are materialized. Windows output
reparse points and hardlinks are refused. A native snapshot broker pins directory and file handles
while capturing a live Windows MCP tree; Node does not follow that live tree during publication.

## Linux

Bubblewrap creates private process, mount and network namespaces, drops capabilities and mounts
system runtime paths read-only. Scratch storage is private. The project mount precedes its read-only
Git controls. Command runtimes include the system paths and resolved Node installation. MCP copies
external runtime, script and explicit data inputs into read-only views. A single data file does not
expose its containing directory. Home roots, workspace ancestors and temporary roots are refused,
including aliases that resolve to those roots.

Network access is absent by default. Only an explicitly permitted network profile shares the host
network and mounts DNS and certificate configuration. Direct MCP sessions have no network access.
Networked MCP integrations need the separately configured Docker gateway or a future reviewed policy.
Inherited credentials are removed; deliberately configured MCP environment values remain explicit grants.
Linux applies those values after namespace setup so loader variables cannot inject code into bubblewrap.

The OS must provide bubblewrap and permit user namespaces. Debian packages declare the bubblewrap
dependency. Restricted Ubuntu hosts need the existing scoped AppArmor integration described in
[AppImage sandbox setup](appimage-sandbox.md). No global namespace setting or Chromium sandbox
fallback is changed. Unavailable isolation refuses the operation.

## Windows

The native helper creates a unique less-privileged AppContainer profile for each launch. Only the
private project, scratch and copied runtime trees receive that profile's file grants. Real project,
home and SDK ACLs are not modified. Read-only runtime and Git grants are separate from workspace
write grants. UNC, device, alternate-stream and drive-relative paths are refused before parent I/O.

The child starts suspended, joins a kill-on-close Job Object, then resumes. The job limits descendants
to 64 processes and restricts UI access. Only the three standard I/O handles are inherited. Parent
exit, cancellation and timeout terminate the job. Network capability is absent unless explicitly
requested; AppContainer loopback remains subject to Windows policy.

Availability requires a successful child probe of both AppContainer and LPAC token properties.
The helper's SHA-256 and ABI pin live inside the packaged app.asar; the helper executable is a separate
resource. The existing ASAR integrity and fuse policy protects the pin. Packaged apps have no
unsandboxed environment override. The implementation follows Microsoft's
[LPAC launch contract](https://learn.microsoft.com/en-us/windows/win32/secauthz/implementing-an-appcontainer)
and [handle-based ACL rules](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setsecurityinfo).

Building the helper needs Zig 0.17.0. The package wrapper builds it before packaging and requires
Windows x64 to execute the engine capability check. A cross-built candidate remains unqualified
until the native checks pass on Windows with a normal non-administrator account and local NTFS storage.

## MCP lifecycle

Calls serialize publication, recheck configuration before and after a response, and refresh their
private workspace when the host changes. Signed manifests are checked again after preparation,
including the actual private launch-code bytes. Reconfiguration and disconnect cancel pending
launches and publication. Shutdown waits for owned process and workspace cleanup. The child-close
listener regression has a focused real-process test.

## Native qualification

Run these tests with Node 22 on the target platform. Install bubblewrap on Linux and build the
native helper on Windows. Do not set an unsandboxed override.

```sh
TENSORFOLD_REQUIRE_PORTABLE_SANDBOX=1 node node_modules/jiti/lib/jiti-cli.mjs electron/agent/portable-sandbox.test.ts
TENSORFOLD_REQUIRE_PORTABLE_SANDBOX=1 node node_modules/jiti/lib/jiti-cli.mjs electron/agent/mcp-isolation-native.test.ts
node node_modules/jiti/lib/jiti-cli.mjs electron/agent/sandbox-git-roots.test.ts
node node_modules/jiti/lib/jiti-cli.mjs electron/agent/isolated-workspace.test.ts
```

The required flag turns a missing backend into a failure. Default shared-source runs can skip the
native suites on an unsupported host. Those skips do not count as confinement evidence.
Real account authorization, GPU execution, desktop automation and the deferred Mac window-close
check remain separate platform qualification work.
