# Dependency setup in the coding assistant

An approved `npm install` or `npm ci` now prepares dependencies in isolation and publishes them into
the selected project. A successful result means subsequent commands in that project can resolve
its installed packages. The response identifies the destination and reports that lifecycle scripts
were skipped.

## Supported commands and projects

- `npm install`, `npm i`, `npm add`, and `npm ci`.
- Registry package names and versions, including scoped packages and npm aliases.
- Common save flags, `--omit`, `--include`, peer-dependency options, and offline preferences.
- npm workspaces using relative directory names and `*` within path segments, with bounded discovery.
- Internal `file:` directory dependencies whose package directories stay inside the selected project.
- Explicit `bin` entries in discovered workspace and internal `file:` packages on POSIX hosts.
- npm lockfiles with `lockfileVersion` 2 or 3. Published download references must use HTTPS at
  `registry.npmjs.org`; internal package links must name a discovered local package.

Select the folder containing the root `package.json` before running the command. Root workspace
installation includes its discovered packages; selecting individual workspaces with CLI flags is
not implemented. Recursive `**`, negation and brace workspace patterns are refused with an error.

Other package managers, `npm exec`/scaffolding, global installations, command composition, prefix
overrides, custom registries, external directory dependencies, tarballs, Git sources and remote URL
dependencies are refused. They do not return a simulated installation success. Private registry
credentials and user/project `.npmrc` files are not copied into the installation stage.

## Network and lifecycle boundaries

The existing saved approval requirement for package installation remains in place. `npm ci` and
`npm i` use the same package-install classification as `npm install`.

The networked process receives the selected package manifests, npm lockfiles, empty npm
configuration files and inert placeholders for declared local executables. It uses a private
temporary directory and cache in an owned sibling stage beside the selected project. The parent
directory must permit creating this private stage. Keeping it outside the denied project lets
macOS resolve the process working directory without granting access to project files. Existing source files,
credentials and the user's npm cache are not imported. The macOS profile explicitly denies reads
and writes to the original project while allowing the owned stage, including when the project is
inside an otherwise writable temporary directory.

Every supported invocation forces `--ignore-scripts`, disables audit/funding requests and retains
directory links for local packages. The installer never executes a package's lifecycle scripts
with network access. Packages that require native setup need a separate, approved `npm rebuild`
or project-specific setup command. That later command keeps the normal build/test profile, with
network disabled; native setup requiring downloads needs a separately configured workflow.

The isolated cache is removed after each operation. Offline registry installation cannot use the
user's pre-existing npm cache. Self-contained workspace and internal-directory dependency fixtures
can install offline. The supported macOS runtime roots remain those of the command sandbox; a
custom runtime outside them is not granted the user's whole home directory as a workaround.

### Local package commands

Local packages can declare a `bin` string or a map of command names to relative paths. The target
must already exist within the declaring package as a regular file with no symlink parents or
hardlinks. Absolute paths, paths leaving the package, hidden directories, `node_modules`, and the
implicit `directories.bin` form are refused. Discovery and executable counts are bounded.

The installer records file identity and permissions without reading its contents into the stage.
For each declared target, it writes an inert, fixed placeholder into the matching staged path.
This lets npm create its usual relative `node_modules/.bin` links. After publication, those links
resolve to the original workspace files. Source bytes remain on the host, outside the networked
stage. The placeholder is not the executable that the project ultimately runs.

On a POSIX host, publication adds executable permission bits to a validated local target when
needed and reports that path in the changed-path list. The existing contents and other permission
bits are preserved. Targets still need a valid interpreter or executable format for that host;
installation does not manufacture a working runtime or execute them as a health check. Permission
changes are rolled back with the dependency transaction when the original file remains unchanged.
If a newer edit makes rollback unsafe, the new edit is preserved and the path is reported for review.

Windows local-package commands are refused explicitly. npm's Windows command shims can depend on
the actual target's interpreter, which cannot be inferred honestly from an inert placeholder.
Windows shim generation needs a separate implementation and qualification; this branch does not
report success for that case.

Linux still has no qualified command sandbox in this branch. Package installation therefore fails
closed there, just like other agent commands. The explicit unsandboxed-development override has
not become a product default or a Linux production qualification.

## Publication and failure handling

The main process checks staged manifests, lockfile destinations and dependency symlinks before
publication. Package manifests, npm lockfiles and dependency directories are eligible, together
with the executable permission changes described above. Unrelated project files and changed
lifecycle scripts are refused. Workspace and internal-directory
links point at the real local packages after publication, so their source files remain available.

Installations from one app process are serialized per canonical project directory. Before every
publication step and again before reporting success, the publisher rechecks **all** canonical
manifest hashes and file identities, plus their parent directories. This includes manifests that
the install did not change. An already-published manifest is compared with this transaction's exact
published version, so a later user edit is not mistaken for an installer-owned change. Local
executable identities are also checked against their expected state.

Changed manifests or replaced parents stop publication rather than overwriting newer work. Existing
dependencies are moved to owned recovery paths, and ordinary failures roll back completed steps.
If a newer edit prevents safe rollback, the edit is preserved and recovery paths are reported.

Stage cleanup retries a small bounded set of transient filesystem errors. Every retry rechecks the
stage's identity before removing it; a concurrent replacement is preserved. A persistent failure
includes the stage path and the underlying cleanup error in the result, rather than silently
claiming that cleanup completed.

This is not a filesystem-wide compare-and-swap operation. As with other portable Node workspace
mutations, a non-cooperating process can race the final filesystem operations. Do not run another
package manager or mutate dependency directories during installation. A process or power failure
can leave private staging/recovery entries; automatic recovery of an interrupted multi-file
publication is not claimed. Keep project manifests in version control and inspect retained recovery
copies before removing them.

## Validation

`electron/agent/package-install.test.ts` runs real offline npm installation and clean installation
against both a workspace package and an internal `file:` package, then imports them from the actual
project. It also checks lifecycle suppression, configuration isolation, symlink rejection,
concurrent edits, parent replacement, cancellation, serialized installation and rollback retention.
Regression cases mutate an unchanged manifest just before dependency publication and mutate a
manifest after the last publication step. Both must fail, restore dependencies and retain the newer
edit. Injected transient cleanup errors verify retry, and a replaced stage verifies ownership checks.

`electron/agent/package-install-bins.test.ts` uses real offline npm to install a workspace command
and an internal `file:` command, then executes their final `.bin` paths from the actual project.
It checks that only inert placeholders enter the stage, source contents stay intact, lifecycle
scripts remain off, executable permissions are applied, and cancellation restores both permissions
and the previous dependency tree. It also covers unsafe paths, symlinks and hardlinks.

`electron/agent/sandbox.test.ts` checks the sandbox integration and platform refusal. macOS runtime
confinement and native-package behavior still require a live Mac qualification; passing the portable
tests does not establish those platform results.
