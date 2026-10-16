# Isolated bot workspaces

Choose **Isolated Linux VM** for a saved bot to keep its file and command tools in a separate Linux machine on an Apple silicon Mac with macOS 26 or later.
The model connection remains in TensorFold and must run on this Mac.
VM bots refuse cloud model connections and external Cortheon research so guest tool results remain local.

## Setup

Save a bot with the **Isolated Linux VM** target and explicitly start managed setup.
TensorFold downloads Apple's exact signed [container 1.5 package](https://github.com/apple/container/releases/tag/1.5.0), verifies its pinned SHA-256, Apple installer identity and notarization, and extracts only the required components into this app's private data.
Each component has a pinned checksum; all six native executable helpers must also pass Apple's code-signature check.
A bounded archive decoder rejects traversal, links, privileged files and unexpected members.
It never executes an installer hook or requests an administrator password, and it preserves existing Apple runtime services.
If another Apple runtime is already registered, managed setup stops before replacing anything and offers the manual runtime path.

Setup shows its current phase and download bytes, supports cancellation and retries, starts owned private runtime services, then creates and starts the selected bot's workspace.
Pausing or archiving the bot aborts its pending setup before it can boot later.
Installation requires at least 6 GiB free, and starting the first bot requires the disk reservation described below.
Downloads happen only after the owner's explicit setup action, never when a model chooses a tool.

For an existing Apple installation, install Apple's signed package directly, choose its `container` executable with **Select runtime**, and start the runtime and bot through their manual controls.

TensorFold checks the executable's signature against Apple Containerization's Developer ID and pins its bytes after native selection.
Replacing or upgrading a selected executable requires selecting it again.
Only the 1.5 release family has been qualified for this adapter.
The runtime may download its Linux kernel and initial filesystem during first setup.
The Apple package is approximately 118 MB and its verified private runtime files use approximately 380 MB.
Creating a workspace explicitly downloads a digest-pinned official Python ARM64 image, approximately 20 MB compressed, plus Apple's kernel and initialization files.

## What a bot can do

Each bot has a separately owned container and an ext4 workspace volume at `/workspace`.
File tools support bounded UTF-8 reads, atomic writes, unique exact-text edits, directory listing, literal text search and glob matching.
File tools refuse symlinks, hard links, parent traversal and paths outside the workspace.
Shell commands run as Linux user 1000 with no Linux capabilities, in a VM limited to two CPU cores and 1 GiB of memory.
The root filesystem is read-only; a 128 MiB temporary filesystem is available at `/tmp`.
The private workspace disk cannot exceed 8 GiB.
Starting a VM also checks that free disk space covers the maximum size of all owned workspaces plus 2 GiB of headroom.
This reservation is deliberately conservative and may require more free space than the current sparse disk usage.

Commands default to 30 seconds and cannot exceed 120 seconds or 64 KiB of output.
After a command completes, TensorFold restarts that bot's VM to remove detached processes while preserving workspace files.
Cancelled commands and commands that hit a limit leave the VM stopped for review.
Long-running background servers are not supported by this command adapter.

## Isolation and lifecycle

No host folder, SSH agent, credential, port or network is shared with the bot.
The adapter checks ownership labels, image, user, capabilities, mounts and network settings before executing work or changing lifecycle state.
It never falls back to host tools when the VM is unavailable.
Host browser, computer, phone, plugins, connectors, agent CLIs and arbitrary delegation tools are unavailable to VM bots.
Only local API model providers are supported for these bots.
Read-only and review-before-change approvals still apply before a tool reaches the VM.

Stop and application Quit stop owned VMs and preserve their disks.
Quit also unregisters the managed runtime's exact owned user services; a manually selected Apple installation remains under its own app's control.
Rebuild replaces a bot's machine while retaining its workspace volume.
The application's explicit erase-local-data action removes verified owned containers, disks, managed services and runtime cache; failed erasure retains ownership records for a later repair.
TensorFold never prunes all containers or volumes, stops Apple's shared service, or deletes a foreign resource.

There is no guest graphical desktop, remote browser, screenshot or human takeover support in this adapter.
Those capabilities are not advertised as ready.
The design is an original implementation informed by [Apple's container reference](https://github.com/apple/container/blob/1.5.0/docs/command-reference.md) and the separation described in [OpenBot's VM contract](https://github.com/ashhart/OpenBot/blob/main/docs/VM_CONTRACT.md).

## Verification

Automated checks cover foreign ownership refusal, no-host-fallback dispatch, mount and network tampering, command cancellation, output limits, disk limits, symlink and hard-link refusal, persistence, rebuild, quiescence and owned-data purge.
Installer checks cover release and component hashes, publisher refusal, download bounds, cancellation, redirect restrictions, archive path and link refusal, failed activation rollback, pause cancellation and foreign-service preservation.
The adapter also has a real Apple container 1.5 qualification run covering create, start, file and shell work, restart, rebuild, no-network and zero-capability checks, cancellation and descendant cleanup, shutdown, and verified disk erasure.
This does not certify arbitrary model behavior or third-party images; the adapter accepts only its pinned image.
