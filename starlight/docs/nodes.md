# Starlight Node

Starlight Node runs TensorFold engines on one machine for TensorFold Starlight. It enrols once, keeps a long-poll open to
Starlight for its desired state, starts and supervises one engine process per assigned replica, and serves gateway
traffic on a mutually authenticated listener. It never runs as root and never exposes an engine directly.

| Platform | Service manager | Engine | Default state directory |
| --- | --- | --- | --- |
| macOS 13+ on Apple silicon | launchd LaunchDaemon, dedicated user | Homebrew `tensorfold` or the release archive | `/Library/Application Support/TensorFold/Starlight Node` |
| Linux x86_64 or aarch64, glibc 2.28+ | systemd unit or container | Release archive `bin/tensorfold-native` | `/var/lib/starlight-node` |
| Windows | Service control manager (LocalSystem) | TensorFold engine for Windows | `%ProgramData%\TensorFold\StarlightNode` |

## How it works

```
gateway ──mTLS (gateway cert) + permit──▶ :7443 starlight-node ──loopback + per-process key──▶ engine (127.0.0.1, port 0)
                                               │
                                               └──mTLS (node cert)──▶ Starlight node listener: enrol, sync, renew
```

- **Identity.** Enrolment generates a P-256 key on the node, sends a CSR with a one-time token, and receives a
  certificate naming `spiffe://<trust-domain>/node/<id>`. The private key never leaves the state directory.
- **Control.** `POST /node/v1/sync` reports facts, replica states and cached artifacts, and returns the complete desired
  state. A local state change interrupts the long-poll so Starlight hears about it at once. Anything Starlight does not list
  is drained and stopped.
- **Engines.** Each replica gets its own working directory, a fresh 0600 key file with three keys (`starlight[inference]`,
  `starlight-read[read]` and an unscoped `starlight-responses`), and an engine started with typed flags only: `serve <model> --host 127.0.0.1 --port 0 --name <served>
  --api-key-file <file> --no-update-check`, plus the deployment's context, parallelism, KV cache, reasoning, drafter,
  vision and prompt-cache settings. The node learns the port from the engine's `serving … at http://127.0.0.1:PORT/v1`
  line and declares the replica ready only after `/health` answers and `/v1/models` lists the served name.
- **Engine version.** Before every start the agent runs `--version` on the engine and refuses one older than
  TensorFold 1.2.0, the release Starlight is qualified on; releases before 1.1.8 do not honour key scopes at all, so every
  key in the key file would have full access. The replica reports the refusal as its error.
- **Rank groups.** A `tensor_parallel` replica runs one engine process per rank (see
  [tensor parallel](fleet.md#tensor-parallel)). On the node that leads the group, rank 0 serves and the agent proxies to
  it; a node that only follows runs its ranks, reports their state and serves nothing for that replica.
- **Crashes.** A failed start or a crash restarts after 1, 2, 4 … 60 seconds. Five failures within ten minutes mark the
  replica `failed` until Starlight sends a new assignment generation.
- **Gateway traffic.** Requests arrive at `/starlight/v1/replicas/<replica>/<engine path>` from a client certificate naming a
  gateway in the same trust domain. Each carries an Ed25519 permit bound to the node, replica, deployment revision,
  method, path and the SHA-256 of the exact body, valid for at most two minutes and usable once. Only the engine's
  inference routes are reachable; Living Weights, imprint, metrics and dashboard routes are not.
- **Live stats.** Every five seconds the node reads each ready engine's `/metrics` with the read key and posts one
  report to `POST /node/v1/stats`: generated and prompt tokens per second from counter deltas (streaming replies count
  through `generation_tokens_running`, so long replies show throughput while they run), requests running and waiting,
  and the fullest KV cache pool. A first sample or a restarted engine reports no rate yet. Reports go out on their own
  goroutine, newest first; while Starlight is unreachable they are dropped during a backoff instead of queueing.
- **Refusals.** Anything refused before the engine sees it carries `Starlight-Refused: 1`, so a gateway may retry another
  replica: `429 replica_busy` when every slot is taken, `503` when the replica is unknown, not ready or draining. A
  failure after the engine received the request never carries the header.

## Install on macOS

Create a dedicated service account, install the agent, then enrol as that account so it owns its state.

```bash
sudo sysadminctl -addUser _tensorfold -fullName "TensorFold Starlight Node" -shell /usr/bin/false -home /var/empty -roleAccount
sudo install -m 0755 starlight-node /usr/local/bin/starlight-node
sudo /usr/local/bin/starlight-node service install --user _tensorfold -- --engine /opt/homebrew/bin/tensorfold --prevent-sleep
sudo -u _tensorfold /usr/local/bin/starlight-node enroll \
  --state-dir "/Library/Application Support/TensorFold/Starlight Node" \
  --server https://starlight.example.com:9443 --token sle_... --ca-sha256 <hex from the console>
sudo launchctl kickstart -k system/dev.tensorfold.starlight-node
```

`service install` writes `/Library/LaunchDaemons/dev.tensorfold.starlight-node.plist` (the reference copy is
[`deploy/node/dev.tensorfold.starlight-node.plist`](../deploy/node/dev.tensorfold.starlight-node.plist)), creates the state and log
directories owned by the service user, and loads the job. The daemon starts at boot without anyone logged in, restarts
if it exits, and gets 180 seconds to drain on shutdown. Logs go to `/Library/Logs/TensorFold/starlight-node.log`; each
engine logs to `engine.log` in its replica directory.

Metal from a LaunchDaemon is a qualification gate: confirm on each macOS release that an engine started by the daemon
reports `backend: metal` and serves at its expected speed. Where a logged-in session is required, install a user agent
instead with `--scope user`, which runs at that user's login.

### Sleep

`--prevent-sleep` holds a `caffeinate -i -s` assertion for as long as the agent runs, so idle sleep never interrupts a
replica; `-s` applies on AC power. Also set **Energy → Prevent automatic sleeping when the display is off** and
**Start up automatically after a power failure** on Mac Studio servers. On Windows the flag requests
`ES_SYSTEM_REQUIRED`; Linux servers should disable suspend in the OS.

## Install on Linux

```bash
sudo useradd --system --home-dir /var/lib/starlight-node --shell /usr/sbin/nologin starlight-node
sudo install -m 0755 starlight-node /usr/local/bin/starlight-node
sudo /usr/local/bin/starlight-node service install --user starlight-node -- \
  --engine /opt/tensorfold/bin/tensorfold-native --hf-home /var/lib/starlight-node/hf
sudo systemctl stop starlight-node
sudo -u starlight-node /usr/local/bin/starlight-node enroll --state-dir /var/lib/starlight-node \
  --server https://starlight.example.com:9443 --token sle_... --ca-sha256 <hex from the console>
sudo systemctl start starlight-node
```

The unit ([`deploy/node/starlight-node.service`](../deploy/node/starlight-node.service)) runs with `NoNewPrivileges`, a
read-only system, no home directories, private `/tmp`, no capabilities and a 0077 umask; only the state directory is
writable. It lifts the locked-memory limit (`LimitMEMLOCK=infinity`), which RDMA needs when ranks talk over RoCE or
InfiniBand. Keep models in the state directory or a path the service user can read. AMD ROCm hosts need the service user
in the `render` group (`SupplementaryGroups=render` in a drop-in); the agent finds their GPUs with `rocminfo`, on
`PATH` or in `/opt/rocm/bin`.

### Container

[`deploy/node/container/Dockerfile`](../deploy/node/container/Dockerfile) builds the agent and installs a TensorFold
release archive verified against the SHA-256 you pass in. The NVIDIA Container Toolkit supplies `libcuda.so.1` from the
host driver; the archive embeds its CUDA kernels. Vision towers additionally need CUDA 13 cuBLASLt, so pass a CUDA 13
runtime image as `BASE` for those deployments.

```bash
docker build -f deploy/node/container/Dockerfile --build-arg TENSORFOLD_VERSION=1.2.0 \
  --build-arg TENSORFOLD_SHA256=<published archive sha256> -t starlight-node-cuda .
docker run --rm -v starlight-node:/var/lib/starlight-node starlight-node-cuda enroll --state-dir /var/lib/starlight-node \
  --server https://starlight.example.com:9443 --token sle_... --ca-sha256 <hex> --advertise gpu-01.example.com:7443
docker run -d --name starlight-node --restart unless-stopped --gpus all --ipc host -p 7443:7443 \
  -e STARLIGHT_NODE_ADVERTISE=gpu-01.example.com:7443 -v starlight-node:/var/lib/starlight-node starlight-node-cuda
```

`--gpus all` hands the node every GPU and Starlight pins each replica to its own (see [GPUs](fleet.md#gpus)); pass a device
list instead to keep some back. `--ipc host` gives NCCL the shared memory that `tensor_parallel` ranks exchange through;
Docker's default 64 MiB `/dev/shm` is too small for them. A node whose ranks join groups across nodes also needs the
host's network (`--network host` in place of `-p`), and for RoCE its InfiniBand devices (`--device /dev/infiniband`)
and unlimited locked memory (`--ulimit memlock=-1`).

## Install on Windows

From an elevated PowerShell in the folder holding `starlight-node.exe`:

```powershell
.\install-windows.ps1 -Server https://starlight.example.com:9443 -Token sle_... -CASha256 <hex> `
  -Engine "C:\Program Files\TensorFold\bin\tensorfold-native.exe"
```

[`deploy/node/install-windows.ps1`](../deploy/node/install-windows.ps1) copies the agent to Program Files, enrols it,
registers an automatic service with restart-on-failure recovery and opens inbound TCP 7443 for the agent only. The state
directory's ACL grants SYSTEM, Administrators and the enrolling account. Engines run in a kill-on-close job object, so
none outlives the agent.

## Enrolment

The Starlight console prints the exact command for a new node, including a one-time token and `--ca-sha256`, the SHA-256 of
Starlight's workload CA certificate. Enrolment accepts the server only if its chain contains that CA and the leaf is a
`control` identity issued by it. `--ca-file` accepts the CA as PEM instead; without either, system roots are used, which
only works when IT has installed Starlight's CA in the operating system's trust store.

After enrolment the node trusts only the returned workload CA and a `control` identity in its trust domain for every
later connection, whatever was used to enrol. Tokens may require approval: such a node reports `pending` and receives no
work until an administrator approves it.

| Flag | Meaning |
| --- | --- |
| `--server` | Starlight's node listener, for example `https://starlight.example.com:9443` |
| `--token` | One-time enrolment token (`sle_…`) |
| `--ca-sha256` / `--ca-file` | How to recognise Starlight's CA during enrolment |
| `--name` | Node name; defaults to the host name |
| `--advertise` | `host:port` gateways dial; defaults to the host name and listen port |
| `--force` | Replace an existing identity |

Certificates renew automatically when Starlight's `renew_after` passes or a third of their life remains. Renewal rotates the
key, verifies the new certificate against the pinned CA and swaps certificate and key in one atomic rename; the
listener serves the new certificate to the next connection without a restart.

## Running

| `run` flag | Default | Meaning |
| --- | --- | --- |
| `--state-dir` | per platform | Private state: identity, credentials, CA, permit keys, replica directories |
| `--listen` | `:7443` | Gateway-facing mTLS listener |
| `--advertise` | host name and port | Address reported to Starlight |
| `--engine` | `tensorfold` on `PATH` | Engine binary |
| `--models-dir` | `state-dir/models` | Where HTTPS archive artifacts unpack |
| `--hf-home` | inherited `HF_HOME` | Hugging Face cache for pulled artifacts |
| `--hf-endpoint` | none | Internal Hugging Face mirror |
| `--prevent-sleep` | off | Keep the machine awake (macOS, Windows) |
| `--startup-timeout` | `30m` | Longest an engine may take to load |
| `--drain-timeout` | `2m` | Longest to wait for in-flight requests before stopping an engine |
| `--stats-interval` | `5s` | How often live replica stats are measured and reported |
| `--rank-host` | the `--advertise` host | Address other nodes reach this one at for tensor-parallel ranks, such as its RoCE interface; also `STARLIGHT_NODE_RANK_HOST` |
| `--env KEY=VALUE` | none | Extra engine variables: `TENSORFOLD_*`, `TF_*`, `NCCL_*` except `NCCL_COMM_ID`, and the device lists `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` |

A device list given with `--env`, other than `NVIDIA_VISIBLE_DEVICES`, also applies to the agent, so it reports only
the GPUs Starlight may assign. Engines get a minimal environment. `TENSORFOLD_API_KEY` and `TENSORFOLD_REQUEST_LOG` are never
passed, so no other key reaches an engine and prompts are never logged by it. `TENSORFOLD_NO_UPDATE_CHECK=1` and `TENSORFOLD_NO_LIVE=1` are always
set, and `HF_HUB_OFFLINE=1` when the installation is offline.

## Artifacts

| Kind | Source | Offline |
| --- | --- | --- |
| `hf` | `tensorfold pull repo@<40-hex revision>`, served from that exact snapshot directory | Only through `--hf-endpoint`, or when already cached |
| `path` | An absolute directory holding `config.json` | Always |
| `url` | An HTTPS `.tar` or `.tar.gz`, streamed through SHA-256 and refused on mismatch | Allowed; point it at an internal mirror |

Archives may contain only directories and regular files: absolute names, `..`, backslashes, links and devices are
refused, and extraction stops at the size and file-count caps. Unpacking happens in a temporary directory that is
renamed into place only when complete. Query strings, which often carry signed-URL credentials, are dropped from logs.

## Status and troubleshooting

```bash
starlight-node status
starlight-node status --json
starlight-node service status
```

| Symptom | Check |
| --- | --- |
| `starlight refused this node's identity` | The node was revoked or disabled, or its certificate expired while offline; re-enrol with `--force` |
| Replica stuck in `starting` with an error | `engine.log` in `state-dir/replicas/<id>/`; the startup timeout, model path, and engine version |
| `engine … is older than 1.2.0` | Upgrade TensorFold on the node; every start checks the version again, so no agent restart is needed |
| A group replica never becomes ready | Rank 0 waits for every rank: check `engine.rank*.log` on each member, that the members reach the lead's `--rank-host` on TCP 29500 to 30499, and the engine's interconnect |
| Replica `failed … waiting for a new assignment generation` | Five failures in ten minutes; fix the cause, then redeploy or bump the deployment in Starlight |
| `artifact is not on this node and the installation is offline` | Stage the snapshot, or configure `--hf-endpoint` for an internal mirror |
| Gateways see `permit_mismatch` | Clock skew over five seconds between gateway and node, or a body rewritten after signing |
| Gateways cannot connect | Port 7443 reachability, `--advertise`, and that the gateway certificate comes from the same CA |
| Leftover engine after an agent crash | The next start stops it by PID after checking its command line names that replica's key file |

## Security notes

- The state directory is 0700 (owner only; a protected DACL on Windows) and every file in it 0600. A directory owned by
  another account is refused.
- Engines bind only to loopback with a key file the agent writes per process; nothing on the node can reach an engine
  without that key, and the agent forwards only inference routes with an allowlisted set of request headers.
- Forwarded requests use the scoped `inference` key. The unscoped `starlight-responses` key is used only where the engine
  requires it: a Responses POST that stores (`store` absent or true) or continues (`previous_response_id`), and GET or
  DELETE of a stored response. Starlight checks that the caller owns the response id before sending any of these.
- Client `Authorization`, cookies and `Origin` never reach an engine; the agent sets the engine key itself.
- Permits are single use: a replay cache rejects a request id seen before, until its permit expires.
- A node whose identity Starlight refuses keeps its engines but receives no permits; Starlight answering `revoked` or `disabled`
  stops every engine.
