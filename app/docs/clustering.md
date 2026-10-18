# Clustering

The cluster workspace manages explicit hosts, SSH trust, headless services, immutable model revisions, per-machine admission and reviewed deployments. Discovering a machine or an RDMA adapter does not make it trusted or prove a distributed model path.

## Boundaries

The management agent and inference endpoint use different credentials. Management starts on loopback; remote access uses verified SSH forwarding or an explicitly configured TLS endpoint. Advertising a node never advertises a credential. New hosts must be enrolled and inspected before a plan can use them. A forwarded remote endpoint remains remote in provider policy, including phone, Google and bot restrictions.

The released engine's generic cluster commands do not launch a distributed model. The app must refuse those paths, including heterogeneous Metal/CUDA execution, until an implemented engine profile supports them. Specific native speed-up profiles must declare their supported architecture, rank count, model types, transport and whether they actually divide model weights; a replicated speed-up must not be presented as pooled model capacity.

All remote mutations have a durable command identity and receipt. A lost acknowledgement is reconciled before retry, not treated as proof that nothing happened. Restart does not replay a model load, setup command or inference request. Headless supervision is independent of an SSH login; the GUI may disconnect without killing a reviewed remote service.

## Setup flow

1. Add a saved SSH host, configured TLS endpoint or untrusted discovery candidate.
2. Compare the SSH host fingerprint, choose an existing key or generate an app-owned key, and inspect prerequisites.
3. Review a per-user headless installation. System services, driver installation and OS approval remain explicit separate actions.
4. Inspect engine version, cached checkpoints, memory, transport availability and any independently verified link result.
5. Select an immutable model revision and an implemented execution profile; review each machine's admission before downloading or loading.
6. Stage the model and start the ranks as a transaction, then require an authenticated canary before making the reviewed endpoint usable.
7. Inspect health, stop owned work or reconcile uncertain receipts without falling back silently to another model or transport.

MCDMA driver installation may require physical access and OS approval. This app must not weaken system security or load an experimental kernel driver automatically. Ethernet visibility, an SSH connection, a NIC name and a TCP canary each establish different facts from a byte-verified RDMA transfer.

## Qualification

Use synthetic profiles, fake SSH/process boundaries and local HTTP peers for normal tests. Exercise restart, failed writes, changed host identity, expiring discovery, malformed reports, unavailable transports, memory refusal, partial staging, cancelled loads, peer disappearance and uncertain command acknowledgements. Real hardware and matched model-output tests are separate qualification; no simulation can establish RDMA bandwidth, distributed correctness or a supported hardware combination.

No new runtime dependency is planned; the headless entry uses Node built-ins and the installed TensorFold engine, while SSH uses the app's existing transport.

## Current engine support

The shipped adapter implements one native process per endpoint (`single-native`), with native capability/format checks, a pinned cache revision, host-memory floor budgeting, a second admission check before load, and an authenticated inference canary. That memory floor is an estimate, not a guarantee for device memory or context growth; native load refusals remain visible.

Engine 1.0.12 has particular two-Mac speed-up implementations, but does not supply a complete versioned admission, checkpoint, link and session contract for this headless adapter. They therefore remain unavailable here, as does mixed Mac/CUDA weight distribution. The app never substitutes generic `cluster` commands or presents independent servers as pooled model capacity. A declaration alone cannot enable a path that the native adapter has not implemented. Multi-rank scheduling is covered through fake peers only; native peer rendezvous needs a separate readiness contract before it can be enabled.

Linux RDMA device hints are passive inventory only. A data-link check needs matching peer identities, positive byte counts and a matching native proof; no Ethernet, NIC inventory or successful HTTP request manufactures that proof. No driver or OS security configuration is changed.

## Headless installation and discovery

The transferred `cluster-node.js` is a single built-in-only Node bundle produced by the normal app build. Remote setup requires Node.js 22 or later and an approved installed `tensorfold-native` binary; it installs neither runtime nor engine. Linux requires an existing systemd user manager and administrator-enabled lingering, and macOS requires the user's existing launchd login domain. Setup preserves unknown or partial installations and asks for deliberate repair instead of overwriting them.

The private JSON configuration contains `schema: 1`, an absolute `dataDir`, the approved absolute `enginePath`, `host`, and `port`; an optional `cacheDir` selects the node's model cache. Loopback HTTP is the default. A non-loopback listener requires `tls: {certFile, keyFile}` with ordinary verified certificates; there is no insecure LAN switch. Management and inference tokens are created as separate private files, never inline configuration, command arguments or discovery metadata.

Discovery is optional: explicitly set `advertise: true` in a reviewed private node configuration to publish a protocol-only Bonjour suggestion through the host's existing discovery tool. The default and guided SSH installation keep advertising off. Missing discovery tools are reported; no package or system service is installed automatically. SSH suggestions select the saved-SSH review flow, never a fictitious externally reachable HTTPS service. The GUI's bounded scan expires suggestions and never trusts or enrolls them automatically.

## Recovery and credentials

Protected desktop state is stored per app profile in `state/cluster-workspace.json`; app-owned SSH keys live under that profile's `clusters/ssh` directory. Global workspace replacement and local-state erasure refuse unresolved remote operations. Stop and reconcile work before erasing the desktop's receipts; erasing local state does not uninstall a remote service.

The desktop commits a command intent before every remote mutation, binds credentials to the actual captured SSH session and saved target, and never retries an uncertain mutation automatically. A GUI restart can reconcile a surviving service's exact deployment and receipts. A node process restart cannot assume ownership of a lost native process: uncertain native work stays blocked for deliberate host repair, rather than signalling an unverified PID. Partially staged checkpoints remain separate from successful loads.

Managed providers retain remote privacy classification even when their address is a loopback forward. Inference resolves the protected deployment reference on every request, verifies the pinned model/revision/rank/profile and current consumer credential, and uses the exact authenticated SSH byte channel. It never sends a prompt or bearer token to a stale or reused local forwarding port. Direct endpoints retain TLS and DNS-to-socket pinning. External clients copying an SSH forward's URL must reconnect when that desktop-owned forward closes.

## Management protocol

The authenticated `/api/cluster/v1` interface provides `GET /status`, `GET /identity?nonce=...`, `POST /admission`, `POST /models/ensure`, `POST /deployments/start`, `POST /deployments/stop`, `GET /jobs`, `GET /jobs/:id`, `POST /jobs/:id/cancel`, `POST /canary` and `GET /endpoint-credentials`. Mutations require `expectedNodeId` and a durable `commandId`; using a command identity for a different payload is an explicit conflict. The identity challenge echoes a fresh nonce only after management authentication.

A launch includes a deployment identity, exact `{repo, revision}`, implemented profile, memory role, bounded generation settings, transport, rank/world geometry and explicit peer links. Admission distinguishes a missing checkpoint, memory refusal and unsupported execution path. A cancelled HTTP request is never interpreted as cancellation of already accepted physical work; cancel has its own durable intent and waits for the owned work to drain.

The separate consumer credential permits only `/v1/models`, `/v1/chat/completions` and `/v1/responses` for the verified deployment. Management responses, incoming bodies, output streams, queues, cache scans and journals have explicit bounds. Redirects are refused, partial streams do not become finished requests, and a lost durable write blocks further effects.

The Mac package declares the exact Bonjour service and explains local-network use in its permission description, following [Apple’s local-network privacy guidance](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy); an OS permission remains the user’s decision.
