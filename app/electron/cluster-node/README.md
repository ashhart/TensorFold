# Standalone headless node

`createClusterNodeServer` is a pure Node 22 service with an injectable `EngineAdapter`, journal and publisher; its entry is `electron/cluster-node-main.ts`, bundled separately as `cluster-node.js`.
It does not import Electron, install services, open SSH sessions or change drivers.

The private JSON configuration has `schema: 1`, an absolute `dataDir`, `host`, `port`, and optional absolute `enginePath`, `cacheDir`, `fabricLibraryPath`, `tls: {certFile, keyFile}` and `advertise` fields.
The default host is loopback; a nonloopback host requires explicit TLS and a private key file.
The config and owned data directory must be private to the current user, with no symlink namespaces.
Credentials are generated into separate private `management.token` and `inference.token` files, never inline config or discovery records.

## HTTP contract

All management routes use the management Bearer credential and the `/api/cluster/v1` prefix.
`GET /identity?nonce=...` returns the caller nonce and stable opaque node ID; `GET /status` returns a typed snapshot.
`POST /admission` accepts `{expectedNodeId, role, model, profileId, generation, transport, rank, worldSize, links}`.
`POST /models/ensure` accepts `{commandId, expectedNodeId, model}`; `POST /deployments/start` accepts `ClusterNodeLaunch`.
`POST /deployments/stop` and `POST /canary` accept `{commandId, expectedNodeId, deploymentId}`.
Mutations return an accepted job; `GET /jobs`, `GET /jobs/:id` and `POST /jobs/:id/cancel` expose lifecycle and cancellation, with the cancel body `{commandId, expectedNodeId}`.
Repeated command IDs with the same canonical payload reuse their receipt; changed payloads are refused.

`GET /endpoint-credentials` succeeds only after an owned deployment is ready and passes a separate canary, returning `{apiKey, model}`.
The exposed model is its friendly repository name; the runtime rewrites that exact alias to the owned engine session and refuses other models.
The inference key is derived from a separate private seed and deployment identity, so replacing a deployment invalidates old endpoint credentials even for the same repository.
Only `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/responses` accept that inference credential.
Requests are bounded, authenticated and origin-refused; incomplete streams are failed rather than accepted as finished results.

## Ownership and recovery

The service persists a receipt before an effect, admits one physical mutation or inference lane at a time, and keeps cancellation busy until the owned work actually drains.
It stores content-free request digests rather than prompt or completion text.
Owned native processes serve only on loopback with a separate internal credential and must acknowledge physical idleness before another request.
An exclusive process lease prevents two node services sharing a data directory.
On restart, interrupted jobs, requests or deployments become `needs-review` and are never replayed automatically.
An uncertain process stop is not ownership proof; inspect and repair the owned engine and receipt state manually before enabling more work.

## Optional discovery

Discovery stays off unless the reviewed config explicitly sets `advertise: true`.
macOS uses the installed `dns-sd` utility; Linux requires installed `avahi-publish-service` and its daemon.
The owned publisher uses literal arguments without a shell and advertises only an opaque hashed label, `_tensorfold-cluster._tcp`, the bound port, `protocol=1` and `transport=https` for TLS nodes or `transport=ssh` otherwise.
These are untrusted enrollment hints, not authentication, a remote inference address, distributed support or RDMA proof.
Startup waits for registration acknowledgement; a missing utility or timeout is an explicit failure, and service shutdown waits for the publisher to exit.
The app never installs a publisher or changes the default to advertise automatically.

## Native support boundary

The native adapter inspects CPU-only capabilities and immutable model-cache metadata before staging or loading.
Current source-supported single-host families and formats must appear in the installed engine capability table; unknown and pair-required model types are refused.
The reported weights-plus-runtime host-memory floor is only a conservative preflight against the node role budget, not full context or device admission.
Admission is repeated before load; final native allocation and context checks may still refuse a job.
The child environment is a small explicit runtime allowlist, with experimental native control prefixes excluded and optional Hugging Face credentials available only to model download.

Released generic cluster flags do not establish distributed weights support.
Distributed activation remains blocked until an implemented native adapter has versioned per-family capability, admission, immutable checkpoint, byte-matched link and session-health contracts.
Passive Linux adapter hints never claim an active RDMA path; no MCDMA status command is run by inspection.
Tests use fake engines, generated temporary TLS keys, synthetic caches, fake processes and local HTTP servers; they do not qualify real models or hardware.
