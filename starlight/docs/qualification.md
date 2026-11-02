# Qualification

This page separates what the automated tests prove from what still needs real hardware, real identity tenants or real
load. Run the suites with Postgres 15 or newer:

```bash
export STARLIGHT_TEST_DATABASE_URL='postgres://starlight@127.0.0.1:5432/postgres?sslmode=disable'
go test -race -count=1 ./...
go run ./tools/leancheck .
```

Tests that need Postgres skip when `STARLIGHT_TEST_DATABASE_URL` is unset; each creates and drops its own database from a
migrated template.

## What the automated tests prove

### End to end (`internal/e2e`)

These run Starlight in process against Postgres, enrol a real `starlight-node` agent over mutual TLS with the CA pin, and let the
agent supervise a fake engine binary that implements the TensorFold engine's HTTP API and key-scope rules (it does not run
a model).

| Test | Proves |
| --- | --- |
| `TestLocalFleetServesEveryProtocol` | Enrolment, placement of a warm deployment, engine start-up and readiness, chat through an alias, streaming with the usage chunk hidden, Anthropic Messages with `x-api-key`, `/v1/models` with aliases, `model_not_found`, usage records |
| `TestConcurrencyLimitAndRevocation` | A concurrency limit holds across concurrent streams (one 200, one 429); a revoked key stops working within the cache window |
| `TestSSOToConnectToInference` | OIDC sign-in through an IdP behind a private CA (`identity.ca_file`), an IdP group mapped to a project role, Connect's PKCE approval and token exchange, `/oauth/userinfo`, inference with the Connect token, and an Entra-shaped SCIM deactivation revoking that token |
| `TestConnectAgainstRealStarlight` | Connect's device sign-in and its local endpoint against the real Starlight |
| `TestNodeOnlineControls` | Draining the only node stops its engine and requests answer 503; making it active again brings a ready replica back; revoking it stops routing |
| `TestStoredResponsesBelongToTheirOwner` | Stored Responses can be read, continued and deleted only by their owner |
| `TestLiveThroughputReachesStarlight` | A node's live tokens-per-second for its replica reaches Starlight within seconds |
| `TestFrontierPlanLocalExecution` | A plan drafted by a provider route, the egress hash matching the bytes the provider received, approval by hash, a one-time task token, local execution, and the task token unable to reach the frontier model |
| `TestCredentialsNeverLeaveAndOfflineMeansOffline` | A credential in a prompt never reaches a provider and is audited; offline mode blocks cloud routes while local inference works |
| `TestTensorParallelGroupAcrossTwoNodes` | A `tensor_parallel` deployment with `nodes: 2` placed on two enrolled nodes: rank 0 on the lead and rank 1 on the other meet at the lead's rendezvous address, chat is served through rank 0, and killing rank 1 brings the group back with a new process |
| `TestEightThousandUsers` (`STARLIGHT_SCALE=1`) | 8,000 distinct users, each with an API key and project membership, each served exactly once through one or more gateways and 128-slot replicas: no errors, no leaked leases, one usage row per user |

### The 8,000-user run

`STARLIGHT_SCALE=1 go test -run TestEightThousandUsers -v ./internal/e2e` sends one chat request from each of 8,000 users,
from 512 concurrent workers by default, through the real gateway, admission, node agent and settlement path to a fake
engine that answers in 15 ms. `STARLIGHT_SCALE_CLIENTS` sets the workers, `STARLIGHT_SCALE_GATEWAYS` the Starlight instances they
spread over and `STARLIGHT_SCALE_NODES` the nodes, each running one replica. Each worker holds one keep-alive connection, as
a load balancer would. CI runs it on Linux. On one Apple M3 Ultra with Postgres 17 on the same machine, while other
builds shared it, five runs of each single-gateway setup and three of the two-gateway setup gave:

| Setup | Requests per second | p50 | p95 | p99 | Errors |
| --- | --- | --- | --- | --- | --- |
| 512 workers, one gateway, one replica: four per slot, so most requests queue | 3,180 to 3,335 | 134 to 139 ms | 312 to 342 ms | 448 to 485 ms | none |
| 100 workers, one gateway, one replica: under the slot count | 4,333 to 4,454 | 20 to 21 ms | 30 to 33 ms | 52 to 56 ms | none |
| 512 workers, two gateways, two replicas | 3,466 to 3,656 | 118 to 124 ms | 302 to 335 ms | 476 to 514 ms | none |

Every run served all 8,000 users and left no lease behind. The earlier admission design gave 1,207 to 1,291 and 2,379 to
2,524 requests a second for the first two setups. The queued latency is the queue doing its job: 512 requests over 128
slots wait about four service times. Extra gateways and replicas on one machine share its CPU with Postgres and the load
generator, so they show less here than on separate hosts. These numbers measure Starlight's own overhead with an instant
engine; [admission.md](admission.md#throughput) has the admission benchmark. A real profile needs real nodes and the
gate below.

### A real engine

The suites use a fake engine. Starlight was also run against the TensorFold 1.2.0 release on an Apple M3 Ultra, serving
Qwen3.5-2B through an enrolled `starlight-node`: `/v1/models`, chat completions, streaming with the usage chunk and
`[DONE]`, completions, Anthropic Messages, Responses (store, read back, delete), the error for an embeddings request to a
chat model, 401 for a bad key, and one usage row per request all worked. The engine logged the agent's health and
metrics reads under the scoped `starlight-read` key, and the console's dashboard showed live decode speeds of 230 to 408
tokens a second with four requests running.

### Deployment assets

| Tool | Version | Result |
| --- | --- | --- |
| `helm lint deploy/helm/starlight` | 4.3.1 | No failures |
| `helm template` with the defaults | 4.3.1 | 10 objects |
| `helm template` with ingress, autoscaling, ServiceMonitor, disruption budget and network policy enabled | 4.3.1 | 13 objects, no errors |
| `promtool check rules deploy/prometheus/rules/starlight.yaml` | 3.15.1 | 14 rules |
| `promtool check config` on `prometheus.yml`, and on the token example with `--syntax-only` | 3.15.1 | Valid |

### Packages

| Area | Suite | Covers |
| --- | --- | --- |
| Admission | `internal/admission` | 48 requests across four gateways never exceed three slots; interactive work overtakes queued batch work; concurrency and budget limits with refunds; dead leases reaped and charged, live ones left alone; revision pins and classification, for every node of a rank group; isolated projects kept off shared prompt caches, and caches restarted when a grant would share them; admitting and settling under related limits never deadlocking; negative or overflowing amounts refused; a caller who leaves mid-admission leaving no lease; heartbeats only for what a gateway holds; failover counting a request once; route limits read at admission; wake-ups handed on; stalled or terminated databases answering promptly; throughput that grows with replicas (`STARLIGHT_SCALE=1`) |
| Scheduling | `internal/scheduler` | Warm and on-demand modes, idle unload, surge-first rollouts and the one-at-a-time fallback on a full node, placement constraints, GPU placement (spread, best fit, per-GPU memory, a GPU per replica on a B300, nothing pinned on unified memory), rank groups on distinct nodes and stopped when a member fails, a lead is lost or a member's node drains, rollouts that wait for the new replica to be ready and for every draining seat to let go of its GPUs, memory and ports, a node whose GPU probe failed never treated as unified memory, failure timers that routine syncs do not reset, AMD nodes holding whole GPUs and never their integrated GPU, leader election and fencing |
| Rate limits and database | `internal/ratelimit`, `internal/db`, `internal/httpx` | Instances sharing one budget through Postgres, each keeping its own bucket when Postgres is down; keys that cannot be reversed without the master key; IPv6 counted per /64; forwarded addresses with ports, and unreadable ones failing closed; a dialer that holds its slot until the server answers; an unreachable database reported as unavailable, not as a bad credential |
| Gateway | `internal/gateway` | Duplicate-key refusal and stable re-encoding (fuzzed), request inspection with bounded output limits and choices, stream relay and usage metering for every protocol, interrupted streams, streams stopped when the caller loses the project, outages answered 503 |
| Credentials | `internal/authn`, `internal/admin/keys` | Key constraints, CIDR allowlists, project selection, membership loss, task-tree revocation, rotation with grace, IdP tokens and project lookups during a database outage answered as outages |
| OAuth | `internal/identity/oauth` | PKCE, single-use codes burned on a bad verifier, refresh rotation with family revocation on reuse, the device grant with `slow_down` |
| Single sign-on | `internal/identity/{oidc,saml,enrich,mapping,login,connections,presets}` | OIDC and SAML flows against in-process IdPs, forged and replayed tokens and assertions, Entra overage through a fake Graph, Google domain and groups through fake APIs, account linking rules, back-channel logout, SAML single logout started by Starlight or the IdP over both bindings with forged, replayed, oversized, SHA-1 and nested-signature messages refused, response trees bounded before signature checks, `private_key_jwt` with RSA and P-256 keys including Entra's certificate thumbprint and Graph, certificate fields that never keep a pasted key, connections whose removal signs their people out, setup and recovery codes with every spare setup link spent once an owner is claimed |
| SCIM | `internal/identity/scim` | Okta- and Entra-shaped request sequences, filters (fuzzed) compiled to SQL, PATCH variants, uniqueness, connection isolation, rate limits shared across instances, the last owner never suspended or deleted, even by concurrent requests |
| Nodes | `internal/node/...` | Permit checks, slot limits, streaming proxy, least-privilege engine keys, supervisor lifecycle, the engine release check (and an engine upgraded in place), GPU pinning on CUDA and HIP, rank layouts for every `tensor_parallel` family, rank groups that start and fail together on one node or as followers of another, GPU discovery from `nvidia-smi` and `rocminfo` under CUDA and HIP device lists, safe archive extraction, enrolment, sync and renewal |
| Connect | `internal/connect/...`, `internal/mdns` | PKCE and device flows, refresh, the local endpoint's Host, Origin and app-key checks, managed configuration, mDNS |
| Cryptography | `internal/crypto/...`, `internal/trust` | Sealing with bound context, token shapes, permit signatures and expiry, mTLS identity checks, permit keys published before they sign and retired after two days, data keys re-wrapped under a new master key |
| Audit | `internal/audit`, `internal/identity/connections` | Chain integrity under concurrent writers, tamper and gap detection; a sign-in connection change whose audit fails is rolled back |
| Guard | `internal/guard` | Credential kinds found in JSON bodies, no false positives on plain text (fuzzed) |
| Console | `internal/console/...` | Every page renders for an owner; read-only roles see no controls; members get a personal view; project admins manage only their project; refused forms keep what was typed; the live feed is filtered to the viewer; node controls, write-only provider credentials, plan approval by hash, audited actions, CSV export |
| Configuration as code | `internal/declare` | Strict parsing, a dry run equal to the apply, a second apply that changes nothing, a failure that leaves nothing behind |
| Lean rules | `tools/leancheck` | Files under 600 lines, one-line comments |

## Open gates

Each gate needs equipment or accounts the automated suites do not have. Record the result, versions and date when one
passes.

### Metal under a LaunchDaemon

On a Mac Studio with Apple silicon: install `starlight-node` as a LaunchDaemon for `_tensorfold` with `--prevent-sleep`,
deploy a checkpoint the engine qualifies on Metal, then confirm the replica becomes ready, the engine reports the Metal
backend, and its decode speed matches the same engine started from a login session on the same machine. Reboot without
logging in and confirm the replica returns. Check `pmset -g assertions` shows the agent's assertion.

### NVIDIA B300 or GB300 in the CUDA container

Build `deploy/node/container/Dockerfile` with the engine release, run it with the NVIDIA Container Toolkit, and deploy a
checkpoint the engine qualifies on CUDA for that GPU. Confirm readiness, compare throughput with the engine run directly,
and confirm a restart of the container returns the replica. Then deploy eight one-GPU replicas and confirm each engine
sees only its own GPU (`nvidia-smi` lists one process per GPU), and deploy a `tensor_parallel: glm` model with `gpus: 2`
and confirm both ranks start, rank 0 serves, and killing rank 1 restarts the pair. The tests prove the placement,
pinning and rank wiring with a fake engine; they do not run NCCL.

### Rank groups across nodes

On two GB10s joined by RoCE, set `--rank-host` on each node to its RoCE address and deploy a `tensor_parallel` model the
engine qualifies across nodes with `nodes: 2`. Confirm both members start, the lead serves at the engine's two-node
speed, killing the follower's rank brings the group back, and stopping the follower node's agent for 90 seconds stops
the group and places it again. Repeat with `gpus: 8, nodes: 2` on two B300s.

### AMD Instinct

On an MI300X host with ROCm, confirm `starlight-node status` lists the GPUs `rocminfo` reports, less any withheld with
`HIP_VISIBLE_DEVICES`; that a one-GPU replica's engine sees only its GPU; and that `tensor_parallel: rocm` with `gpus: 2`
starts both ranks and serves.

### Windows nodes and clients

On Windows Server 2022 or Windows 11: install with `deploy/node/install-windows.ps1`, confirm the service starts at boot,
the state directory's ACL admits only the service account and administrators (`icacls`), and an engine serves. For
Connect, confirm tokens land in Credential Manager, the tray runs, and Group Policy or Intune values from the ADMX lock the
settings.

### Linux desktop

On GNOME and KDE: run `tensorfold-connect tray`, confirm tokens are stored in the Secret Service (`secret-tool search`) and
that the file fallback is used only where no Secret Service exists.

### Real identity tenants

For each IdP you will support (at least Okta, Microsoft Entra ID, Google Workspace and ADFS), follow its guide in
[sso/](sso/README.md) and check: sign-in, group mapping to a project role, just-in-time and SCIM-provisioned users,
deactivation, back-channel logout where the IdP supports it, the SAML variant, and for Entra a user in more than 200 groups
(overage through Microsoft Graph). For Google, check nested groups through the Directory API.

### MDM rollouts

Deploy Connect and its configuration with Jamf Pro and Microsoft Intune on macOS, and with Group Policy or Intune on
Windows. Confirm the managed server URL is used and cannot be changed, and that the root-CA profile makes a private CA
trusted.

### Load at the 8,000-user profile

The automated run above covers Starlight's own path. For your fleet, define the profile first: for example 10% of 8,000
people active at peak, each starting a request every 30 seconds, with 600 output tokens on average and your real prompt
lengths. Drive it with any OpenAI-compatible load generator through two
or more Starlight instances against real nodes, and record queue wait, time to first byte, errors by code, Postgres CPU and
connections, and gateway CPU and memory. Confirm limits are never exceeded and that killing a gateway, killing a node and
failing over Postgres stay within your error budget.

### High availability and recovery drills

Kill the controller leader and confirm a new epoch within about 15 seconds with no duplicate placement. Restore a backup
into a new database with the master key and confirm `starlight doctor` and `starlight audit verify` pass.

### Images

Docker was not available where these assets were written, so the images have not been built. Build
`deploy/starlight/Dockerfile` (the CI workflow does, for amd64 and arm64) and the node CUDA image, and run the compose
quick start once on Linux and once on macOS.

### Independent security review

Commission a review and penetration test of the gateway, identity, node and console surfaces before production use; see
[security.md](security.md#residual-risks-and-open-review-items).
