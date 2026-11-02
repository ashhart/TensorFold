# TensorFold Starlight

TensorFold Starlight gives an organisation one OpenAI- and Anthropic-compatible endpoint for its private AI fleet. It signs
people in with the organisation's identity provider, decides who may use which model and where each request may run,
places models on Macs and GPU servers that run the TensorFold engine, and accounts for every token. It runs fully
offline, or brokers approved frontier providers under the same identity, policy and budget.

| Component | Binary | Runs on | Job |
| --- | --- | --- | --- |
| Starlight | `starlight` | Container, Kubernetes or a Linux service | Gateway, control plane, admin console, SSO, SCIM, metrics |
| Starlight Node | `starlight-node` | macOS, Linux (service or container), Windows | Runs and supervises TensorFold engines on one machine |
| TensorFold Connect | `tensorfold-connect` | macOS, Windows, Linux | Sign-in, a local endpoint for desktop tools, a tray app |

Postgres 15 or newer is the only stateful dependency. Every Starlight instance is stateless; run one container or several
behind a load balancer.

## Quick start with Docker Compose

This brings up Starlight and Postgres on one machine for a trial. Production installs are covered in
[docs/install.md](docs/install.md).

```bash
cd deploy/compose
./bootstrap.sh
docker compose build
docker compose run --rm starlight init --org "Example Corp"
docker compose up -d
```

`bootstrap.sh` writes the master key (what `starlight init --write-master-key` does on a bare host), the database password and
a `.env` file, all readable only by you. Back up `secrets/master-key`: without it Starlight cannot read the secrets it seals.

`starlight init` prints a one-time setup link. Open it, connect your identity provider with the setup wizard, and sign in;
the first person to finish signing in becomes the owner. To try single sign-on without your own IdP, add
`--profile idp` to `docker compose up`; `bootstrap.sh` printed the development Keycloak's issuer, client secret and test
user.

Enrol a machine that runs the TensorFold engine. Starlight prints the exact command, including the CA fingerprint the node
pins:

```bash
docker compose exec starlight starlight nodes token --pool default --auto-approve
```

Run the printed `starlight-node enroll ...` command on the node, then `starlight-node run` (or `starlight-node service install`).
Add a model and a deployment in the console at `/console`, and point a client at Starlight:

```bash
tensorfold-connect login --server http://localhost:8443
tensorfold-connect bridge &
tensorfold-connect env openai
```

or use an API key from the console with any OpenAI SDK (`OPENAI_BASE_URL=http://localhost:8443/v1`).

## The console

`/console` opens on a live operations view: tokens per second across every node, requests in flight and queued, active
users, nodes online, models available and today's tokens and spend, with an hour of throughput and requests. Below it,
who is in flight on which model and node, who is connected, every node's live load and KV cache with Drain, Take offline
and Revoke controls, and every model's availability and routing. The same console manages projects, models,
deployments, providers, limits, frontier subscriptions, plans, single sign-on, users, groups, administrators and the
audit log; people without an organisation role see only their own usage. `starlight apply -f` keeps the same configuration
in version control.

## What it does

| Area | Summary | Read |
| --- | --- | --- |
| Architecture | Request path, identity, authorisation, fleet, agents | [docs/architecture.md](docs/architecture.md) |
| Installation | Compose, Helm, systemd, TLS, nodes, Connect rollout | [docs/install.md](docs/install.md) |
| Operations | Backup, HA, upgrades, key rotation, offline installs, monitoring, troubleshooting | [docs/operations.md](docs/operations.md) |
| Security | Threat model per boundary, controls, revocation bounds, residual risks | [docs/security.md](docs/security.md) |
| API | Inference endpoints, headers, errors, discovery, OAuth, tasks and plans | [docs/api.md](docs/api.md) |
| Single sign-on | OIDC and SAML with 18 presets, groups, logout, break-glass | [docs/sso/README.md](docs/sso/README.md) |
| Provisioning | SCIM 2.0 for Okta, Entra ID and others | [docs/scim.md](docs/scim.md) |
| Fleet | Catalogue, deployments, placement, GPUs and tensor-parallel ranks, rollouts, artifacts, configuration as code | [docs/fleet.md](docs/fleet.md) |
| Nodes | Installing and running Starlight Node | [docs/nodes.md](docs/nodes.md) |
| Admission | Limits, budgets, priorities, the queue, failover | [docs/admission.md](docs/admission.md) |
| Frontier providers | Providers, classification, the credential guard, offline mode, frontier-planned local work | [docs/frontier.md](docs/frontier.md) |
| Connect | Desktop sign-in, the local endpoint, MDM | [docs/connect.md](docs/connect.md) |
| Metrics | Every Prometheus series and its labels | [docs/metrics.md](docs/metrics.md) |
| Qualification | What the tests prove and what still needs real hardware or tenants | [docs/qualification.md](docs/qualification.md) |

## Security posture

- Identity comes only from a verified credential: an API key, a Connect token, a task token, or an IdP-issued JWT.
- A request names one project; model grants, limits and data classification are checked before any byte is forwarded.
- Gateways reach nodes over mutual TLS with SPIFFE-named certificates and an Ed25519 permit bound to the exact request.
- Engines listen on loopback with a per-process key, so nothing on a node can bypass the agent.
- Secrets at rest are sealed with AES-256-GCM under a master key; audit events form a SHA-256 hash chain.
- Offline mode and per-provider allowlists stop cloud egress at the HTTP client, not only in policy.

## Licence

Apache-2.0, as the rest of TensorFold. Third-party licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
