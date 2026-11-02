# TensorFold Starlight architecture

TensorFold Starlight turns a fleet of TensorFold engines into one governed, OpenAI- and Anthropic-compatible endpoint for an
organisation. It runs fully offline, or brokers approved frontier providers under the same identity, policy and budget.

| Component | Binary | Runs on | Job |
| --- | --- | --- | --- |
| Starlight | `starlight` | Container (Linux), or any Go platform | Gateway, control plane, console, SSO, SCIM, metrics |
| Starlight Node | `starlight-node` | macOS (launchd), Linux (systemd or container), Windows (service) | Runs and supervises engines on one machine |
| TensorFold Connect | `tensorfold-connect` | macOS, Windows, Linux | Tray app, sign-in and a local endpoint for desktop tools |

Postgres is the only stateful dependency. Every Starlight instance is stateless and can run any subset of its roles
(`gateway`, `control`, `console`, `scim`), so an installation grows from one container to several behind a load balancer.

The inference engine stays the TensorFold native engine. Starlight never changes its arithmetic, batching or scheduling; it
decides who may send which request to which engine, and it accounts for what came back.

## Request path

```
client ──HTTPS──▶ Starlight gateway ──mTLS + permit──▶ Starlight Node ──loopback + engine key──▶ tensorfold serve
                      │
                      └──HTTPS (egress-controlled)──▶ approved frontier provider
```

1. **Authenticate.** A bearer credential is one of: an API key (`slr_`), a Connect access token (`sla_`), a task token
   (`slt_`), or a JWT from an IdP connection that accepts bearer tokens. Identity comes only from the credential.
2. **Canonicalise.** The JSON body is parsed once with duplicate keys refused, policy-relevant fields are rewritten, and
   the exact bytes forwarded are the bytes hashed into the permit.
3. **Resolve.** The requested name resolves through project aliases, then global aliases, to a catalogue model. A model
   the caller may not use answers 404, the same as one that does not exist.
4. **Classify.** The request's classification is the strictest of the project default, the key's floor, a client
   `X-Starlight-Classification` header (raise only), and the secret scanner's verdict. Offline installations make every
   cloud route ineligible.
5. **Admit.** One Postgres statement checks concurrency for every scope the request counts against (organisation,
   project, member, user, service account, key, task tree), counts rate limits, reserves budget and leases a slot on an
   eligible replica, locking only that replica. Without capacity the request waits in a shared priority queue with
   ageing.
6. **Dispatch.** Local routes carry an Ed25519 permit bound to node, replica, deployment revision, body hash and a
   two-minute deadline. Provider routes use the organisation's sealed credential through an egress-allowlisted client.
7. **Settle.** Usage from the final event replaces the reservation. An outcome the gateway cannot observe is charged in
   full. Leases heartbeat every 15 seconds; a crashed gateway's leases free themselves within 60 seconds.

## Identity

Starlight is an OpenID Connect relying party and a SAML 2.0 service provider towards the organisation's identity providers,
and an OAuth 2.0 authorisation server towards its own clients.

- **Upstream SSO.** Any number of connections, each OIDC or SAML, created from presets (Okta, Microsoft Entra ID, Google
  Workspace, Auth0, OneLogin, Ping, JumpCloud, Keycloak, Authentik, Zitadel, AWS IAM Identity Center, ADFS, Shibboleth,
  Duo, generic). Email domains route users to their connection. Groups come from claims, from SCIM, or from directory
  APIs where the IdP does not put them in tokens (Entra overage, Google Workspace).
- **Stable identity.** A user is linked by the connection's immutable subject (Entra `tid` + `oid`), never by email alone.
  Linking a new subject to an existing user by email requires that the IdP proves the email.
- **Provisioning.** SCIM 2.0 creates, updates and deactivates users and groups. Deactivation revokes sessions, refresh
  and access tokens, API keys, tasks and pending device grants in the same transaction.
- **Own clients.** Connect signs in with authorization code + PKCE on a loopback redirect; headless tools use the device
  grant. Refresh tokens rotate, and a reused refresh token revokes its whole family.

## Authorisation

Projects are the unit of access and budget. IdP groups map to project roles; a request always acts in one project.

| Role | Scope | Can |
| --- | --- | --- |
| `owner` | organisation | Everything, including other owners |
| `security_admin` | organisation | SSO, SCIM, sessions, egress, providers' data policy, audit |
| `fleet_operator` | organisation | Nodes, enrolment, placement, draining |
| `model_publisher` | organisation | Catalogue, deployments, aliases |
| `budget_admin` | organisation | Limits, budgets, usage, subscriptions |
| `auditor` | organisation | Read everything except secrets |
| `project_admin` | project | Members, keys, service accounts, plan approval |
| `member` | project | Use the project's models, own keys, tasks and plans |
| `viewer` | project | Usage only |

Model grants say which models a project may use and with what context and output caps. Route policy (`local_only`,
`local_first`, `cloud_first`, `cloud_only`) and classification ceilings on providers and node pools say where a request
may run. Grants are unioned; route and data rules always intersect.

## Fleet

Nodes enrol with a one-time token and a CSR; Starlight's internal CA issues a 30-day certificate naming
`spiffe://<trust-domain>/node/<id>`. Gateways hold `spiffe://<trust-domain>/gateway/<id>` certificates. Each side checks
the other's SPIFFE ID, not hostnames.

A deployment is a model revision plus a typed engine spec (artifact, drafter, context, parallelism, KV cache, backend).
Nodes build engine argv from that spec only; there is no free-form flag field. Artifacts must pin a Hugging Face commit, an
absolute local path or an HTTPS URL with a SHA-256.

The controller runs on one Starlight instance at a time under a Postgres lease with a fencing epoch. It places replicas on
nodes by pool, labels, backend, memory and cached artifacts; keeps warm deployments at their minimum; loads on-demand
deployments when requests queue; and unloads idle ones after their residency window.

Nodes long-poll `POST /node/v1/sync`, report replica states and receive the full desired state. Every five seconds they
also post each replica's live load (generated and prompt tokens per second, running and waiting requests, KV-cache use)
to `POST /node/v1/stats`, which the console's live dashboard reads. Engines listen on loopback with a per-process key
file, so nothing on the node can reach an engine without going through the node agent; an unscoped key in that file is
used only for stored Responses, which Starlight has already checked belong to the caller.

## Agents and frontier planning

A task token (`slt_`) is a scoped credential for an agent run: a model subset, local-only routing, a concurrency cap, an
expiry, a delegation depth and a token budget shared by the whole tree. Children are minted from a parent, never wider
than it, and revoking a task revokes its subtree.

A plan is a frontier-drafted plan executed locally. The brief and explicitly attached context pass the secret scanner,
their exact bytes are hashed and recorded, the frontier model writes the plan, and an approver approves its SHA-256.
Approval issues a task token pinned to the local executor's deployment revisions; if the executor's route changes to a
different checkpoint or a cloud route, the token stops working.

## Operations

- The admin console at `/console` is server-rendered and works without JavaScript; a small same-origin script refreshes
  the live dashboard (throughput, requests in flight, who is connected, node and model state) every few seconds.
- `starlight doctor` checks configuration, database, schema, clock skew, keys, CA, permit keys, controller leadership, nodes,
  identity providers and providers, and prints the fix for each problem.
- Prometheus on a separate listener; usage and audit stay in Postgres.
- Audit events form a SHA-256 hash chain; `starlight audit verify` recomputes it.
- Secrets at rest are sealed with AES-256-GCM data keys wrapped by a master key; `starlight keys rotate` adds a data key.
- Migrations are embedded, versioned, checksummed and applied under an advisory lock.
