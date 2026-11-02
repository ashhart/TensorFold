# Security model

This document lists what Starlight defends at each trust boundary, the control that does it and where it lives in the code,
and what remains for operators or for review. File paths are relative to `starlight/`.

## Credentials

| Credential | Prefix | Lifetime | Stored as |
| --- | --- | --- | --- |
| API key | `slr_` | Until expiry or revocation; rotation keeps the old key for a grace of up to 7 days | SHA-256 |
| Connect or CLI access token | `sla_` | 10 minutes (`sessions.access_ttl`) | SHA-256 |
| Refresh token | `slf_` | 30 days idle, 90 days absolute; rotated on every use | SHA-256 |
| Task token | `slt_` | Up to 7 days, never beyond its parent or the key that minted it | SHA-256 |
| Node enrolment token | `sle_` | 24 hours by default, single use by default | SHA-256 |
| SCIM token | `sls_` | Optional expiry | SHA-256 |
| Console session | `slz_` | 30 minutes idle, 12 hours absolute | SHA-256 |

Every token carries 280 random bits, at least 200 of them never displayed, and a checksum so that leaked tokens are
recognisable by secret scanners
(`internal/crypto/secret/secret.go`). Starlight never stores a usable token.

## Client to gateway

| Threat | Control | Code |
| --- | --- | --- |
| Stolen or guessed credentials | High-entropy tokens, expiry, CIDR allowlists on keys, per-address limits on the token, device, sign-in, setup and recovery endpoints | `internal/authn`, `internal/ratelimit` |
| Spreading attempts across instances | The limits on the token, device, sign-in, setup, recovery, SAML, SCIM and enrolment endpoints check an in-memory token bucket first, then count a fixed window in Postgres (`counters` rows keyed `rl:`), so all instances together allow one budget. Keys are HMAC-SHA256 values under a key derived from the master key, never addresses or tokens; IPv6 clients count per /64, and requests whose address cannot be read share one budget. If Postgres cannot answer, each instance keeps its own bucket and logs once a minute | `internal/ratelimit` |
| Refresh-token theft | Rotation with reuse detection: presenting a spent refresh token revokes its whole family | `internal/identity/oauth/tokens.go` |
| Privilege escalation through headers | Clients can lower priority and raise classification, never the reverse; identity never comes from headers | `internal/gateway/policy.go` |
| Policy bypass with ambiguous JSON | One top-level object, duplicate keys refused; the bytes forwarded are the bytes the permit hashes | `internal/gateway/body.go` |
| Probing the catalogue | An ungranted model answers exactly like an unknown one | `internal/catalog/catalog.go` |
| Reading another person's stored response | Ownership recorded per response; reads, deletes and continuations check principal and project | `internal/gateway/responses.go` |
| Exhaustion | Body limit, concurrency, rate and budget limits, queue timeouts, request timeout | `internal/admission`, `internal/config` |
| An outage read as a bad credential | Every database call on the request path is bounded (10 seconds; connections time out after 10 seconds unless `connect_timeout` says otherwise). When Postgres or an IdP's keys cannot be reached, are too slow, drop the connection or refuse work, authentication and admission answer `503 temporarily_unavailable` with `Retry-After: 1`, never 401 or 403, and nothing is admitted until they answer | `internal/authn`, `internal/db`, `internal/gateway/serve.go` |
| Learning another project's prompts from cache timing | Projects with `isolate_prompt_cache` are served only by deployments that keep no prompt states or whose model no other project is granted; granting such a model to another project, or withdrawing it from its isolated project, restarts its replicas that keep prompt states, so none carries a prefix across | `internal/admission/target.go`, `internal/admin/fleet/projects.go` |
| Credentials pasted into prompts | The credential guard keeps such requests local or refuses them | `internal/guard`, `internal/gateway/policy.go` |

Revocation is bounded:

- A revoked key, token, user or task stops authenticating new requests on every instance within `sessions.auth_cache`
  (15 seconds by default, at most one minute). The instance that made the change from the console clears its cache at once.
- A request already streaming re-checks its credential every 10 seconds, with its project access and its scope and grant
  in the request's project, and is cancelled when any of them no longer holds, so a revoked stream ends within the
  watchdog interval plus the cache window (`internal/gateway/serve.go`).
- Deactivating a user (console or SCIM) revokes sessions, refresh and access tokens, API keys, tasks and pending device grants
  in one transaction (`internal/identity/directory/users.go`).

## Gateway to node

| Threat | Control | Code |
| --- | --- | --- |
| Rogue node joining | One-time enrolment tokens, a CSR whose key never leaves the node, manual approval unless the token pre-approves, certificate serials checked on every control call | `internal/control` |
| Impersonating a node or gateway | An internal CA name-constrained to the trust domain; both sides check the peer's SPIFFE ID, not a hostname; TLS 1.3 | `internal/crypto/pki` |
| Tampering or replay | An Ed25519 permit binds node, replica, deployment revision, method, path, the body's SHA-256 and a 30-second deadline; nodes remember request ids until expiry | `internal/crypto/permit`, `internal/node/proxy` |
| Calling a node directly | The node accepts only gateway certificates and valid permits | `internal/node/proxy` |
| Revoked node | Certificates revoked with the node; the node is told it is revoked at its next sync and stops its engines; gateways stop routing to it at once | `internal/control/sync.go`, `internal/admission/target.go` |

A node processes prompts and model state in plaintext. Treat every node as trusted with the data classifications it may
receive: enrolment tokens and nodes carry a `max_classification`, and admission never places a request on a node below the
request's classification. For a tensor-parallel group that spans nodes, every member's node must be cleared too.

## Node to engine

| Threat | Control | Code |
| --- | --- | --- |
| Local processes using the engine | Engines listen on `127.0.0.1` with a fresh per-process key file (0600); the agent's state directory is 0700 and owned by the service account | `internal/node/engine`, `internal/node/state` |
| Reaching engine admin routes | The node forwards only an allowlist of inference routes; the unscoped engine key is used only for stored Responses | `internal/control/wire/wire.go`, `internal/node/proxy/keys.go` |
| Leaking keys through the environment | Engines start with a minimal environment and never receive `TENSORFOLD_API_KEY` | `internal/node/engine` |
| Engines that ignore key scopes | Every start checks the engine's release and refuses one older than TensorFold 1.2.0, the release Starlight is qualified on; releases before 1.1.8 would give every key in the key file full access | `internal/node/engine/version.go` |

Anyone with administrator rights on a node can read its key files and its traffic. Run nodes on managed machines, and
keep the network between the nodes of a pool that runs rank groups private to that pool.

## Starlight to frontier providers

| Threat | Control | Code |
| --- | --- | --- |
| Data leaving against policy | Offline mode, route policy, provider classification ceilings and task scopes decide eligibility before dispatch | `internal/gateway/policy.go` |
| Egress to unapproved hosts or internal services | The only outbound client allows the hosts of enabled providers, requires HTTPS unless a provider is marked internal, refuses internal addresses unless a provider is internal, always refuses the metadata address, and does not follow redirects | `internal/providers/egress.go` |
| Provider keys at rest | Sealed with the keyring, bound to the provider row, cached in memory for five minutes | `internal/providers/providers.go` |
| Engine extensions confusing providers | TensorFold-only fields are dropped on cloud routes | `internal/gateway/serve.go` |
| Unrecorded egress of planning context | Plans record the SHA-256 and size of the exact messages sent to the planner, and an audit event | `internal/plans` |

What a provider retains is governed by your contract with it; Starlight cannot enforce it.

## Starlight to identity providers

The checks are listed in [sso/README.md](sso/README.md#what-starlight-checks). In short: PKCE S256, single-use state and
nonce bound to the starting browser, asymmetric signature algorithms only, issuer, audience, `azp` and lifetime checks;
Microsoft Entra users identified by `tid` and `oid` with email trusted only when the tenant proves it; Google Workspace
domain and `email_verified` enforced; SAML responses must answer a request Starlight sent, with a strict audience check,
assertion replay protection and IdP-initiated sign-in off by default; post-login redirects stay on Starlight
(`internal/identity/session/guard.go`). SAML responses are bounded in depth, width and size, and their signature
placement is checked, before any signature work; SAML signatures must use SHA-256 or stronger. Logout messages in
either direction must be signed by a current IdP certificate, name the IdP and Starlight's logout URL, be fresh, stay under
32 KiB, and be accepted once. Disabling or deleting a connection ends its sessions and revokes the Connect and CLI
tokens of the people who sign in through it. Clients can authenticate to the IdP
with `private_key_jwt` (RS256 or ES256) instead of a shared secret, which Microsoft Entra ID recommends. A private CA
for on-premises IdPs is trusted only through `identity.ca_file`.

## SCIM

| Threat | Control | Code |
| --- | --- | --- |
| One connection changing another's people | Tokens are per connection and stored as hashes; a token cannot read or change another connection's users or groups | `internal/identity/scim/auth.go` |
| Unrecorded or injected changes | Every write is audited in its transaction; filters compile to parameterised SQL from a fuzzed parser | `internal/identity/scim` |
| Flooding | Each token is rate limited on all instances together, with `Retry-After` on 429 | `internal/identity/scim/handler.go`, `internal/ratelimit` |
| Locking everyone out | Suspending or deleting the last active owner answers 409 and is audited as denied; owners count whether they hold the role directly or through a group | `internal/identity/scim/owners.go`, `internal/admin/people/owners.go` |

A stolen SCIM token can still deprovision that connection's users, so store it only in the IdP and rotate it if exposed.
Group changes are never refused, so removing the last owner from an owner group in the IdP still removes the role; keep
at least one owner who holds it directly.

## Admin console

Sessions use an `HttpOnly`, `SameSite=Lax` cookie (`__Host-` prefixed over HTTPS) with idle and absolute expiry, and a new
sign-in replaces any existing session. Unsafe methods need a per-session CSRF token and a same-origin request
(`internal/identity/session`). Pages are rendered with `html/template` under a restrictive Content Security Policy and
cannot be framed (`internal/web/web.go`, `internal/console/request.go`). Every page and action checks the signed-in user's
permissions (`internal/console/console.go`), and administrative changes are audited. Suspending or deactivating the last
active owner, or removing the role binding that makes them one, is refused (`internal/admin/people/owners.go`).

## Database and master key

Secrets at rest are sealed with AES-256-GCM data keys wrapped by the master key; the additional data names the row and
column, so a sealed value copied to another row does not open (`internal/crypto/keyring`). Audit events form a SHA-256
chain verified by `starlight audit verify` (`internal/audit`). Migrations are checksummed and refuse edited history
(`internal/db/migrate.go`).

Anyone who can write to the database can change policy directly, and the audit chain protects audit rows, not policy
rows. Give Starlight a dedicated database role, require TLS to Postgres (`sslmode=verify-full`), and keep the master key out
of the database's backups and hosts.

## Residual risks and open review items

- Two adversarial internal reviews found issues that are all fixed with regression tests. The first found ten:
  cross-connection account claims, task tokens outliving their minting key, credential-guard bypass through JSON
  escapes, the reaper freeing live leases, open redirects through backslashes, private-address egress for public
  providers, the device grant's rate limit, tree concurrency, `n` and `best_of` reservations, and a race in bearer-token
  connections. The second, of rank groups across nodes, admission, sign-in and shared rate limits, found twenty-seven,
  among them a deadlock between admitting and settling under related limits, budgets disabled by an overflowing
  `max_tokens`, leases kept alive after their gateway lost them, failure timers reset by every node sync, groups left
  running after their lead was lost, follower nodes' classification ignored, a pasted private key kept in plain text,
  client addresses lost behind proxies that send ports, and CPU exhaustion through SAML logout. No third-party review
  or penetration test has been done; commission one before production use.
- Single sign-on has been tested against in-process IdP fakes, not real tenants; see [qualification.md](qualification.md).
- A deployment that keeps prompt states shares them among the projects its model is granted to, so timing can reveal
  to one of them whether another recently sent a prefix. Set `isolate_prompt_cache` on projects that must not share,
  or `prompt_cache: off` on the deployment; see [fleet.md](fleet.md#prompt-caches).
- Rank groups that span nodes depend on the network between them, which Starlight does not protect: a host that reaches a
  lead's rendezvous ports can disrupt its group. Give such pools a private network.
