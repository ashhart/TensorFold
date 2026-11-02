# Operating Starlight

## Troubleshooting starts with the doctor

```bash
starlight doctor
```

The doctor reads the same configuration as `starlight serve` and checks, in order:

| Area | Fails or warns when |
| --- | --- |
| Configuration and TLS | The certificate files do not load or the public certificate expires within 30 days; TLS terminates elsewhere but `trusted_proxies` is empty; `node_url` is unset on an instance with the `control` role |
| Database | Postgres is unreachable (the doctor stops there) or older than 15 |
| Clock | This host and Postgres differ by more than 2 seconds (permits tolerate 5 seconds) |
| Schema | Migrations are pending |
| Keys | The master key is missing or cannot unwrap the stored data keys |
| Setup | The installation is not initialised, or it has no active owner |
| Trust | The workload CA is missing or expires within a year, or there is not exactly one active permit key |
| Controller | No instance holds the controller lease |
| Nodes | No node is online, or nodes are stale (not seen for 90 seconds), awaiting approval, or hold certificates expiring within 7 days |
| Replicas | Replicas have failed |
| Single sign-on | No connection is enabled, or an enabled connection's discovery document or SAML metadata is unreachable (it honours `identity.ca_file`) |
| Providers | A provider has no credential or its host does not resolve; offline installations skip this |

It prints a hint for each problem and exits non-zero when anything fails, so it also works as a scheduled check.

| Symptom | Usual cause |
| --- | --- |
| `403 project_required` | The person belongs to several projects; send `X-Starlight-Project` or pick a default in Connect |
| `404 model_not_found` | The model, alias or grant is missing for this project, or the key's model list excludes it |
| `503 no_route` | No active deployment for a local model, or the request's classification, offline mode or a task scope rules out every route |
| `503 capacity_unavailable` | Replicas exist but every slot stayed busy until the queue timeout; add replicas or raise `max_replicas` |
| `503 temporarily_unavailable` | Postgres could not be reached or did not answer in time, for example during a failover; clients retry after a second |
| `400 credentials_detected` | The prompt contains what look like credentials and only cloud routes exist; see [frontier.md](frontier.md) |
| Node stays `pending` | Approve it with `starlight nodes approve NAME` or in the console |
| Replica `failed` | The engine exited during start-up; the console shows the engine's last error and its last log lines, and `starlight-node status` its logs |
| Replica error `engine … is older than 1.2.0` | Upgrade TensorFold on that node; Starlight runs TensorFold 1.2.0 and newer |
| Sign-in loops back to the login page | Clock skew, a wrong redirect URI at the IdP, or a proxy stripping cookies; check the doctor and the audit log |

## Backup and restore

Back up two things, separately:

1. **Postgres.** `pg_dump --format=custom` for logical backups, or continuous WAL archiving with your usual tooling for
   point-in-time recovery. Everything Starlight knows lives in this one database.
2. **The master key.** Keep it in a secret manager or an offline vault, never next to the database backups.

Secrets in the database are sealed with AES-256-GCM data keys that the master key wraps: identity-provider client
secrets and SAML signing keys, provider API keys, the workload CA's private key, permit signing keys and plan texts.
Without the master key none of these can be opened and Starlight refuses to start; you would have to re-enter every IdP and
provider secret and re-enrol every node under a new CA. API keys, session and refresh tokens, enrolment and SCIM tokens are
stored only as SHA-256 hashes and do not depend on the key.

To restore, load the dump into an empty database, start one instance with the same master key, and run:

```bash
starlight doctor
starlight audit verify
```

## High availability

Run two or more instances behind a load balancer. Every role is safe to run on several instances at once:

- **Gateways** share all admission state through Postgres. A request holds a lease that its instance heartbeats every
  15 seconds while it still holds it; if the instance dies, or loses track of a lease, the lease expires after 60
  seconds, its reservation is charged in full and the slot frees.
- **The controller** runs on one instance at a time under a 15-second Postgres lease renewed every 5 seconds. Every
  scheduling write checks the lease's fencing epoch in the same transaction, so a paused former leader cannot overwrite
  its successor's decisions. `starlight_controller_epoch` is non-zero only on the current leader.
- **Console, SSO and SCIM** keep their state in Postgres; sessions work on any instance.
- **The node listener** can be served by every instance with the `control` role. Nodes long-poll whichever instance their
  TCP load balancer chooses, and changes reach them through Postgres notifications.

Keep every Starlight host and node within a few seconds of each other and of Postgres with NTP or chrony. Permits allow
5 seconds of skew, and the doctor fails at 2 seconds.

Set the load balancer's idle timeout above the longest stream you allow (`gateway.request_timeout`, 30 minutes by
default); streams are server-sent events over one long response. Sticky sessions are not needed.

Postgres is the availability floor. Use a managed service or a replicated cluster with automatic failover; Starlight's pool
reconnects after a failover, and in-flight requests that lose their database mid-settlement are charged in full when
their leases are reaped.

## Upgrades

1. Read the release notes for migrations and configuration changes.
2. Apply migrations: `starlight migrate`, the Helm pre-upgrade hook, or `starlight serve --migrate` on one instance. Migrations
   are embedded, checksummed and applied under an advisory lock; Starlight refuses to run against edited history.
3. Replace instances one at a time. A new instance refuses to start while any of its migrations are pending, and
   `/readyz` answers 503 until they are applied. An older instance refuses to start against a newer schema and its
   `/readyz` answers 503, so finish the rollout rather than restarting old instances. To roll back, restore the
   backup taken before the migration.

Nodes and Connect are upgraded separately from Starlight; check the release notes for changes to the node protocol. Nodes
need TensorFold 1.2.0 or newer, and an upgraded engine is picked up at the next start of each replica.

## Keys and certificates

| What | How | Effect |
| --- | --- | --- |
| Data key | `starlight keys rotate` | New seals use a new data key; values sealed earlier stay readable with their own key |
| Permit signing key | `starlight keys rotate-permits` | The new key is published to nodes at once and gateways sign with it two minutes later, after every node has synced it (syncs land within 30 seconds); the old key stays verifiable for two more days, so no permit is refused during a rotation |
| Master key | Set the new key as `keys.master_key_file`, list the old one in `keys.previous_key_files`, run `starlight keys rewrap`, then remove the old key from `previous_key_files` | Every data key is re-wrapped under the new master key; values sealed by those data keys need no change; rate-limit windows start afresh |
| Node certificates | Automatic | Nodes renew at two thirds of their 30-day lifetime; revoking a node revokes its certificates at once |
| Public certificate | Replace the files at `tls.cert_file` and `tls.key_file` | Each instance picks up the new pair within a minute, with no restart; a pair that fails to load leaves the current one in service |
| Workload CA | Created by `starlight init`, valid for ten years | Replacing it means re-enrolling every node |

`starlight_certificate_expiry_timestamp_seconds` reports when the public certificate and the workload CA expire, and the
bundled `StarlightCertificateExpiring` alert fires 14 days ahead.

## Scaling

- Each database call on the request path gives up after 10 seconds, and a new connection after 10 seconds unless the
  database URL sets `connect_timeout`, so a stalled Postgres turns into `503 temporarily_unavailable`, not hung requests.
- Add gateway instances for request volume; each holds up to `database.max_conns` connections (32 by default) and keeps
  `database.min_conns` (4) open. An instance opens at most eight new connections at once, so a burst of traffic does
  not flood Postgres with connection attempts.
- Each inference request costs two Postgres statements, one to admit and one to settle, plus a shared heartbeat per
  instance. Admission locks only the replica or route it leases on, so throughput grows with replicas; see
  [admission.md](admission.md#throughput). Queued requests retry admission when capacity frees and poll about once a
  second as a fallback.
- `usage_events` is partitioned by month; the controller creates the current and next month's partitions. Drop old
  partitions (`DROP TABLE usage_events_YYYYMM`) to apply your retention policy for usage records.
- Model capacity grows by adding replicas: raise a deployment's `max_replicas` and enrol more nodes into its pool.

## Offline and air-gapped installations

Build a bundle on a connected machine and carry it in:

```bash
deploy/offline/build-bundle.sh --version 1.0.0 --with-image
```

The bundle holds `starlight`, `starlight-node` and `tensorfold-connect` for macOS arm64, Linux amd64 and arm64, and Windows amd64,
the Starlight image as a `docker save` archive when `--with-image` is given and Docker is available, every deploy asset and
document, `SHA256SUMS` and `manifest.json`. Verify it with `shasum -a 256 -c SHA256SUMS` before use. The Mac build of
Connect includes the tray only when the bundle is built on a Mac.

Then:

- Initialise with `starlight init --offline`, or switch offline mode on later under Settings in the console. Offline mode
  makes every cloud route ineligible and the provider HTTP client refuses all requests, whatever the providers table says.
- Nodes learn offline mode in each sync and start engines with Hugging Face access disabled. Use `path` artifacts staged
  on the node, `url` artifacts on an internal HTTPS server with a SHA-256, or an internal Hugging Face mirror with
  `starlight-node run --hf-endpoint https://hf-mirror.example.internal`.
- Single sign-on needs an identity provider reachable from Starlight; offline mode governs model traffic, not IdP traffic.
- Starlight makes no update checks or telemetry calls.

## Monitoring

`deploy/prometheus` has a scrape configuration and alert rules; `deploy/grafana` has a provisioned dashboard. Fleet
gauges (`starlight_nodes`, `starlight_replicas`, `starlight_queue_depth`) are sampled from Postgres by every instance, so aggregate
them with `max`, not `sum`. The series are described in [metrics.md](metrics.md).

## Audit

```bash
starlight audit verify
starlight audit export --since 2026-01-01T00:00:00Z > audit.jsonl
```

`verify` recomputes the SHA-256 chain and reports the sequence range and the anchor hash it started from. `export` writes
JSON lines for a SIEM. Retention is set under Settings in the console: 0 keeps everything, otherwise 30 to 3,650 days
(400 by default); the controller prunes hourly, and verification then starts from the oldest retained event.

## Break-glass access

When the identity provider is down, an operator with database access can sign in an existing owner:

```bash
starlight admin recovery --email owner@example.com
```

The link works once, for 15 minutes, behind a confirmation page, and both issuing and using it are audited. While no
owner exists, `starlight admin setup-code` issues a new first-owner setup link instead.
