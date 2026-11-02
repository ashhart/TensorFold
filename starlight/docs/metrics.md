# Metrics

Starlight serves Prometheus metrics on the metrics listener (`listen.metrics`, `127.0.0.1:9090` by default, `:9090` in the
container image) at `GET /metrics`. Set `metrics.bearer_token_file` to require `Authorization: Bearer <token>`. Labels
come from the catalogue and fixed vocabularies, never from people, keys or prompts, so cardinality stays bounded by the
number of models and deployments.

## Request series (per instance)

| Series | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `starlight_requests_total` | counter | `model`, `protocol`, `route`, `status`, `outcome` | Finished inference attempts |
| `starlight_request_duration_seconds` | histogram | `model`, `route` | Arrival to last byte |
| `starlight_first_byte_seconds` | histogram | `model`, `route` | Arrival to the first byte sent to the client |
| `starlight_queue_wait_seconds` | histogram | `model`, `priority` | Time queued for capacity, for requests that queued |
| `starlight_tokens_total` | counter | `model`, `route`, `direction` | Tokens; `direction` is `prompt`, `completion` or `cached` |
| `starlight_admission_rejections_total` | counter | `reason` | Requests refused before dispatch |
| `starlight_inflight_requests` | gauge | `route` | Requests this instance is streaming now |

Label values:

- `route`: `local` or `cloud`.
- `protocol`: `openai.chat`, `openai.completions`, `openai.embeddings`, `openai.responses`, `anthropic.messages`,
  `anthropic.count_tokens`.
- `status`: `2xx`, `4xx`, `5xx`, or `other` for attempts that never received a status.
- `outcome`: `ok`, `client_error`, `upstream_error`, `upstream_unreachable`, `interrupted`, `cancelled`, `revoked`, or
  `not_dispatched` for an attempt refused by a node or provider before it started and then retried elsewhere.
- `priority`: `interactive`, `standard`, `batch`.
- `reason`: the error code for policy refusals (for example `invalid_json`, `model_not_found`, `credentials_detected`,
  `context_length_exceeded`, `permission_denied`), `unauthenticated`, `project_required`, `concurrency_limit`,
  `rate_limit`, `budget_limit`, `capacity`, `no_route`, `client_left`, `internal`.

Histogram buckets are 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300 and 600 seconds.

## Fleet series (installation-wide)

Every instance samples these from Postgres every 15 seconds, so all instances report the same values. Aggregate them with
`max`, not `sum`.

| Series | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `starlight_nodes` | gauge | `status` | Nodes by status: `active`, `pending`, `draining`, `disabled`, `revoked`, and `stale` for active nodes not seen for 90 seconds |
| `starlight_replicas` | gauge | `deployment`, `state` | Replicas by deployment and state |
| `starlight_queue_depth` | gauge | `model` | Requests waiting for capacity across all gateways |
| `starlight_controller_epoch` | gauge | none | The fencing epoch on the current leader, 0 on every other instance |
| `starlight_certificate_expiry_timestamp_seconds` | gauge | `certificate` | When the `public` certificate and the `workload_ca` expire, as Unix time; the public pair is re-read from disk within a minute of changing |

The Go runtime and process collectors (`go_*`, `process_*`) are also exported.

## Not exported as series

- **Per-replica throughput.** Nodes post decode and prompt tokens per second, running and waiting requests and KV-cache use
  every 5 seconds; Starlight keeps the latest sample for the console's dashboard, but there is no Prometheus series for it
  yet. Engines expose their own `/metrics` on loopback only.
- **Certificate expiry.** No series reports certificate lifetimes. `starlight doctor` warns when a node certificate expires
  within 7 days, the public TLS certificate within 30 days or the workload CA within a year, and fails once the TLS
  certificate or the CA has expired; run it on a schedule.
- **Usage and spend.** Per-project and per-person usage live in Postgres (`usage_events`) and the console, not in metrics.

## Alerts and dashboard

`deploy/prometheus/rules/starlight.yaml` alerts on an instance down, no controller leader, stale or pending nodes, failed or
lost replicas, a deep queue, a server-error ratio above 5%, a surge in admission refusals, capacity exhaustion,
`no_route` refusals, high queue waits, a surge in invalid credentials, and a certificate within 14 days of expiry. `deploy/grafana/dashboards/starlight.json` charts
every series above. Thresholds are starting points; tune them to your traffic.

Example queries:

```promql
sum by (model) (rate(starlight_tokens_total{direction="completion"}[5m]))
histogram_quantile(0.95, sum by (le, model) (rate(starlight_first_byte_seconds_bucket[5m])))
sum(rate(starlight_requests_total{route="cloud"}[1h])) / sum(rate(starlight_requests_total[1h]))
max by (deployment) (starlight_replicas{state="ready"})
```
