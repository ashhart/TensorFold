# Admission: limits, budgets, priorities and the queue

Every inference request is admitted by one Postgres statement, a call to `starlight_admit` in its own short transaction. It
checks concurrency for every scope the request counts against, counts rate limits, reserves budget, and leases a slot on
one eligible replica or provider route. If any check fails, the statement raises an error and nothing is kept. Several
gateways share this state, so limits hold across all of them.

## Limit kinds and scopes

| Kind | Window | Counts |
| --- | --- | --- |
| `concurrency` | none | Requests in flight |
| `requests_per_minute` | calendar minute (UTC) | Requests started |
| `tokens_per_minute` | calendar minute | Reserved prompt and output tokens, corrected when the request settles |
| `tokens_per_day`, `tokens_per_month` | UTC day, UTC month | Prompt plus completion tokens |
| `cost_per_day`, `cost_per_month` | UTC day, UTC month | Micro-dollars at the route's prices |

| Scope | Applies to |
| --- | --- |
| `org` | Every request |
| `project` | All requests in a project together |
| `member` | Each person separately, within a project |
| `user` | One person across projects |
| `service_account` | One service account |
| `api_key` | One key |

A limit may name a model, in which case it counts only that model. A limit can carry a temporary boost (`boost_value`
until `boost_until`) for a deadline or an incident. Tasks add their own scopes automatically: each task's
`max_concurrency`, the root's concurrency for the whole tree, and the root's `budget_tokens` for the whole tree.

## Reservations and settlement

Admission reserves before any work starts:

- **Tokens:** an estimate of the prompt (a third of the request's bytes, plus 1,600 tokens for each inline image) plus the
  output limit.
- **Cost:** those tokens at the highest price among the routes the request may take, local chargeback included.

When the request ends, one more statement (`starlight_settle`) replaces the reservation with the usage the engine or
provider reported, priced at the route that served it, writes the usage record and wakes waiting gateways. If usage
never arrives (the stream broke, or a gateway died), the reservation is charged in full. Each gateway heartbeats the
leases and queue entries it still holds every 15 seconds, and only those; the controller reaps leases silent for 60
seconds and charges them, so one a gateway lost track of is freed within about 75 seconds. Settlement is retried when a
failure proves nothing was written. A request whose caller leaves while it is being admitted has its lease released at
once, with its reservation refunded.

Reservations are bounded so they cannot overflow: an output limit (`max_tokens` and its aliases) above 1,048,576, or `n`
or `best_of` above 16, answers `400 invalid_request_error`, and costs saturate rather than wrap.

A refused request answers `429` with `concurrency_limit`, `rate_limit` or `budget_limit` and a `Retry-After` until the
window resets (1 second for concurrency).

## Priority

| Class | Default for | Queue timeout |
| --- | --- | --- |
| `interactive` | Connect and CLI access tokens | `gateway.queue_interactive`, 30 seconds |
| `standard` | API keys (unless the key says otherwise), IdP tokens, service accounts, tasks | `gateway.queue_standard`, 2 minutes |
| `batch` | Keys or requests that ask for it | `gateway.queue_batch`, 15 minutes |

A request may lower its class with `X-Starlight-Priority` or `"priority": "background"`. It may raise it only up to its
credential's ceiling (an API key's `priority`; interactive for people's access and IdP tokens; standard for service
accounts and tasks) and the project's `max_priority`. Batch requests on local engines are sent
with the engine's `priority: "background"`, so the engine also yields to foreground work.

## The queue

When no eligible replica or route has a free slot, the request waits in a queue shared by every gateway. Its rank is its
class (interactive 2,000, standard 1,000, batch 0) plus 100 for every `gateway.ageing_step` (30 seconds) it has waited, up
to one class, so batch work cannot starve forever and never overtakes interactive work. Ranks are computed in Postgres
on the database's clock, so every gateway ranks every waiter alike. Equal ranks go first in, first
out. A request may take a free slot only when fewer higher-ranked waiters could use that slot.

Waiters wake when capacity is released anywhere, through Postgres notifications: each release wakes the best-ranked
waiter not already woken, so slots freed together are refilled together; a woken waiter that finds its slot owed to a
better-ranked one hands the wake-up on. A waiter polling on its own first looks for
room without taking any lock, so a deep queue does not slow admissions, and it polls about once a second as a fallback.
A waiter that misses its queue timeout answers `503 capacity_unavailable` with `Retry-After: 2`. Queued requests for an
on-demand model wake the controller, which loads a replica.

## Throughput

Admission locks only what it counts: each scope with a concurrency limit, the rate and budget counters' rows, and the
one replica or route it leases on. Requests for different replicas of a model never wait for each other, so adding
replicas adds admission throughput. With 64 callers against Postgres 17 on one Apple M3 Ultra, four runs of
`STARLIGHT_SCALE=1 go test -run TestAdmissionScalesWithReplicas ./internal/admission` gave:

| Replicas | Admit-and-settle cycles a second |
| --- | --- |
| 1 | 7,776 to 10,056 |
| 4 | 18,514 to 22,261 |
| 8 | 19,757 to 22,901 |

The earlier design, one lock per model and several statements per admission, managed about 4,100 to 4,600 at every
replica count. Eight replicas on one machine are bound by that machine's CPU, which Postgres and the callers share.

## Choosing a target

Among eligible targets, admission follows the model's route policy (local first unless `cloud_first` or `cloud_only`).
Among local replicas it prefers the one that last served the same prompt prefix, by rendezvous hashing of the first 4 KiB
of the conversation, while that replica has at least a quarter of its slots free; otherwise it takes the replica with the
most headroom. Among provider routes it takes the lowest `priority` number with capacity.

Eligibility is checked again inside the admitting statement, so a queued request never acts on a stale view: the
replica must still be ready on its current revision, on an active node cleared for the request's classification, and
for a rank group every member's node must be active, recently seen and cleared too. A provider route must still be
enabled, its provider enabled and cleared for the classification, and its `max_concurrency` is read there and then.

## Failover

A request is retried on another target, at most three attempts in all, only when it provably never started:

- the connection to the node or provider failed before the request was written, or
- the node answered `429 replica_busy` or `503` with `Starlight-Refused: 1`, or a provider answered `429`.

Each retry excludes the targets already tried and is admitted again; it counts against limits again, except
`requests_per_minute`, which the first attempt already counted. A failure after the request reached an engine or
provider is never retried, because the work may have run.
