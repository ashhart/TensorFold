# Fleet: catalogue, deployments and nodes

Starlight separates what people ask for (models and aliases), who may use it (grants), and where it runs (deployments on
nodes, or provider routes). Model publishers manage the catalogue; fleet operators manage nodes.

## Catalogue

| Object | Fields | Notes |
| --- | --- | --- |
| Model | `name`, display name, description, `kind` (`chat` or `embedding`), capabilities, `route_policy`, chargeback prices, status | `name` is what clients send, for example `qwen3.8-27b` |
| Alias | alias, model, optional project | A project alias overrides a global one with the same name; aliases let clients keep the model names they already send |
| Grant | model, project or every project, `max_context`, `max_output_tokens` | Without a grant a model is invisible to the project |
| Provider route | model, provider, upstream model, protocols, priority, prices, `max_concurrency` | See [frontier.md](frontier.md) |

`route_policy` decides between local deployments and provider routes: `local_only`, `local_first` (the default),
`cloud_first` or `cloud_only`. Chargeback prices put a cost on local use in the same micro-dollar units as provider
prices, so budgets can cover both.

## Deployments

A deployment runs one model on a pool of nodes. Its engine spec is typed; nodes build the engine's command line from these
fields only, so there is no way to pass an arbitrary flag.

| Spec field | Engine flag | Bounds |
| --- | --- | --- |
| `model` | the model argument to `serve` | An artifact, below |
| `drafter` | `--drafter` | An artifact |
| `backend` | `--backend` | `metal` (engine `mlx`), `cuda`, `hip` (engine `rocm`) or `xpu` (no flag); empty lets the engine choose |
| `context` | `--context` | 0 to 4,194,304 |
| `parallel` | `--parallel` | 0 to 128; also the replica's slot count |
| `kv_cache` | `--kv-cache` | `bf16` or `fp8` |
| `reasoning_effort` | `--reasoning-effort` | `low`, `medium`, `high`, `xhigh` |
| `thinking_budget` | `--thinking-budget` | 0 to 1,048,576 |
| `max_tokens` | `--max-tokens` | 0 to 1,048,576 |
| `prompt_cache_gib` | `--prompt-cache-gib` | 0 to 4,096 |
| `prompt_cache` | `--prompt-cache-gib 0` when `off` | `shared` (the default) or `off`, which keeps no prompt states; see [prompt caches](#prompt-caches) |
| `vision` | `--vision` | boolean |
| `no_drafts` | `--no-drafts` | boolean |
| `memory_gib` | none | Memory the replica needs, split evenly over its nodes: node memory on Macs and GB10s, and an even part of each node's share per GPU on nodes with dedicated GPU memory; 0 skips the check |
| `gpus` | none | 0 to 16; GPUs one replica holds on each node with dedicated GPU memory (0 means 1) |
| `nodes` | none | 0 to 16; nodes one replica spans (0 means 1); above 1 needs `tensor_parallel` |
| `tensor_parallel` | rank variables or flags | `glm`, `nemotron`, `minimax` or `deepseek` on CUDA, `rocm` on AMD: one engine rank per GPU, rank 0 serving; needs 2 or more ranks (`gpus` times `nodes`), and `rocm` runs 2, 4 or 8 |

Every engine also gets `--host 127.0.0.1 --port 0 --name <model> --api-key-file <per-replica file> --no-update-check`.

Deployment settings:

| Setting | Meaning |
| --- | --- |
| `pool` | Nodes in this pool are candidates |
| `selector` | Labels a node must carry, for example `rack: a` |
| `mode` | `warm`, `on_demand` or `batch` |
| `min_replicas`, `max_replicas` | Bounds on running replicas |
| `idle_seconds` | Residency window before an on-demand or batch replica may stop, at least 60 |
| `status` | `active`, `paused` or `retired` |

Changing the engine spec creates a new revision; changing placement or scaling does not.

### Modes

| Mode | Behaviour |
| --- | --- |
| `warm` | Keeps at least `max(min_replicas, 1)` replicas running |
| `on_demand` | Runs `min_replicas` (often 0); starts a replica when requests for the model are queued, adds one when the queue is more than twice the ready replicas and all are ready, and stops one after `idle_seconds` without requests and at least `idle_seconds` after it became ready |
| `batch` | As `on_demand`; give it a long `idle_seconds` and send its traffic at batch priority |

### Placement

The controller runs every 5 seconds, and at once when a request queues for a model. A node is a candidate when it is
active, has synced in the last 90 seconds, is in the deployment's pool, carries every selector label, matches the backend
and has not had a replica of this deployment fail in the last 10 minutes. Then the node's kind decides:

- **Dedicated GPU memory** (a B300, H200, RTX or AMD Instinct box: CUDA or ROCm GPUs that report their own memory). A
  replica holds whole GPUs: `gpus` of them, each untaken and able to hold its part of the memory. Several replicas, of
  one deployment or many, share a node on separate GPUs, so an eight-GPU B300 runs up to eight one-GPU replicas.
- **Unified memory** (Macs, GB10s, AMD APUs): a replica needs its memory free after other replicas' reservations, takes
  no GPU of its own, and a node runs at most one replica of each deployment.

A replica with `nodes` above 1 takes that many distinct nodes, each passing these checks for its share.

Among candidates, a node not yet running this deployment wins (replicas spread before they double up), then a node that
already holds the artifact, then dedicated GPUs over unified memory. Ties go to the best fit on GPU nodes, which keeps
whole nodes free for wide models, and to the most free memory elsewhere.

### GPUs

A node reports its GPUs, from `nvidia-smi` on NVIDIA and `rocminfo` on AMD: physical index, UUID, name and memory. To
keep GPUs back from Starlight, start the agent with a device list (`CUDA_VISIBLE_DEVICES`, or `HIP_VISIBLE_DEVICES` or
`ROCR_VISIBLE_DEVICES` on AMD) in its environment or as `--env`; the node reports only the GPUs that list leaves
visible, read with the driver's own rules. The scheduler records each replica's GPUs by UUID (by physical index where a
driver reports no UUID), and the node starts the engine with exactly those GPUs visible: `CUDA_VISIBLE_DEVICES` (with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`) and `HIP_VISIBLE_DEVICES` both name them, overriding any list the agent itself was
started with, so the pin holds on NVIDIA and AMD alike. Engines never share a GPU by accident, and a GPU
counts as taken until the replica that held it has stopped. An AMD APU's integrated GPU has no memory of its own and is
never assigned. If `nvidia-smi` or `rocminfo` fails, the node keeps reporting the GPUs it last saw, so one failed probe
never makes a GPU server look like unified memory.

### Tensor parallel

A `tensor_parallel` replica runs one engine process per rank, each on its own GPU, and rank 0 serves. With `nodes` above
1 the replica spans that many nodes, `gpus` ranks on each: two GB10s over RoCE are `nodes: 2`, and two eight-GPU B300s
are `gpus: 8, nodes: 2`. The first node chosen leads: it runs rank 0 and serves, and the other members run the
following ranks and serve nothing.

| `tensor_parallel` | Engine family | Backend | How each rank finds rank 0 |
| --- | --- | --- | --- |
| `glm` | GLM | `cuda` | `TF_GLM_RANK`, `TF_GLM_WORLD` and `NCCL_COMM_ID` |
| `nemotron` | Nemotron | `cuda` | `TF_NEMOTRON_RANK`, `TF_NEMOTRON_WORLD` and `NCCL_COMM_ID` |
| `minimax` | MiniMax | `cuda` | `TF_MINIMAX_RANK`, `TF_MINIMAX_WORLD` and `NCCL_COMM_ID` |
| `deepseek` | DeepSeek | `cuda` | `TF_TP_RANK`, `TF_TP_WORLD`, `TF_TP_MASTER` and `TF_TP_PORT` |
| `rocm` | Families the ROCm engine serves | `hip` | `--tp`, `--rank`, `--master` and `--master-port`, with `--device` naming the rank's GPU |

Ranks meet on the lead. On one node they use a free loopback port. Across nodes they dial the lead's `--rank-host`
address (by default the host it advertises) on a block of 10 ports: 29500 for its first group, 29510 for the next, up
to 100 groups a node. Open TCP 29500 to 30499 on each node to the other nodes of its pool, and give the pool a private
network for the engine's own rank traffic (NCCL sockets or RoCE), which Starlight neither carries nor encrypts.

Followers start first. If any rank exits, rank 0 cannot serve without it, so the group stops and each node restarts
its ranks within the crash budget until they meet again; each rank logs to its own file (`engine.rank1.log` and on). A
member that stays `failed` for 30 seconds, or whose node stops syncing for 90 seconds or is disabled or revoked, stops
the whole replica, and the scheduler places a new one, avoiding nodes where it failed in the last 10 minutes.

### Prompt caches

An engine keeps recent prompt states so that a request repeating a prefix starts faster. Timing can then reveal to one
caller whether another recently sent that prefix. A project with `isolate_prompt_cache` set (in the console or with
`starlight apply`) is served only by deployments that keep no prompt states (`prompt_cache: off`) or whose model no other
project is granted; when no such deployment exists, its requests answer `503 no_route` rather than share. Routes to
providers are unaffected. Granting such a model to another project, or withdrawing it from its isolated project,
restarts its replicas that keep prompt states, so no prefix crosses over. Turning isolation on protects requests from
then on; to clear prompt states already cached, restart the deployments it now avoids.

### Rollouts and failures

- Each pass makes at most one start and one stop per deployment, so changes roll gradually.
- A new revision starts surge-first: a replica of the new revision must be ready before an old one is stopped. When no
  node has room for the extra replica (every GPU taken), the controller stops one old replica first, waits for it to
  release its GPUs and memory, and stops another only once every new replica is ready. If a replica of the new
  revision fails, no more old replicas are stopped.
- A replica's GPUs, memory and rendezvous ports stay reserved until its process has stopped on every node, so a new
  replica never starts beside one that is still draining.
- A replica that stays `failed` for 2 minutes is stopped and replaced on another node. The timers count from the state
  change, not from the node's latest report.
- When a node stops syncing for 90 seconds its replicas become `lost`, leave routing, and are replaced if the mode needs
  them; lost records are removed after an hour. A group whose lead or member is lost is stopped on every node at once.
- Draining a node stops every replica with a seat on it, including groups it only follows in.
- Replica states are `assigned`, `staging`, `starting`, `ready`, `draining`, `stopped`, `failed` and `lost`.

## Artifacts

| Kind | Fields | Rule |
| --- | --- | --- |
| `hf` | `repo`, `revision` | `owner/name` and a 40-character commit; the node runs `tensorfold pull repo@revision` and serves that snapshot |
| `path` | `path` | Absolute, on the node, containing `config.json` |
| `url` | `url`, `sha256` | HTTPS and a SHA-256 of the archive; the node verifies while downloading and extracts safely |

Offline installations use `path`, internal `url` artifacts, or `hf` through an internal mirror (`starlight-node run
--hf-endpoint`). Nodes report the artifacts they hold so placement can prefer them.

## Nodes

| Status | Meaning |
| --- | --- |
| `pending` | Enrolled, waiting for approval |
| `active` | Eligible for placement and routing |
| `draining` | Its replicas are stopped and replaced elsewhere; nothing new is placed |
| `disabled` | Kept but given no work |
| `revoked` | Certificates revoked; the node is told to stop its engines |

`starlight nodes approve NAME` activates a pending node and also returns a drained node to service; `starlight nodes drain NAME` and
`starlight nodes revoke NAME` do what they say, and the console offers the same actions. Enrolment and node installation are
in [install.md](install.md#nodes) and [nodes.md](nodes.md).

Nodes post live load for each replica (decode and prompt tokens per second, running and waiting requests, KV-cache use)
to `POST /node/v1/stats` every 5 seconds (`starlight-node run --stats-interval`). Starlight keeps the latest sample on the replica
for the console's dashboard.

## Configuration as code

`starlight apply -f FILE` makes the fleet match a file you keep in version control: settings, providers, models with their
aliases, deployments and routes, projects with their members, service accounts, model grants and limits, organisation
limits and organisation roles. [`deploy/examples/fleet.yaml`](../deploy/examples/fleet.yaml) shows every section.

- The whole file applies in one transaction, or nothing does. `--dry-run` prints exactly the changes a real run would
  make and writes nothing, so a CI job can post the plan on a pull request and apply it on merge.
- Parsing is strict: an unknown key, a second document or a server configuration file passed by mistake is refused with
  its line number. Deployment specs are checked with the same rules as the console and the scheduler.
- Apply creates and updates only what the file names; it never deletes. The one exception is `everyone: false` on a
  model, which withdraws its organisation-wide grant.
- Credentials are references, never values: `env:NAME` or `file:/absolute/path`. An inline secret is refused.
- A second apply of the same file changes nothing. Every change is audited with the actor `operator`.
