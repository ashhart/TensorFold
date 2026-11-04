# Flash Next EXL3: CUDA graphs and the device planner

With the expert offload pool, a decode round of Flash Next EXL3 plans its experts on the device
(`flashnext_exl3_dplan.cu`) instead of waiting for the host, and can replay the whole round as a CUDA graph. Acceptance
stays in the core lane engine; the device code proposes and plans.

## Switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `TF_FLASHNEXT_EXL3_GRAPH` | `0` | `0` uses the host planner; `eager` uses the device planner, launching kernel by kernel; `1` uses the same device planner and replays graphs for decode rounds, verify windows, the MTP draft chain and prefill layers. All modes must preserve output tokens. |
| `TF_FLASHNEXT_EXL3_GRAPH_ASYNC` | `1` | `0` instantiates a captured graph on the step's thread; `1` on a worker thread while that step runs eagerly. |

Everything else is fixed in the code: a shape is captured at its second sighting, a device-planned round holds at most
16 rows (the MTP window), and the graph memory model (reservation, headroom, per-graph charges) follows the stream
count. `tools/exl3_serve_tp2.sh` sets `GRAPH=1` and `GRAPH_ASYNC=0`.

## A graph that cannot be built

A capture or an instantiation that fails runs its round eagerly and leaves other graphs usable. The failed owner/shape
backs off (64, 128, 256 graph-cache ticks) and is tried at most three more times. After the fourth failure it stays eager
while its failure entry is retained. This is per key, not a switch that disables graphs for the whole process.

The shared failure table holds at most 64 entries and can evict old entries when full. Owner teardown clears that owner's
entries, so a later engine reusing an address starts without the old failures. Prefill layers and draft chains follow the
same retry and retirement rules.

On two ranks the capture outcome is agreed over the rank wire (`graphAgree`) before either rank launches: a graph is
kept only when both ranks hold one, otherwise both run the round eagerly, so the communicator is never left
half-launched. A communicator error (a lost peer, a timeout) still ends the step, as before.

## State the engine owns

The device planner, the plan active during a round and the MTP head's pool belong to the engine
(`Engine.exl3_dplan`), so two engines in one process share none of them.

Closed streams may retain sequence and draft allocations for reuse. Before either is actually freed, every captured
graph borrowing its allocations is retired, including pending instantiations and prefill graphs. DeltaNet state sets
move between sequences and draft windows, so retirement follows both input and output state identities rather than
only the request that first captured a graph. Graphs for other allocations or CUDA drivers remain cached.

Engine teardown retires its decode, prefill and chain cache entries, releases their graph-admission tickets and removes
their failure records. It leaves other engines' graph executables and retry state intact; cache updates cannot borrow
another engine's key.
