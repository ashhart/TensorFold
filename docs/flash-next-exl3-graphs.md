# Flash Next EXL3: CUDA graphs and the device planner

With the expert offload pool, a decode round of Flash Next EXL3 plans its experts on the device
(`flashnext_exl3_dplan.cu`) instead of waiting for the host, and can replay the whole round as a CUDA graph. Acceptance
stays in the core lane engine; the device code proposes and plans.

## Switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `TF_FLASHNEXT_EXL3_GRAPH` | `0` | `1` plans decode rounds on the device and replays graphs for decode rounds, verify windows, the MTP draft chain and prefill layers. Output is identical to `0`. |
| `TF_FLASHNEXT_EXL3_DPLAN` | `0` | `1` plans on the device without graphs (each planned round launches kernel by kernel). Implied by `GRAPH=1`. This is the eager arm of an eager-vs-graph comparison. |
| `TF_FLASHNEXT_EXL3_GRAPH_ASYNC` | `1` | `0` instantiates a captured graph on the step's thread; `1` on a worker thread while that step runs eagerly. |

Everything else is fixed in the code: a shape is captured at its second sighting, a device-planned round holds at most
16 rows (the MTP window), and the graph memory model (reservation, headroom, per-graph charges) follows the stream
count. `tools/exl3_serve_tp2.sh` sets `GRAPH=1` and `GRAPH_ASYNC=0`.

## A graph that cannot be built

A capture or an instantiation that fails runs its round eagerly and the process keeps its other graphs. The failed
shape backs off (64, 128, 256 rounds) and is tried at most three more times before it stays eager. The prefill layer
shape and the draft chain do the same.

On two ranks the capture outcome is agreed over the rank wire (`graphAgree`) before either rank launches: a graph is
kept only when both ranks hold one, otherwise both run the round eagerly, so the communicator is never left
half-launched. A communicator error (a lost peer, a timeout) still ends the step, as before.

## State the engine owns

The device planner, the plan active during a round and the MTP head's pool belong to the engine
(`Engine.exl3_dplan`), so two engines in one process share none of them.
