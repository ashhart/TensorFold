# Exact Fluid Subgraph Residency

The [FSR design](../design/fluid-subgraph-residency.md) defines TensorFold's exact native expert-residency contract. This is an implementation of published techniques from EdgeMoE, MoE-Infinity, Fiddler and Fate. Slotstream supplies an outside comparison only. No native throughput or memory tier is qualified by a host test.

## Analyze routing locality

The development tool consumes complete JSONL routing traces with a `tensorfold-fsr-v1` header, checkpoint and workload identifiers, model layer/expert counts, physical expert-slot bytes and complete layer lists for each execution mode. Each `begin` has an ordered `route` for every declared layer and an `end`; `closed` records the completed window count. Every actual route is needed, including hits. The tool refuses missing layers, missing closes and forecasts presented as actual routes. This trace format is an analysis interface; it does not claim that a serving build already records it.

```sh
python3 tools/fsr_profile.py routing.jsonl --budget-bytes 17179869184
```

Repeat `--budget-bytes` to compare budgets. `--ssd-bytes-per-second` accepts measured uncached bandwidth and adds a storage-time lower bound. The output reports union growth, usable slots, unused byte tails and protected-demand CLOCK misses. A fits verdict describes the observed trace; it is not a prediction of future routes or a speed receipt.

## Host foundations

`zig build test-fsr-host test-fsr-source-host` checks cache ownership, bounded slab reads, asynchronous forecast publication, mandatory cache bypass on macOS, PLE row staging and the recorded expert address substitution. These steps do not compile Metal sources or execute a GPU. Run `python3 -m unittest discover -s tools -p test_fsr_profile.py` for the profiler controls.

The source substitution is an address-only preparation over the recorded expert kernels. It does not install a serving path or qualify the device. Native CLI integration, whole-path admission and the synthetic/real checkpoint device receipts are required before advertising streamed inference. Preserve the full [qualification contract](../design/fluid-subgraph-residency.md#qualification), including every logit and complete token sequence.
