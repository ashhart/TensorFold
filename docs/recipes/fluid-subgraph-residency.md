# Exact Fluid Subgraph Residency

[Fluid Subgraph Residency](../design/fluid-subgraph-residency.md) is TensorFold's implementation of published techniques from EdgeMoE, MoE-Infinity, Fiddler and Fate. Slotstream is an outside comparison only. The native phase covers Flash Next (`qwen4_exp`) on one Metal device. Other families, backends and the paired speed-up path refuse these options.

## Serve stored experts from SSD

```sh
tensorfold serve CHECKPOINT --ssd-experts 16 --ple-on-ssd
```

`--ssd-experts GIB` is the routed-expert slot budget, not the whole process budget. Shared experts, non-routed weights, the original execution scratch, admitted sessions, prompt states and the runtime reserve still need memory. The startup plan includes them, all nine stored components per slot, the persistent expert map, source metadata and the bounded reader stack. It reports usable slots and unused budget bytes. A budget must hold one layer's maximum expert union. A shortfall refuses; it never changes top-k, context, drafts or concurrency to fit.

The loader leaves routed target and MTP experts in their original checkpoint files. Demand protects all selected experts before CLOCK eviction. All nine components load into an exclusive slot before its map entry is published. The existing expert kernels use the persistent map only on this path. Mixed stored widths retain their original formulas and combine order. Missing or malformed spans, short reads, changed file length and a failed cache-bypass setup stop the request before partial data can be used.

Readers require macOS `F_NOCACHE` and use positional reads directly into shared slots. Native PLE streaming can be selected independently with `--ple-on-ssd`. It deduplicates requested rows, coalesces adjacent rows within each shard and feeds staged IDs and unchanged stored bytes through the resident lookup arithmetic. Logical n-gram IDs and history stay unchanged.

Decode prefetch uses the next layer's own router on the current state. One bounded worker reads predictions into unleased slots behind submitted expert compute. The later real router remains authoritative. A wrong prediction wastes reads and cannot change routes or output. Prompt chunks load each layer's complete demand union once per chunk. Streaming chunks use one command stream so host storage cuts cannot wait on an unsubmitted staggered sibling; concurrent request windows and MTP still use the lane core.

## Bound read overhead

The demand path deduplicates route IDs and reserves missing experts in logical order. Adjacent stored experts merge only when both the source bytes and their destination slots are contiguous. A request is at most 4 MiB. The reader queues at most nine requests through macOS asynchronous IO, drains every submitted request on errors, and checks each touched file's length before and after the batch. Queue pressure falls back to positional reads after submitted work drains. Cache bypass remains required.

An expert has nine stored components in separate tensor ranges. They cannot become one contiguous read without changing the checkpoint layout. Dense prompt unions can merge neighboring experts within each component; sparse decode routes often still need nine requests per missing expert. Mixed-width slot padding prevents merging across that padding. No gap bytes or unselected expert payloads are read to manufacture larger requests.

A private pending-read index avoids scanning the pool for each route ID. Forecasts consume only cold, unleased slots and do not clear actual demand's second-chance bits. An unused prediction remains cold. When actual demand arrives, the worker finishes its current expert, skips unstarted unrelated predictions and finishes predictions the actual union needs. The actual router and complete-demand lease remain authoritative. Shutdown reports lifetime bytes, read requests, file checks, reservations, canceled forecasts, raw and unique demand IDs, worker joins and command waits, including startup.

Router readback and storage ownership still require command-buffer boundaries. These changes reduce unnecessary forecast splits when no cold slot exists; they do not remove the real router's dependency or claim resident-class throughput at every budget. The worker remains bounded and is joined before cache publication or teardown.

## Analyze routing locality

```sh
tensorfold serve CHECKPOINT --fsr-route-trace routing.jsonl
python3 tools/fsr_profile.py routing.jsonl --budget-bytes 17179869184
```

Stop the server cleanly before analyzing the trace. The native writer excludes startup warm-up and calibration. It records every actual expert demand, including hits, and distinguishes prompt chunks, solo decode, shared batches and MTP. Rows and stream counts distinguish plain and drafted windows. A kernel that executes an encoded upper bound includes those rows; a gated kernel that skips them excludes them. Forecasts are never actual-route observations. Checkpoint identity is a hash; prompt text is not written.

The `tensorfold-fsr-v1` format has a header, ordered complete layer observations for every window, window ends and a final close/count. The analyzer refuses missing layers, reordered or duplicated observations, forecasts and truncated traces. Repeat `--budget-bytes` to compare pool budgets. `--ssd-bytes-per-second` accepts measured uncached bandwidth and reports a storage-time lower bound. The report includes union growth, usable slots, unused tails and a protected-demand CLOCK simulation without forecast. A fit verdict applies to the observed workload and predicts neither future locality nor native speed.

For an explicitly synthetic planning example at the real 49-layer, 512-expert, top-10 geometry:

```sh
python3 tools/fsr_synthetic.py --locality hot80 --windows 512 > synthetic-routes.jsonl
python3 tools/fsr_profile.py synthetic-routes.jsonl \
  --budget-bytes 17179869184 --budget-bytes 23622320128 \
  --budget-bytes 35433480192 --budget-bytes 51539607552 \
  --tokens 512 --resident-tps 123 --ssd-bytes-per-second MEASURED_BYTES_PER_SECOND
```

The generator offers uniform, skewed and sticky route scenarios, preserves all top-10 IDs per row, and includes target and MTP windows. It does not load weights or predict a real model's routing. `--rows` and `--streams` describe batch width; `--prompt` labels prompt windows. One synthetic step includes one MTP row per target row; measured draft acceptance needs a real trace. `--tokens` is explicit because a trace window is not necessarily one emitted token. The synthetic header declares uniform component widths so the analyzer can estimate contiguous request counts. Native traces without those widths report bytes and locality without inventing read geometry. Use lifetime IO counters for the actual layout and forecasts.

Optional `--read-latency-ns` adds an explicitly uncalibrated latency scenario. The reported serialized estimates add resident compute time, mandatory byte time and per-request or per-nine-request-group latency. They are planning estimates, not measured native speed. Actual queue depth, layout fragmentation, forecasts and overlap can change them. Measure uncached bandwidth at the same request sizes on the tested storage before interpreting any throughput ceiling.

## Qualification

Host steps are `test-fsr-host`, `test-fsr-source-host`, `test-fsr-cli-host` and `test-flashnext-fsr-owner-host`. They check ownership, source transformations, actual native option forwarding, admission, complete multi-row demand, forecast joins, PLE staging and file failures. Run `python3 -m unittest discover -s tools -p test_fsr_profile.py` for the locality tool's controls. `compile-fsr-metal-gate-host` and `compile-fsr-metal-contract-host` compile the real gate bodies without executing a shader compiler or device.

Device qualification requires `zig build test-fsr-metal` on a Mac with the Metal toolchain and GPU. Its nonzero two-layer MoE has 32 experts per layer and four slots. It compares every observed node and all 32 logits with resident execution at four and six bits, repeats five times with uncached reads and requires a wrong-map control to differ. The Metal kernel contract includes exact stored-view replacements, wrong-map controls and the serving variants' source-reserved launch bounds. The address-view contract does not claim to requalify the unchanged resident accumulation precision.

Synthetic and real-device results remain required before advertising a qualified memory tier. Real checkpoint receipts must compare complete token arrays for cold/warm prefill, serial decode, drafted/plain, concurrent/solo, resumed/fresh, prompt cuts and cancellation/reset at 16, 22, 33 and 48 GiB pool budgets. Record actual RAM, physical footprint, device allocations, SSD traffic and timings with cache bypass. A budget on a larger-memory device does not prove operation on a smaller-memory device. Follow the complete [qualification contract](../design/fluid-subgraph-residency.md#qualification).
