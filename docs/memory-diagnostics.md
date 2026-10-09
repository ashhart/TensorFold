# Runtime memory diagnostics

`GET /memory` and `GET /v1/memory` return a read-only JSON snapshot with `schema_version: 1`.
Both routes require an API key when authentication is enabled, including with `--metrics-open`.
Neither route resets counters; `reset_peak=1` has no effect here. `/health` retains its existing contract.

All sizes are integer **bytes**. Optional objects and fields are omitted when their source is unavailable;
zero means a measured or configured zero. The endpoint publishes no paths, UUIDs, keys, hostnames or prompts.

| Field | Meaning |
| --- | --- |
| `context_window` | Loaded engine's prompt-plus-reply token limit; omitted when unspecified. |
| `context_fitted` | Whether startup memory fitting reduced the context. |
| `process.source` | `darwin_rusage_info_v4`; this scope is currently available on macOS. |
| `process.physical_footprint_bytes` | Kernel physical footprint, including charged Metal buffers. |
| `process.lifetime_peak_physical_footprint_bytes` | Kernel lifetime maximum, including peaks between polls. Never reset by TensorFold. |
| `device.active_bytes`, `device.cache_bytes` | Existing backend allocator counters, where implemented. |
| `device.peak_bytes` | Backend peak since startup or the last `/health?reset_peak=1`. |
| `device.peak_scope` | `since_start_or_health_reset`; distinct from the process lifetime peak. |
| `prompt_cache_plan` | Immutable retained-prefix budget applied at startup, currently reported by Flash Next. |

The cache plan contains `source`, `budget_bytes`, and `explicit_budget`. With OS readings,
`source` is `physical_footprint` and the plan also includes the exact sizing inputs: `ram_bytes`,
`ready_footprint_bytes`, `cap_bytes`, `room_bytes`, and `margin_bytes`. The cap, margin, budget arithmetic,
refusal and override behavior are unchanged. `over_cap` records an applied budget above the computed room,
rather than merely the presence of an override flag.

Without both OS readings, the existing fallback uses an explicit budget (`source: explicit`) or the spare
Metal working set (`source: metal_working_set`). Its OS inputs and unknown `over_cap` remain omitted. A disabled retained-prefix
cache still reports a known `budget_bytes: 0`. Backends without an applied plan omit `prompt_cache_plan`.
This plan is a cache allowance, not a process limit, current cache occupancy, or an admission guarantee.

`/metrics` keeps `tensorfold:process_footprint_bytes` and adds `tensorfold:process_footprint_peak_bytes`
for the same kernel lifetime peak. Both gauges are omitted when the OS reading is unavailable. Existing
device gauges retain their backend scope and reset behavior. Process and device counters overlap on
unified memory: do not add them. RSS, allocator peaks and physical footprint are different measurements.
The Darwin structure follows Apple's `sys/resource.h` `rusage_info_v4` layout; host tests check its size
and both footprint offsets, and exercise the kernel reader on macOS.

Route/auth regression checks use the CPU-only fake engine:

```sh
python3 zig/tests/server/test_memory_http.py zig-out/server/fake_serve zig/tests/server/fixtures
```

For certification, record startup, prefill, decode and cleanup boundaries; sample current footprint and
retain the kernel peak. A lifetime peak includes startup and all earlier workloads, so start a fresh
process for each independent matrix cell. Record context, concurrency and cache state alongside the
memory data. A constrained run on a larger Mac is budget emulation, not a measurement on a smaller Mac.
The API does not predict a smallest compatible RAM class or certify a workload by itself.
