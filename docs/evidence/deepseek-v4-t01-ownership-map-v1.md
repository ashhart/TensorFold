# DeepSeek-V4-Flash-0731 — Agreed File Ownership Map (v1)

Agreed ownership for the serial CUDA implementation, consolidating the plan doc's
"Reuse map and remaining work" section (`docs/recipes/deepseek-v4-cuda-plan.md`)
against the actual tree. Lanes: **audit/G0** (evidence + contract + test pins),
**TensorFold reference** (existing MLX family), **serial-implementation** (the CUDA port),
**consolidation** (this T01 manifest + ownership map).

## Anchored files (committed, in scope)

| Path | Owner lane | Status | Serial-implementation responsibility |
| --- | --- | --- | --- |
| `src/tensorfold/families/deepseek_v4/*.py` (config, weights, compressor, moe, dense, mtp, runtime, prompts, convert + `vendor/encoding_dsv4.py`) | TensorFold reference | committed | Source of truth for decomposition math; adapt/verify against the serial engine (D1/D4). MTP/drafting files stay reference-only for R3 (D5). |
| `tests/test_deepseek_v4_g0_discovery.py` | audit/G0 | committed (909221c) | Serial-rejection API pins (tp=1, drafter='', rank=0, master='', master_port=29551, no_drafts=False, mtp_drafts=None, **options); 4 no-load rejection pins; RED->GREEN gate for the engine signature. |
| `tests/test_deepseek_v4_donor_pins.py` | audit/G0 | committed (ebdbd50) | Donor revision + license pins; drift fails RED (U4 notice record). |
| `tests/test_deepseek_v4_unsupported_options.py` | audit/G0 | committed (ebdbd50) | Unsupported-option pins for the 0731 GGUF (U2 reader decision). |
| `docs/evidence/deepseek-v4-t01-evidence.json/.yaml` | audit/G0 | committed (ebdbd50) | Upstream audit: sources/revisions/licenses, unsupported options, U1-U4. |
| `docs/evidence/deepseek-v4-t01-arch-contract-v1.json/.md` | audit/G0 | committed (5d22c7a) | D1-D5 architecture contract + gates R3/R6/R7/R8/G11. |
| `docs/evidence/deepseek-v4-g0-testfirst.md` | audit/G0 | committed (a4edf63) | Test-first plan + RED->GREEN exit criteria. |
| `docs/evidence/deepseek-v4-t01-decision-manifest-v1.json/.md` | consolidation | this card | Single versioned decision/evidence manifest (merges audit + contract + G0 plan). |
| `docs/evidence/deepseek-v4-t01-ownership-map-v1.md` | consolidation | this card | This ownership map. |
| `docs/recipes/adding-a-cuda-family.md` | serial-implementation | committed | Family registration + engine signature contract (quoted by g0-testfirst.md). |
| `docs/recipes/deepseek-v4-cuda-plan.md` | serial-implementation | committed | Build/run plan + reuse map + ownership source. |

## Planned files (serial CUDA implementation, to be created)

| Path | Owner lane | Status | Serial-implementation responsibility |
| --- | --- | --- | --- |
| `src/tensorfold/families/deepseek_v4/cuda_engine` (the serial engine; exact module path per `adding-a-cuda-family.md` signature) | serial-implementation | planned | Implement donor serial eager per-layer loop (D5), slab KV (D3), attention kernels (D4). Gated by G0 RED->GREEN. |
| `tests/cuda/` (deepseek_v4 serial CUDA tests; new files under the existing harness) | serial-implementation | planned | Fixture-oracle exactness (R6), window/compress/indexer equality (G11), T16/T19/VMM evidence. |
| `tools/build_deepseek_v4_cuda.py` | serial-implementation | planned | Build + run the serial engine on GB10; record R7 build/reproducibility. |
| donor C kernel imports (per U3/U4 resolution) | serial-implementation | pending | Pick cuda/mmq vs gguf-tools/quants (U3); record MIT/ggml notices before merge (U4). |

## Ownership rules

1. The **TensorFold reference** lane owns the decomposition math; the serial engine must match it exactly (D4 fp32 accumulation, exact masking).
2. The **audit/G0** lane owns the pins and contracts; any drift in the donor/revision/license pins fails RED via `test_deepseek_v4_donor_pins.py`.
3. The **serial-implementation** lane owns the engine, its CUDA tests, and the build tool. Nothing in this lane merges until U3/U4 are resolved and T16 evidence is recorded.
4. No file in this map may load the full model on the live stack without measured admission (R1/R4 development rule).

## Status gates

G0 SATISFIED (24 passed, 6 intended xfails, 0 failed, 0 skipped on the three anchored suites).
Pending: U1-U4 (unresolved), T16, T19, VMM — see `deepseek-v4-t01-decision-manifest-v1.json`.
