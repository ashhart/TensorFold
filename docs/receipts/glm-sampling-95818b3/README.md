# GLM sampled decoding: qualification receipt

Tested implementation: `95818b3276d314e6a1597727629982ff8a0b71f7`.
Implementation tree: `dba292097c3d46ef5df5c3e078882374cd2149b2`.
Base: `c7137bb31a7dad924fb541a7068422127b9a94b2`.
Test date: 2026-10-15. Platform: Apple M3 Ultra, 256 GB RAM (owner-provided capacity), macOS 26.6.2 / 25G83.
Compiler requirement: Zig 0.17.0; an independent compiler-version capture was not included in the uploaded archive.
Checkpoint: GLM-5.3-Flash MLX 4-bit MTP, 45/45 layers and MTP loaded, 180.58 GB in buffers.
Checkpoint revision: not captured. Standalone load limit: GLM_LOAD_LIMIT_GB=235.

## Scope

This is a feature-correctness receipt, not a performance comparison. No throughput improvement or absence of
performance regression is claimed. Raw tools retain their measured timing fields as evidence, without using them
as a qualification condition. The receipt-only commit does not alter the tested implementation.

## Results

| Check | Result |
| --- | --- |
| macOS host/golden build | 164/164 steps succeeded; 2000/2008 tests passed, eight skipped |
| Frozen goldens | 309/319 equal; ten known differences, zero unexpected differences |
| Synthetic shared Metal sampler | 15/15 passed with Metal API validation enabled |
| Python served-gate client tests | 9/9 passed |
| Served plain/restored/drafted | Four cases, temperatures 0, 1, 0.7, 1, all native token hashes match |
| Restoration | Cold cached_tokens=0; restored cached_tokens=11318 in every case |
| Actual drafting | Drafted counts 68, 64, 30, 61; none silently ran plain |
| Mixed-mode concurrency | Each served-gate case matches its solo baseline in both request orders |
| Additional concurrent benchmark | Eight cells: code/chat, temperatures 0/1, one/four streams; zero failed requests or mismatches |
| Standalone sampled and greedy | Two prompts, depths 0/3, two repetitions per depth: all 16 runs agree within each mode/prompt |
| Standalone cancellation recovery | Four checks passed; next fresh reply equals the baseline |

The served JSON and concurrent JSON were independently rechecked while packaging; passing log text alone was not
used to establish token-hash equality, cache restoration or positive drafting telemetry.
The sampled hashes differ from greedy hashes, establishing that sampled requests did not all fall back to argmax.

## Reproduction

The exact public standalone token arrays are in standalone-prompts.json. Exact served request bodies, settings,
responses, native token hashes and speculative/cache telemetry are in served-sampling.json. Concurrent settings
and results are in concurrent.json. The existing docs/recipes/glm-sampling-target-rows.md describes the commands.

Standalone environment: GLM_SAMPLING=12,0.7,40,0.9,0.05 (sampled) or 12,0,0,1,0 (greedy), GLM_DEPTHS=0,3,
GLM_RUNS=2, GLM_MAX=64, GLM_CANCEL_TEST=1, GLM_LOAD_LIMIT_GB=235.

build.log and format.log are empty, consistent with quiet successful commands, but do not independently capture
exit status. Native model execution is evidenced by the served/standalone results. The host/golden log includes
its explicit successful build summary. Warning diagnostics containing 'failed command' are preserved; they do not
replace or contradict that final summary.

## Remaining coverage limits

Served streaming disconnect/cancellation recovery was not captured. Standalone cancellation recovery passed.
MTP-omitted model execution and two-Mac greedy/refusal behavior were not run. Sampled two-Mac execution remains
unsupported. These are disclosed coverage limits, not claims that every platform/configuration was qualified.
No before/after performance or peak physical/device-memory receipt is supplied. There is no performance claim.

## Publication handling

This is a separate sanitized copy. User-specific checkout/model paths were replaced with placeholders; raw
result values and token hashes were preserved. ZIP resource forks and filesystem metadata are excluded.
The original upload was not modified. SHA256SUMS pins every published receipt file except itself.
