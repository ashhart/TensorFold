# Code proofs: TensorFold

## Proof record — 2026-09-30T17:38:28Z

**Change:** CUDA admission cap, reserve removal, refusal diagnostics, Nemotron workspace estimate, and uv lock.
**Worktree:** `/Users/arivera/repos/TensorFold`
**Head SHA under test:** `1a3b68721bf3892a4f9aff41e6ec697775c359db`
**Pre-change state:** `9cd52ab4daba68ddd09be89be8f23ad43175e821` (upstream/main, v0.5.0)
**Verdict:** NOT PROVED
**Confidence:** 4/5 for the observed regression; CUDA runtime behavior remains untested.

Changed files: CHANGELOG.md, README.md, RUNBOOK.md, docs/recipes/cuda.md,
src/tensorfold/cuda/capacity.py, src/tensorfold/cuda/geometry.py,
tests/test_cuda_geometry.py, tests/test_cuda_unified_memory.py, uv.lock.

All Python commands used the existing project `.venv`. The old-code run used a temporary
detached worktree at the pre-change SHA; that worktree was removed after testing.

| Proof/check | Command | Exit | Expected | Observed | Verdict |
|---|---|---:|---|---|---|
| Old-code budget regression tests | In base worktree: `python -m pytest -q /Users/arivera/repos/TensorFold/tests/test_cuda_unified_memory.py -o 'pythonpath=/tmp/tensorfold-review.Bw1z78/src /Users/arivera/repos/TensorFold/tests'` | 1 | New behavior absent | 13 failed, 3 passed | PASS: tests distinguish the budget/diagnostic change |
| Patched budget tests | `python -m pytest -q tests/test_cuda_unified_memory.py` | 0 | Pass | 16 passed | PASS |
| Focused admission suite | `python -m pytest -q tests/test_cuda_unified_memory.py tests/test_cuda_geometry.py tests/test_cuda_admission.py tests/test_cuda_nemotron_admission.py tests/test_cuda_capacity.py tests/test_cuda_failed_admission.py tests/test_cuda_parallel_admission.py` | 0 | Pass | 79 passed, 76 skipped | PARTIAL: torch tests skipped |
| Old-code cache bound | Python inline assertion below, importing base worktree source | 0 | Workspace covers cache lower bound | 1.9250 GiB estimate >= 1.7500 GiB caches | PASS |
| Patched cache bound / forbidden underestimation | Same inline assertion on reviewed head | 1 | Workspace covers cache lower bound | 1.6345 GiB estimate < 1.7500 GiB caches | FAIL |
| Full repository suite | `python -m pytest -q` | 0 | Pass | 3159 passed, 749 skipped in 197.03s | PARTIAL: GPU/end-to-end paths skipped |
| Lock consistency | `uv lock --check` | 1 | Current lock | Lock needs updating | FAIL |
| Patch hygiene | `git diff --check upstream/main...HEAD` | 2 | No conflict markers | CHANGELOG.md lines 30, 32, 34 contain leftover markers | FAIL |
| Ruff | `uvx ruff check --config pyproject.toml src/tensorfold/cuda/capacity.py src/tensorfold/cuda/geometry.py tests/test_cuda_geometry.py tests/test_cuda_unified_memory.py` | 1 | No lint errors | Five findings; same five categories/sites on base | Existing debt, not a new blocker |

### Reproduce the cache bound

Run with the project's Python environment and `PYTHONPATH=src` (use the base checkout's
src directory to reproduce the passing pre-change assertion):

```python
from tensorfold.cuda.geometry import hybrid_geometry
from tensorfold.cuda.capacity import GIB

text = {
    "hidden_size": 512, "vocab_size": 1024,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64,
    "mamba_num_heads": 8, "mamba_head_dim": 64, "n_groups": 2,
    "ssm_state_size": 64, "conv_kernel": 4, "n_routed_experts": 8,
    "num_experts_per_tok": 2, "moe_intermediate_size": 128,
    "layers_block_type": ["mamba", "attention", "moe", "mamba", "attention", "moe"],
}
length = 262144
geometry = hybrid_geometry(text, 1, 16, rows=16, chunk=512, drafts=True, draft=0)
one_kv = 2 * length * text["num_key_value_heads"] * text["head_dim"] * 2
required_cache = (5 * text["layers_block_type"].count("attention") + 4) * one_kv
estimate = geometry.bytes_at(length - 16)
print(estimate / GIB, required_cache / GIB)
assert required_cache <= estimate
```

The inputs follow the existing Nemotron allocation test's geometry, extending the window.
Five main cache sets comprise the live engine, lazy serial engine, and three snapshots;
four MTP sets comprise its live caches and snapshots. This is a synthetic allocation
lower bound, not a GPU measurement. The runtime keeps two completed snapshots and can
create the third while finishing a request. The serial engine persists after first use.

No absence of weight loading or dispatch was independently proved on a real CUDA engine.
Invalid environment values are refused in unit tests. Real startup, graphs, prefill,
serial requests, retained snapshots, multi-rank operation, and near-capacity decode
remain runtime proof gaps because PyTorch/CUDA are unavailable in this environment.
No temporary test files were created. No source files were changed.

The budget and diagnostic tests pass on the reviewed SHA; the Nemotron allocation
bound fails on that same SHA. The change as a whole is NOT PROVED.

## Branch review

**Verdict: DO NOT MERGE AS-IS**
**Disposition:** BLOCK WITH REWRITE PATH
**Reviewed head:** `1a3b68721bf3892a4f9aff41e6ec697775c359db`
**Confidence:** 4/5
**Decisive reason:** The reduced Nemotron estimate omits a serial engine that production still allocates.

The cap and better diagnostics are useful PR material. The current bundle also includes
a default memory policy change, an incorrect Nemotron estimate, broken release notes,
and a stale lockfile. Separate the dependency lock from CUDA behavior to simplify review.

### Hard gates

| Gate | Status | Evidence / impact |
|---|---|---|
| No P0/P1 | FAIL | P1 Nemotron cache bound fails on exact head |
| Essential exact-head proof | FAIL | Allocation lower bound fails; CUDA runtime proof unavailable |
| Required CI | UNKNOWN | No open PR or branch runs; branch-protection API returned 404 |
| Mergeability against upstream/main | PASS locally | Current upstream SHA equals merge base and is an ancestor of HEAD |
| Reviewed SHA is published | FAIL | Remote branch points at c0548655014c7abdf488ec5d1d25a347512fd7f9, not reviewed head |
| Patch/lock consistency | FAIL | Conflict markers and stale uv lock |
| Security authority | PASS for scope | No credential, ingress, or authorization changes |

### Old contract and behavior delta

| Caller/component | Operation | Before | After | Shipped / evidence |
|---|---|---|---|---|
| CUDA operators | Startup admission | Free GPU/host memory less max(4 GiB, 10%); unified host reserve | No default reserve; optional absolute GiB cap | Old behavior verified at v0.5.0 in capacity.py; new lines 178–193 |
| Nemotron callers | Drafted then serial requests and retained prefixes | Five main cache sets plus four MTP sets budgeted | Four main sets budgeted, while lazy serial engine remains | geometry.py:339–351; app.py:189–192, 248–249; engine.py:43–44, 90 |
| CUDA operators | Startup refusal | Token capacity only | Local plan's budget also printed | capacity.py:240–255; budget regression tests |

There is no authority change. The first release containing the exact old reserve formula
was not established; v0.5.0 demonstrably contains it. The capacity module originated in
v0.3.5, which alone does not establish when this particular formula shipped.
The new release number is not established. TENSORFOLD_CUDA_RESERVE_GB exists only in
intermediate branch commits in the reviewed change; no stable support was established.

### Full criteria assessment

Scores use 1=absent/contradicted, 2=incomplete, 3=plausible, 4=adequate, 5=direct proof.
A blocker cannot be averaged away. For unaffected security criteria, passing means
no relevant new authority or security behavior was found in this diff.

| # | Criterion | Score | Evidence / threshold | Impact |
|---:|---|---:|---|---|
| 1 | Claimed behavior correct | 1 | Serial twin remains; cache bound fails | BLOCK |
| 2 | Owning components | 4 | CUDA changes in capacity/geometry | Pass |
| 3 | Scope/unrelated behavior | 2 | Dependency lock and default reserve removal bundled | Rewrite |
| 4 | Error/concurrency/cleanup | 2 | Lazy serial allocations omitted; torch paths skipped | BLOCK |
| 5 | Reported final effects | 2 | Workspace estimate can be below actual caches | BLOCK |
| 6 | Authority delta | 5 | No authority changes in complete diff | Pass |
| 7 | Enforcement boundary | 4 | Cap applied in shared admission | Pass |
| 8 | Invalid/failure paths | 4 | Six invalid env values refused | Pass for tested values |
| 9 | Residual authority | 4 | No authority delta | Not applicable |
| 10 | Equivalent prohibited effects | 4 | No authorization prohibition introduced | Not applicable |
| 11 | Escalation/secrets/destruction | 4 | No relevant new operations in diff | Pass |
| 12 | Stable old behavior | 5 | Reserve and geometry inspected at v0.5.0 | Established |
| 13 | Reasonably usable contract | 4 | Automatic admission used by existing engines | Preserve default reliability |
| 14 | Fix versus contract change | 5 | Cap/diagnostics plus default reserve removal | Owner decision warranted |
| 15 | Silent losses | 2 | Default protection removed without opt-in | Rewrite |
| 16 | Compatibility control | 2 | New cap allows manual reduction; old default lost | Rewrite |
| 17 | First affected release | 2 | v0.5.0 old behavior proved; exact first release unknown | Evidence gap |
| 18 | New intent stated | 3 | README clear, CHANGELOG contradictory | Rewrite |
| 19 | Owner approval | 1 | No open PR/maintainer discussion | Policy decision pending |
| 20 | Independent product intent | 2 | Tests demonstrate mechanics only | Evidence gap |
| 21 | Maintainer objections | 3 | No open PR to inspect | Unknown |
| 22 | Old fail/new pass tests | 5 | 13 old failures; all 16 budget tests pass on head | Pass for cap/diagnostics |
| 23 | Allowed/forbidden paths | 2 | Invalid limits covered; allocation bound fails | BLOCK |
| 24 | Absence of dispatch/state mutation | 2 | CUDA refusal tests skipped; no real loading proof | Gap |
| 25 | Operation variants | 2 | Unified/discrete unit paths pass; serial/multi-rank not proved | Gap |
| 26 | End-to-end exact-head proof | 1 | No real CUDA runtime available | BLOCK for memory regression |
| 27 | Post-rebase retest | 4 | Full suite run on current clean head | Pass within hardware limits |
| 28 | Required checks trustworthy | 2 | No PR/check results; local tests available | Unknown CI |
| 29 | Accurate docs | 1 | CHANGELOG markers and contradictory reserve claims | Rewrite |
| 30 | Affected operators identified | 4 | CUDA docs name cap and memory risk | Pass |
| 31 | Replacement workflow | 3 | Manual GiB cap documented; no real startup proof | Partial |
| 32 | Release/version accuracy | 1 | Unlanded commit headings inserted among released sections | Rewrite |
| 33 | Missing compatibility path explicit | 3 | Reserve removal documented, but inconsistent CHANGELOG | Rewrite |
| 34 | Branch mergeability | 4 | Upstream main ancestor; no competing base changes | Locally mergeable |
| 35 | Reviewed landing SHA | 2 | Published branch is different SHA | Publish after fixes |
| 36 | Required checks complete | 2 | Local full suite passes; no required CI established | Unknown |
| 37 | Unresolved review threads | 3 | No PR exists | Not applicable yet |
| 38 | Conflict/generated updates invalidate proof | 2 | CHANGELOG and lock still need edits | Rerun on corrected head |

### Findings and flip conditions

1. **P1: Nemotron workspace omits its serial twin.** geometry.py:339–351 removes
   state/KV allowance, but app.py:189–192 constructs and retains a second full engine
   when draft=False. Restore its live-state/cache accounting or change the engine's
   lifetime so those allocations cannot coexist. Add allocation coverage for a drafted
   engine, serial use, and retained snapshots at 262144 tokens, one and two ranks,
   then verify actual peak allocations on CUDA. The synthetic lower-bound test above
   must pass on the corrected exact head.
2. **P2: Default reserve removal lacks runtime evidence.** capacity.py:186–193 grants
   all available memory without a limit. This affects every CUDA family, including
   operators who never set the new variable. Preserve the old default and make larger
   grants explicit, or obtain a maintainer decision supported by near-capacity startup,
   long-prefill, decode, snapshot, and parallel tests across affected families.
3. **P2: Broken changelog.** Remove the conflict markers at lines 30/32/34, duplicate
   and obsolete intermediate entries, and the contradictory claim that the reserve
   remains. Summarize final behavior in an unreleased section consistent with the
   repository's release headings. `git diff --check upstream/main...HEAD` must pass.
4. **P2: Stale lock.** uv.lock's root package lists test/ssd/vision but omits pyproject's
   grammar extra and xgrammar dependency. Regenerate it until `uv lock --check` passes,
   or remove it from this CUDA PR and address locking separately.

Silent passes: no uncommitted source changes, upstream main current by ls-remote,
budget tests distinguish old/new code, complete available suite passes, memory caps
stay bounded by available memory in tested cases, and existing lint debt is not
promoted to a branch regression. The large lock was inspected for metadata consistency,
not audited package-by-package for provenance. Changelog calibration measurements
were not treated as independent GPU proof. Required CI policy, private owner decisions,
and remote review threads could not be established.

After corrections, publish the corrected SHA and rerun checks on it. No PR was opened,
no commits were made, and no branch was pushed during this review.

**Final maintainer recommendation:** Do not merge as-is; apply the rewrite path above
and re-review the resulting SHA.

## Follow-up verification — 2026-09-30

The preceding review describes the committed head before the corrections below.
The user explicitly chose to retain reserve-free admission and authorized removal of
the lockfile. Reserve removal is therefore intentional for this work, not an outstanding
request for permission. The revised change is suitable for opening a PR for review;
CUDA peak-memory measurements remain unverified.

**Change:** Restore Nemotron's serial-engine cache/state accounting, test the full
31 GiB budget without a reserve, remove uv.lock, and replace broken changelog entries
with an Unreleased section.
**Base HEAD:** `1a3b68721bf3892a4f9aff41e6ec697775c359db`
**State under test:** Uncommitted tracked corrections on that HEAD, not a new commit.
**Tracked diff SHA-256:** `1207d1a0ae732a51789acae88717b4d0fced281106d652e325e81ba1b8c18d6a`
**Verdict:** PARTIALLY PROVED (CPU regressions proved; real CUDA runtime unavailable).

| Check | Command | Exit | Observed |
|---|---|---:|---|
| Regression before geometry correction | `python -m pytest -q tests/test_cuda_nemotron_admission.py tests/test_cuda_unified_memory.py -k 'long_window or 31_gib'` | 1 | Four cache/state cases fail; two 31 GiB budget cases pass |
| Same regression after correction | Same command | 0 | All six cases pass |
| Admission and changelog coverage | Prior focused admission command plus `tests/test_update.py` | 0 | 101 passed, 76 skipped |
| Full corrected suite | `python -m pytest -q` | 0 | 3165 passed, 749 skipped in 197.12s |
| Syntax/undefined-name lint | `uvx ruff check --config pyproject.toml --select E4,E7,E9,F --output-format concise src/tensorfold/cuda/capacity.py src/tensorfold/cuda/geometry.py tests/test_cuda_geometry.py tests/test_cuda_nemotron_admission.py tests/test_cuda_unified_memory.py` | 0 | All checks passed |
| Patch hygiene | `git diff --check upstream/main` | 0 | No errors |

The new tests cover one and two ranks, drafting enabled and disabled, and all
snapshot state fields. The budget tests prove that 31 GiB is granted on discrete
and unified devices with sufficient available memory. This is an admission budget,
not a requirement to allocate exactly 31 GiB or a hard runtime allocator limit.
Existing broader lint findings remain; no new findings were introduced by these
corrections. The dependency lock no longer appears in the diff against upstream/main.
No commits, pushes, or PR creation were performed.
