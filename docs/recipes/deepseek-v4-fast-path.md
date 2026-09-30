# Short path to the existing DeepSeek GGUF in TensorFold

Effective2026-09-30 after the operator requested fewer cards and unnecessary
tests removed. This specification supersedes the expanded implementation/test
matrix in deepseek-v4-cuda-plan.md and granular-task-plan.json for the first
serial release. Keep the original documents as optional later reference.

## Target and constraints

Serve the existing0731 IQ2_XXS/Q2_K/Q8_0 mixed GGUF, unchanged, with TensorFold's
CUDA family/HTTP interfaces. One Spark, one request/session, context262144,
no drafting, no new quant, no concurrency, no graphs or prefix-reuse project.
Keep live DeepSeek, Hunyuan and CUDA moderation running during development.
The final deployment command is `make tensorfold-deepseek-3d` in ml-infra;
Codex owns that integration independently. Workers must not edit ml-infra or
run its deployment command while using live DeepSeek for their own inference.

## Prefer the already-working native implementation

The pinned ds4 donor has an actual public engine/session boundary in ds4.h:
engine open/close, session create/free, sync/eval, logits and tokenization.
Reuse that native implementation behind TensorFold's engine contract when
compatible. Build isolated copied/pinned sources or a packaged library/shim;
never rebuild or modify /home/josh/code/ds4 or the installed ds4 binary.
TensorFold must own the HTTP server, request lifecycle and shared sampling
contract. An HTTP proxy or renamed ds4-server is not this deliverable.
Do not rewrite hc/Sinkhorn/routing/attention/cache math or create a second
independent miniature model just to duplicate the donor. Adapt only the gaps
needed to expose the serial engine. If the native public API cannot satisfy a
required interface, record the specific gap and implement the narrow adapter.
Native fatal errors, ABI mismatches and callbacks must fail predictably rather
than crash the caller silently. Preserve mapped packed-weight behavior.

Reuse the GGUF's embedded tokenizer through a native adapter if practical.
Do not require a replacement model download or invent a tokenizer source.
If the selected TensorFold interface needs generated sidecars, derive them
from the validated0731 GGUF/local assets and retain provenance. Any change to
the helper's tokenizer arguments must be handed to Codex's integration owner.

## Minimal verification policy

Keep existing passing parser/schema/tokenizer tests; do not expand them into
another exhaustive suite. Do not delete behavioral assertions merely to mask a
real bug. The quant fixture repair corrected the Q2_K physical layout and
removed contradictory duplicated checks; it retains all16 stored vectors.

For new work, add a failing test only for a new failure-prone boundary or an
actual regression. Combine cases in a small parameterized check where useful.
No additional test-only cards, coverage quota, repeated audit cards, huge sign/
codebook enumeration, every-bit mutation campaign, full NumPy model, DSpark
rollback matrix or optional-feature qualification in this release.

Required evidence is narrow:
- Existing malformed GGUF/schema/tokenizer tests continue to pass.
- Quant adapter uses donor-correct layouts; the four Q8/Q2 stored-vector tests
  and a representative IQ2_XXS donor comparison protect the formats actually used.
- Native interface rejection/lifetime tests cover no-allocation bad options,
  wrong ABI, cleanup, cancellation and no callbacks after free.
- Memory admission precedes major allocations, preserves explicit262144 context,
  accounts for companions once and refuses insufficient memory. No full BF16 copy.
- One serial inference contract test covers prompt/decode/EOS/seed/callback flow;
  one HTTP integration check covers chat, streaming and a tool followup through
  the existing server. Use tiny/mock engines until a candidate is admitted.
- Built wheel/console script works from its candidate environment; read-only
  preflight neither installs nor compiles nor starts a service.
- Real GGUF chat/code/tool behavior and actual Hunyuan shape/moderation coexistence
  need measured hardware evidence; fixture/health-only passes cannot replace it.

Run only the tests affected by the changed boundary. Repeat a test to diagnose a
specific remaining failure; otherwise commit and move on. At most one focused
smoke pass over prior relevant boundaries when assembling the release. Record
counts and pending hardware checks in the card result; a separate large evidence
report is unnecessary unless it stores a hardware measurement or artifact pin.

## Execution and completion

Ten implementation milestones replace58 future cards. No separate aggregation
reviews, no optional DSpark/graph/concurrency work. Each card produces one usable
increment with its necessary tests, short completion result and local commit.
Use existing results and avoid re-reading a source twice unless it changed.
120 tool turns and a two-hour run cap remain; aim below30 tool iterations.
Use the Hermes Python with pytest, not system python3 if pytest is absent:
`/home/josh/.hermes/hermes-agent/venv/bin/python -m pytest <affected tests> -q`.

Tiny CUDA checks are unattended only after measured admission: <=256MiB tensor
allocation plus separately accounted runtime/driver overhead, one worker and
>=8GiB remaining system headroom. Bound CPU/build work to the existing worker
limits and <=4 jobs. Full candidate loads on loopback18000 are attempted only
if live DeepSeek+Hunyuan+moderation and actual staging/cache footprint fit with
headroom. Do not stop live DeepSeek to make development fit.

If a second full model cannot fit, deliver the ready build, measured capacity
refusal and final deployment smoke harness. Label full-model qualification
pending, never passed. The independent one-command launcher can perform the
explicit final backend replacement when the user runs it; workers do not run
that cutover. No human approval holds on routine work or resource evaluation.
A real missing dependency, runtime defect or capacity limit is still reported.
