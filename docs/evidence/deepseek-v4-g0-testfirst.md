# G0b — Serial rejection test API and pending status (test-first evidence)

Gate: G0b (part of G0). Scope: pin the future CUDA port's serial-rejection test
API and its pending (RED) status. No CUDA engine is implemented here.

Status: RED / pending — intentional, tracked. The port lands at T16.

## Contract pinned

The future `deepseek_v4.cuda_engine` factory must conform to
`docs/recipes/adding-a-cuda-family.md`:

    def cuda_engine(model_dir, *, drafter="", tp=1, rank=0, master="",
                    master_port=29551, no_drafts=False, mtp_drafts=None,
                    **options):

and must reject, at factory-call time (before any engine is built or any weight
loaded — the no-load sentinel):

| config | rejection |
| --- | --- |
| tp != 1 | ValueError (`tp.*1` / `one GPU` / `serial`) |
| --parallel > 1 | ValueError (`parallel` / `concurrent` / `one request`) |
| incompatible drafter | ValueError (`drafter` / `draft` / `no drafting`) |
| unknown storage format | ValueError (`format` / `storage` / `unsupported`) |

The unknown-format case uses `format="onnx"` — a genuinely unsupported storage.
`gguf` is the intended supported storage for this family and is deliberately
NOT used as the negative case (a supported format must be accepted, never
rejected).

## Test-first pins (tests/test_deepseek_v4_g0_discovery.py)

RED (xfail-strict, reason "pending deepseek_v4.cuda_engine serial contract (T16)"):
- test_serial_engine_rejects_tp_neq_1_before_load
- test_serial_engine_rejects_parallel_gt_1_before_load
- test_serial_engine_rejects_incompatible_drafter_before_load
- test_serial_engine_rejects_unknown_format_before_load
- test_serial_engine_signature_pins_cuda_family_contract

GREEN (unchanged, must stay green):
- test_importing_family_discovery_pulls_no_torch_or_mlx
- test_mlx_selection_remains_valid
- test_unsupported_backend_rejected_before_load

## Why the RED tests are not qualification

These tests xfail today because `deepseek_v4.cuda_engine` does not exist. That
bounded expected failure is NOT evidence the future engine qualifies. Each RED
test asserts the documented contract itself (signature defaults, keyword-only
shape, **options catch-all, and the four no-load rejections); when the factory
lands with the documented signature, each must flip to GREEN and its xfail
mark removed. An XPASS-strict is the signal the contract is satisfied and the
mark is stale.

## Exit criteria (RED -> GREEN)

- `deepseek_v4.cuda_engine` exists with the adding-a-cuda-family.md signature
  (keyword-only `tp=1`, `drafter=""`, `rank=0`, `master=""`, `master_port=29551`,
  `no_drafts=False`, `mtp_drafts=None`, plus `**options`).
- `tp=2`, `parallel=8`, `drafter="dflash"`, `format="onnx"` each raise ValueError
  before any load; no engine is returned for a non-serial config.
- `format="gguf"` is accepted (supported storage), so it is never the negative case.
- The 5 RED tests pass with their xfail marks removed.
