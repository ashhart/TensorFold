# A fresh gfx1151 run with all three #1001 branches

The run offered on [#1001](https://github.com/ashhart/TensorFold/issues/1001): GPU, CPU and checkpoint identity and
each request's drafted and accepted tokens, measured after the fixes in #1043, #1044 and #1045, with all three merged
into one binary. The 18 October receipts in `../out/` and `../out-r5b/` were made with earlier heads of these
branches, before the visible-device commits `82237515` and `fb8eaf85` (#1043) and the last fixes of #1044 and #1045.

## Where and when

A Bosgame M5, 19 October 2026, 17:31 to 17:47 UTC (the `t=` stamps in `run.log`). The receipts record the CPU
(`AMD RYZEN AI MAX+ 395 with Radeon 8060S`), one GPU (`AMD Radeon 8060S Graphics`), Linux and the `rocm` backend; the
binary was built for `gfx1151` (`prep.sh`). The checkpoints are the 18 October ones, offline:
`TensorFold/Qwen3.8-27B-MLX-4bit` at `70ae7fac63274ff2eac54152031433374cb80f2f` and
`TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` at `f84b054c5677b6e59bdb969ae61915b0983d68bb`.

Source (`heads.txt`, `prep.log`): main `76aafcf2` with `benchmark-rocm-host-identity` at `c949cc98` (#1043),
`hip-refuses-drafter` at `f2c78eeb` (#1044) and `benchmark-drafter-counts` at `134a5f43` (#1045), merged on the box
(`7f4db1cf`, not published; git merged the three without a conflict) and built with `-Doptimize=ReleaseFast`
(`binary.sha256`). `prep.sh` merged and built; `run.sh` ran R1 to R6 on the box. `match.py` and `check.mjs` were run
afterwards on a Mac.

## The runs

Each benchmark ran 20 attempts, all `ok`; the counts are summed over the 20 and every attempt reported them.

| Run | What it is | Exit | Accepted of drafted, rounds |
| --- | --- | --- | --- |
| R1 `27b-auto` | 27B, default drafting, no drafter passed | 0 | 70 of 262, 5030 |
| R2a `27b-dflash2` | `benchmark --drafter z-lab/Qwen3.8-27B-DFlash2`: refused, no receipt | 1 | none |
| R2b `27b-serve-drafter` | `serve --drafter z-lab/Qwen3.8-27B-DFlash2`: refused | 1 | none |
| R3 `35b` | 35B-A3B-MTP, MTP drafts | 0 | 2466 of 4251, 2642 |
| R4 `35b-serial` | the same, `--serial` (plain decode) | 0 | 0 of 0, 5100 |
| R5 `35b-attached` | the same served, benchmarked attached (`35b-manifest.json`) | 0 | 2626 of 4537, 2483 |
| R6 `35b-hipvis0` | R3 with `HIP_VISIBLE_DEVICES=0` set by `run.sh` | 0 | 2601 of 4500, 2507 |

R1's counts are the same as 18 October's (`../out/27b-auto`): 70 of 262 over 5030 rounds. This checkpoint has no
MTP tensors (its file list in `../pull.log` has no `mtp-4bit.safetensors`), so these are not MTP drafts; a managed run
keeps no server log, so the receipts do not show where they come from.

## What the receipts show

- Identity: every receipt names the CPU, the one GPU (40 compute units, 128,849,018,880 bytes), unified memory with
  source `detected`, and the checkpoint's `revision`, `config_sha256` and `tokenizer_sha256` with `local` false. R5's
  attached receipt takes them from the server's manifest, which carries no `family` or `quantization`, so those two are
  null there, as in `../out-r5b/`.
- Per-request counts: `match.py` is `../match.py` with `RUNS` set to these five runs and its two controls pointed at
  R5. It finds every count matching: each file's samples sum to its totals and to the run's printed line, and each of
  R5's 20 samples has the counts the server printed for that request (`35b-serve.log`; the four warm-ups are left out).
  A drafted count off by one and the server log read one request out of step are both refused (`match.txt`).
- R6 names the same GPU as R3. The receipts do not record the environment; `run.sh` sets the variable for R6 only.
  On a box with one GPU this shows that the visible-device changes keep the GPU when
  `HIP_VISIBLE_DEVICES` is set; it cannot tell the old probe from the new one. The host tests and negative controls in
  #1043 (`docs/receipts/rocm-host-identity/host-controls/`) do that.
- The board: `board-validator.log` runs the board's `protocol.js` (the same file, by sha256, as on 18 October) through
  `check.mjs` from #1043. Each receipt is refused as `rocm` ("Unsupported benchmark runtime.") and accepted when only
  `runtime.backend` is changed to `cuda`, in memory.

The box's own `llm-server` was stopped for the run and restarted by `run.sh`'s exit trap (stopped by 17:31:37 UTC,
answering again at 17:50:29, in `run.log`).

## Sanitizing

Home paths are written as `~` in the text files (`*.log`, `binary.sha256`), and the trailing spaces of line 1 of
`35b-serve.log` were removed, as in `../out/`. The receipts, draft-count files and the manifest are as the run wrote
them. `SHA256SUMS` covers every file in this folder except itself
(`shasum -a 256 -c SHA256SUMS` from this folder).
