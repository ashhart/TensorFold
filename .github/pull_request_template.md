<!-- Read CONTRIBUTING.md first. A small pull request with a complete receipt lands fastest. -->

## What this changes

<!-- One or two sentences: the behavior before and after. One change to a pull request. -->

- Issue: <!-- "Fixes #N" when this completes it, "Refs #N" for part of it, or "n/a" and why. A new family, backend or arithmetic change links the issue where its gate was agreed. -->
- Stacked on: <!-- "main", or the parent pull request and the commits this one adds. -->
- `atc overlap --json`: <!-- The result against main, and how each conflict was resolved. -->

## Checks on this head

- Head commit:
- Lean checker, `python3 tools/zig/lean_check.py`, on the base and on this head: <!-- Both counts, and new findings, which must be 0. -->
- Tests run, with the command and pass, fail and skip counts:
- CI: <!-- Green, or each failure and whether this change caused it. -->
- Not run, and why:

## Receipt

<!-- A change that cannot reach inference (documentation, CI, clients, the app) writes "n/a" and why here and skips the inference checklist. -->

- Environment, with the TensorFold commit, Zig and backend toolchain, machine and GPU, checkpoint and revision:
- Paths and platforms the change reaches, which I ran and which I could not:
- Exactness, from `tools/bench_concurrent.py --alone --serial` or `token_sha` against `"draft": false`:
- Decode speed before and after, from `tools/bench_openai.py`:
- Prompt speed before and after: the cold first request's `ttft_s` from `tools/bench_cache.py --require-cold` on a long prompt, or "n/a" with the reason when the change cannot reach prompt processing:
- Licence of any new weights or drafter, or "n/a":

<details><summary>Kernel contract, only when a GPU kernel is new or changed</summary>

<!-- See docs/kernel-contract.md. Delete this section when no kernel changes. -->

- Kernels registered with the contract, and the kernel each one replaces:
- A planted control for each new kernel, and each one fails as planted:
- Each backend's `SUMMARY` line and exit status:
- Chips I could not run:

</details>

## Checklist

- [ ] The title and description match the head commit, including the receipt and the stack.
- [ ] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [ ] Comments and docstrings are one line. No measurements or history in the source.
- [ ] No personal data, machine names, internal hosts or local paths in the diff or its commits. No AI attribution lines.

<!-- For a change that reaches inference: -->

- [ ] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [ ] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo.
- [ ] No precision traded for speed. If the bits change, the description says which sums change and why.
- [ ] Decode and prompt processing are no slower than main at one stream and the widest admitted; with `--drafter`, no slower than without it, greedy and sampled
