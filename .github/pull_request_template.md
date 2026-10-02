<!-- Read CONTRIBUTING.md first. A small pull request with a complete receipt lands fastest. -->

## What this changes

<!-- One or two sentences, and the issue it closes if there is one. One change to a pull request. -->

## Receipt

<!-- A change to documentation alone needs no receipt. -->

- Environment, with the TensorFold commit, MLX or PyTorch version, machine and GPU, checkpoint and revision:
- Exactness, from `tools/bench_concurrent.py --alone --serial` or `token_sha` against `"draft": false`:
- Decode speed before and after, from `tools/bench_openai.py`:
- Prompt speed before and after, from `tools/prefill_cold.py`:
- Tests run, with pass, fail and skip counts:
- Not run, and why:

## Checklist

- [ ] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [ ] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo.
- [ ] No precision traded for speed. If the bits change, the description says which sums change and why.
- [ ] Prompt processing is no slower than the last release.
- [ ] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [ ] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [ ] Comments and docstrings are one line. No measurements or history in the source.
- [ ] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
