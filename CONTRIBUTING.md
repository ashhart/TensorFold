# Contributing to TensorFold

Thank you for wanting to make TensorFold faster. This page says what a pull request needs to land in the next
release, and what happens to it after you open it. A small pull request with its measurements attached lands
fastest.

## Before you write code

- Read the pinned issue "Landing next" and the open pull requests. Work that is integrated for the next release is
  listed there before it reaches `main`, so you don't build it twice.
- Open an issue first for a large change: a new backend, a GPU generation, a model family, or anything that changes
  the arithmetic a model runs. We will say what it needs to land before you spend the time.
- Keep one change to a pull request. A fix and an unrelated speed-up land separately. In a stack of dependent pull
  requests, say which one each sits on.

## What every change must keep

1. **Lanes.** Every model decodes through the shared lane rounds, where one forward verifies the drafted tokens
   together. No serial path, no feature that works only with drafts off, no second batcher beside the engine.
2. **Exact output.** A drafted reply equals the same engine's `"draft": false` reply, token for token. A resumed
   prompt equals a fresh one, and each concurrent stream equals its solo run. The README's "Exact decoding" section
   has the contract.
3. **No precision traded for speed.** No lower-precision activations or sums, no skipped keys, no pruned tokens. A
   reorder, or another kernel at equal or better precision, is fine. Say which sums change and that the bits change.
   A precision mode that a checkpoint format defines is a separate conversation, so open an issue.
4. **Prompt processing no slower.** Measure cold prompts against the last release for any change that touches
   prefill, a kernel, the engine, a family or the server.
5. **Every platform it touches.** That is Metal on Apple Silicon from M1 to M5, and CUDA. Say which you ran and
   which you could not.
6. **Lean code.** One job to a module, files under about 600 lines, and one-line comments and docstrings that say
   what the code can't. Measurements, history and benchmark tables belong in the pull request, not in the source.

## The receipt

Paste these into the pull request. We rerun what we can, and a complete receipt lets a small change land on review
without waiting for a machine. A change to documentation alone needs no receipt.

- **Environment.** The TensorFold version or commit, the MLX or PyTorch version, the machine and GPU, the checkpoint
  and its revision.
- **Exactness.** Drafted against serial, and concurrent against solo, by token hash:

  ```
  python3 tools/bench_concurrent.py http://127.0.0.1:8080 MODEL --alone --serial
  ```

  For a server change, also send one request twice, the second with `"draft": false`, and compare `token_sha`.
- **Decode speed**, before and after, on the same machine in the same session:

  ```
  python3 tools/bench_openai.py http://127.0.0.1:8080 MODEL
  ```

- **Prompt speed**, before and after. The tool builds cold prompts of 2k, 8k, 16k, 32k and 64k tokens:

  ```
  python3 tools/prefill_cold.py build MODEL_DIR prompts.json
  python3 tools/prefill_cold.py run http://127.0.0.1:8080 MODEL prompts.json out.json
  ```

- **Tests** you ran, with the pass, fail and skip counts.
- **What you did not run**, and why. We need this as much as the rest.

When the difference is a few percent, alternate the runs: before, after, after, before. One run each way cannot tell
a 2% change from a warm machine.

## Running the tests

TensorFold needs Python 3.11 or newer.

```
python -m pip install -e '.[test,tui]'
python -m pytest -q tests
```

On a Mac that is the host suite. The CUDA tests are `tests/cuda` and `tests/test_cuda_*.py`. Those that need PyTorch
or a GPU skip without one.

A new test skips cleanly where its dependency is missing. Use `pytest.importorskip`, so the suite still collects on
a machine without PyTorch or without MLX.

A test should fail before your change and pass after it. Say so in the pull request.

## Commits and identity

- Your commits keep your authorship. We record every author under a GitHub noreply address, such as
  `12345+yourlogin@users.noreply.github.com`, so no personal email lands in the history. "Keep my email addresses
  private" in your GitHub email settings does this for you.
- No AI tool attribution lines, and no co-author trailers for tools. We remove them when we land a commit.
- No personal data, machine names, internal hosts or local paths in code, tests, fixtures or commit messages.
- A data file built from text, such as an n-gram table, comes from public text only. Name the corpus in the pull
  request.

## How a pull request lands

We review it, port it onto the release branch ourselves and run it through the release checks. We do not ask you to
rebase. If `main` has moved, we carry your commits forward.

Your change lands on `main` as your own commit, under your name, inside a release. We then close the pull request
with a comment that links the release and says what we changed on our side, if anything. GitHub may show the pull
request as closed and not merged, because the commit that lands is a port of yours. It still carries your name, and
you appear in the repository's contributors.

Our changes on top of yours, such as a style pass or an extra test, go in separate commits under our name.

When a pull request overlaps work that is already integrated, we close it with a note that says where the work lives
and credits what your report or measurements added. That is not a rejection, and the "Landing next" issue exists to
make it rare.

## What we can check ourselves

We test on an M3 Ultra, an M5 Max, one DGX Spark and two together, and an RTX PRO 6000. A change for other hardware
needs a complete receipt, because we cannot rerun it, and it takes longer to land. Keep such a change in its own
files where you can, so it cannot affect the paths we do test.

## Reporting a bug

Open an issue with the TensorFold version, the machine, the checkpoint and its revision, the exact command, the
startup lines the server printed, and what you expected against what happened. A request that reproduces it is worth
more than a description.

## License

TensorFold is Apache-2.0. A contribution you submit is under that license, as its section 5 says. Model weights keep
their own licenses.
