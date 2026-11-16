# Quality suite receipts

Each file here is one run of TensorFold Web's quality suite against one checkpoint, made with
[`tools/quality_suite.py`](../../../tools/quality_suite.py). `tensorfold models` shows each checkpoint's score from
these files, and a host test checks that the two agree.

- **The suite.** 228 items from [`web/tensorfold_web/quality.py`](../../../web/tensorfold_web/quality.py), digest
  `d21a62442423`: 100 GSM8K problems and 100 MMLU questions drawn with a fixed seed, 20 instruction checks and 8
  long-context items. Each item has a fixed reply length, and replies are scored by the Bench's strict rules. A run
  under another digest is not comparable.
- **How it runs.** One request at a time, greedy (temperature 0, top_p 1). Thinking is off when the model's
  `capabilities.thinking_switch` says an off request works, and left at the model's default otherwise. Drafts stay as
  shipped, because TensorFold's replies are the same with them or without.
- **What a file holds.** The suite digest, the checkpoint as `REPO@REVISION` and how that revision was established,
  the engine commit, the chip, the summary overall and per category, and every item's verdict. A missed item keeps
  the first 200 characters of its reply.
- **Names.** `REPO.CHIP.json`, with the repository's `/` written as `--`.

Rerun one:

```sh
tensorfold serve REPO --port 8080
python3 tools/quality_suite.py http://127.0.0.1:8080 out.json --checkpoint REPO@REVISION
```
