# Decision scoring regression on two CUDA hosts

Run `glm_decision_tp.py` with `PYTHONPATH=src:tests/cuda` on both hosts,
rank 1 first, rank 0 second, with the same `--master` and `--port` (29627).
Requires pytest for the existing synthetic-checkpoint helpers.
Stop only the test's rank-1 process after rank 0 prints four PASS records;
`follow()` deliberately stays available for another request.

The harness uses real rank-specific weights, CUDA kernels and NCCL. It compares
chat continuations with serial fresh prefills after scoring and after switching
conversations, with MTP and DFlash2, at zero and 64 MiB cache budgets. The budget
is synchronized through the bootstrap store in a test-only engine subclass.
It also checks that saved MTP prefixes actually resume and DFlash2 falls back.

Verified on both GB10s in `glm53-tf-r0` / `glm53-tf-r1` using an isolated copy
of the PR source, without changing the running production source or restarting
production. Four single-GPU CUDA cases and all four real two-rank cases passed.
The checkpoint is synthetic; this is not full-checkpoint EXL3 qualification.

Production occupied ~110 GiB of 119 GiB unified memory. Normal admission
correctly refused another model because of its reserve. For these tiny tests
only, `capacity.available_bytes` was overridden in the test process to 1 GiB;
the startup estimate was 0.27 GiB per rank. No production admission setting was
changed. Extension builds used `/cache/pr127_extensions`, not production's cache.
