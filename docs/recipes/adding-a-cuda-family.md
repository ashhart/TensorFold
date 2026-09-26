# Adding a CUDA family

A family package (`src/tensorfold/families/<name>/`, see [adding a family](adding-a-family.md)) serves NVIDIA
GPUs when it has a `cuda_engine` function. `tensorfold serve` calls it on Linux (or with `--backend cuda`) and
puts `tensorfold.cuda.server` in front of what it returns: the OpenAI routes, the model's chat template,
streaming, tool calls and thinking. A package with `load` and no `cuda_engine` runs on Apple Silicon only; one
with `cuda_engine` and no `load` runs on NVIDIA GPUs only (GLM-5.3-Flash). Keep the CUDA code in the family's
`cuda/` folder with a README that lists its kernels, as `families/qwen3_5/cuda/README.md` does.

## The package

```python
# src/tensorfold/families/mymodel/__init__.py
MODEL_TYPES = ("mymodel",)
TITLE = "My Model"
MODELS = ("owner/checkpoint",)          # what `tensorfold models` lists and your kernels expect
DRAFTER = "owner/draft-model"           # optional: `--drafter auto` uses it once it has been pulled

def cuda_engine(model_dir, *, drafter="", tp=1, rank=0, master="", master_port=29551,
                no_drafts=False, mtp_drafts=None, **options):
    from .cuda.engine import MyEngine   # import torch inside, never at package import
    return MyEngine(model_dir, ...)

CUDA_APP = None                         # optional: a tensorfold.cuda.server.App subclass
```

`cuda_engine` sets the engine up the way your measured recipe runs it, so `tensorfold serve MODEL` with no
flags reproduces your numbers. Refuse settings you do not support (GLM refuses `tp=1`: it does not fit one
Spark). Import torch inside the engine module only: the CLI imports every family package on the Mac too.

## The engine

| Member | What it does |
| --- | --- |
| `eos` | the token ids that end a reply |
| `generate(prompt, max_tokens, sampling, on_tokens)` | decode up to `max_tokens` tokens after `prompt` (token ids). `sampling` is an `engine.exact_sampling.Sampling`, or None for greedy. Call `on_tokens(new_ids)` as each round commits tokens, and stop when it returns True. Return a dict of stats (it goes into the response). |
| `follow()` | with `tp=2`, rank 1 runs this forever: receive each request from rank 0 and run the same calls |

The 27B's engine (`families/qwen3_5/cuda/engine.py`) is the template: prefix reuse from the last prompt and
reply, the two-rank request header, the start-up check that both ranks were given the same settings.
`CUDA_APP` lets a family change what a request can ask for. GLM's picks a draft policy from a suffix of the
model name.

## Proving row invariance

Every kernel on the verify path must give a row the same bits alone and in a window. Test each kernel first,
then the model:

1. For every projection shape, random inputs at 1, 2, 3, 16, 17, 64 and 128 rows: each row of the window
   equals the same row computed alone (`torch.equal`, not `allclose`).
2. A tree window through the whole forward: every node's logits equal serial steps along its path, and the
   committed state after the accepted path equals the state after serial steps (`tests/cuda/test_qwen27_forward.py`).
3. The real model: drafted tokens equal serial tokens by SHA-256 on every benchmark run, sampled and greedy,
   code and chat, short and long prompts.
4. Two GPUs: the share protocol on CPU with gloo (`tests/cuda/test_qwen27_share.py`), then drafted equals
   two-GPU serial on the real model.

What breaks it on CUDA:

- cuBLAS, `torch.matmul` and `F.linear` choose algorithms and K splits by the row count. Use them only where
  the row count never changes (for example a draft model's internals, which only propose).
- A split-K that depends on the row count, or a reduction over rows.
- Attention whose chunking depends on the window. Chunk by absolute key position and merge in key order.
- Router logits in bf16, where ties at the top-k cut decide which expert runs. Use fp32 and break ties by id.
- An all-reduce across GPUs. Its summation order is NCCL's choice. All-gather the fp32 partials and add them
  in rank order.
- A different Triton version, which can fuse multiply-adds differently. Serial and drafted decoding always run
  the same build, so this changes your reference, not exactness. Regenerate stored hashes after upgrading.

## Checking quality

Row invariance proves drafted equals serial; it says nothing about whether serial is right. Compare against a
plain fp32 PyTorch forward (`families/qwen3_5/cuda/reference.py` is one) by teacher-forced NLL and top-1
agreement over a few thousand tokens. Run the reference with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` inside
NVIDIA's container, which otherwise runs fp32 matmuls in TF32.

## Measuring

1. The floor: weight bytes read per token over the measured read bandwidth (about 240 GB/s on GB10).
2. Host against GPU: kernels per forward, host time per forward, GPU busy time. If the host is slower than the
   GPU, fuse kernels or capture CUDA graphs. If it is not, graphs will not help.
3. The cost of an extra verify row, and tokens a round against width. Promote a wider window only when
   committed tokens divided by the complete round time goes up.
4. A per-round trace: the 27B's `draft_decode(trace=...)` records the rows, the deepest node, nodes per depth,
   accepted drafts, why the round stopped (a wrong draft, or nothing deeper), and whether the target's token
   was among the drafter's candidates.
5. Against a baseline: `tools/bench_openai.py` against both servers, the same prompts, seeds and reply length,
   medians over seeds, nothing else on the GPUs.

## Tests to write

Put GPU tests in `tests/cuda/` (the folder is skipped where PyTorch sees no GPU) and name them after the
family (`test_mymodel_*.py`): kernel row invariance, window against serial, commit against serial state, and
for two GPUs the share protocol. Keep one test that loads nothing and runs anywhere: the family package imports
and `cuda_engine` refuses a bad setting before touching the GPU.
