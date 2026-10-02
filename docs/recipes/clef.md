# Clef decision models

[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (Qwen3.8-27B) and
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B) are decision models: a `qwen3_5`
backbone plus a small joint schema head. The head reads the backbone's final hidden state at every prompt row and
returns one logit per allowed option of every question. The CUDA engine serves the release's bf16 checkpoint as
stored, on one GPU, with drafts off.

```bash
tensorfold pull Cloudflare/clef-flash
tensorfold serve Cloudflare/clef-flash --context 16384
```

The server answers `POST /v1/decisions` and `POST /v1/systemone` ([API](../api.md#decision-models)). Chat routes
still answer from the backbone's LM head, which the release does not train for chat. `--parallel` and `--tp 2`
refuse decisions.

## Fidelity

The prompt encoding, the option order and the head follow the release's `joint_schema_model.py`. On one DGX Spark
(GB10), 2026-10-02, 45 records (128 questions; choice, score and noul; states of 230 to 14,149 tokens) were
scored by the release code (torch 2.13, transformers 5.10.2, bf16) and by this engine:

| | Clef-Flash |
| --- | --- |
| Token ids and spans equal | 45/45 |
| Same top option | 128/128 |
| Largest probability difference | 0.027 |

A decision's prompt takes cuBLAS for the bf16 projections. Nothing resumes from a decision's prompt state, so the
row-count invariance chat prompts keep does not apply there.

## Speed

One request at a time on GB10, Clef-Flash, release code on the same machine for comparison:

| Prompt tokens | TensorFold | Release code |
| --- | --- | --- |
| 230 | 95 ms | 174 ms |
| 457 | 116 ms | 235 ms |
| 3,439 | 726 ms | 1,836 ms |
| 7,849 | 1,639 ms | 4,214 ms |
| 14,149 | 3,219 ms | 7,681 ms |

The server runs one small decision at startup, so the first request does not pay the kernel builds.
