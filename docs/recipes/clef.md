# Clef decision models

[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (Qwen3.8-27B) and
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B) are decision models: a `qwen3_5`
backbone plus a small joint schema head. The head reads the backbone's final hidden state at every prompt row and
returns one logit per allowed option of every question. The CUDA engine serves the release's bf16 checkpoint as
stored, or an EXL3 pack of it, on one GPU, with drafts off.

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

## Clef in an EXL3 pack

The 27B release needs 54 GB in bf16. An EXL3 pack of its backbone serves in 16 GB, beside another model on one
GB10. Convert with exllamav3 (1.4.7 here), then copy the head files into the pack:

```bash
# exllamav3 reads the tower's settings from preprocessor_config.json: write the image_processor block of
# processor_config.json there, with image_processor_type "Qwen2VLImageProcessorFast"
python convert.py -i clef -w work -o clef-exl3-4.0bpw -b 4.0 -hb 8
cp clef/joint_head.safetensors clef/joint_head_config.json clef-exl3-4.0bpw/
tensorfold serve clef-exl3-4.0bpw --context 16384
```

The head reads LM head rows for its option vectors. A quantized head has no rows to index, so the engine runs its
matmul on identity rows once at startup and keeps the result as a bf16 table (vocabulary x hidden, 2.5 GB for the
27B). The 8-bit head (`-hb 8`) keeps those rows close to bf16.

The same 45 records, Clef 27B, 4.0 bpw EXL3 pack on this engine against the release code in bf16 (2026-10-02):

| | Clef 27B, EXL3 4.0 bpw |
| --- | --- |
| Same top option | 128/128 |
| Probability difference, median / 90th percentile / largest | 0.003 / 0.028 / 0.142 |
| Questions differing by more than 0.05 | 5/128 |

The closest top-two margin in the reference was 0.036, and the pack kept that question's answer too.

| Prompt tokens | EXL3 pack | Release code, bf16 |
| --- | --- | --- |
| 230 | 598 ms | 437 ms |
| 457 | 728 ms | 678 ms |
| 3,439 | 3,304 ms | 5,441 ms |
| 14,149 | 14,274 ms | 22,618 ms |

EXL3 prompts run on the pack's own matmul. Short prompts are slower than bf16 on cuBLAS, and long ones are faster.
