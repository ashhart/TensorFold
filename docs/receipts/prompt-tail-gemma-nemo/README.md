# Prompt chunks nothing reads: Gemma 4 and Nemotron on CUDA (#2432)

The tools and raw runs behind #2432's cold TTFT table, and the drafted runs, main against branch. A non-final prompt
chunk whose rows nothing reads stops at its last state write; the prompt's last chunk is unchanged.

## Source and environment

- One NVIDIA GB10 (sm_121), 128 GB unified memory, Linux 7.0, driver 615.71.09, CUDA 13.0 (V13.0.88), Zig 0.17.0.
  `zig build native -Dsm=121`, one build per arm, fresh server per arm.
- The change is the same in every pair: `git diff BASE HEAD -- zig` is identical for all three pairs below.

| Pair | Base (main) | Head (branch) | Binary md5, base / head |
|---|---|---|---|
| a | `5a45539af031726f68f6dacdf713dbdff8f55b60` | `14d3f3289f7e76d286a61c057745298336ed5b77` | `671fd710…` / `d885103e…` |
| b | `9b15a958d9d88b7200d3f483f80ce86d614b18ce` | `7f9f50dbf6dc922418224acf3cfce059651192f6` | `c6dafa52…` / `eaa871ae…` |
| d | `c16a296407deda1de8d1266c63cd141705643312` | `39e33dd33fb3829adb39796b84d43f1ed2d4dffc` | `651968f9…` / `d82474b6…` |

Every arm's JSON carries its binary's full md5.

- Checkpoints: `mlx-community/gemma-4-31b-it-4bit` (revision not recorded); `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`,
  revision `d9d758fb83953437f7263256b0d96157e2a348b8`. Gemma 4's drafter for the drafted runs:
  `z-lab/gemma-4-31B-it-DFlash`, `model.safetensors` sha256
  `b3f28580103d6e8a0a1d9389646957b9309df5982558afd754b9810357f425b7`. Nemotron drafts with its own MTP head
  (`mtp-4bit.safetensors` in the checkpoint).
- Serve flags: `--parallel 1 --context 40960 --no-update-check`; TTFT runs add `--no-drafts --prompt-cache-gib 0`;
  Nemotron runs b, c, 2k12 and drafted add `--no-thinking`.

## Tools

- `tools/ab.py` (Nemotron) and `tools/abn.py` (Gemma 4, which has no logprobs on CUDA): one arm. Drops the page cache,
  starts a server, sends one warm-up, then three cold prompts each at 2K, 16K and 32K (synthetic word lists, seeded),
  16 tokens at temperature 0. Then decode (128 tokens), one seeded sampled reply, Nemotron's prompt_logprobs, and two
  prompt-cache turns on a second server. TTFT is read from the server's own `done` line.
- `tools/gate.py` (Nemotron) and `tools/gaten.py` (Gemma 4): the comparison. Arms in the order base, new, new, base.
  TTFT change = median of the six new prompts over the median of the six base prompts, per length; content (reply,
  token sha, logprobs, prompt_logprobs, seeded, kept turns) must be equal in every arm; the range per length was set
  before each run.
- `tools/ab2k.py` and `tools/gate2k.py`: Nemotron at 2K, 12 prompts an arm, level = median change <= +0.5 %.
- `tools/abba.py`: the base, new, new, base driver (the runs used an equivalent shell loop), e.g.
  `abba.py ab.py BASE_BIN NEW_BIN MODEL OUT gate.py -0.7 0.7 -2.2 -0.8 -2.3 -0.8`.
- `tools/abd.py` and `tools/gated.py`: the drafted run. One arm with drafts on, on the TTFT table's nine prompts,
  64 tokens at temperature 0; the gate requires drafts on, drafts accepted on every request, and equal replies.
- The only edit to the tools as run: the page-cache drop reads its command from `PT_DROP_CACHES`.

## Runs

Each `runs/<name>/` holds the arms' JSON, the server logs (`*-cold`, `*-kept`) and `gate.txt`, the gate's output as run.

| Table cell | Run | Pair | Lengths | Result |
|---|---|---|---|---|
| Gemma 4, earlier base | `gemma4-a` | a | 2K, 16K, 32K | -1.32, -3.09, -3.83 %; gate in |
| Gemma 4, this head | `gemma4-b` | b | 32K | -3.88 %; gate in |
| Nemotron, earlier base, first | `nemotron-a` | a | 2K, 16K, 32K | +0.00, -1.90, -2.11 %; gate out on its empty guard only: replies were all reasoning, so no top logprobs to compare |
| Nemotron, earlier base, second | `nemotron-b` | a | 2K, 16K, 32K | +0.52, -2.20, -2.29 %; gate out on 2K only, +0.52 past the +0.5 limit |
| Nemotron 2K, 12 prompts | `nemotron-2k12` | a | 2K | -0.35 %, shas equal; gate in |
| Nemotron, this head | `nemotron-c` | b | 32K | -2.31 %; gate out by 0.01 pp on the fast side of the preset -2.3 % bound |
| Drafted, Gemma 4 | `drafted-gemma4` | d | 2K, 16K, 32K | replies and token shas equal 9 of 9; gate in |
| Drafted, Nemotron | `drafted-nemotron` | d | 2K, 16K, 32K | replies and token shas equal 9 of 9; gate in |

## Drafted runs

One server an arm, main then branch, drafts on at the family's own drafter, the TTFT table's nine prompts, 64
tokens at temperature 0, `--prompt-cache-gib 0`, page cache dropped before each server. Every request drafted and
had drafts accepted.

| Family | Drafter | Replies equal | Accepted / drafted, main | Cold TTFT branch vs main, 2K / 16K / 32K |
|---|---|---|---|---|
| Gemma 4 31B | DFlash, `z-lab/gemma-4-31B-it-DFlash` + copies | 9 of 9 | 40-49 of 97-192 | -9.2..-11.7 / -2.9..-6.8 / -3.8..-4.0 % |
| Nemotron 3.5 Lightning | its MTP head + copies | 9 of 9 | 29-36 of 39-49 | -1.0..+0.4 / -0.1..+0.4 / -0.4..0.0 % |

Nemotron's MTP head absorbs the prompt rows, so every chunk is read and runs whole: TTFT is level. Gemma 4's DFlash
drafter taps layers 1, 12, 23, 35, 46 and 57 of 60 (`target_layer_ids` in its config), not the last layer, so a
non-final chunk is still unread and stops at its last state write with the drafter on too; the replies are equal and
the TTFT gain holds. These are single arms, one server each, not a base, new, new, base run.

Drafted and accepted counts differ slightly on some prompts between the arms while the tokens are equal (Gemma 4 on
all nine, by up to 19 drafted and 1 accepted; Nemotron on one, 32 against 31 accepted). The cause was not
measured; there is no main against main control.

## Redaction

Local paths, user and host names are replaced (`redaction-report.json` counts them per file). Nemotron's
`prompt_lp.plp` (prompt_logprobs, about 3.7 MB an arm) is replaced by the sha256 of its sorted JSON and its row
count; the gate compared it in full. Everything else is as recorded. `SHA256SUMS` covers every other file here.
