# Supported models

Which checkpoints the native release serves, on which machines, and how to manage the models on disk.
Install TensorFold first with the [quick start](../README.md#quick-start); the serve flags are in the [run guide](run-guide.md#serve-flags).

<a id="qualified-models"></a>

## Pick a model

These are the native release's serving paths. A recipe may also describe the older Python 0.6.6 engine;
use its native section and the current [changes](../CHANGELOG.md) together.
Hardware below records the documented scope, not an unmeasured minimum. Context, parallel requests, prompt caches
and drafters need memory beyond the weights. `tensorfold info REPO` reports the checkpoint's memory floor.
Where no RAM capacity is published, inspect that floor before serving a large checkpoint.
CUDA release images include sm_80, sm_86, sm_120 and sm_121. The A100 (sm_80) has kernel-check evidence
([#846](https://github.com/ashhart/TensorFold/pull/846)), but no served checkpoint on record, so it is not model-qualified.
Nemotron and dense Qwen have NVFP4/FP8 fallback code for sm_80 to sm_89 (#901, #1470); no such model run on sm_80 to sm_89 is recorded.

Run a table command after installation. On a Linux archive installation, use the `tensorfold` shell function from the [quick start](../README.md#linux-with-an-nvidia-gpu).
`REPO@revision` pins a pull; serve the repository name without the revision after pulling it.
Conversions are explicitly marked: their upstream weights need the linked conversion before native serving.

| Model / format | Download | Machine and memory scope | Native recipe / limits |
| --- | --- | --- | --- |
| dots.mocr, BF16 OCR | Use the local `dots-studio/dots.mocr` checkpoint. | Native HIP; AMD R9700 (`gfx1201`) checked, one GPU. | [Native OCR recipe](recipes/dots-mocr-hip.md); full model-card demo matches HF tokens. Native CUDA serves text prompts only ([CUDA recipe](recipes/dots-mocr-cuda.md)). No video, MTP head, or Metal qualification. |
| Qwen3.5-2B, MLX 4-bit | `tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3` | One Mac (M5 Max checked), or one GB10; no RAM minimum published. | [2B](recipes/qwen3.5-2b-native.md); [CUDA release checks](../CHANGELOG.md#110). Text only. |
| Qwen3.8-27B, MLX 4-bit | `tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit`<br>`tensorfold pull z-lab/Qwen3.8-27B-DFlash2` | One Mac (M5 Max/M3 Ultra), or one GB10; a 32 GB Mac needs `iogpu.wired_limit_mb` raised (the startup refusal names the value). | [27B](recipes/qwen3.8-27b.md). Add `--drafter z-lab/Qwen3.8-27B-DFlash2`; without it decoding is plain. |
| Nemotron 3.5 Lightning 30B-A3B, MLX 4-bit | `tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | One Mac (M1-M5), or one GB10; also checked on RTX 3090 / RTX PRO 6000. | [Lightning](recipes/nemotron-3.5.md). Includes MTP. |
| Lightning, mixed oQ4 / oQ6 / oQ8 | `tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ4`<br>`tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ6`<br>`tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ8` | One M3 Ultra checked. | [Width and MTP instructions](recipes/nemotron-3.5.md#run). Metal only; copy the 4-bit checkpoint's MTP head to draft. |
| Lightning, NVIDIA NVFP4 | `tensorfold pull nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | One GB10 checked. Non-cluster fallback code exists for sm_80 to sm_89; no NVFP4/FP8 serving run is recorded on sm_80. | [Lightning](recipes/nemotron-3.5.md); [CUDA changes](../CHANGELOG.md#110). |
| Nemotron 3.5 Super VL 120B-A12B, NVFP4 | `tensorfold pull nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3` | One or two GB10s; weights 70.8 GiB, 73.2 GiB with vision on one. | [One/two-GB10 commands](recipes/nemotron-3.5-super-vl.md). Add `--vision` for images. |
| Nemotron 3.5 Bolt 74B-A9B | [NVIDIA's checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) | Native CUDA; 24 GB checkpoint, checked on RTX PRO 6000. | [Bolt/Omni video](recipes/nemotron-omni-video.md); [release qualification](../CHANGELOG.md#110). |
| Qwen3.8 Flash Next, MLX 6-bit | `tensorfold pull TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP` | One M5 Ultra checked; RAM minimum not published. | [Metal](recipes/qwen3.8-flash-next.md), [runbook](../RUNBOOK.md#flash-next-and-paired-metal-serving). |
| Qwen3.8 Flash Next, MLX 4-bit | `tensorfold pull TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` | One GB10; 73.3 GiB uploaded, plus mapped tables and caches. Metal has a native 4-bit path too (M3 Ultra checked). | [Native CUDA](recipes/flash-next-cuda.md). NVFP4/two-local-rank and vision paths are experimental. Metal's shared-session limits are in [known issues](known-issues.md#models). |
| GLM-5.3-Flash, MLX 4-bit | `tensorfold pull TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` | Two M5 Ultras; single-Mac native serving is not qualified. | [Paired Metal runbook](../RUNBOOK.md#flash-next-and-paired-metal-serving). Do not use the Python recipe's single-Mac command. |
| GLM-5.3-Flash, EXL3/TR3 | `tensorfold pull Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | Two CUDA ranks; the release checks the two-GB10 setup. | [Release scope](../CHANGELOG.md#110), [checkpoint](recipes/glm-5.3-flash.md#exl3). Experimental checkpoint; use [native rank options](recipes/glm-5.3-flash-cuda.md#run), not the Python installation. |
| GLM-5.3-Flash, NVFP4 | `tensorfold pull nvidia/GLM-5.3-Flash-NVFP4@423acf37583782c51c142d145aef733d72943d93` | Two GB10s at TP=2: @Bizuayeu loaded the whole checkpoint on 1.1.0 in 302.1 s and served it. | [Native CUDA](recipes/glm-5.3-flash-cuda.md); [contributed run](https://github.com/ashhart/TensorFold/issues/1040): one token hash per prompt across three runs and a passphrase found in 199,307 tokens; scored prompts and more than two ranks were not covered. |
| GLM-5.3-Flash, FP4-Local | `tensorfold pull nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` | Two GB10s at TP=2, or three at TP=3 over a ring of direct links: @Bizuayeu loaded it as stored and served it at both sizes. 165.3 GiB of weights, 13% less than NVFP4's. | [Native CUDA](recipes/glm-5.3-flash-cuda.md), in 1.1.5 (#1050). NVIDIA's mixed-precision checkpoint: NVFP4 routed experts, FP8 for most of the rest. As close to BF16 as NVFP4 at its smaller size ([receipt](receipts/glm-fp4local-80517fc3/README.md)). |
| GLM-5.3-Flash, XSFP4 | `tensorfold pull Bizuayeu/GLM-5.3-Flash-XSFP4@88736f916f8939607fde8a4b881dcae749f7551d` | Two GB10s at TP=2, or three at TP=3 over a ring of direct links: @Bizuayeu served it at both sizes. 165.7 GiB of weights. | [Native CUDA](recipes/glm-5.3-flash-cuda.md), in 1.1.5 through FP4-Local's path. FP4-Local with its routed experts refitted from the BF16 weights and a smoothing folded in, on the same kernels. Closer to BF16 than FP4-Local and NVFP4 by all three measures in [its card](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-XSFP4#measured). |
| GLM-5.3-Flash, Jammed XSFP4 | `tensorfold pull Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4@0a22bb46ecef3d0f5f96e41dc6011be12974fbf1` | Two GB10s at TP=2, or three at TP=3 over a ring of direct links: @Bizuayeu served it at both sizes. 160.2 GiB of weights; at TP=2 each GB10 holds [6.6 GiB less](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4#memory) than at block 16. | [Native CUDA](recipes/glm-5.3-flash-cuda.md), in 1.2.3 (#2041). Refitted from BF16 at block 32 with XSFP4's smoothing, each row's scales packed (`csf0_v1`, read only by this path), at XSFP4's speed. Its KL and top-1 are nearer BF16 than NVIDIA's two checkpoints', short of XSFP4's ([its card](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4#measured)). |
| GLM-5.3-Flash, Jammed XSFP4 NZ5 | `tensorfold pull Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4-NZ5@1c900ea0294e28a8e9ab56c1156c0338eed60aef` | Two GB10s at TP=2: @Bizuayeu served it. 160.2 GiB of weights. | [Native CUDA](recipes/glm-5.3-flash-cuda.md), on main after 1.2.7 (#2894). Jammed XSFP4 refitted with E2M1's -0 code read as +5 (`NVFP4_NZ5`, read only by this path). Nearer BF16 than Jammed XSFP4 ([its card](https://huggingface.co/Bizuayeu/GLM-5.3-Flash-Jammed-XSFP4-NZ5#measured)). |
| GLM-5.3, EXLR8 `home` | `tensorfold pull kindlingai/glm-5.3-exlr8-k2-k3-k4@f459f5d0272a529403dd9b3fd32f771f4250a0d4 --exlr8 home --node 0/2 --dir ~/models/glm-5.3-exlr8-home` | Two M5 Ultra Macs, 256 GB each; about 160 GB resident per Mac at 8k context / four streams. | [Two-Mac recipe](glm-5.3-exlr8.md). Use `--node 1/2` on the other Mac; Thunderbolt 5 and MCDMA required. No CUDA backend. |
| LFM2.5-24B-A2B, MLX 4 / 6 / 8-bit | `tensorfold pull LiquidAI/LFM2.5-24B-A2B-MLX-4bit@667408d494bde191c2cb9dd4f4549827816b9275`<br>`tensorfold pull LiquidAI/LFM2.5-24B-A2B-MLX-6bit@1deec53eb49571e21cdcbf82fd5839248f07ed88`<br>`tensorfold pull LiquidAI/LFM2.5-24B-A2B-MLX-8bit@67ab417dd02d0c9545802a726df668620eae049d` | One Mac (M4 Pro checked), or one NVIDIA GPU with the 4-bit export (GB10 checked). | [Native recipe](recipes/lfm2.5-24b-a2b.md) ([CUDA](recipes/lfm2.5-24b-a2b.md#cuda)), [release history](../CHANGELOG.md). Weights: LFM Open License v1.0. |
| Qwen3.6-35B-A3B, MLX with MTP | `tensorfold pull TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP@81169a9bc511a27c1b4eedb77a2cd98ced431847` | One GB10; 20.9 GB checkpoint. | [Qwen3.6](recipes/qwen3.6-moe.md). Native CUDA receipt here. Metal also registers this model type; the recipe's Metal commands describe Python 0.6.6. |
| Qwen3.6-35B-A3B, NVIDIA NVFP4 with MTP | `tensorfold pull nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5` | One GB10; 20.3 GiB on the device, 20.9 GiB with the drafter. | [Native CUDA](recipes/qwen3.6-moe.md#nvidias-nvfp4-checkpoint-native-cuda). Serve with `--drafter` naming the same checkpoint for MTP drafts. |
| Gemma 4 31B, MLX 4-bit | `tensorfold pull mlx-community/gemma-4-31B-it-4bit@696d436c404745a59f30e4939a658162b0a9e57f`<br>`tensorfold pull z-lab/gemma-4-31B-it-DFlash@eabd648301ce28583cc14757912e5e0f84e152e1` | One Mac (M5 Ultra checked), or one GB10 on CUDA. Pre-M5 Metal has a row path; its assistant and DFlash drafters require M5 tensor units. | [Native 31B on Metal](recipes/gemma-4-native.md), [CUDA qualification](../CHANGELOG.md#114). Name the drafter with `--drafter`; images with `--vision`. |
| Gemma 4 12B, converted MLX 4-bit | `hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --local-dir ~/models/gemma-4-12b-hf`<br>`hf download google/gemma-4-12B-it-assistant --revision 46d4c6f13f0ac0ad827b915669b8df9b81c64c51 --local-dir ~/models/gemma-4-12b-assistant` | One Mac (M5 Ultra checked); converted text weights 6.3 GB, image weights 100 MB, plus assistant/caches. MLX 4-bit also served on one GB10. Pre-M5 Metal has a row path; the MTP assistant requires M5 tensor units. | [Convert and serve](recipes/gemma-4-12b-native.md). Upstream download needs `hf`; conversion needs Python, mlx and mlx-lm. Serving is native; [CUDA qualification](../CHANGELOG.md#114) includes the MTP assistant. |
| Step-5-Preview, converted affine | `hf download stepfun-ai/Step-5-Preview --local-dir ~/models/step5-preview-fp8` | One 256 GB Mac; text 166.9 GiB, MTP adds 2.5 GiB and vision 3.7 GiB. | [Convert and serve](recipes/step-5-preview.md). The documented 256 GB setup serves converted affine weights; converting the upstream FP8 download needs Python and mlx. |
| MiniMax-M3.1-Flash, community affine 3/4-bit | `hf download BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34 --revision 9524999057e0d277d26438606e033e2191fb465c --local-dir ~/models/MiniMax-M3.1-Flash-MLX-mix34`<br>`hf download BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k --revision 916e7ca1044f1c3e7566163096243c940327224d --local-dir ~/models/MiniMax-M3.1-Flash-MLX-mix256k` | One 256 GB M5 Ultra; 203.7 GiB (32,768-token window) or 189.2 GiB (147,456), plus caches. | [MiniMax-M3.1-Flash](recipes/minimax-m3.1-flash.md). Set `MINIMAX_LOAD_LIMIT_GB`; images with `--vision`. Community checkpoints: lossy conversions of MiniMax's NVFP4 release (KL 0.244 / 0.295). MiniMax Community License. |
| Recursive Flint 40B-A2.5B, MLX INT4 | `hf download RecursiveSI/flint-int4 --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 --local-dir ~/models/flint-int4` | One M5 Mac; M5 Ultra checked; 21.4 GiB resident plus caches. M1-M4 untested. | [Flint](recipes/flint.md). Text and tools; always reasons; no draft head. |
| Kolibri-1, block FP8 (research) | `tensorfold pull Aleph-Alpha/Kolibri-1@e52eb4627d11516b0c01de49210ab5a4e4061444` | One NVIDIA GPU; GB10 measured; 74 GB checkpoint. | [Kolibri](recipes/kolibri-1.md). Text only; copy drafts, no EAGLE-3/neural drafter. |
| Liquid AI d1-3B, decisions | Not documented yet. | Native CUDA; RAM minimum not published. | [API section](run-guide.md#serve-flags). `/v1/systemone` and `/v1/decisions`. |
| Fish Audio Drama 3, speech | Not documented yet. | Native Metal or CUDA; GB10 measured. | [Release audio instructions](../CHANGELOG.md#110), [reference voices and dialogue speakers](drama-http-voices.md). `/v1/audio/speech` with [sampling controls](speech-sampling.md); checkpoint and `codec.pth` required. |
| Drama 3 pruned NVFP4, speech | `hf download plotarmordev/drama-3-NVFP4-Pruned --revision 494db08481db952f834b2241bce63fb3dfa72554 --local-dir drama-3-NVFP4-Pruned` | Native Metal; roughly 21 GiB checkpoint plus runtime state. Not a minimum-RAM or cross-backend quality claim. | Packed experts, BF16 dense/tower weights and FP32 accumulation; no conversion. Use `serve drama-3-NVFP4-Pruned --cfg-gamma 1.5 --cfg-batched` with an attested catalog voice. |

### GLM-5.3: one machine or two?

The native release does not qualify GLM-5.3 on one machine. GLM-5.3 EXLR8 `home` needs two 256 GB M5 Ultra Macs;
[its recipe](glm-5.3-exlr8.md) gives both pull and serve commands. GLM-5.3-Flash is a different model:
its native Metal setup needs two M5 Ultras. Native CUDA has recorded TP=2 and TP=3 setups, including
FP4-Local and XSFP4 on two and three GB10s. The single-256-GB-Mac
GLM-5.3-Flash command in the older recipe is for Python 0.6.6, not this release.
For the NVFP4 path, see [GLM-5.3-Flash on CUDA](recipes/glm-5.3-flash-cuda.md) and what
[the contributed run](https://github.com/ashhart/TensorFold/issues/1040) on two GB10s covered.

DeepSeek checkpoints are **not served yet** by any release binary: whole-model gate [#827](https://github.com/ashhart/TensorFold/pull/827) is still open.
MiMo's native backend is not in a release yet. Bonsai's ternary pack and Gemma 4 26B-A4B remain Python-only;
they are not native download recommendations. [Known issues](known-issues.md) lists current limits,
including Flash Next shared-session cache/performance behavior and the 27B's drafter limits.

## Models on disk

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold models
tensorfold models --json
tensorfold info TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

`pull` downloads a checkpoint into the Hugging Face cache and checks what it downloads against the Hub: large files by sha256, small files by their git blob sha1.
Large files come down in pieces over up to 16 connections, and an interrupted pull resumes with the pieces it lacks.
It refuses a checkpoint whose `model_type` has no native family, and fetches the 27B's DFlash2 drafter for `--drafter`. `models` lists the cached checkpoints with a model type it knows, and `info` shows one checkpoint's family, format, context and memory floor.
`models --json` prints the same list as one document for programs: `schema` 1, and each checkpoint's `repo`, `family` and `model_type`; `family` is a display title and `model_type` is the key to match on.
`tensorfold serve REPO` serves a cached checkpoint by its repository name.

## Model discovery

`GET /v1/models` (also `/models`) advertises each served name and alias. When the engine reports a finite
context window, each entry includes equal `context_length` and `max_model_len` integer extensions. These
are the loaded engine's effective prompt-plus-reply token limit, including any startup memory fitting;
they are not the default reply limit. An unspecified engine window omits both fields. Clients should
still handle per-request memory refusals: an advertised context is not a guarantee that every workload fits.

A model that thinks also carries an OpenRouter-style `reasoning`: `default_enabled`, `default_effort` (only
with `--reasoning-effort`), and `supported_efforts`, the levels its template names, highest first, then `none`
when the template has a thinking switch. A template that names no level leaves `supported_efforts` out. These
describe chat and Responses requests.
