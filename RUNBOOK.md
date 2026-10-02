# Installation runbook

Use the backend that matches the host. TensorFold needs Python 3.11 or newer, Apple Silicon for MLX,
or a supported NVIDIA CUDA environment. Choose one checkpoint from the [model table](README.md#models)
and check disk space and available memory before downloading it.

## Apple Silicon

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold --version
tensorfold models
```

Choose a model explicitly. This example uses Nemotron with its included MTP head:

```bash
tensorfold info Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold pull Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --name local-model --context 8192
```

`info` reads configuration only. `pull` downloads weights; `serve` completes a missing download.
The server prints whether Nemotron's MTP head is active. A failed row check disables drafting without
changing the serial reference; keep MLX within the package requirements.

For Qwen3.8-27B, optionally pull `z-lab/Qwen3.8-27B-DFlash2` too. M1 through M4 use the 4-bit row-exact
simdgroup decoder; the M5 tensor-unit path also reads the documented lower and higher affine widths.
Model-specific requirements are in the [recipes](docs/recipes/README.md).

## Check the endpoint

Leave the server running and use another terminal:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128}'
```

Use the ID returned by `/v1/models` if the server was started without `--name local-model`.
The client base URL is `http://127.0.0.1:8080/v1`. Reasoning can appear separately from the answer.
See [API fields](docs/api.md) for streaming and tool calls.

<a id="dgx-spark"></a>

## NVIDIA GPUs

Start NVIDIA's container, then install and serve inside it:

```bash
nvidia-smi
docker run -it --gpus all --ipc=host --network host nvcr.io/nvidia/pytorch:26.07-py3
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --name local-model --host 0.0.0.0 --port 8080
```

The first start compiles kernels. Container removal discards an unpersisted installation and cache;
use a retained container or configure persistent storage when downloads should survive removal.
There is no `tensorfold[cuda]` extra. Qwen3.8-27B, Flash Next and Nemotron have one- and two-rank CUDA
engines; GLM requires two ranks. Nemotron CUDA uses its included MTP head and 4-bit/group-64 weights.
Qwen3.8-27B CUDA requires DFlash2 unless `--no-drafts` selects the serial reference. Flash Next also reads
the NVFP4 (ModelOpt FP4) checkpoint `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` as it ships — its PLE layer
included, whose table ships as BF16 rows with no per-shard `scales`, a layout the reader takes as it is.

CUDA `--parallel auto` serves one request at a time. To share rounds, set `--parallel N` greater than
one for Qwen3.8-27B on one or two ranks, or Flash Next on one rank. Pass the same N on both Qwen ranks.
Flash Next rejects parallel two-rank execution; GLM and Nemotron CUDA keep serial request scheduling.

For two ranks, start a container on each host with network devices and locked-memory support:

```bash
docker run -it --gpus all --ipc=host --network host --device /dev/infiniband \
  --ulimit memlock=-1 --cap-add IPC_LOCK nvcr.io/nvidia/pytorch:26.07-py3
```

Install and pull the same checkpoint and drafter on both ranks. Configure `NCCL_SOCKET_IFNAME` and
`NCCL_IB_HCA` for the actual link if automatic selection fails. Two DGX Sparks on their direct cable expose two
RoCE devices for the one port; list both, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`. With `rocep1s0f1` alone, the
27B read a 7k-token prompt about 8% slower in our runs, and decoded at the same speed. Start rank 1 first, then
rank 0:

```bash
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name local-model --host 0.0.0.0
```

Replace the documentation address with rank 0's reachable address. Both ranks must agree on context and
drafting settings. The default rendezvous port is 29551. The rendezvous port and the link between the ranks
are not authenticated: keep them on a private link, or firewall the port to the peer. GLM requires two CUDA
ranks; Flash Next can use one or two and needs `--no-drafts` when its checkpoint lacks an MTP head.

### RTX cards without Docker

On an RTX 40 or 50 series card or an RTX PRO Blackwell, pip alone is enough. torch comes from PyPI and the CUDA
compiler from NVIDIA's own wheels, all in a virtual environment, with no root and no container:

```bash
python3 -m venv ~/tf-venv && . ~/tf-venv/bin/activate
python -m pip install torch ninja "cuda-toolkit[nvcc,cccl]==13.0.*"
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --name local-model --host 127.0.0.1 --port 8080
```

Match the compiler wheel to torch's CUDA version, which `python -c "import torch; print(torch.version.cuda)"`
prints; PyPI's torch 2.14 uses CUDA 13.0. The first start compiles the kernels and names the compiler it found. On
a card other jobs share, set `TENSORFOLD_MEMORY_RESERVE_GIB` to the memory TensorFold should leave free and pass an
explicit `--context`.

<a id="win-nvidia"></a>

## Windows with an NVIDIA card

Native Windows is experimental: its host layer is in, but it has not served a request on a Windows PC yet. It runs
one GPU a process, since CUDA on Windows has no NCCL for two ranks; it reads weights through pinned buffers where
Linux uses O_DIRECT, sizes memory with Windows' own API, and prints every thread's stack on Ctrl+Break. GPUs below
compute capability 8.9 are refused at startup. WSL2 runs the Linux engine instead: inside Ubuntu, follow
[RTX cards without Docker](#rtx-cards-without-docker). We have not run it under WSL2 yet either.

## Memory and context

Omit `--context` on MLX to fit the default window to the model and memory budget, then inspect the
reported capacity. CUDA targets the affordable native capacity for Qwen, 2,051 tokens for GLM,
and 16,384 for Nemotron; the capacity estimate can lower these defaults. On CUDA, `--context 0` targets the affordable native capacity; on MLX it
removes the metadata cap while memory admission still applies. A positive context that cannot fit
is refused at startup.

On MLX, `TENSORFOLD_MEMORY_LIMIT_GB` sets the process budget in GiB in place of the default 70% of RAM.
It can raise or lower the budget, within physical RAM and the GPU's recommended working set:

```bash
TENSORFOLD_MEMORY_LIMIT_GB=110 tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
```

On a 128 GiB M4 Max this gives 110 GiB to the process and 107 GiB to MLX after the 3 GiB reserve.
The same budget reaches concurrent admission; context and request memory checks still apply.

On CUDA, a discrete card's admission budget is its own free memory; on a unified GPU it is the host's
available memory less a floor of a tenth of RAM, at least 4 GiB. `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` caps that
grant from above in GiB, an absolute budget like the MLX one; free memory still caps it:

```bash
TENSORFOLD_CUDA_MEMORY_LIMIT_GB=31 tensorfold serve nvidia/Qwen3.8-27B-NVFP4
```

A budget close to a shared pool can end requests with CUDA errors mid-reply, which is why a unified GPU keeps
its floor; `TENSORFOLD_MEMORY_RESERVE_GIB` moves that floor. A discrete card's host need is its loading
buffers, which startup checks on its own.
Requested replies need cache space too. Reduce context, reply length, retained prefixes on MLX, or
checkpoint size after a memory refusal. The MLX process budget reserves 3 GiB outside the allocator.
Release-qualified memory and speed results are TBD [release-0.3.5]; see the
[memory-class table](README.md#context-and-memory). Do not assume model-file size is the whole process
footprint. Prompt caching uses token-derived message boundaries; `--prefill-grid` is no longer an option.

## Updating

Run `tensorfold update --check`, then `tensorfold update` when ready, and restart the server.
An editable checkout must be clean and able to fast-forward; run `python -m pip install -e .` afterwards
to refresh installed metadata and dependencies. Update inside the container when serving CUDA.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Command not found | Activate the installation environment |
| Download failure | Repository ID, access and free disk space |
| `info` succeeds but `serve` downloads | `info` reads only configuration |
| Rejected checkpoint | Quantization, model family and draft-head requirements |
| Client cannot connect | Server process, `/health`, base URL and model ID |
| Two-rank startup waits | Link reachability, rendezvous port, NCCL devices and matching settings |
| CUDA start stops after `loading …`, GPU idle | `kill -USR1 <pid>` prints every thread's Python stack. A wait in the extension build is a build lock, whose path the start log names: when no other build is running, stop the start, delete the lock and start again |

Unsupported architectures or formats need a family implementation. See [adding a family](docs/recipes/adding-a-family.md)
or [adding a CUDA family](docs/recipes/adding-a-cuda-family.md); forcing an unsupported checkpoint to load
is not an installation fix.
