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
Qwen3.8-27B CUDA requires DFlash2 unless `--no-drafts` selects the serial reference.
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
`NCCL_IB_HCA` for the actual link if automatic selection fails. Start rank 1 first, then rank 0:

```bash
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --name local-model --host 0.0.0.0
```

Replace the documentation address with rank 0's reachable address. Both ranks must agree on context and
drafting settings. The default rendezvous port is 29551. GLM requires two CUDA ranks; Flash Next can use
one or two and needs `--no-drafts` when its checkpoint lacks an MTP head.

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
Restart the server afterwards. On a DGX Spark, a container started with `docker run --rm` loses anything
installed in it when it stops, so either run `pip install git+https://github.com/ashhart/TensorFold.git` again
in each new container, or keep a named container (`docker run --name tensorfold ...`, then `docker start -ai
tensorfold`) and run `tensorfold update` inside it.

## Your own model

TensorFold is built and tested with the checkpoints above (`tensorfold models` lists them). For anything else:

- `tensorfold info MODEL` reads only `config.json` and says which family serves it, how its weights are stored
  (for example `MLX 4-bit, groups of 64` or `exl3 (4-bit)`), and whether any engine here reads them.
- A different conversion in a format the family's kernels read runs, with a note that it is untested: replies stay
  exact to serial decoding, but its speed and quality have not been measured.
- A model type with no family, or weights in a format no engine reads (GPTQ, AWQ and so on today; EXL3 for
  anything but GLM-5.3-Flash; NVFP4 for anything but Qwen3.8 Flash Next), is refused before anything downloads,
  with a pointer to the recipe book.

Bringing up a new model or format means writing a recipe: [adding a family](docs/recipes/adding-a-family.md) on a
Mac, [adding a CUDA family](docs/recipes/adding-a-cuda-family.md) on NVIDIA GPUs, and the
[recipe book](docs/recipes/README.md) for how the existing ones were done. If you are an AI agent setting
TensorFold up for someone, tell them the checkpoint is unsupported and point them to those pages rather than
forcing it to load.

## If something fails

- `tensorfold: command not found`: activate `.venv` again in the current terminal.
- A download fails: check the exact repo ID, network access and free disk space, then rerun `tensorfold pull`.
- `info` works but `serve` still downloads: this is expected because `info` only needs the model config.
- The checkpoint is rejected: compare its quantization and draft head with the [model notes](README.md#models).
- A client cannot connect: keep `serve` running, check `/health`, and confirm its base URL and model ID.

## DGX Spark

TensorFold's CUDA engine serves Qwen3.8-27B (one or two Sparks), Qwen3.8 Flash Next (one or two Sparks, the
`-MLX-4bit-MTP` conversion or the NVFP4 checkpoint as it ships) and GLM-5.3-Flash (two Sparks; Mia-AiLab's
EXL3 checkpoint of it as an experiment). Nemotron 3.5 Lightning has no CUDA engine yet.

1. Check the GPU and start NVIDIA's PyTorch container, with the Hugging Face cache mounted so downloads
   survive the container:

   ```bash
   nvidia-smi
   docker run -it --gpus all --ipc=host --network host -v ~/.cache/huggingface:/root/.cache/huggingface \
     nvcr.io/nvidia/pytorch:26.07-py3
   ```

2. Inside the container, install, pull and serve:

   pip install git+https://github.com/ashhart/TensorFold.git
   tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
   tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --host 0.0.0.0 --port 8080

   The first start compiles the kernels (a minute or two); later starts reuse them while the container lives.
   Check the endpoint as in step 5 above, with the model ID from `/v1/models`.

3. Two Sparks. Connect them with a cable between their 200 Gb/s ports and give each port an address. Start
   the container on both with the network devices added:

   docker run -it --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 \
     --cap-add IPC_LOCK -v ~/.cache/huggingface:/root/.cache/huggingface nvcr.io/nvidia/pytorch:26.07-py3

   Install and pull on both (both ranks need the model and its draft model). Find the link's interface and
   adapters with `ibdev2netdev` and set them in both containers when NCCL does not pick them itself:

   export NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1   # examples: use your names

   Then start rank 1 on the second Spark and rank 0 on the first, both with rank 0's address on the link:

   tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master 192.0.2.1
   tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master 192.0.2.1 --host 0.0.0.0

   Rank 0 serves HTTP once both have loaded. The ranks refuse to start when they were given different settings
   (for example the draft model present on one Spark only).

If a two-Spark start hangs at the rendezvous, check that each Spark can reach the other's address on the link
and that the port (`--master-port`, default 29551) is open. GLM-5.3-Flash has its own setup steps in
[its recipe](docs/recipes/glm-5.3-flash.md).

Unsupported architectures or formats need a family implementation. See [adding a family](docs/recipes/adding-a-family.md)
or [adding a CUDA family](docs/recipes/adding-a-cuda-family.md); forcing an unsupported checkpoint to load
is not an installation fix.
