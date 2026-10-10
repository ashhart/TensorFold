# Living Weights: how to use it

Living Weights teaches a running model new facts. Tell it "I like blue." and a few minutes later it answers "What colour
do I like?" with blue. The fact goes into the model's own weights, so it is still there after a restart, without
`--slide`, and on any TensorFold server you start from the same folder, Mac or NVIDIA.

It is not a prompt cache or a retrieval index: each fact it keeps becomes a small change to the model's output-projection
weights, written into the checkpoint's safetensors files. It shipped in 1.0.3 as Sliding Weights (experimental), and its
flag and API use the name `slide`.

## What you need

- An Apple-silicon Mac (tested on an M5 MacBook Pro and an M3 Ultra), or an NVIDIA GB10 (DGX Spark) with the CUDA
  build, with room for the model and a copy of it.
- The Nemotron 3.5 Lightning MLX 4-bit checkpoint, the only model it learns into today.
- TensorFold 1.0.4 or later on a Mac (1.0.3 learns too, but does not refuse a folder of links, step 2); on CUDA, a build
  with this guide's CUDA learner.

A folder that learned on either backend serves on the other.

## Quick start

### 1. Install or upgrade TensorFold

```sh
brew install ashhart/tensorfold/tensorfold    # or: brew upgrade tensorfold
```

On a GB10, install the `linux-aarch64` archive as the [RUNBOOK](../RUNBOOK.md#linux-and-cuda) shows.

### 2. Get the model and make a copy to teach

Learning rewrites files in the model folder, and nothing undoes a learned fact. Always teach a copy and keep the
original.

```sh
tensorfold pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
src=$(ls -d ~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/* | head -1)
mkdir -p ~/models
cp -cRL "$src" ~/models/nemotron-living
```

Use exactly these `cp` flags. The pulled model lives in the Hugging Face cache, whose folders hold links to shared
files: `-L` copies the real files, so learning can never write into the cached original, and `-c` makes APFS clones, so
the copy takes no extra disk until learning changes it. `--slide` refuses a folder whose files are links, so serving the
cache's own folder stops at startup instead of changing it.

On Linux, copy without `-c` (there are no APFS clones), which takes the checkpoint's full 18.5 GB:

```sh
cp -RL "$src" ~/models/nemotron-living
```

### 3. Start the server with `--slide`

```sh
tensorfold serve ~/models/nemotron-living --slide --slide-graph ~/models/nemotron-living-facts.json
```

`--slide` turns learning on. `--slide-graph` keeps the list of learned facts in a file, so it survives a restart;
without it the list lives in memory, though the weights keep what they learned either way. The server listens on
`http://127.0.0.1:8080` unless you pass `--host` or `--port`.

### 4. Teach it a fact

```sh
curl -N http://127.0.0.1:8080/v1/slide/learn \
  -H 'Content-Type: application/json' \
  -d '{"text": "I like blue."}'
```

The reply streams progress as server-sent events: `fact` when it finds a fact in your text, `learning` when it starts on
it, and `learned` when it is done. `"recalled": true` means it answered its held-out questions with the fact, and its
change is kept. `false` means none of it is kept: the lesson is taken out whole before anything is written, and `message`
says why, for example that the fact did not come back on its held-out questions or that it disturbed another answer. A fact takes
two to four minutes.

### 5. Ask

Use the normal chat API, from any client:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What colour do I like?"}]}'
```

## Teaching it more

- Several facts at once: put them in one text, as in `{"text": "I like blue. My sister is called Ana."}`. Each is
  learned in turn.
- A file: send its contents as the text, for example
  `jq -Rs '{text: .}' notes.md | curl -N http://127.0.0.1:8080/v1/slide/learn -H 'Content-Type: application/json' -d @-`.
- A web page: fetch its text first, or use omp's `/learn <url>` below, which does it for you.

The server keeps answering chat requests while it learns: learning runs when no reply is being generated. Short,
distinct facts work best.

## From omp

The omp extension in `packaging/omp/sliding-weights.ts` adds a `/learn` command. Copy it into
`~/.omp/agent/extensions/`, then select your TensorFold model in omp.

| Command | What it does |
| --- | --- |
| `/learn` or `/learn on` | Learn from this session as you work: each finished turn is sent to the model |
| `/learn I like blue.` | Learn the facts you type |
| `/learn @notes.md` | Learn a file |
| `/learn https://example.com/page` | Learn a web page's readable text |
| `/learn graph` | Open the fact graph in your browser |
| `/learn off` | Stop learning from the session |

A panel shows each fact as queued, learning, learned or missed.

## See what it learned

- Open `http://127.0.0.1:8080/slide` in a browser for the fact graph.
- `curl http://127.0.0.1:8080/v1/slide/graph` returns it as JSON.
- `curl -X DELETE http://127.0.0.1:8080/v1/slide/graph` clears the graph. The weights keep what they learned.

## Keep, move or reset what it learned

- Keep: learned facts live in the folder. Restart the server, with or without `--slide`, and they are still there.
- Move: copy the folder to another Mac or to an NVIDIA machine and serve it there. It answers the same way.
- Reset: delete the copy and clone a fresh one from the original (step 2). There is no per-fact undo.

## When something goes wrong

| You see | What it means |
| --- | --- |
| `501 this engine does not learn: serve its model with --slide` | Start the server with `--slide`, on a Mac or a GB10 |
| `503 the engine cannot take a learn request now` | Another learn request is running; send yours when it ends |
| `400 text is required` | The JSON body needs a `"text"` field |
| `503 a lesson that did not come back could not be taken out: restart the server` | The engine refused to take a failed lesson out, so nothing of that request was saved and the server learns nothing more until it restarts. Until then, answers may still show the failed lesson |
| `learned` with `"recalled": false` | Nothing of the fact was kept, and `message` says why: for example it did not come back on its held-out questions, it disturbed a related question, or there were too few clean answers to learn from |
| `--slide rewrites the model's own files, and … is a link to data another file shares` at startup | The folder holds links into the Hugging Face cache (or hard links); make the copy with `cp -cRL` (Linux: `cp -RL`) as in step 2 |
| `--slide learns on a GB10 (sm_121) so far` at startup | CUDA learning is qualified on a GB10 only; serve this GPU without `--slide`, or teach on a Mac or a GB10 |
| `--slide runs a prompt one chunk at a time` at startup | Serve it without `--segments` or `TF_CUDA_SEGMENTS` |
| The answer has not changed yet | Wait for the `learned` event before asking |

## What to expect

Living Weights is experimental. Measured on Nemotron 3.5 Lightning:

- After learning "I like blue." on a Mac, it answers "Do you remember what colour I like?", "What's my favourite
  colour?" and "What colour do I like?" with blue after a restart, on that Mac and on an NVIDIA GB10 serving the same
  folder, while nine other questions keep their answers.
- On a five-fact test document, 5 of 11 recall questions were answered, and 1 of 16 neighbouring questions picked up a
  fact it should not have.
- On a GB10, a training step takes 48 to 436 ms and a fact two to four minutes. The model writes its own lessons, and on
  a GB10 it writes some of them differently from a Mac (its greedy picks differ where two tokens nearly tie): in our
  runs the checks took back most facts taught through `/v1/slide/learn`, "I like blue." among them. While `--slide`
  serves, a decoded token takes about 2% longer with no lessons kept, 6% with one and 11% with four.

Before keeping a fact, it checks related questions it never trained on, and checks again after its last training steps.
If one of them starts answering with the new fact, or the fact itself does not come back, the whole lesson is taken out
before anything is written. Even so, it remembers short, distinct facts best, can miss facts in long documents, and
a fact can occasionally bleed into a related question. Keep the original model for anything that matters.
