# Living Weights: how to use it

Living Weights teaches a running model new facts. Tell it "I like blue." and a few minutes later it answers "What colour
do I like?" with blue. The fact goes into the model's own weights, so it is still there after a restart, without
`--slide`, and on any TensorFold server you start from the same folder, Mac or NVIDIA.

It is not a prompt cache, a retrieval index or an adapter: each fact it keeps becomes about ten new neurons inside the
model, written into the checkpoint's own safetensors files. Nothing the model already had is changed. It shipped in 1.0.3
as Sliding Weights (experimental), and its flag and API use the name `slide`.

## What you need

- An Apple-silicon Mac (tested on an M5 Ultra and an M3 Ultra), or an NVIDIA GB10 (DGX Spark) with the CUDA build,
  with room for the model and a copy of it. On a GB10, `--slide` also keeps about 26 GiB of GPU memory for its
  trainer; on a Mac the trainer works on the model's own weights.
- The Nemotron 3.5 Lightning MLX 4-bit checkpoint, the only model it learns into today.
- A TensorFold build with the pocket learner described here (the release after 1.0.6). A folder taught by 1.0.3 to
  1.0.6 still serves and answers from what it learned, but `--slide` refuses it: those releases saved lessons into
  the shared experts the pocket widens. Teach a fresh copy instead.

A folder that learned on either backend serves on the other.

## Quick start

### 1. Install or upgrade TensorFold

```sh
brew install ashhart/tensorfold/tensorfold    # or: brew upgrade tensorfold
```

On a GB10, install the `linux-aarch64` archive as the [RUNBOOK](../RUNBOOK.md#linux-and-cuda) shows.

### 2. Get the model and make a copy to teach

Learning rewrites files in the model folder, and so does deleting a fact. The server keeps the folder's original
files before its first lesson or delete, so `tensorfold slide rollback` can undo them all at once
([Undo what it learned](#undo-what-it-learned)), but teach a copy and keep the pulled model as it is.

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

### 3. Make a key and start the server with `--slide`

```sh
(umask 077; openssl rand -hex 24 > ~/models/slide-key)
tensorfold serve ~/models/nemotron-living --slide --api-key-file ~/models/slide-key \
  --slide-graph ~/models/nemotron-living-facts.json
```

`--slide` turns learning on, and it starts only with a key, because any client that reaches the server could otherwise
rewrite the model's weights. Give it `--api-key-file FILE` (one key a line, readable only by you), `--api-key KEY` or
`TENSORFOLD_API_KEY`. The server reads the key before it loads the model. It stops with one line naming the cause when
there is no key, when the key file is missing, empty or readable by other users (`chmod 600` it), or when a key has a
space in it. Every request then sends the key, as the commands below do; [API keys](api.md#api-keys) has the rest.

`--slide-graph` keeps the list of learned facts in a file, so it survives a restart; without it the list lives in
memory, though the weights keep what they learned either way. The server listens on `http://127.0.0.1:8080` unless you
pass `--host` or `--port`.

### 4. Teach it a fact

```sh
curl -N http://127.0.0.1:8080/v1/slide/learn \
  -H "Authorization: Bearer $(cat ~/models/slide-key)" \
  -H 'Content-Type: application/json' \
  -d '{"text": "I like blue."}'
```

The reply streams progress as server-sent events: `fact` when it finds a fact in your text, `learning` when it starts on
it, and `learned` when it is done. `"recalled": true` means it answered its held-out questions with the fact, and its
change is kept. `false` means none of it is kept: the lesson is taken out whole before anything is written, and
`message` says why, for example that the fact did not come back on its held-out questions or that it disturbed another
answer. A fact takes 1.5 to 3 minutes. The folder's first lesson starts by keeping its original files, which adds a
few seconds on a Mac.

### 5. Ask

Use the normal chat API, from any client:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $(cat ~/models/slide-key)" \
  -H 'Content-Type: application/json' -d '{"messages": [{"role": "user", "content": "What colour do I like?"}]}'
```

## How it works

Every MoE layer's shared expert gets 64 extra neurons, zero until a fact uses them. The first save widens the shared
expert's tensors in the shards (3712 to 3776 neurons) and records `tensorfold_learned_neurons` in `config.json`; any MLX
loader reads the folder as a slightly wider model. A fact takes one neuron at each of about ten layers.

- A fact about you ("I like blue.") fires its neurons at the first row of the reply to a question about it.
- A fact about someone or something ("My sister is called Ana.", "Quillmoor Labs was founded in Sheffield.") fires them
  at the last token of the subject where a question names it, in the middle layers, where models keep what they know of a
  subject. A question has to name the subject: "When does the Quillmoor team deploy?" reaches the fact, "When does the
  team deploy?" does not.
- Each neuron's input weights are fitted to fire on the fact's questions and stay at zero everywhere else. Its output is
  then fitted through the model so the answers come back, with thinking on too, while other answers that name the same
  subject stay as they were.
- The model writes the lesson itself: a dozen questions, its own answers with the fact in front of it, the same answers
  under client system messages and after an unrelated exchange, and near misses it must leave alone.

Before keeping a fact, the server asks held-out wordings it never trained on, with thinking off and on, plus near misses
and unrelated prompts. A fact that does not come back, or changes an answer it should not, is taken out and leaves
nothing behind. An earlier fact's neuron that the new lesson would disturb is refitted only away from its own rows.

## Teaching it more

- Several facts at once: put them in one text, as in `{"text": "I like blue. My sister is called Ana."}`. Each is
  learned in turn.
- A file: send its contents as the text, for example:

  ```sh
  jq -Rs '{text: .}' notes.md | curl -N http://127.0.0.1:8080/v1/slide/learn \
    -H "Authorization: Bearer $(cat ~/models/slide-key)" -H 'Content-Type: application/json' -d @-
  ```

- A web page: fetch its text first, or use omp's `/learn <url>` below, which does it for you.

The server keeps answering chat requests while it learns: learning runs when no reply is being generated. Short,
distinct facts work best.

## From omp

The omp extension in `packaging/omp/sliding-weights.ts` adds a `/learn` command. Copy it into
`~/.omp/agent/extensions/`, then select your TensorFold model in omp and give that model the server's key as its API
key; `/learn` sends it.

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

Open `http://127.0.0.1:8080/slide` in a browser while the server runs. It draws the model in 3D and updates live:

- The model as a tower of its 52 layers. Each MoE layer is a ring carrying its 64 pocket neurons, and a neuron glows in
  the colour of the fact that owns it.
- The checkpoint's shard files beside the tower, each linked to the layers it holds, with how full each pocket is.
  A save shows which files it wrote, how many bytes and how long it took.
- The facts and the people and things they name, as a graph around the tower: queued, learning, learned or missed.
- The lesson running now, phase by phase: writing, reading every layer, fitting the neurons' inputs, fitting their
  outputs (with the loss after each step), checking each question, then kept or taken out.

Drag to turn, scroll or pinch to zoom, and click a fact for its neurons, the questions it was checked with and its
rounds. Phones get a bottom sheet. Without WebGL2, or with `/slide?flat`, the page draws a flat graph from the same
data. The page asks for the server's key once and keeps it for that browser tab.

- `curl -H "Authorization: Bearer $(cat ~/models/slide-key)" http://127.0.0.1:8080/v1/slide/graph`
  returns the same data as JSON: each fact's state, neurons (layer, index and key), questions and rounds, and the
  model's layers, pocket and shards.
- The same with `-X DELETE` clears the graph. The weights keep what they learned. To take one fact out of the weights,
  see [Delete one fact](#delete-one-fact).

## Delete one fact

```sh
curl -N -X DELETE http://127.0.0.1:8080/v1/slide/fact/2 \
  -H "Authorization: Bearer $(cat ~/models/slide-key)" \
  -H 'Content-Type: application/json' -d '{"reason": "my sister changed her name"}'
```

Take the fact's `id` from `GET /v1/slide/graph`. The body is optional; its `reason` goes into the graph. Like a
lesson, the reply streams server-sent events. With a few facts learned it takes seconds; every learned fact's
questions are asked twice, so it grows with them:

1. `fact`: the fact and its state.
2. `checking` with `"when": "before"`: every learned fact's held-out questions are asked as its lesson checked them,
   with thinking off and on, and the wordings its lesson never trained on with thinking off.
3. `zeroed`: the input weights of the fact's neurons are set to zero in the running model, so they never fire, and
   their layers are rounded to the 4-bit values the shards will hold, so the checks after read what will be written.
4. `checking` with `"when": "after"`, then one `checked` event a fact, saying whether it came back before and after.
5. `forgot` with `"deleted": true`: every other fact still comes back, and the deleted fact's questions no longer give
   its answer. The change is written into the shards, the graph marks the fact `deleted` with the time and reason, and
   its neurons at the top of each layer's pocket are free for later lessons.
6. Or `forgot` with `"deleted": false`: the delete did not complete, because a check failed or a write failed (below).
   For a failed check, the neurons are put back exactly as they were, nothing is written, and `message` names the
   fact and question that failed and what the model said.

`forgot` also carries `state`, the fact's state in the graph file when the request ends. A delete writes two things,
the shards and the graph file, in this order, so that together they never claim more than is on disk:

1. The graph file marks the fact `deleting`.
2. The shards are written.
3. The graph file marks the fact `deleted`.

`"deleted": true` means all three were written. When one fails, `forgot` says `"deleted": false` and `message` says
what happened:

| What failed | What is on disk | `state` |
| --- | --- | --- |
| The graph file, before the shards | Nothing changed: the delete was put back | `learned` |
| The shards | The delete was put back and the shards written again as they were, so nothing changed | `learned` |
| The shards, and writing them back too | The folder may hold part of the delete | `deleting` |
| The graph file, after the shards | The shards hold the delete; the graph file does not say so yet | `deleting` |

A put-back restores the pocket in memory at once, and on the GPU once every layer's upload succeeds. If one does not
(on CUDA, an upload or a graph capture that fails), the server keeps what it needs to put it back, and refuses every
reply until it lands: a reply says "a Living Weights delete could not be put back on the device" (one that was
already waiting ends with the same words, unanswered), and a learn or delete request answers 503. Each learn or
delete request tries the put-back again first, and replies come back once it lands. A restart reloads the folder,
which a put-back never wrote.

A fact left `deleting`, by one of these or by a crash, is settled by sending the delete again, after a restart too.
If its neurons are silent already, an earlier delete reached the shards, and the graph file now records it. If not,
the delete runs again, checks and all. A fact left `deleting` is not checked as a kept fact meanwhile, and a lesson
placed on one of its neurons takes that neuron over. Without `--slide-graph` the graph lives in memory only, and only
the shards are written.

A kept fact passes when each answer that came back before the delete still does, and at least half of its other
wordings do if they did before. The deleted fact passes when none of its questions gives its answer, or when the model
as shipped gives that answer too (a likely guess, such as a common colour, is no trace of the fact).

- A fact a newer one replaced has its neurons at zero already: deleting it answers "nothing to delete" and changes
  nothing. Deleting the newer fact does not bring back the one it replaced.
- A fact learned before this release kept its questions but not their answers: its check holds each question to the
  answer the model gave just before the delete.
- A delete silences a fact; it does not erase it. A deleted neuron's output weights stay in the shards, unused, until
  a later lesson takes the neuron. Zeroing them too would round again the 4-bit groups that other facts' neurons in
  the same layer share, and move their answers.
- A model with a fact deleted is close to the same model never taught it, but not the same: the lessons after it
  were learned with it in place.
- Only freed neurons at the top of a layer's pocket go to the next lesson. One below another fact's neuron stays
  silent, and taken, until that fact is deleted too.
- While a delete runs, chat replies already see the fact gone; a delete that is put back brings it back once the GPU
  takes the put-back (above).
- Before its first write, a delete keeps the folder's original files as a lesson does, so a rollback undoes deletes
  along with lessons.
- A delete changes the weights only. Recall's records and saved conversations (`--recall`, `--learn-conversations`)
  keep what the conversations said, the fact's words included.

## Keep, move or reset what it learned

- Keep: learned facts live in the folder. Restart the server, with or without `--slide`, and they are still there.
- Move: copy the folder to another Mac or to an NVIDIA machine and serve it there. It answers the same way. A copy of
  the whole folder takes its snapshot (`.tensorfold-original/`) along; leave that folder out if you will not roll the
  copy back.
- Delete one fact: `DELETE /v1/slide/fact/{id}`, as [above](#delete-one-fact).
- Reset: stop the server and run `tensorfold slide rollback ~/models/nemotron-living`, below, or delete the copy and
  clone a fresh one from the original (step 2). A rollback undoes every lesson and delete since the snapshot.

## Undo what it learned

Before a folder's first lesson, the server keeps every file a lesson can change in a hidden folder inside it,
`.tensorfold-original/`: each `*.safetensors` file, `model.safetensors.index.json` and `config.json`. On a Mac, and
on Btrfs or XFS, each kept file is a clone that shares the original's blocks, which takes no disk at first. The first
lesson writes new shard files, and from then on the snapshot keeps the original blocks on disk; a copy cloned from the
pulled model with `cp -c`, as in step 2, shares them with the pulled model, so they cost nothing more. On other file
systems, such as ext4, each kept file is a plain copy, so the snapshot takes the model's size again (18.5 GB for
Nemotron 3.5 Lightning); the server checks the disk has room for the copies and the largest file again first, and
refuses the lesson with HTTP 507 and the space it needs when it does not. The snapshot's `SHA256SUMS` lists every kept
file's SHA-256 (`shasum -a 256 -c SHA256SUMS` inside the folder checks it by hand). The server checks the kept files
against it as it makes the snapshot, and removes the snapshot and refuses the lesson if one differs.

Later lessons reuse the snapshot, which is never refreshed. Each server checks it against `SHA256SUMS` at its own
first lesson, and refuses to teach while it does not match ("the way back is broken"). Model loaders read only the
folder's own files, never a hidden folder, and `tensorfold info` leaves it out of the model's size. A folder taught
before it had a snapshot (by 1.0.7 to 1.0.11) gets one at its next lesson, holding the folder as it is then; the log
says so, and a rollback returns there, not to the model as pulled.

To undo every lesson, stop the server and run:

```sh
tensorfold slide rollback ~/models/nemotron-living
```

The rollback:

- refuses while any TensorFold server has the folder open. Each server holds its folder while it runs (shared, or
  exclusive with `--slide`), and a server in turn refuses to start while a `--slide` server or a rollback holds it.
- checks every kept file against `SHA256SUMS` first, and changes nothing when one does not match.
- clones every kept file beside its own where the file system makes clones, and checks the disk has room for the
  largest file it must copy instead, before it changes any file: a refusal for space, or a clone that fails, leaves
  the folder as it was. It then puts each file back (a copy is made beside the file and checked against its SHA-256
  before it is renamed over it) and checks every file's SHA-256 against the manifest.
- only then removes what the lessons added: lesson files (`living-weights-00001.safetensors` and on, or an older
  `living-weights.safetensors`), writes a lesson left cut short (`NAME.tmp` beside a kept or lesson file), and the
  fact graph files that `--slide-graph` named while the folder learned. A fact graph can live outside the folder; the
  rollback removes it all the same, since it describes weights that are gone, and prints every path it removes.
- leaves everything else: files you added after the snapshot, files a lesson never writes (the tokenizer, templates,
  a download's `.cache`), and the snapshot itself, so you can teach the folder again and roll back again.

It prints how many files it put back, their size, how many were clones and how long it took. When a put-back file does
not match, it says so and leaves the lessons' files where they are; run it again.

## Who can teach it

A key stops clients that do not have it from teaching the model. It does not stop someone who has the key from
teaching it something wrong, and a learned fact stays until you delete it or roll the folder back, which undoes every
lesson since the snapshot. Give the key only to clients you trust, and keep the original model. The folder hold works
between TensorFold processes only: other tools that write files (`cp`, `rsync`, an editor) do not take it, so leave
the folder's files alone while a server or a rollback has it.

## When something goes wrong

| You see | What it means |
| --- | --- |
| `501 this engine does not learn: serve its model with --slide` | Start the server with `--slide`, on a Mac or a GB10 |
| `503 the engine cannot take a learn request now` | Another learn request is running; send yours when it ends |
| `503 a learn or delete request is running` | A delete does not wait for another learn or delete: send it when that one ends |
| `forgot` with `"deleted": false` | A check or a write failed; `message` says which, and `state` what the graph file holds (see Delete one fact) |
| `a Living Weights delete could not be put back on the device` on a reply | The GPU did not take a delete's put-back: send the delete again to retry it, or restart the server |
| `400 text is required` | The JSON body needs a `"text"` field |
| `learned` with `"recalled": false` | Nothing of the fact was kept (its neurons were taken out), and `message` says why: for example it did not come back on its held-out questions, it kept answering questions about something else, or the model wrote too few clean questions to learn from |
| `--slide rewrites the model's own files, and … is a link to data another file shares` at startup | The folder holds links into the Hugging Face cache (or hard links); make the copy with `cp -cRL` (Linux: `cp -RL`) as in step 2 |
| `--slide learns on a GB10 (sm_121) so far` at startup | CUDA learning is qualified on a GB10 only; serve this GPU without `--slide`, or teach on a Mac or a GB10 |
| `--slide runs a prompt one chunk at a time` at startup | Serve it without `--segments` or `TF_CUDA_SEGMENTS` |
| `--slide adds learned neurons beside every shared expert, and … is unquantized` at startup | The folder was taught by TensorFold 1.0.3 to 1.0.6; serve it without `--slide`, or teach a fresh copy |
| `--slide lets any client that reaches this server rewrite the weights in …` at startup | Add a key: `--api-key-file FILE`, `--api-key KEY` or `TENSORFOLD_API_KEY` (step 3) |
| `--slide needs a key this server can read, and --api-key-file … gives none: …` at startup | Make the file (one key a line), `chmod 600` it, and check its path; the end of the line gives the reason |
| `--slide needs a key this server accepts, and --api-key or TENSORFOLD_API_KEY gives one it refuses: …` at startup | A key has a space or a control character in it; use one without |
| `` … is held by a --slide server or a `tensorfold slide rollback` … `` at startup | Another server is teaching the folder, or a rollback is running; stop that server or let the rollback finish |
| `--slide rewrites the files in …, and another TensorFold process has that folder open` at startup | Stop the other server, or teach another copy |
| `… cannot be locked, so a rollback could rewrite it while --slide teaches it` at startup | The folder is on a file system without file locks (some network shares); teach a copy on a local disk |
| `… is on a file system without clones, and copying its original files needs … GB free` | At a lesson (HTTP 507), the snapshot does not fit; from `slide rollback`, the copies back do not fit and nothing changed. Free that much space, or teach a copy on APFS, Btrfs or XFS |
| `500 … the way back is broken …` | The snapshot does not match its manifest; move `.tensorfold-original` out of the folder to keep the folder as it is now, then teach again (a new snapshot starts from the folder as it is) |
| `a TensorFold server or another rollback has … open` from `slide rollback` | Stop the server first |
| `… has no .tensorfold-original snapshot …` from `slide rollback` | The folder never learned with a release that keeps one; reset it from the original model |
| `… does not match its line in SHA256SUMS, so … was left as it is` from `slide rollback` | The snapshot is damaged and the folder is unchanged; reset it from the original model |
| The answer has not changed yet | Wait for the `learned` event before asking |

## What to expect

Living Weights is experimental. Measured on Nemotron 3.5 Lightning (MLX 4-bit) on an M5 Ultra, an M3 Ultra and a GB10:

- **Facts you tell it.** "I like blue.", "My sister is called Ana." and "My dog is called Ginger.", taught one after
  another, each came back on both test questions on all three machines, with a client system message in front too,
  and after a plain restart. Unrelated questions kept their answers and no fact leaked into questions it does not
  answer. With thinking on, a fact sometimes misses one wording ("Do you remember what colour I like?"): the model
  recalls the fact in its reasoning and stops there without a reply, or says it keeps no memory of past
  conversations.
- **Short documents.** From a three-sentence handbook (six facts) it answered 3 to 6 of 6 questions across six runs on
  both backends, with no drift. In three runs the founder's name also answered a question about another role ("Who is
  the head of marketing?"), and in one the team chat tool answered a question about video calls. A fact it cannot
  learn is left out, and `learned` says why.
- **Long documents about one subject are the weak spot.** From a 21-fact company handbook it kept about half the facts
  and answered 6 of 16 questions, and with ten facts about the company kept, its answers about things the handbook does
  not say got worse, most with thinking on. Teach a few facts per subject, and always teach a copy.
- **Time.** A fact takes about 1.5 to 2.5 minutes on an M5 Ultra or M3 Ultra and 2.5 to 3 minutes on a GB10, a good
  part of it the model writing its own lesson. A document takes that per fact.
- **Memory.** On a Mac the trainer works on the model's own weights. On a GB10, `--slide` keeps about 26 GiB for its
  trainer, which is ready a few seconds after the server starts.
- **Undo.** On an M3 Ultra, the snapshot of Nemotron 3.5 Lightning (7 files, 18.5 GB) took 5.3 s as clones, a new
  server's check of it 2.6 s, and a rollback 5.6 to 5.7 s. On a GB10's ext4, with a build for the GB10's own CPU, the
  snapshot took 35 s as copies and 18.5 GB of disk, a check 2.6 s, and a rollback, which copies one file at a time, 58
  to 91 s; the release's Linux build computes SHA-256 about five times slower (0.39 GB/s against 2.1 GB/s), so expect
  its checks to take longer. Rolled back, the folder matched the untaught copy byte for byte and, served again, gave
  its replies token for token.
- **Serving.** Without learned facts the server gives the same replies, token for token, at the same speed as without
  Living Weights. A folder holding many learned facts decodes slower on a Mac (about 16% with 21 facts in the pocket),
  and drafted replies still equal plain ones.

While it learns, near misses and free writing about the fact's subject are trained to stay as they were. Before
keeping a fact, it checks wordings it never trained on, with thinking off and on, near misses and unrelated prompts.
If the fact does not come back, or another answer changes, the whole lesson is taken out before anything is written.
It also asks a few questions about the user that nothing told it, such as a cat's name or the user's own name; a lesson
that makes it answer one with something made up ("Your cat's name is Whiskers.") is taken out too. Keep the original
model for anything that matters.
