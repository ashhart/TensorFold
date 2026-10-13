# Recall

`tensorfold serve MODEL --recall` keeps a record of the token ids the server has read and written, and drafts from it.
When a reply starts to repeat text the server has seen before, the lanes verify the earlier continuation in a few rounds instead of drafting it token by token.
A regenerated or repeated request copies its earlier reply, an agent copies code and tool calls it wrote in an earlier session, and boilerplate copies itself.

The target model verifies every copied token, so replies are the same tokens with Recall on, off or read-only.
Recall is off unless a flag turns it on.

Recall is not the prompt cache and not `--learn`.
Those keep the model's state after a prompt so a later request can skip reading it again.
Recall keeps only token ids, and it changes how fast a reply is written, not how a prompt is read.

## Turning it on

| Flag | Meaning |
| --- | --- |
| `--recall` | Keep the record and draft from it. |
| `--recall-dir DIR` | The record's own directory. Without it the record lives in `~/.cache/tensorfold/recall`, in a directory named by the tokenizer. Implies `--recall`. |
| `--recall-gib GIB` | The record's disk, 1/16 GiB (64 MiB) by default; the oldest ids go first past it. Implies `--recall`. |
| `--recall-read-only` | Draft from the record and add nothing to it. Implies `--recall`. |

`GET /v1/recall` reports the record: its state, size on disk and in memory, how many copies it offered and how many of their tokens landed.
`DELETE /v1/recall` empties it.
Both need the server's API key when it has one.

## What stays on disk

The record holds token ids only, never text, in files of up to 1,048,576 ids (4 MiB) under one directory, made with mode 0700, each file 0600.
A file named `identity` names the tokenizer that wrote the record: a digest of its vocabulary, merges and added tokens, its vocabulary size, and the model the server first served.
Models whose tokenizers have the same identity share one record, and a server whose tokenizer differs refuses to start on that directory and says which tokenizer wrote it.
Each request adds the part of its prompt that no earlier request read, from where the prompt cache resumed it, and its reply when the reply ends.
Runs of 128 or more ids the record already holds are not added again, so a request sent twice adds nothing the second time.
Past `--recall-gib` the oldest 4 MiB file is deleted.
While the disk has under 4 GiB free the server stops adding to the record, and serving goes on.
A second server that opens a directory another server is writing reads the record as it was and adds nothing.
`DELETE /v1/recall`, or deleting the directory while no server uses it, removes everything.
Nothing in the record leaves the machine: no request, reply or route sends it anywhere.

In memory the record takes about 8 MB for every million ids it holds, plus a key table fixed by `--recall-gib` (8 MiB at the default 64 MiB).

## Which engines read it

Recall drafts in the lane engine's copy window and its grafted lanes, for streams that share rounds and streams alone.
On CUDA a lone Nemotron stream's own driver asks the record when its context has nothing to copy.
Flash Next's replay engine copies from its own history on the GPU and reads no record, so a Flash Next server refuses `--recall` before it loads.
A lone Nemotron stream on a Mac and a lone Qwen3.8-27B stream on a Mac with tensor units run their own drivers, which do not read the record yet.

## Benchmarks and gates

A populated record makes any benchmark that repeats its prompts faster than a fresh server would be.
Run release gates and speed comparisons with Recall off, or with `--recall-read-only` on an empty record, and say which.
The server's startup line and `GET /v1/recall` state the record's state and size.

## How it decides

A copy is offered when the record's continuation of the stream's last 16 tokens or more is expected to land more tokens than the stream's own drafts have lately.
The expected tokens come from how the record's earlier copies landed, kept apart for matches of 16 to 31 tokens, 32 to 63, and 64 or more.
A record whose copies keep missing stops offering them, apart from a rare probe, and a stream whose copy landed nothing leaves the record alone for four rounds.
The stream's own context is asked first, and wins ties.
