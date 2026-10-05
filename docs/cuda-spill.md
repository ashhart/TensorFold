# CUDA spill tier (`--spill-gib`)

The Mac server writes conversations evicted from its prompt cache to disk (`--spill-gib`, PR #68). On CUDA the same
flag turns on a disk tier for the prompt states an engine keeps (GLM-5.3-Flash with `--parallel 1` today; it refuses
`--parallel` 2 or more until its concurrent decoder has a hook): a kept state that leaves memory is written to
`--snapshot-dir`, unless a later turn of its conversation that can be written is still kept (the Mac's rule), and a
later request that extends the same tokens reads it back instead of prefilling, also after a restart.

```
tensorfold serve <GLM-5.3-Flash> --tp 2 --rank R --master ADDR --spill-gib 64 --snapshot-dir /fast/nvme/tf-spill
```

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--spill-gib N` | 0 (off) | disk cap per rank |
| `--snapshot-dir DIR` | `~/.cache/tensorfold/prefix-snapshots` | where it writes (one folder per build and rank) |
| `--spill-highwater F` | 0.70 | past this fraction of kept-prompt memory, the states eviction takes next are written early, in the background; 1.0 writes only at eviction |
| `--spill-min-tokens N` | 8192 | shorter states are not written |
| `--spill-min-free-gib N` | 50 | no write leaves less disk free (any rank) |
| `--spill-keep-builds N` | 1 | other builds' folders kept at start |

Environment: `TF_SPILL_WRITERS` (2 writer threads), `TF_SPILL_DIRECT` (1: O_DIRECT where the file system takes it),
`TF_SPILL_FLUSH_S` (60: a clean shutdown's budget for writing what is kept).

## How it works

- **Format.** One safetensors file per stored prompt, the header padded to 4 KiB: the snapshot object's fields
  (`cuda/spill_format.py` `encode`: tensors, plain values, lists, tuples, string dicts; the class by import path, read
  back only as a class the engine names, never imported from the file; a class's `transient` names left out). The
  metadata names the format, model and token ids, as `engine/prefix_snapshots.py` does on the Mac. A file whose size
  does not match its header, or a `.partial` left by a crash, is dropped at start.
- **Model-agnostic core.** `cuda/spill.py` (the store and its index), `spill_format.py` (files and keys) and
  `spill_io.py` (staging buffers, writer threads, reads) know no model. A family supplies its snapshot object
  and, if its rows live in the engine's caches, a `materialize` callable that attaches views of them before the write
  (GLM: `_spill_rows` in `families/glm5_next/cuda/spill_hooks.py`). Restores decode the object and hand it to the
  family's existing resume path.
- **Writes run beside the engine.** Copies off the device run on a writer's own CUDA stream through two pinned
  staging buffers per writer. A state written from saved rows never holds the engine; one written from the live caches
  holds the next prefill until its last bytes are off the device, about the time it takes to write (0.4 s for 200k
  tokens of GLM-5.3-Flash on a GX10's NVMe). At most 4 GiB of writes wait in a rank's queue; one past that first
  waits for the oldest.
- **DFlash2.** A state is written with its drafter's ring window, so a drafting request can resume from it. The copy
  `_take_over` keeps in memory drops that window, so the state is written as it is copied, from that copy. Without a
  ring (`TF_GLM_DRAFT_RING=0`) no state is written. Reads go to an entry whose stored drafter fields fit the request.
- **Several ranks.** Each rank writes its own shard to its own folder. Every rank takes the same decisions in the same
  order: rank 0 picks what to write and read and rank 1 follows through the request header, and eviction under the cap
  reads only the index both hold, never how far a rank's own writes got; sizes and the free-disk floor are agreed in
  one all-gather a save; a read is used only if it worked on both ranks (one all-gather), else both prefill and forget
  the entry; at start only entries every rank holds are kept, and a rank that cannot use its folder stops every rank.
  One rank needs no communication.
- **Compatibility.** Files live under `spill-<hash>/rank<N>/`, the hash over the format, the weights' path, the
  TensorFold version and the family's and CUDA sources (`build_id`, as MLX's snapshot key), the runtime (torch, CUDA,
  GPU and driver), each rank's own copy of the weights and drafter (`weights_id`: every `*.safetensors` and `*.json`
  file's name, size, mtime and inode), the drafter's path, KV layout,
  tensor-parallel layout and the start settings the ranks agree on (GLM: the draft model, MTP head, latent cache,
  draft ring, prompt chunk rows, long context and `--no-drafts`; not the window's cache slots); another build never
  reads them, and older builds' folders are pruned at start. Ranks compare the shared part at start and log any rank
  that differs from rank 0.
- **Integrity.** Every 8 MiB block of a file has a CRC-32 in the header (zlib, which uses the CPU's CRC instructions
  where it was built with them), computed beside the write and checked beside the read; a block that differs fails the
  read (`crc_failed`), every rank forgets the entry and the request prefills. The files are 0600 and the folders 0700,
  owned by `--snapshot-dir`'s owner: they hold prompts' token ids.
- **Stopping.** A clean stop of rank 0 (SIGTERM, Ctrl-C) writes what is kept, with every rank, within
  `TF_SPILL_FLUSH_S`. Stop rank 0 first: a rank stopped before it leaves that flush waiting.

## Adding a family

1. Keep prompt states as an object whose fields are tensors and plain values (most CUDA families already keep a
   `Snapshot`); name runtime-only fields in a `transient` class attribute, and pass its class to `SpillStore`
   (`classes`), the only one a read decodes.
2. If its attention rows stay in the engine's caches, give `materialize` views of them (written from there; the
   engine waits for the copy before those caches change).
3. Call `SpillStore.save` where the engine drops a state (and for the oldest ones past `--spill-highwater`),
   `find` + `load` where it looks for a resume point, and send the decision to the other ranks the way its requests
   already travel. Set `CUDA_SPILL = True` in the family package.

Recurrent state (GLM's KDA, Qwen's GDN, Mamba) exists only at the position it was taken, so a state resumes exactly
there; pure attention rows could be cut at any page, which this tier does not do (one file per stored prompt).
