# GLM-5.3 (EXLR8) on two Macs

TensorFold serves the whole GLM-5.3 model from kindlingai's EXLR8 release on two Macs joined by Thunderbolt. Each Mac
holds half of every expert and every attention head, and rank 0 serves the OpenAI-compatible API while rank 1 runs each
request beside it. The release is
[kindlingai/glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4), pinned at revision
`f459f5d0272a529403dd9b3fd32f771f4250a0d4`. We have run its `home` profile on two M5 Ultra Macs with 256 GB each.

## What you need

- Two Macs with 256 GB each. A served Mac holds about 160 GB: 157.2 GB of weights with the MTP layer, and 159.6 GB at
  `--context 8192 --parallel 4`. The server refuses to load past 66% of RAM.
- About 270 GB of free disk on each Mac. A Mac's share is 139.5 GiB of expert halves plus 11 GB of whole files, but the
  expert files keep their full size with only that Mac's halves written, and APFS gave those folders 256 and 264 GB.
- A Thunderbolt 5 cable between the Macs and MCDMA's fabric library, set up as in steps 3 and 4 of the
  [speed-up mode guide](speed-up-mode.md). The settings files are the same.

The weights come under the release's own terms: the GLM-5.3 license, and CC BY 4.0 for the EXLR8 format.

## Pull each Mac's share

`pull --exlr8 PROFILE --node R/2` fetches only what node R of two needs: every whole file, and its half of each
expert's intermediate channels. Run it on each Mac with its own node number:

```sh
tensorfold pull kindlingai/glm-5.3-exlr8-k2-k3-k4@f459f5d0272a529403dd9b3fd32f771f4250a0d4 \
  --exlr8 home --node 0/2 --dir ~/models/glm-5.3-exlr8-home
```

On the second Mac, pass `--node 1/2`. Every whole file must equal the hub's sha256, every expert file's published hash
must equal the hub's, and each expert half must equal its xxhash64 before it is written. A pull that stops resumes:
run the same command again, and it checks what is on disk and fetches only what is missing. Our two Macs took 43
minutes over a shared uplink, at about 55 MiB/s each.

## Serve

Start rank 1, then rank 0 within five minutes. Each Mac loads in three to four minutes.

```sh
tensorfold serve ~/models/glm-5.3-exlr8-home --speed-up rank1.json --context 8192 --parallel 4 --temperature 0
tensorfold serve ~/models/glm-5.3-exlr8-home --speed-up rank0.json --context 8192 --parallel 4 --temperature 0 \
  --name glm-5.3
```

Run each command on its own Mac, and send requests to rank 0. `--context` defaults to 16,384 tokens and takes up to
262,144, and `--parallel` sets the streams, 8 by default and 16 at most. Rank 0 keeps prompt states for later turns in
up to 8 GiB by default, inside the load limit; `--prompt-cache-gib` sets another size and 0 turns them off. The engine
decodes greedily, and it refuses `--learn` and `--slide`.

## Measured

On our two M5 Ultras over one cable, one stream decodes at 21 tokens a second with `"draft": false`, and at 31 to
32 with the MTP layer's drafts, of which 60 to 74% land. Prompts run at about 200 tokens a second, 205 at 837 tokens
and 184 at 3,251, and the first prompt after a load at about 120.

Drafted replies equal `"draft": false`, a resumed conversation equals a fresh one, and concurrent requests equal their
solo replies, idle and with 36 busy threads on each Mac.

## Known limits

- Decode is slow. A one-row round takes about 45 ms of GPU time. At one row, the attention output projection's 48
  trellis jobs fill 48 of the GPU's 80 cores, and the routed experts' short down-projection jobs run unsplit, so part of
  the GPU stays idle.
- No CUDA serving yet. The CUDA kernels decode EXLR8 weights bit for bit as the Metal ones do, but no CUDA engine serves
  GLM-5.3.
- Contexts past 2,048 tokens have not been checked against a whole-model reference. Past 2,048 tokens the sparse
  indexer picks which keys each query reads; the layer checks covered 2,304 tokens, and a 3,251-token prompt served,
  but nothing compared a whole reply at that length.
- Only the `home` profile, two nodes, and M5 Ultras have been run.
