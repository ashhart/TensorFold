# Nemotron Omni video on native CUDA

This increment adds video to the maintained Nemotron Bolt family and its shared RADIO tower.
It uses the engine's existing vision kernels and GEMMs. It needs no Python vision process.
The supplied #940 video work is by @MiaAI-Lab, with literal marker shielding by @cwschroeder.
The Bolt family comes from @BobClawblaw's #897 and the maintained lane integration.

Start an admitted Omni checkpoint with `--vision`. Video is offered only when its config declares
`video_temporal_patch_size: 2`, its tokenizer has the reserved video marker, and its checkpoint contains
`vision_model.radio_model.model.patch_generator.video_embedder.weight` with the expected BF16 shape.
Image-only checkpoints keep their image path.

Requests use the chat content part `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}`.
Public HTTPS URLs require `--vision-urls`. The native decoder library, libtfvideo with FFmpeg, must be
available beside the server. A missing library gives a named refusal. See the server's video build instructions.

`--vision-max-videos` defaults to 2 and accepts 1 through 16. `--vision-video-tokens` defaults to 16384
and accepts 64 through 262144 visual tokens shared by the request's videos. Image tokens have a separate
budget. The engine reserves the checked sum of both budgets in its whole-request visual rows buffer.
A larger budget increases device memory admission and can make a checkpoint too large to load.

Encoded input is bounded to 16 MiB per clip and 20 MiB across the request. Dimensions are bounded to
8192 per side and 16 Mi pixels, with at most 3600 seconds per clip. Source frame counts and rate components
must fit the decoder's signed integer range. At most 256 frames are sampled per clip, or 128 above
720 rows. The visual token budget can lower that count. Frames are sampled at 2 fps, or 1 fps above
720 rows, and paired for the temporal projection. An odd last frame is repeated in its pair.
The prompt includes each pair's timestamp and visual rows. Literal marker text stays ordinary text.

This patch's host checks cover preparation, limits, ownership, prompt marker handling, and row planning.
Real decoder compatibility, tower reference accuracy, model outputs, drafting, concurrency and prompt
reuse across video boundaries still need final-source GPU qualification before release.
