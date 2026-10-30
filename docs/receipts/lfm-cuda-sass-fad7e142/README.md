# Liquid CUDA kernels: SASS identity of the refactored images

`tools/zig/cuda_sass_identity.py --repo . -j 3 --receipt sass.tsv 465e48b62 fad7e142c`: main 1.1.9 (`465e48b62`) against this stack's NVFP4 head (#1492, #1503, #1508 at `fad7e142c`), nvcc 13.0 (V13.0.88), sm_80, 86, 89, 120 and 121. `sass.tsv` has every function's SASS sha256 on both sides; `summary.txt` is the tool's output.

| image | functions on main | identical | changed | added |
|---|---|---|---|---|
| nemotron_attention | 35 | 35 | 0 | 10 (`tf_paged_attn_chunk_segs_64`, `tf_paged_attn_merge_segs_64`) |
| prefill_attention | 40 | 40 | 0 | 10 (`pattn_kernel<64, 8, 4, 8>`, `pattn_kernel<128, 8, 4, 8>`) |
| contract_controls | 220 | 220 | 0 | 45 (the stack's planted controls) |

`lfm2_moe` and `lfm3` are new images, on one side only. The tool exits 1 because it counts added functions as differing, and because its `fp4pair` control has no use in these two images to change (0 of 2); its `warp_sum` control changes 10 functions, so a changed source does show.
