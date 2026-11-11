//! CUDA test runner help, kept separate from command dispatch.
pub const text =
    \\usage: tf-cuda-test <command>
    \\  qwen4                     synthetic native Qwen4: binary64 logits/tokens, state seams, lanes and copied drafts
    \\  qwen-draft-frames MODEL DRAFTER IDS OUT  native 27B DFlash frames: raw bytes equal across partitions, controls red
    \\  experts-rows              experts_rows.cu's windows and chunks against the decode kernel's one-row launches
    \\  route-rows                route_rows.cu's router logits and softmax top-k against float64 (synthetic rows)
    \\  flashnext-fidelity-single MODEL FIXTURE OUTPUT  single-rank reference isolation
    \\  flashnext-fidelity-rank MODEL RANK PORT FIXTURE OUTPUT  fixed-history full score rows
    \\  flashnext-window-rank MODEL RANK PORT IDS_FILE  two-rank serial/window and rollback parity
    \\  flashnext-windows MODEL IDS_FILE  several streams' windows in one shared verify against each alone
    \\  flashnext-mtp-parity MODEL IDS_FILE  several streams' MTP heads in one shared pass against each alone
    \\  flashnext-score MODEL IDS_FILE  decisions' label scores against a plain pass's last row
    \\  flashnext-model-rank MODEL RANK PORT IDS_FILE  explicit full-model replay experiment (no drafts)
    \\  flashnext-rank RANK PORT [sum|eof|timeout|missing]  two-process collective and failure probes
    \\  flashnext-draft-affine    proposal-head affine projection against FP64 and row-count invariance
    \\  flashnext-shared          separate MXFP8 shared expert and reserved-slot scatter against FP64
    \\  flashnext-nvfp4-tp        rank-local expert cuts and FP32 partials against FP64 (one GPU)
    \\  flashnext-fp8-experts     the MTP head's block-FP8 experts (512 x 640 x 2560) through moe_native.fp8 against FP64
    \\  moetile                   whole shared down writes only slot 10 with identical FP32 bytes
    \\  moetile-contract          shared down replacement gates and the planted control
    \\  moetile-bench             alternate shared-down plus scatter and direct-slot GPU timings
    \\  nvfp4-fact                NVFP4 experts' prompt tiles with the decode form's sums: its bytes at every row, FP64
    \\  nvfp4-b32                 block 32 experts with csf0 scales: block 16's bytes on equal scales, each form and width
    \\  lane-half                 lane half tiles (MXFP8, BF16): bytes equal to the 64-column tile, FP64, cold-weight time
    \\  flashnext-dense-hc        dense connection weights and FP32 normalization against FP64
    \\  flashnext-hc              Flash Next's own hyper-connection and n-gram glue against float64 (synthetic rows)
    \\  flashnext-exl3 DIR names|norms|router [layers] [rows]|layer [layers]  Flash Next on an EXL3 pack vs float64
    \\  flashnext-exl3-tp RANK PORT DIR layer [layers]|sum|head|memory  EXL3 expert parallel on two ranks vs tp=1 and float64
    \\  flashnext-exl3-engine DIR tf|gen GOLDEN [LINES] | bench [LINES]  the family's Engine with the offload pool vs the reference golden
    \\  flashnext-exl3-tpe RANK WORLD PORT DIR tf|gen GOLDEN DUMP | bench  the Engine at tp=1/2 with per-rank pools
    \\  flashnext-attn            Flash Next's own attention glue, pooling and block scores against float64 (synthetic)
    \\  flashnext-gdn-heads [BASELINE]  DeltaNet full/half bytes, optionally against a frozen flashnext_batch.fatbin
    \\  flashnext-tp-heads MODEL  each rank's cut in-projections and attention heads equal the whole layer's bytes
    \\  flashnext-pages           the attention glue's caches through a page table: paged == contiguous, float64, controls
    \\  info                      driver, device and library versions
    \\  smoke                     copies, fills, launches, argument packing, module globals
    \\  shared-context            Context.shared and Stream.borrowed as a library inside a host's process sees them
    \\  graph                     stream capture, explicit graphs, node and whole-exec updates
    \\  overhead [n] [reps]       dependent one-thread kernels: plain stream vs one graph
    \\  overhead-pdl [n] [reps]   the same with programmatic dependent launch on every kernel
    \\  launch-ex                 cuLaunchKernelEx: clusters, cooperative grid, PDL on a stream and in a graph
    \\  ptx                       hand-written PTX through the driver JIT, and a refused broken image
    \\  symbols                   every gdn fatbin instantiation resolves by its listed symbol
    \\  cublaslt                  bf16 GEMM through cuBLASLt against an fp64 reference
    \\  nccl                      one-rank NCCL all-reduce and all-gather
    \\  rank-scatter <rank> <n> [id]  rank-order reduce-scatter (NCCL_COMM_ID or an id file) against an all-gather
    \\  roce-gather <rank>        two ranks (NCCL_COMM_ID): the RoCE all-gather against NCCL's, graph replays, time a call
    \\  rank-sum                  the reduce-scatter's sum kernel alone, 2 to 5 ranks as each rank (one GPU, no NCCL)
    \\  nccl-pair <rank> <world> <idfile>  one rank of a multi-host communicator: send/recv with every peer, all-reduce, all-gather
    \\  gdn-replay <dir>          replay_kernel bits against the Python oracle's fixture
    \\  gdn-tree <dir>            tree_kernel bits against the Python oracle's fixture
    \\  triton <dir>              a Triton cubin's bits against the Python oracle's fixture
    \\  norms                     RMS norm and the residual step against float64 references (synthetic)
    \\  norm-rows                 bf16-residual and gated norms, token rows against float64 (synthetic)
    \\  norm-tail                 bf16 residual tails (branch norm, scalar, MLX rounding) against float64 (synthetic)
    \\  embed-rows                MLX affine token rows (2 to 8 bits, bf16 scale) against float64 (synthetic)
    \\  tree-round                tree_round.cu's picks, batched verified trees and commits against core/tree_round
    \\  act-rows                  SwiGLU, output gates, group sums, DFlash2's dynamic conv against float64 (synthetic)
    \\  gdn-prep                  Gated DeltaNet conv, norms, g and beta against float64 (synthetic)
    \\  qk-rows                   QK norm, RoPE and the cache write against float64 references (synthetic)
    \\  sample                    sample.cu's keyed draws against the Metal rule's host references (synthetic rows)
    \\  glue                      Nemotron's glue kernels against their host references at its shapes (synthetic rows)
    \\  lfm2-kernels              LFM2's glue kernels against their host references at d1's shapes (synthetic rows)
    \\  pocket [reps]             the pocket's two kernels against tf_pocket_rows byte for byte; reps > 0 times them too
    \\  lfm2-attention            LFM2's GQA prompt attention (head dim 64), causal and packed trees, against float64 (synthetic rows)
    \\  lfm2-vision [MODEL]       LFM2-VL's vision kernels against host references; MODEL adds features against CPU goldens
    \\  radio REF MODEL           RADIO's tower and projector rows against float32 references on the same pixels
    \\  radio-processor REF PICS MODEL  decoded RGB, processor pixels, float resizes and grids against the reference
    \\  radio-kernels             RADIO's float-patch cast and position rows against host references (synthetic rows)
    \\  radio-attention           the head-dim-80 tower attention's time at five sizes, outputs finite (synthetic rows)
    \\  gdn-back                  DeltaNet input gradients against FP64
    \\  q4-back                  Affine input gradients against FP64
    \\  moe-back                  Expert and router input gradients against FP64
    \\  attn-back                  Gated attention input gradients against FP64
    \\  fn-hc-back                  Hyper-connection input gradients against FP64
    \\  fn-moe-back                  MoE and pocket input gradients against FP64
    \\  fn-pocket [reps]                  Pocket kernels against the row reference
    \\  fn-back                  All component gradient cases
    \\  window-profile MODEL WIDTHS IDS_FILE   a decode window's GPU ms by kernel class (WIDTHS like 1,4,16), beside its graph
    \\  tree-accept MODEL IDS_FILE COUNT OUT   a greedy reply's MTP chain and level-1 runner-up branches at every position (JSON)
    \\  chunk-costs MODEL IDS_FILE WIDTHS   r rows as the decode window graph (r up to its window) and as a prompt chunk (any r)
    \\  resume MODEL IDS_FILE     prompt states kept at off-grid rows: each resumed pass against a fresh one, byte for byte
    \\  marks MODEL IDS_FILE N PATTERNS   states kept inside chunks against chunks cut at the same rows, byte for byte
    \\  prompt-cuts MODEL IDS_FILE   a prompt keeping states inside chunks and at their ends (background passes too), and resumed from each, equals one uncut pass
    \\  prompt-segments MODEL IDS_FILE   a prompt run as 2 to 4 staggered segments a call equals one uncut pass, byte for byte
    \\  cached-plans-tp2 MODEL IDS_FILE   rank-zero cached plans and prompt resumes with the existing expert rank
    \\  shared-fork MODEL         the latent MoE's shared expert on the side stream (held back), lone and in shared windows of 2/4/8 streams, against one stream, byte for byte
    \\  recapture MODEL           graphs captured again while another sequence is bound: both keep their sampling rules
    \\  paged MODEL IDS_FILE      a paged sequence against the own one: prompt pass, decode, resumes across page edges
    \\  grouped-plan <dir>        the shared expert plan against experts.route's bytes (oracle/grouped_plan.py)
    \\  fp8-lane <dir>            block-FP8 projections against the Python lane matmul's bytes (oracle/fp8_lane.py)
    \\  fp8-experts <dir>         grouped block-FP8 experts against recorded fixture bytes
    \\  nvfp4-experts <dir>       grouped NVFP4 experts against recorded fixture bytes
    \\  projection-f32            FP32 projection outputs against FP64, rounding and row/reduction invariance
    \\  mxfp8                     MXFP8 strided projections against FP64 and row-count invariance
    \\  quant-ref                 FP8 and NVFP4 projections and experts against float64 references (synthetic)
    \\  affine8                   MLX affine 2 to 8-bit tiles and the 8-bit lane_gemv: widened codes, float64, rows (synthetic)
    \\  native-fp                 -Dnative-fp-decode: hardware e2m1/e4m3 pair decode equals software on every finite input
    \\  affine4-decode            affine-4 decode at the Qwen3.5 shapes: row counts and segmented launches byte for byte
    \\  w4a8                      int8-activation prompt matmuls on affine-4 tiles: float64 of their formula, rows (synthetic)
    \\  carveout [MiB] [card]     GB10 display memory: round trips and bandwidth (SKIP without the card)
    \\  nvfp4 <dir>               NVFP4 projections against recorded lane-matmul and prompt-GEMM fixture bytes
    \\  moe                       softmax routing and SwiGLU experts against host references, then moe-live
    \\  moe-live                  a captured bucket's padding rows: never routed, planned or combined (synthetic)
    \\  moe-router-bench          tf_moe_router against its register tiles by row count, alternating (35B-A3B shape)
    \\  experts3 [reps]           affine-3 packing and outputs against affine-4 and fp64; reps > 0 times decode
    \\  ggml <dir>                Bolt's IQ2_XXS experts and Q8_0 / IQ2_XXS dense rows on real tensors against fp64 (tools/zig/bolt_ggml_fixture.py)
    \\  ggml-bench                IQ2_XXS expert and Q8_0 lm_head decode GB/s (synthetic blocks)
    \\  modelopt                  ModelOpt repack, FP8 and BF16 lane rows and relu^2 NVFP4 experts against fp64
    \\  checkpoint                checkpoint prompt tiles, NVFP4 up rows and share sums byte-equal to the decode path (synthetic FP8 and NVFP4)
    \\  exlr8                     EXLR8 projections: every codebook state, FP64, row windows, the Metal frozen digests
    \\  exlr8-bench DIR [MANIFEST] [LAYER] [SETS]   a real EXLR8 layer: FP64, row windows, then GB/s by stage
    \\  exlr8-dense [DIR]         EXLR8 dense formats: decode bit-equal, FP64, row windows; DIR adds the real matrices
    \\  tree                      tree attention (committed keys and paths) against float64 references (synthetic)
    \\  block                     DFlash block attention over windowed contexts against float64 (synthetic)
    \\  window                    sliding-window attention against float64 references (synthetic)
    \\  window-rows               window attention at head dims 256 and 512 against float64 ranges (synthetic)
    \\  window-pages-512          head dim 512 decode and prompt rows over shuffled pages: the contiguous bits (synthetic)
    \\  window-bench WHICH        window attention's GPU time: callers (Kolibri-1's shapes) or gemma (head dims 256, 512)
    \\  moe-glue                  sigmoid top-k routing and the weighted combine against float64 references (synthetic)
    \\  dense-rows                bf16 rows times weight rows (routers, heads) against float64 references (synthetic)
    \\  exl3 [dir]                EXL3 decode-row linear: float64 checks, row invariance, the Python engine's fixture (dir none skips it)
    \\  exl3-shapes [rows]        the 27B's EXL3 shapes once each at plan()'s launch (1 and 16 rows), for ncu
    \\  exl3-midm                 the mid-M kernels, linear_wc and the projection groups against the linear, byte for byte
    \\  exl3-plans [shape]        the 27B's EXL3 shapes: every (K splits, warps) and layout timed at 1 and 16 rows
    \\  exl3-contract             EXL3's replacements of linear_body through gates 1-4, each control red (kernel-contract's subset)
    \\  exl3-tp2 <rank> <host:port> [dir]  EXL3 tensor parallel: rank shards vs the Python engine's, all-gathered (dir none skips them)
    \\  exl3-pack DIR [REPS]       EXL3 Dense and grouped experts on a real EXL3 pack's shapes: float64 error and timing
    \\  exl3-ple2 DIR [ROW] [N]    EXL3 n-gram table through cuda_ple.Ngram (ids, rows, RSS) and the PLE kernels vs float64
    \\  exl3-draft DIR [THREADS]   EXL3 draft head rows dequantised from lm_head vs the full head (Dense) and float64
    \\  exl3-prefill DIR experts|copy [LAYER]|dense  EXL3 prefill pieces on the pack: grouped experts by rows, expert-major passes, line copies, prompt Dense
    \\  exl3-prefill DIR engine IDS N[,N] [LINES|-] [cmp] [ROWS[,ROWS]] [--decode N]  whole prefills through the deep chunks, timed; N greedy decode rounds after each (default 32)
    \\  exl3-prefill DIR equal IDS N ROWS[,ROWS] [LINES] [--equal-first-size]  deep prefill at each chunk size vs the batched pass: last-row logits bytes and greedy tokens
    \\  exl3-prefill DIR mixed IDS N STEP[,STEP] [LINES]  prompt cut into deep and shallow segments vs the batched pass and serial decode: logits bytes, tokens
    \\  exl3-prefill DIR resume IDS N STEP[,STEP] CUT[,CUT] [LINES]  state kept at any row (snap grid off), resumed vs a fresh prompt; CUT<0: rows left
    \\  exl3-tpprefill RANK WORLD PORT DIR IDS NS ROWS CAP deep|both DUMP [--greedy N] [--both-max N]  EXL3 deep prefill at tp=1/2 vs the batched pass and tp=1
    \\  exl3-pool DIR [equal|bench|hit FILES] [LAYERS]  Flash Next EXL3 expert offload pool on Experts' pointer tables vs fully resident
    \\  exl3-tiles [DIR]          EXL3's tile candidate: codebooks and tiles bit-equal, FP64, rows through qlinear and Dense
    \\  train-pack MODEL EXAMPLES [COUNT]   training passes holding several sequences against one a pass: differences, time
    \\  exl3-prefill-contract     the prefill expert tiles alone through the contract (gates 1-4 and the control)
    \\  exl3-promptd-contract     the prompt tiles in decode's bytes (tf_exl3_promptd_*) alone through the contract
    \\  exl3-tppf-contract        the TP prefill's rotate, combine, finish and place launches alone through the contract
    \\  exl3-kvt-contract         the host tier's kvt_plan, kvt_fetch and kvt_remap alone through the contract
    \\  kernel-contract [WORD...]  the contract's gates, each planted control red (-Dnvcc); WORDs: kernels named so
    \\  qmm-prefill-many-contract the GB10 bit-exact multi-projection prompt launch against separate launches
    \\  qmm-prefill-tile22-contract the GB10 large-prompt 2x2 tile against the default tile
    \\  qmmf-prompt              BF16 K32 prompt staging: parent K64 bytes at current Flash Next shapes
    \\  qmmf-prompt-contract     K64 and K32 replacement gates with both new output controls
    \\  qmmf-prompt-repro        isolate M=65 N=8240 K=2560 SK=4 BF16 without clusters
    \\  qmmf-prompt-bench        rank0 normal-main weighted A-B-A timings, plus separate MTP hidden timings
    \\  qmm-prefill-many-bench    same-process A-B-A screening for the GB10 launch fusion
    \\  mtp-rows                  tf_q36m_attn_rows: gates 1-4 (byte-equal to tf_q36m_attn a segment) and float64 alone
    \\  qwen-pages                qwen35.cu's attention cache through a page table: paged == contiguous, float64, red controls
    \\  qwen-norm-shift           qwen35.cu's _shift norms (BF16 checkpoints): 1 + w in FP32 within a BF16 ulp of float64, red controls
    \\  clef tiny DIR | head MODEL CASES OUT [--piece N]  Clef's joint schema head: the tiny fixture, gate 3's logits
    \\  qwen-pending              qwen35's pending kept rows: the slot scans and tf_q35k_apply_pend == tf_q35k_apply, red controls
    \\  rope-yarn                 YaRN rope in tf_q35_rope, tf_q35_rope8 and tf_q36m_norm_rope against float64, to 1M
    \\  kernel-contract-limits DIR   each registered launch against release SMs or sm_110 in DIR's fatbins
    \\  pocket-contract           Nemotron's pocket kernels (learned neurons) through gates 1-4, controls red
    \\  flashnext-modelopt-contract  Flash Next NVFP4's BF16 dense and FP8 MTP-expert downs through gates 1-4
    \\  minimax-contract          MiniMax-M3.1-Flash's indexer and block-sparse attention through gates 1-4, controls red
    \\  moss-contract             MOSS-Image's own kernels and its full attention through gates 1-4, controls red
    \\  glm-exl3                  GLM's EXL3 routed experts, decode and prompt, against float64 (synthetic)
    \\  glm-kda-pad-split         TF_GLM_KDA_PAD's padding of the bf16 and split KDA stacks at TP=3 (synthetic)
    \\  glm-vision MODEL F32 BF16 IMG  GLM's image tower and preparation against vision_ref.py's torch runs
    \\  glm-mm-fixed              Prompt K policies: FP64, every row alone/batched, offsets and guarded tails
    \\  glm-mm-bench              GLM's bf16 linears: b16.matmul against glm_mm, by rows (one GPU)
    \\  glm-index-check           GLM's indexer with glm_index against tf_glm_scores and tf_glm_select_rows, byte for byte (one GPU)
    \\  glm-kv-host-slots         TF_GLM_KV_HOST's gather past its slots is refused as error.GlmKvHostSlots, then one within (one GPU)
    \\  glm-ref                   GLM projections (matmuls, router, attention, indexer, hc) against float64 (synthetic)
    \\  glm-lanes-cover <ckpt> <dc.json> <rank> <world> <idfile>  copy drafts, cancels, discards, eos and max_new on one lanes.Engine across requests
    \\  glm-lanes-concurrent <ckpt> <dc.json> <rank> <world> <idfile>  concurrent streams on one lanes.Engine, each against its solo run
    \\  glm-shared-check <ckpt> <dc.json> <rank> <world> <idfile>  shared verify windows against lone twins, logits and states byte for byte
    \\  glm-pair-check <ckpt> <dc.json> <rank> <world> <idfile>  paired prompt steps against twins a chunk a step, logits and states byte for byte
    \\  glm-split-check <ckpt> <dc.json> <rank> <world> <idfile>  prompt chunks with their indexer rows split across ranks against whole twins, byte for byte
    \\  glm-solo-check <ckpt> <dc.json> <rank> <world> <idfile>  solo prompt passes against twins a chunk a step, logits and states byte for byte
    \\  glm-tap-split-check <ckpt> <drafter> <rank> <world> <idfile>  DFlash2 prompt taps: split and ring retained taps against gather, byte for byte, every rank
    \\  glm-paged-check <ckpt> <rank> <world> <idfile>  a paged stream (pages out of order, NaN first) against a whole-window twin
    \\  dory MODEL LOOPS N TEXT     Dory (NVFP4): N greedy tokens from TEXT, then verify windows of 2 to 16 rows bit-equal
    \\  dory-serve MODEL LOOPS TEXT N   Dory served: eager == graphs, copy drafts == plain, 2/4/8 streams == solo
    \\  dory-states MODEL LOOPS TEXT N  Dory kept prompt states at 8k and 32k resumed into wiped caches == fresh
    \\  dory-bench MODEL LOOPS TEXT N [STREAMS...]  Dory greedy decode tok/s at 1/2/4/8 streams (graphs on)
    \\  dory-pages MODEL LOOPS TEXT N   Dory with its full caches in pages == contiguous: fp32 logits bytes, then replies
    \\  dory-store MODEL LOOPS TEXT N   Dory prompts as pool pages (DORY_PAGED=1): resumes and shared streams == fresh
    \\  dory-prefill MODEL LOOPS TEXT N  Dory prompt time alone at 512, 2k, 8k and 16k rows (DORY_PAGED=1: on pages)
    \\  gemma4-states MODEL DRAFTER TEXT N   Gemma 4 + DFlash: stepped passes keep the drafter's context, resumed == fresh
    \\  spark25-pages MODEL TEXT N   Spark2.5: full-layer caches in pool pages == linear ones, replies and logits
    \\  gemma4-pages MODEL TEXT N [DRAFTER]  Gemma 4: global caches in pool pages == linear ones, replies and logits
    \\  gemma4-ple                         Gemma 4 E-series: per-layer input rows bit-equal to the host steps
    \\  gemma4-e4b-force MODEL REF OUT NAME...  Gemma 4 E4B: a reference's tokens forced, logits as .npy
    \\  kolibri-reference MODEL IDS  Kolibri-1's layer 0 and head against float64 from the checkpoint (prompt and decode rows)
    \\  kolibri-layers MODEL IDS    Kolibri-1's window edge (513 keys) and its fifth layer's full reach, on the checkpoint
    \\  kolibri-full MODEL IDS N    Kolibri-1: eager rows, steps, drafted, sampled, shared rounds, cuts and kept states alike
    \\  kolibri-pages MODEL TEXT N  Kolibri-1: full layers in pool pages == linear caches, replies and logits
    \\  kolibri-greedy MODEL PROMPTS N OUT  Kolibri-1 greedy from each prompt (JSON id lists): ids and top-2 margins
    \\  kolibri-logits MODEL IDS OUT [PROMPTS]  Kolibri-1's teacher-forced fp32 logits of 2,048 prompt rows; each prompt's last
    \\  glm-synth <dir> [fp8pb|local]  a synthetic GLM-5.3-Flash checkpoint (2 layers, 16 experts) and its dc.json for the lane checks
    \\  glm-loader-gate <ckpt>   every rank's uploads at TP 1, 2 and 3 against the checkpoint cut on the host, byte for byte
    \\  glm-fp8g-shares          block-FP8 TP shares packed as GLM's loader packs them, on the lane, against FP64 of the cut
    \\  aot-fallback <dir>        a launch's 1 from the runtime-parameter variant: the folded variant's bits (#1003)
    \\
;
