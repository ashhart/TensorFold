//! Dense Qwen text configuration and checkpoint contracts. GPU forward pieces follow separately.
pub const dflash = @import("dflash/root.zig");
pub const yarn = @import("yarn.zig");
pub const loop = @import("loop.zig");
pub const gguf = @import("gguf.zig");
pub const config = @import("config.zig");
pub const tied_head = @import("tied_head.zig");
pub const affine = @import("affine.zig");
pub const widths = @import("widths.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const weights = @import("weights.zig");
pub const attention = @import("attention.zig");
pub const attention_plan = @import("attention_plan.zig");
pub const attention_join = @import("attention_join.zig");
/// The tail attention's rewrite and check before M5, as the runner builds it (tf-qwen27-attention-precision reads it).
pub const simd_attention = @import("simd_attention.zig");
pub const decode_round = @import("decode_round.zig");
pub const forward = @import("forward.zig");
/// The MTP draft head's forward (mtp-4bit.safetensors beside the checkpoint).
pub const mtp = @import("mtp.zig");
/// TF_Q27_MTP_GPU_PICK: the draft pick from a GPU top-1 row.
pub const mtp_pick = @import("mtp_pick.zig");
pub const gdn = @import("gdn.zig");
pub const gdn_contract = @import("gdn_contract.zig");
pub const gdn_hook = @import("gdn_hook.zig");
pub const gdn_library = @import("gdn_library.zig");
pub const gdn_plan = @import("gdn_plan.zig");
pub const gdn_pool = @import("gdn_pool.zig");
pub const gdn_pool_gpu = @import("gdn_pool_gpu.zig");
pub const glue = @import("glue.zig");
pub const glue_gpu = @import("glue_gpu.zig");
pub const image_rows = @import("image_rows.zig");
pub const gpu_weights = @import("gpu_weights.zig");
pub const gpu_frame = @import("gpu_frame.zig");
pub const model = @import("model.zig");
/// The routed block's kernels (the kernel contract builds its row kernel's source as the engine does).
pub const moe_gpu = @import("moe_gpu.zig");
pub const projection = @import("projection.zig");
pub const prompt_memory = @import("prompt_memory.zig");
pub const kv_grow = @import("kv_grow.zig");
pub const prompt_rows = @import("prompt_rows.zig");
pub const prompt_marks = @import("prompt_marks.zig");
pub const prompt_tail = @import("prompt_tail.zig");
pub const quant_gpu = @import("quant_gpu.zig");
pub const ane_mlp = @import("ane_mlp.zig");
pub const round_plan = @import("round_plan.zig");
pub const state = @import("state.zig");
pub const session = @import("session.zig");
pub const snapshot = @import("snapshot.zig");
pub const disk = @import("disk.zig");
pub const taps = @import("taps.zig");
pub const tap_gather = @import("tap_gather.zig");
pub const tap_ring = @import("tap_ring.zig");
/// Staged references: a round's stage rows copied to their planes (tf-qwen27-run --stages).
pub const stage_planes = @import("stage_planes.zig");
pub const sn_mtp = @import("sn_mtp.zig");
/// The 27B's Metal sources by name (the kernel contract compiles them as the engine does).
pub const runtime_sources = @import("qwen_runtime_sources");
test {
    _ = dflash;
    _ = yarn;
    _ = loop;
    _ = disk;
    _ = @import("core").gpu_profile;
    _ = @import("core").qwen35_stages;
    _ = stage_planes;
    _ = session;
    _ = @import("gdn_test.zig");
    _ = @import("simd_attention.zig");
    _ = glue;
    _ = image_rows;
    _ = @import("rope_test.zig");
    _ = @import("attention_pick_test.zig");
    _ = state;
    _ = projection;
    _ = model;
    _ = quant_gpu;
    _ = @import("quant_rows_test.zig");
    _ = @import("prompt_tail.zig");
    _ = round_plan;
    _ = attention;
    _ = prompt_marks;
    _ = config;
    _ = affine;
    _ = widths;
    _ = checkpoint;
    _ = weights;
    _ = @import("weights_test.zig");
    _ = @import("mtp_load_test.zig");
    _ = @import("mtp_step_test.zig");
    _ = @import("geometry_test.zig");
    _ = @import("slot_pages_test.zig");
    _ = sn_mtp;
}

pub const residency = @import("residency.zig");
