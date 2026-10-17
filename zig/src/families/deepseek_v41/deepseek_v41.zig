//! DeepSeek-V4.1-Flash on the Zig engine: release config, EXL3 pack headers, a rank's TP weight plan (host side).

const std = @import("std");

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const exl3 = @import("exl3.zig");
pub const names = @import("names.zig");
pub const pack = @import("pack.zig");
pub const Pack = pack.Pack;
pub const plan = @import("plan.zig");
pub const Plan = plan.Plan;
pub const stage = @import("stage.zig");
pub const named = @import("named.zig");
pub const prepare = @import("prepare.zig");
pub const engram_host = @import("engram_host.zig");

test {
    _ = config;
    _ = exl3;
    _ = names;
    _ = pack;
    _ = plan;
    _ = stage;
    _ = named;
    _ = prepare;
    _ = engram_host;
    // the device-side weight load, compiled on the host (no GPU work)
    inline for (.{ @import("load.zig"), @import("forms.zig"), @import("forms_inplace.zig"), @import("host_stage.zig"), @import("triton_capture.zig"), @import("ex_stacks.zig"), @import("buffers.zig"), @import("block_calls.zig") }) |t| std.testing.refAllDecls(t);
    // a block's emitter (attention, index, MoE, read-ahead, windows past 16 rows) over the launch bindings
    const block = @import("block.zig");
    inline for (.{ block, block.Emitter, block.Pool, @import("block_attn.zig"), @import("block_index.zig"), @import("block_moe.zig"), @import("block_prefetch.zig"), @import("block_wide.zig"), @import("knob_types.zig"), @import("forward_types.zig") }) |t| std.testing.refAllDecls(t);
    _ = @import("host_test.zig");
    _ = @import("engram_rows_test.zig"); // Engram's shard rows
    _ = @import("phases.zig"); // the phase timers
    // the forward over the block emitter, split KV and the decode graphs (hooks for later stages unset)
    inline for (.{ @import("forward.zig"), @import("forward_engram.zig"), @import("forward_follow.zig"), @import("forward_glue.zig"), @import("forward_graph.zig"), @import("forward_hooks.zig"), @import("forward_sample.zig"), @import("kv_state.zig"), @import("run.zig"), @import("triton_call.zig"), @import("kv_glue_emit_test.zig") }) |t| std.testing.refAllDecls(t);
    _ = @import("graphs.zig"); // the decode graphs' buckets, statics and floor
    _ = @import("greedy.zig"); // greedy's host step
    // the prompt's prefill: segments over the prefill emitter, CED, long prompts, x3gm v2 (hooks for images unset)
    inline for (.{ @import("block_prefill.zig"), @import("block_prefill_attn.zig"), @import("block_prefill_emit.zig"), @import("block_prefill_mhc.zig"), @import("block_prefill_moe.zig"), @import("block_prefill_select.zig"), @import("forward_prefill.zig"), @import("forward_prefill_finish.zig"), @import("forward_prefill_glue.zig"), @import("forward_prefill_k4.zig"), @import("gm2pf.zig"), @import("longpf.zig"), @import("calls_check.zig"), @import("rowmode.zig"), @import("serve_knobs.zig") }) |t| std.testing.refAllDecls(t);
    _ = @import("ced.zig"); // CED's keep and the drafter hand-off
    _ = @import("image_opts.zig"); // image prompts' block options
    _ = @import("rowtab.zig"); // row mode's per-row table
    _ = @import("multi_test.zig"); // several slots' prompt segments in one call
    _ = @import("streamrb_test.zig"); // the stream top-k's read-back twins on the prefill emitter
    // prepared weight folders (TF_DSV41_PREPARED)
    _ = @import("prepared.zig");
}
