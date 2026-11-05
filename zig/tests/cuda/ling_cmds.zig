//! The Ling-3.1-flash commands of `tf-cuda-test` (main.zig hands every `ling-*` command here).
const std = @import("std");
const check = @import("check.zig");

pub const usage =
    \\  ling-route                group-limited select and Ling's SwiGLU limit vs float64 (synthetic)
    \\  ling-mla                  Ling's MLA (128-wide absorb/expand, rope, 576-wide attention) vs float64 (synthetic)
    \\  ling-synth <dir> [--format bf16|fp8|nvfp4] [--layers N] [--tiny]   a tiny Ling checkpoint with the real schema
    \\  ling-weights <dir>        every layer of a Ling checkpoint loaded: projections vs float64, small tensors byte for byte
    \\  ling-forward-synth <dir> <ref> [picks]   a ling-synth checkpoint's forward vs its float64 oracle; cuts, windows
    \\  ling-clamp <dir>          Ling's SwiGLU order min(silu(g), L): glue and the dir's grouped experts vs float64
    \\  ling-lanes <dir> <out> [ref] [--layers a,b] [--mtp]   the lane backend: chunked, drafted, resumed, concurrent
    \\  ling-slots                the lane backend's slots over reused map buckets (no GPU work)
    \\  ling-ckpt-check <dir> [index]   every tensor's name, dtype and shape against the loader's (headers alone)
    \\  ling-exl3-layers <dir> <ref> <out>   real EXL3 layers on exllamav3's inputs (prompt and decode), outputs saved
    \\  ling-init-faults <dir> [--mtp] [--every N]   the backend opened with each driver call, then each allocation, failing
    \\  ling-exl3-clamp <dir> [layer]   an EXL3 MoE layer's grouped gate-up in Ling's clamp order vs float64
    \\  ling-tp <dir> <out> [--layers a,b] [--steps N] [--prompt N] [--capacity N] [--resume M] [--window W] [--split] [--mtp]
    \\                            greedy rows at TP=1, or TP=2 over TF_LING_RANK/WORLD (--split: the latent caches split)
    \\
;

fn arg(rest: []const [:0]const u8, i: usize) ![]const u8 {
    if (i >= rest.len) return error.MissingArgument;
    return rest[i];
}

/// Runs `cmd` when it is one of ours; false otherwise.
pub fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !bool {
    const eq = std.mem.eql;
    if (eq(u8, cmd, "ling-route")) {
        try @import("ling_route.zig").run(gpu);
    } else if (eq(u8, cmd, "ling-mla")) {
        try @import("ling_ref_mla.zig").run(gpu);
    } else if (eq(u8, cmd, "ling-synth")) {
        try @import("ling_synth.zig").run(gpu, rest);
    } else if (eq(u8, cmd, "ling-weights")) {
        try @import("ling_weights_tests.zig").run(gpu, try arg(rest, 0));
    } else if (eq(u8, cmd, "ling-forward-synth")) {
        try @import("ling_forward_synth.zig").run(gpu, try arg(rest, 0), try arg(rest, 1), if (rest.len > 2) rest[2] else null);
    } else if (eq(u8, cmd, "ling-clamp")) {
        try @import("ling_clamp.zig").run(gpu, try arg(rest, 0));
    } else if (eq(u8, cmd, "ling-lanes")) {
        try @import("ling_lanes.zig").run(gpu, rest);
    } else if (eq(u8, cmd, "ling-slots")) {
        try @import("ling_cuda").lanes.slotCheck(gpu.gpa);
        check.pass("ling-slots: new streams take the lowest free slot over buckets holding stale slots ({t})", .{@import("builtin").mode});
    } else if (eq(u8, cmd, "ling-ckpt-check")) {
        try @import("ling_ckpt.zig").run(gpu, rest);
    } else if (eq(u8, cmd, "ling-exl3-layers")) {
        try @import("ling_exl3_layers.zig").run(gpu, rest);
    } else if (eq(u8, cmd, "ling-init-faults")) {
        try @import("ling_faults.zig").run(gpu, rest);
    } else if (eq(u8, cmd, "ling-exl3-clamp")) {
        try @import("ling_exl3_clamp.zig").run(gpu, rest);
    } else if (eq(u8, cmd, "ling-tp")) {
        try @import("ling_tp.zig").run(gpu, rest);
    } else return false;
    return true;
}
