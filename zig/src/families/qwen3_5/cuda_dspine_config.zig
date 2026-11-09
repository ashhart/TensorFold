//! DSpine on the CUDA drafter: core/dspine.zig's admitted shape bound to the target and to these kernels' limits.
const std = @import("std");
const core = @import("core");
const engine = @import("cuda_engine.zig");
const Config = @import("cuda_draft_config.zig").Config;

/// `root` a DSpine config.json (core.dspine.declared): its CUDA config over `target`, at TF_DSPINE_WINDOW.
pub fn parse(root: std.json.Value, target: engine.Geometry) !Config {
    return parseWindow(root, target, try @import("sys").environment.read(usize, "TF_DSPINE_WINDOW", 0));
}

/// parse with a served `window` in place of the checkpoint's sliding_window (0 keeps it, core.dspine.narrowed).
pub fn parseWindow(root: std.json.Value, target: engine.Geometry, window: usize) !Config {
    const s = try core.dspine.narrowed(try core.dspine.parse(root), window);
    if (s.head_dim != 128 or s.layers > 8 or s.code > 4096 or s.rank > 1024) return error.UnsupportedDraftConfig;
    if (s.hidden != target.hidden or s.vocab != target.vocab or s.target_layers != target.layers or s.hidden % 256 != 0) return error.UnsupportedDraftConfig;
    for (s.tapIds()) |tap| if (tap >= target.layers) return error.BadDraftConfig;
    return .{ .hidden = s.hidden, .intermediate = s.intermediate, .layers = s.layers, .vocab = s.vocab, .heads = s.heads, .kv_heads = s.kv_heads, .head_dim = s.head_dim, .block = s.block, .mask = s.mask, .window = s.window, .group = 16, .rank = s.rank, .topk = s.topk, .eps = s.eps, .theta = @floatCast(s.theta), .taps = s.taps, .n_taps = s.n_taps, .kind = .dspine, .code = s.code, .bridge_eps = s.bridge_eps };
}

test {
    _ = @import("cuda_dspine_config_test.zig");
}
