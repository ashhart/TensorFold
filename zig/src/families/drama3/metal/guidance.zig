//! Semantic-only CFG with independent recurrent/KV state and one shared checkpoint weight set.
const std = @import("std");
const cfg = @import("../config.zig");
const Engine = @import("engine.zig").Engine;
const k = @import("kernels.zig");
const fwd = @import("forward.zig");
const H = cfg.hidden;

pub fn validGamma(gamma: f32) bool {
    return std.math.isFinite(gamma) and gamma >= 1 and gamma <= 5;
}

pub fn mix(e: *Engine, unconditional: u64, gamma: f32) void {
    mixLogits(e.g, e.cand_logits, unconditional, e.w.n_cand, gamma);
}

fn mixLogits(g: *@import("gpu.zig").Gpu, conditional: u64, unconditional: u64, count: usize, gamma: f32) void {
    const Args = extern struct { count: u32, gamma: f32 };
    g.go(.d3_cfg_mix, .{ (count + 255) / 256, 1, 1 }, 256, &.{ conditional, unconditional }, Args{ .count = @intCast(count), .gamma = gamma });
}

/// Host lifecycle is separate from GPU storage; no old reference may activate a later plain request.
pub const State = struct {
    active: bool = false,
    gamma: f32 = 1,

    pub fn begin(s: *State, rows: usize, gamma: f32) !void {
        s.clear();
        if (!validGamma(gamma)) return error.BadDramaGuidance;
        if (rows == 0) return;
        s.* = .{ .active = true, .gamma = gamma };
    }

    pub fn clear(s: *State) void {
        s.* = .{};
    }
};

pub const Guide = struct {
    e: *Engine,
    batched: bool,
    state: State = .{},

    pub fn create(base: *Engine, batched: bool) !Guide {
        return .{ .e = try base.forkBackbone(), .batched = batched };
    }

    pub fn destroy(g: *Guide) void {
        g.e.destroy();
    }

    pub fn begin(g: *Guide, rows: []const [cfg.frame_width]i32, gamma: f32) !void {
        try g.state.begin(rows.len, gamma);
        errdefer g.state.clear();
        // A guide is never restored from the conditional reference cache.
        if (g.state.active) try g.e.startPrompt(rows);
    }

    /// One prompt row per scheduler round keeps cancellation and other lanes responsive.
    pub fn promptStep(g: *Guide, rows: []const [cfg.frame_width]i32) !bool {
        if (!g.state.active or g.e.rows_done == rows.len) return false;
        try g.e.backboneChunk(g.e.rows_done, 1, null);
        g.e.rows_done += 1;
        if (g.e.rows_done == rows.len) {
            g.e.g.copy(g.e.hfin, g.e.hlast, H * 2);
            try g.e.finishPrompt(rows);
        }
        return true;
    }

    pub fn clear(g: *Guide) void {
        g.state.clear();
    }
    /// Paired means one command-buffer submission, not a two-row GEMM or simultaneous GPU execution.
    pub fn frame(g: *Guide, conditional: *Engine, sampling: @import("../common.zig").Sampling) !void {
        if (!g.state.active) return conditional.frameStep(sampling, true, null);
        const u = g.e;
        k.linear(u.g, u.w.cand_head, u.hlast, H, H, u.w.n_cand, u.cand_logits, u.w.n_cand, 1, true);
        if (!g.batched) try u.g.commit();
        try fwd.enqueueFrameGuided(conditional, sampling, null, null, .{ .unconditioned = u.cand_logits, .gamma = g.state.gamma });
        if (!g.batched) try u.g.commit();
        // Feed the actual sampled conditional frame to the unconditional backbone, never its codec tower.
        k.nextInput(u.g, conditional.frame, u.bb_input, u.pos, u.pos_bb);
        k.embedBackbone(u.g, u.bb_input, u.w.embed, u.w.cb_emb, u.x, 1, H, u.c.ids.sem_begin, u.c.ids.sem_end, cfg.codebook_size);
        for (0..cfg.layers) |li| fwd.backboneLayer(u, fwd.laneBB(u, u.pos_bb), li, 1);
        k.rmsnorm(u.g, u.x, H, u.w.final_norm, u.hlast, H, 1, H, cfg.eps, true);
        k.copyI32(u.g, u.pos_bb, u.pos, 1);
        try u.g.commit();
    }
};

test "CFG rejects nonfinite and out-of-range gamma before state activation" {
    for ([_]f32{ 0, 0.99, 5.01, std.math.nan(f32), std.math.inf(f32) }) |g|
        try std.testing.expect(!validGamma(g));
}

test "CFG transitions clear old voice on plain request cancellation and invalid begin" {
    var state: State = .{};
    try state.begin(12, 1.5);
    try std.testing.expect(state.active);
    try state.begin(0, 1.5);
    try std.testing.expect(!state.active and state.gamma == 1);
    try state.begin(8, 2);
    state.clear();
    try std.testing.expect(!state.active);
    try state.begin(9, 3);
    try std.testing.expectError(error.BadDramaGuidance, state.begin(9, std.math.nan(f32)));
    try std.testing.expect(!state.active and state.gamma == 1);
    try state.begin(7, 1);
    try std.testing.expect(state.active and state.gamma == 1);
}

/// Synthetic device qualification called only by the explicit Metal quant-check executable.
pub fn check(g: *@import("gpu.zig").Gpu) !void {
    const first = g.bufs.items.len;
    defer g.releaseFrom(first);
    const n = cfg.codebook_size + 1;
    // The trailing sentinels model codec logits, which must never be touched by semantic CFG.
    var conditional: [n + 8]f32 = undefined;
    var unconditional: [n + 8]f32 = undefined;
    var actual: [n + 8]f32 = undefined;
    for (&conditional, &unconditional, 0..) |*c, *u, i| {
        c.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 97)) - 48)) / 8;
        u.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 73)) - 36)) / 16;
    }
    conditional[0] = -0.0;
    const cbuf = try g.alloc(@sizeOf(@TypeOf(conditional)));
    const ubuf = try g.alloc(@sizeOf(@TypeOf(unconditional)));
    for ([_]f32{ 1, 1.5, 5 }) |gamma| {
        // Gamma one must preserve bits even if an unused unconditional value is NaN.
        unconditional[0] = if (gamma == 1) std.math.nan(f32) else -2;
        try g.put(cbuf, std.mem.asBytes(&conditional));
        try g.put(ubuf, std.mem.asBytes(&unconditional));
        mixLogits(g, cbuf, ubuf, n, gamma);
        try g.get(cbuf, std.mem.asBytes(&actual));
        for (actual, conditional, unconditional, 0..) |got, c, u, i| {
            if (gamma == 1 or i >= n) {
                if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(c))) return error.GuidanceIdentityMismatch;
            } else {
                const expected = @as(f64, c) + (@as(f64, gamma) - 1) * (@as(f64, c) - @as(f64, u));
                if (!std.math.isFinite(got) or @abs(@as(f64, got) - expected) > 1e-6)
                    return error.GuidanceFormulaMismatch;
            }
        }
    }
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 0, 5.01 }) |bad| {
        var state: State = .{};
        if (state.begin(2, bad)) |_| return error.GuidanceAcceptedInvalidGamma else |err| {
            if (err != error.BadDramaGuidance or state.active) return error.GuidanceAcceptedInvalidGamma;
        }
    }
}
