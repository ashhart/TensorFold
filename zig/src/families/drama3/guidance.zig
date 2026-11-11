//! Semantic CFG's host lifecycle, shared by the Metal and CUDA guides; each backend owns its frame step.
const std = @import("std");
const W = @import("config.zig").frame_width;

pub fn validGamma(gamma: f32) bool {
    return std.math.isFinite(gamma) and gamma >= 1 and gamma <= 5;
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

    /// The gamma a guided frame mixes with; none at gamma 1, so its logits stay exactly the conditional ones.
    pub fn mixing(s: State) ?f32 {
        return if (s.active and s.gamma != 1) s.gamma else null;
    }
};

/// An unconditional backbone state over the conditional engine's weights, prefilled one row per scheduler round.
pub fn Guide(comptime Engine: type) type {
    return struct {
        const Self = @This();
        e: *Engine,
        batched: bool,
        state: State = .{},

        pub fn create(base: *Engine, batched: bool) !Self {
            return .{ .e = try base.forkBackbone(), .batched = batched };
        }

        pub fn destroy(g: *Self) void {
            g.e.destroy();
        }

        pub fn begin(g: *Self, rows: []const [W]i32, gamma: f32) !void {
            try g.state.begin(rows.len, gamma);
            errdefer g.state.clear();
            // A guide is never restored from the conditional reference cache.
            if (g.state.active) try g.e.startPrompt(rows);
        }

        /// One prompt row per scheduler round keeps cancellation and other lanes responsive.
        pub fn promptStep(g: *Self, rows: []const [W]i32) !bool {
            if (!g.state.active or g.e.rows_done == rows.len) return false;
            try g.e.backboneChunk(rows.len, g.e.rows_done, 1, null);
            g.e.rows_done += 1;
            if (g.e.rows_done == rows.len) try g.e.finishPrompt(rows);
            return true;
        }

        pub fn clear(g: *Self) void {
            g.state.clear();
        }
    };
}

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

test "CFG mixes only an active guide above gamma 1, so gamma 1 keeps the conditional logits exactly" {
    var state: State = .{};
    try std.testing.expectEqual(null, state.mixing());
    try state.begin(4, 1);
    try std.testing.expect(state.active);
    try std.testing.expectEqual(null, state.mixing());
    try state.begin(4, 2.5);
    try std.testing.expectEqual(2.5, state.mixing().?);
    state.clear();
    try std.testing.expectEqual(null, state.mixing());
}

test "CFG guide: a cancelled or changed reference clears the guide, which prefills its own rows again" {
    const Fake = struct {
        rows_done: usize = 0,
        starts: usize = 0,
        finished: usize = 0,
        fn startPrompt(f: *@This(), _: []const [W]i32) !void {
            f.starts += 1;
            f.rows_done = 0;
        }
        fn backboneChunk(f: *@This(), n: usize, c0: usize, t: usize, _: ?*const anyopaque) !void {
            if (c0 != f.rows_done or t != 1 or c0 >= n) return error.BadPromptChunk;
        }
        fn finishPrompt(f: *@This(), rows: []const [W]i32) !void {
            f.finished += 1;
            f.rows_done = rows.len;
        }
    };
    var e: Fake = .{};
    var g: Guide(Fake) = .{ .e = &e, .batched = false };
    const voice_a: [3][W]i32 = @splat(@splat(0));
    const voice_b: [2][W]i32 = @splat(@splat(0));
    try g.begin(&voice_a, 2);
    try std.testing.expect(try g.promptStep(&voice_a));
    g.clear(); // cancelled mid-prompt
    try std.testing.expectEqual(null, g.state.mixing());
    try std.testing.expect(!try g.promptStep(&voice_a));
    try g.begin(&voice_b, 1.5); // the next request names another reference
    while (try g.promptStep(&voice_b)) {}
    try std.testing.expectEqual(2, e.starts);
    try std.testing.expectEqual(1, e.finished);
    try std.testing.expectEqual(1.5, g.state.mixing().?);
    try g.begin(&.{}, 1.5); // a plain request leaves no guidance behind
    try std.testing.expectEqual(null, g.state.mixing());
    try std.testing.expect(!try g.promptStep(&voice_b));
}
