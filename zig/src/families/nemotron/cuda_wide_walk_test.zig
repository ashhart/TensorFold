//! A 32-row Lightning shared round walked through every call's row limit: none refuses it (no WindowTooWide).
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const state = @import("cuda_state.zig");
const wide = @import("cuda_wide.zig");
const kern = @import("cuda_kernels.zig");

/// Lightning's config as the loader parses it (its quantization block, 4-bit in groups of 64), the switch on or off.
fn lightning(on: bool) !config.Config {
    var why: config.Why = .{};
    var c = try config.parse(std.testing.allocator, config.lightning, &why);
    try std.testing.expect(wide.allowed(c, true, false)); // GB10, one rank, non-latent, not Bolt: the switch applies
    c.wide_rounds = on;
    return c;
}

test "wide round walk: Lightning's matrices are affine-4 64-groups, so every dense call takes the affine-4 path" {
    const c = try lightning(true);
    var buf: [96]u8 = undefined;
    try std.testing.expect(c.other(&buf) == null); // no matrix at another width or group (fourOnly passes on CUDA)
    try std.testing.expectEqual(@as(usize, 4), c.bits);
    try std.testing.expectEqual(@as(usize, 64), c.group_size);
    try std.testing.expectEqual(config.Format.mlx, c.format);
}

test "wide round walk: a 32-row shared round passes every row limit on the way, 16 rows a stream" {
    const c = try lightning(true);
    const rows = state.sharedRows(c);
    try std.testing.expectEqual(@as(usize, 32), rows);
    // the forward's window (cuda_forward.zig windowRows): the round's rows; D1050 served RED here
    try std.testing.expect(wide.windowFits(c, rows));
    try std.testing.expect(!wide.windowFits(c, rows + 1));
    // each stream's Mamba conv and scan rows (glue's rmax, the per-sequence stride): its own window, 16 rows
    try std.testing.expectEqual(@as(usize, 16), state.windowRows(c));
    // the dense calls (in_proj, out_proj, qkv, o, the head): affine-4 rows past 16 take verify (verifyRows, PDL)
    var a4: cuda.qlinear.Affine4Kernels = undefined;
    a4.pdl = true;
    a4.wide_verify = false;
    try std.testing.expect(rows <= a4.verifyRows());
    try std.testing.expect(rows > cuda.qlinear.Affine4Kernels.max_rows); // and the swap is what those rows take
    // the routed plan stages the round's rows, the shared expert's plan holds them in 16-row items
    try std.testing.expect(rows <= kern.plan_rows);
    for (1..rows + 1) |r| {
        const w = wide.sharedItems(@intCast(c.experts), 32, r);
        var held: usize = 0;
        for (0..w[12]) |i| held += w[3 * i + 2];
        try std.testing.expectEqual(2 * r, held); // both halves' rows, every one in an item
        for (0..w[12]) |i| try std.testing.expect(w[3 * i + 2] <= 16);
    }
    // the copies of a batched draft or shared draw: a part a stream
    try std.testing.expect(wide.streams(c) <= wide.wide_streams);
}

test "wide round walk: with the switch off every limit stays at today's 16 rows and 8 streams" {
    const c = try lightning(false);
    try std.testing.expectEqual(@as(usize, 16), state.sharedRows(c));
    try std.testing.expect(wide.windowFits(c, 16));
    try std.testing.expect(!wide.windowFits(c, 17)); // as the forward refused before
    try std.testing.expectEqual(@as(usize, 8), wide.streams(c));
}

test "wide round walk: a stream past its own window's rows is refused, however wide the round (REV60 3)" {
    const c = try lightning(true);
    const rows = state.windowRows(c);
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.c = c;
    var seq: state.Seq = undefined;
    const ids = [_]u32{1};
    try std.testing.expectError(error.BadWindow, e.verifyShared(&.{.{ .seq = &seq, .ids = &ids, .rows = rows + 1 }}));
    var h: @import("cuda_mtp.zig").Head = undefined;
    h.e = &e;
    var parts = [_]@import("cuda_mtp.zig").Part{.{ .k_cache = 0, .v_cache = 0, .probs = 0, .ids = 0, .hidden = 0, .sampled = 0, .rule = null, .pos = 0, .keep = rows + 1, .depth = 1 }};
    try std.testing.expectError(error.BadDraftRequest, h.chains(&parts));
    const glue = @import("cuda_glue.zig");
    const g: glue.Glue = .{ .set = null, .f = undefined, .s = undefined, .emb = undefined, .norms = undefined, .moe = undefined };
    const shape: glue.Shape = .{ .proj = 1, .xd = 1, .cd = 1, .heads = 1, .dh = 32, .groups = 1, .state = 128, .rmax = rows };
    const seg: glue.MambaSeg = .{ .proj = 0, .base = 0, .raw = 0, .xc = 0, .dt = 0, .state = 0, .meta = 0, .y = 0, .rows = 1 };
    var past = seg;
    past.rows = rows + 1;
    try std.testing.expectError(error.ScanShape, g.mambaSegs(&.{ seg, past }, 0, 0, 0, 0, 0, 0, 0, shape)); // the batched conv
    try std.testing.expectError(error.ScanShape, g.mambaSegs(&.{past}, 0, 0, 0, 0, 0, 0, 0, shape)); // a stream alone
}

test "wide round walk: with the switch the lanes learn what a row past 16 costs, not a wider fixed overhead (REV60 4)" {
    for ([_]bool{ false, true }) |on| {
        var e: @import("cuda_engine.zig").Engine = undefined;
        e.c = try lightning(on);
        var b: @import("cuda_lanes.zig").Cuda = undefined;
        b.e = &e;
        b.head = null;
        b.copy_drafts = false;
        b.cost_count = 0;
        b.mtp_ms = 0;
        b.mtp_drafts = null;
        b.mtp_confidence = null;
        const f = b.facts();
        // 17-32 rows are priced from the measured 16 linearly: depth.zig's row cost learns what they take here
        try std.testing.expectEqual(on, f.learn_row_costs);
        try std.testing.expectEqual(@as(u32, if (on) 32 else 16), f.batch_rows);
    }
}

test "wide round walk: a round past 8 streams with lessons attached is refused, not served without them (REV74 1)" {
    const weights = @import("cuda_weights.zig");
    const fwd = @import("cuda_forward.zig");
    var blk: weights.Block = .{ .kind = .attention, .norm = 0 };
    blk.attn.adapter = @as(weights.Adapter, undefined);
    var w: weights.Weights = undefined;
    w.blocks = (&blk)[0..1];
    var f: fwd.Forward = undefined;
    f.w = &w;
    var segs: [9]fwd.Seg = undefined;
    try std.testing.expectError(error.LessonsPastRoute, f.windowSegs(&segs));
}

test "wide round walk: with the switch off a direct draft call past 8 streams is refused, and conv checks its rows (REV74 4)" {
    const c = try lightning(false);
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.c = c;
    var h: @import("cuda_mtp.zig").Head = undefined;
    h.e = &e;
    const part: @import("cuda_mtp.zig").Part = .{ .k_cache = 0, .v_cache = 0, .probs = 0, .ids = 0, .hidden = 0, .sampled = 0, .rule = null, .pos = 0, .keep = 1, .depth = 1 };
    var parts: [9]@import("cuda_mtp.zig").Part = @splat(part);
    try std.testing.expectError(error.BadDraftRequest, h.chains(&parts));
    const glue = @import("cuda_glue.zig");
    const g: glue.Glue = .{ .set = null, .f = undefined, .s = undefined, .emb = undefined, .norms = undefined, .moe = undefined };
    const shape: glue.Shape = .{ .proj = 1, .xd = 1, .cd = 1, .heads = 1, .dh = 32, .groups = 1, .state = 128, .rmax = 16 };
    try std.testing.expectError(error.ScanShape, g.conv(0, 0, 0, 0, 0, 0, 0, 17, shape));
}

test "wide round walk: with the switch off every host buffer is sized for main's 8 streams (REV83 2)" {
    const off = try lightning(false);
    const on = try lightning(true);
    const engine = @import("cuda_engine.zig");
    // the engine's pinned words: a shared window's streams before the draw, main's 8 off
    try std.testing.expectEqual(engine.pinDraw(on) - 8 * engine.pin_prompt, engine.pinDraw(off));
    // the lanes' mapped logprob rows: main's 8 streams of 32 rows, then the first token's
    try std.testing.expectEqual(@as(usize, 8 * state.max_rows + 1), @import("cuda_lanes.zig").rowSlots(off));
    // the head's pinned words and device buffers: sized by the switch's streams
    const mtp = @import("cuda_mtp.zig");
    try std.testing.expectEqual(mtp.hostWords(16) - 8 * (1 + state.max_rows + 8 + 1), mtp.hostWords(wide.streams(off)));
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_mtp.zig"), "const S: usize = @import(\"cuda_wide.zig\").streams(c);") != null);
}

test "wide round walk: with the switch off the 17-32-row kernel paths refuse as main did (REV83 2)" {
    const src = @embedFile("cuda_kernels.zig");
    // the dense call routes past 16 rows only with the switch (shared qlinear.decode is main's)
    try std.testing.expect(std.mem.indexOf(u8, src, "if (o.k.wide and q == .affine4 and rows > cuda.qlinear.Affine4Kernels.max_rows) return o.k.affine.verify(") != null);
    // the routed plan admits 32 rows only with it, main's 16 without
    try std.testing.expect(std.mem.indexOf(u8, src, "if (rows > @as(usize, if (o.k.wide) plan_rows else 16) or slots > 8 or count > 128) return error.PlanTooWide;") != null);
    // the kernels' flag is the switch's, set before any buffer, and false until then
    try std.testing.expect(std.mem.indexOf(u8, src, "var k: Kernels = undefined;\n        k.wide = false;") != null);
    try std.testing.expect(std.mem.indexOf(u8, @embedFile("cuda_engine.zig"), "e.k.wide = e.c.wide_rounds;") != null);
}
