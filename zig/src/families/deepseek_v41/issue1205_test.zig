//! Four-rank prefill plans both normal segment sizes and charges the expert views they actually emit.
const std = @import("std");
const t = std.testing;
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const Config = @import("config.zig").Config;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try @import("widths_listing.zig").parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try @import("widths_listing.zig").parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..43) |layer| try w.gm.put(a, @intCast(layer), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    for (0..43) |layer| for (4..8) |g| {
        const first = try std.fmt.allocPrint(a, "L{d}.attn.wo_a.0", .{layer});
        try w.dense.put(a, try std.fmt.allocPrint(a, "L{d}.attn.wo_a.{d}", .{ layer, g }), try w.k2(first));
    };
    return w;
}

test "issue1205: world four plans a 2048-row prefill segment after attention and expert scratch admission" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{ .expert_width = 2560, .vocab = 129536 };
    const cs = try bp.emitPrefill(a, &cfg, &w, .{ .world = 4 }, &.{0}, 2048, 0, true);
    var plan: buffers.Plan = .{ .a = a };
    try plan.add(cs);
    try t.expect(plan.sizes.get("s.ex.z").? >= 2048 * 6 * 5120 * 4);
    try t.expect(plan.sizes.get("s.ex.xg").? >= 2048 * 6 * 640 * 2);
}

test "issue1202: the shipped expert and vocabulary dimensions admit rank zero of four" {
    const cfg: Config = .{};
    const Opts = struct {
        world: u32 = 4,
        rank: u32 = 0,
        expert_rows: i64 = 1024,
        dspark_rows: i64 = 64,
        branch_rows: i64 = 0,
        prefill_rows: i64 = 0,
        image_rows: i64 = 0,
        expert_topp: ?f32 = null,
    };
    try @import("block_admission.zig").options(&cfg, Opts{}, &.{0}, 1);
}

test "issue1205: every rank plans short and normal segments at fresh and deep positions" {
    const cfg: Config = .{};
    for (0..4) |rank| for ([_]i64{ 16, 17, 1024, 2048 }) |rows| for ([_]i64{ 0, 300000 }) |position| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var w = try widths(a);
        const opts: block.Options = .{ .world = 4, .rank = @intCast(rank) };
        const cs = try bp.emitPrefill(a, &cfg, &w, opts, &.{0}, rows, position, true);
        var plan: buffers.Plan = .{ .a = a };
        try plan.add(cs);
        const share = try @import("partition.zig").share(cfg.expert_width, @intCast(rank), 4);
        try t.expect(plan.sizes.get("s.ex.z").? >= @as(u64, @intCast(rows * 6 * cfg.hidden * 4)));
        try t.expect(plan.sizes.get("s.ex.xg").? >= @as(u64, @intCast(rows * 6 * share.len * 2)));
        var rotations: usize = 0;
        for (cs) |call| if (std.mem.eql(u8, call.name, "tf_dsv41_x3gm_v1.rot")) {
            rotations += 1;
            try t.expectEqual(rows * 6, call.args[9].arg.i);
        };
        try t.expectEqual(@as(usize, 1), rotations);
    };
}

test "issue1205: two ranks retain normal prefill and the old non-PF4K large-segment refusal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w = try widths(a);
    const cfg: Config = .{};
    _ = try bp.emitPrefill(a, &cfg, &w, .{ .world = 2 }, &.{0}, 2048, 0, true);
    try t.expectError(error.Unsupported, bp.emitPrefill(a, &cfg, &w, .{ .world = 2 }, &.{0}, 4096, 0, true));
}
