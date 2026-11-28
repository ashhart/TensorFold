//! TF_DSV41_FLOAT_LINEAR: a projection a pack keeps as a float matrix alone, planned and run as that matrix.

const std = @import("std");
const block = @import("block.zig");
const knobs = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;
const Emitter = block.Emitter;

/// The width table's names, under a block, of the projections `names.plainOf` covers.
pub const projections = [_][]const u8{ "wq_a", "comp_wkv", "comp_wgate", "ix_wk", "ix_wq_b" };

/// plain_fp64.BK: the K step `_plain` accumulates in, ascending, whatever the row count.
const plain_bk: i64 = 256;

/// A float projection: `_plain`'s fp64 sums, rounded to fp32 then bf16 in the launch, into `p.y` from column `p.col`.
pub fn plain(e: *Emitter, p: Emitter.Proj) !void {
    const k: i64 = @intCast(p.k);
    const n: i64 = @intCast(p.n);
    const y = p.y.t;
    if (y.dt != .bf16 or y.shape.len != 2 or p.x.t.shape.len != 2) return error.Unsupported;
    // `_plain` has no output stride of its own: it stores at OUT + row * N + output, so N is `y`'s row stride
    const stride = y.stride[0];
    const out = try e.view(y.role, .bf16, &.{ e.n, n }, &.{ stride, 1 }, y.offset + 2 * @as(i64, @intCast(p.col)));
    try e.triton("_plain", .{ e.n, n, 1 }, &.{
        .{ .name = "X", .arg = p.x },                                                 .{ .name = "x_rs", .arg = .{ .i = p.x.t.stride[0] } },
        .{ .name = "W", .arg = try e.weight("{s}.plain", .{p.w}, .f32, &.{ n, k }) }, .{ .name = "w_rs", .arg = .{ .i = k } },
        .{ .name = "OUT", .arg = out },                                               .{ .name = "K", .arg = .{ .i = k } },
        .{ .name = "N", .arg = .{ .i = stride } },                                    .{ .name = "BK", .arg = .{ .i = plain_bk } },
        .{ .name = "PDL", .arg = .{ .b = false } },                                   .{ .name = "BF16", .arg = .{ .b = true } },
    });
}

/// `w` with every projection a backbone block may keep float at width 0, which `block.Widths.plain` reads.
pub fn floatWidths(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths) !block.Widths {
    var out: block.Widths = .{ .dense = try w.dense.clone(a), .experts = w.experts, .gm = w.gm };
    var nb: [64]u8 = undefined;
    for (0..cfg.layers) |L| for (projections) |proj| {
        if (out.dense.getPtr(try std.fmt.bufPrint(&nb, "L{d}.attn.{s}", .{ L, proj }))) |k2| k2.* = 0;
    };
    return out;
}

/// `aot-needs`: the width tables its programs are enumerated over, the second the setting's float projections.
pub fn tables(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, get: knobs.Get, log: *std.Io.Writer) ![]const block.Widths {
    if (!try knobs.floatLinear(get)) return a.dupe(block.Widths, &.{w.*});
    try log.print("aot-needs: TF_DSV41_FLOAT_LINEAR, a second pass with every projection a pack may keep float\n", .{});
    try log.flush();
    return a.dupe(block.Widths, &.{ w.*, try floatWidths(a, cfg, w) });
}

test "the float width table takes a backbone block's five projections to width 0, and nothing else" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    var nb: [64]u8 = undefined;
    for (projections) |proj| {
        try w.dense.put(a, try std.fmt.allocPrint(a, "L2.attn.{s}", .{proj}), 10);
        try w.dense.put(a, try std.fmt.allocPrint(a, "L{d}.attn.{s}", .{ cfg.layers, proj }), 10);
    }
    try w.dense.put(a, "L2.attn.wkv", 10);
    const f = try floatWidths(a, &cfg, &w);
    for (projections) |proj| {
        try std.testing.expect(f.plain(try std.fmt.bufPrint(&nb, "L2.attn.{s}", .{proj})));
        try std.testing.expect(!f.plain(try std.fmt.bufPrint(&nb, "L{d}.attn.{s}", .{ cfg.layers, proj })));
        try std.testing.expect(!w.plain(try std.fmt.bufPrint(&nb, "L2.attn.{s}", .{proj})));
    }
    try std.testing.expect(!f.plain("L2.attn.wkv") and !f.plain("L2.attn.wq_b"));
}

test "aot-needs enumerates the float width table only when TF_DSV41_FLOAT_LINEAR is set" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try w.dense.put(a, "L2.attn.wq_a", 10);
    var log: std.Io.Writer.Allocating = .init(a);
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_FLOAT_LINEAR")) value else null;
        }
    };
    for ([_]?[]const u8{ null, "0", "1" }, [_]usize{ 1, 1, 2 }) |value, count| {
        Env.value = value;
        const ts = try tables(a, &cfg, &w, &Env.get, &log.writer);
        try std.testing.expectEqual(count, ts.len);
        try std.testing.expect(!ts[0].plain("L2.attn.wq_a"));
        if (count == 2) try std.testing.expect(ts[1].plain("L2.attn.wq_a"));
    }
}
