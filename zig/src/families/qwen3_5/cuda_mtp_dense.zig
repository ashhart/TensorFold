//! The MTP layer's dense or routed MLP, sharing the target's transient activation banks on its ordered stream.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const precision = @import("cuda_draft_precision.zig");

pub fn sumWidth(g: geometry.Geometry) usize {
    return @max(g.hidden, g.attnWidth(), if (g.moe()) 0 else g.intermediate);
}

pub fn run(d: anytype, x: u64, out: u64, rows: usize) !void {
    if (rows == 0 or rows > engine.most_verify) return error.BadMtpRows;
    const m = switch (d.w.mlp) {
        .moe => |m| return d.routed(m.router, m.experts, x, out, rows),
        .dense => |m| m,
    };
    const e = d.e;
    const ni = e.g.intermediate;
    if (m.gate.inputs() != e.g.hidden or m.up.inputs() != e.g.hidden or
        m.gate.outputs() != ni or m.up.outputs() != ni or m.down.inputs() != ni or
        m.down.outputs() != e.g.hidden) return error.UnsupportedMtpGeometry;
    const prompt = precision.prompt(d.projection_role, rows, precision.mtp_projection_prompt_cutoff);
    try e.projManySums(&.{ m.gate, m.up }, x, d.xs, &.{ e.gate, e.up }, rows, prompt, 0);
    try e.run(e.fns.mlp_act, .{ (rows * ni + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, @as(c_longlong, @intCast(rows * ni)), @as(u64, 0), @as(c_int, @intCast(ni)), d.xs });
    try d.proj(m.down, e.act, out, rows);
}

test "MTP dense sums cover the full intermediate width before logits rather than stopping at attention width" {
    const g: geometry.Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 16384, .linear = @splat(false) };
    try std.testing.expectEqual(@as(usize, 17408), sumWidth(g));
    try std.testing.expect(sumWidth(g) > @max(g.hidden, g.attnWidth()));
}

test "MTP dense and routed dispatch use the actual variant and reject malformed dense dimensions" {
    const Fake = struct {
        const Self = @This();
        const E = struct {
            g: struct { hidden: usize = 5120, intermediate: usize = 17408 } = .{},
            gate: u64 = 0x2000,
            up: u64 = 0x3000,
            act: u64 = 0x4000,
            fns: struct { mlp_act: u32 = 9 } = .{},
            pairs: usize = 0,
            acts: usize = 0,
            fn projManySums(e: *@This(), ls: []const engine.Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64) !void {
                try std.testing.expectEqual(@as(usize, 2), ls.len);
                try std.testing.expectEqual(@as(u64, 0x1000), x);
                try std.testing.expectEqual(@as(u64, 0x5000), xs);
                try std.testing.expectEqualSlices(u64, &.{ e.gate, e.up }, outs);
                try std.testing.expectEqual(@as(usize, 4), rows);
                try std.testing.expect(!prompt and params == 0);
                e.pairs += 1;
            }
            fn run(e: *@This(), _: u32, _: [3]usize, _: [3]usize, args: anytype) !void {
                try std.testing.expectEqual(@as(c_longlong, 4 * 17408), args[3]);
                try std.testing.expectEqual(@as(c_int, 17408), args[5]);
                try std.testing.expectEqual(@as(u64, 0x5000), args[6]);
                e.acts += 1;
            }
        };
        e: *E,
        w: struct { mlp: engine.Mlp },
        xs: u64 = 0x5000,
        projection_role: precision.Role = .proposal,
        downs: usize = 0,
        routed_calls: usize = 0,
        fn proj(d: *Self, l: engine.Proj, x: u64, out: u64, rows: usize) !void {
            try std.testing.expectEqual(@as(usize, 17408), l.inputs());
            try std.testing.expectEqual(d.e.act, x);
            try std.testing.expectEqual(@as(u64, 0x6000), out);
            try std.testing.expectEqual(@as(usize, 4), rows);
            d.downs += 1;
        }
        fn routed(d: *Self, router: u64, _: cuda.experts.Layer, _: u64, _: u64, _: usize) !void {
            try std.testing.expectEqual(@as(u64, 7), router);
            d.routed_calls += 1;
        }
    };
    const a: cuda.qlinear.Affine4 = .{ .w = 1, .s = 2, .b = 3, .n = 17408, .k = 5120, .npad = 17408 };
    const down: cuda.qlinear.Affine4 = .{ .w = 4, .s = 5, .b = 6, .n = 5120, .k = 17408, .npad = 5120 };
    var e: Fake.E = .{};
    var d: Fake = .{ .e = &e, .w = .{ .mlp = .{ .dense = .{
        .gate = .{ .a4 = a },
        .up = .{ .a4 = a },
        .down = .{ .a4 = down },
    } } } };
    try run(&d, 0x1000, 0x6000, 4);
    try std.testing.expectEqual(@as(usize, 1), e.pairs);
    try std.testing.expectEqual(@as(usize, 1), e.acts);
    try std.testing.expectEqual(@as(usize, 1), d.downs);
    d.w.mlp.dense.down.a4.k = 5120;
    try std.testing.expectError(error.UnsupportedMtpGeometry, run(&d, 0x1000, 0x6000, 4));
    d.w.mlp = .{ .moe = .{ .router = 7, .experts = undefined } };
    try run(&d, 0x1000, 0x6000, 4);
    try std.testing.expectEqual(@as(usize, 1), d.routed_calls);
    try std.testing.expectError(error.BadMtpRows, run(&d, 0x1000, 0x6000, 0));
}
