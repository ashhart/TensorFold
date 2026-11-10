//! DFlash's Qwen3 RMSNorm rounds the normalized row before its BF16 gain; the target's norm differs.
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");

pub fn enabled(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadDFlashNormalization;
}

fn setting() !bool {
    return enabled(try @import("sys").environment.given("TF_DRAFT_QWEN3_NORM"));
}

pub fn open(d: *const cuda.Driver) !?cuda.norms.Norms {
    return if (try setting()) try cuda.norms.Norms.load(d) else null;
}

pub fn close(value: *?cuda.norms.Norms) void {
    if (value.*) |*n| n.unload();
    value.* = null;
}

pub fn run(d: anytype, h: u64, w: u64, out: u64, rows: usize) !void {
    const e = d.e;
    if (try setting()) {
        const n = if (d.qwen3_norm) |*value| value else return error.MissingDFlashNormalization;
        // No residual/branch/scalar: h is unused. Keep the original affine projection's group-sum order.
        try n.addRmsTail(e.stream, h, null, w, 0, out, null, rows, d.c.hidden, d.c.eps, .{ .round = true });
        try e.run(e.fns.sums, .{ rows, (d.c.hidden / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ out, d.xs, @as(c_int, @intCast(d.c.hidden)), @as(u64, 0) });
    } else {
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ h, @as(u64, 0), w, out, d.c.eps, @as(c_int, @intCast(d.c.hidden)), @as(u64, 0), d.xs });
    }
}

test "DFlash Qwen3 normalization accepts explicit modes and refuses malformed switches" {
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(!try enabled("0"));
    try std.testing.expect(try enabled("1"));
    for ([_][]const u8{ "", "true", "2", "01" }) |value| try std.testing.expectError(error.BadDFlashNormalization, enabled(value));
}

test "DFlash Qwen3 normalization dispatches the rounded row before original group sums" {
    const Fake = struct {
        const Norms = struct {
            calls: usize = 0,
            pub fn addRmsTail(n: *@This(), _: void, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: ?u64, rows: usize, width: usize, _: f32, options: anytype) !void {
                try std.testing.expectEqual(@as(u64, 100), x);
                try std.testing.expectEqual(@as(?u64, null), r);
                try std.testing.expectEqual(@as(u64, 200), w);
                try std.testing.expectEqual(@as(u64, 0), h);
                try std.testing.expectEqual(@as(u64, 300), y);
                try std.testing.expectEqual(@as(?u64, null), xs);
                try std.testing.expectEqual(@as(usize, 7), rows);
                try std.testing.expectEqual(@as(usize, 5120), width);
                try std.testing.expect(options.round);
                n.calls += 1;
            }
        };
        const Engine = struct {
            stream: void = {},
            fns: struct { norm: u8 = 1, sums: u8 = 2 } = .{},
            calls: usize = 0,
            last: u8 = 0,
            pub fn run(e: *@This(), function: u8, _: anytype, _: anytype, arguments: anytype) !void {
                e.calls += 1;
                e.last = function;
                if (@typeInfo(@TypeOf(arguments)).@"struct".field_types.len == 4) {
                    try std.testing.expectEqual(@as(u64, 300), arguments[0]);
                    try std.testing.expectEqual(@as(u64, 400), arguments[1]);
                }
            }
        };
        e: *Engine,
        c: struct { hidden: usize = 5120, eps: f32 = 1e-6 } = .{},
        xs: u64 = 400,
        qwen3_norm: ?Norms = .{},
    };
    var e: Fake.Engine = .{};
    var d: Fake = .{ .e = &e };
    _ = sys.env.setenv("TF_DRAFT_QWEN3_NORM", "1", 1);
    defer _ = sys.env.unsetenv("TF_DRAFT_QWEN3_NORM");
    try run(&d, 100, 200, 300, 7);
    try std.testing.expectEqual(@as(usize, 1), d.qwen3_norm.?.calls);
    try std.testing.expectEqual(@as(usize, 1), e.calls);
    try std.testing.expectEqual(@as(u8, 2), e.last);
    d.qwen3_norm = null;
    try std.testing.expectError(error.MissingDFlashNormalization, run(&d, 100, 200, 300, 7));
    _ = sys.env.setenv("TF_DRAFT_QWEN3_NORM", "0", 1);
    try run(&d, 100, 200, 300, 7);
    try std.testing.expectEqual(@as(usize, 2), e.calls);
    try std.testing.expectEqual(@as(u8, 1), e.last);
}
