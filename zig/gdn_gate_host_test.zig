//! Independent float64 central differences and a wrong-family gate witness.
const std = @import("std");
const ref = @import("tests/cuda/gdn_back_ref.zig");

test "GDN gate: both analytic backward modes agree with independent FP64 differences" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d: ref.Dims = .{ .rows = 5, .nk = 1, .nv = 2, .dk = 4, .dv = 4 };
    const p = try a.alloc(f64, d.rows * d.proj());
    for (p, 0..) |*v, i| v.* = 0.7 * @sin(@as(f64, @floatFromInt(i)) * 0.17) - 0.2;
    const cw = try a.alloc(f64, d.conv() * ref.taps);
    for (cw, 0..) |*v, i| v.* = 0.2 + 0.1 * @cos(@as(f64, @floatFromInt(i)));
    const weights = [_]f64{ 0.75, 1.25, 0.5, 1.5 };
    const dout = try a.alloc(f64, d.rows * d.nv * d.dv);
    for (dout, 0..) |*v, i| v.* = @cos(@as(f64, @floatFromInt(i)) * 0.31);
    var in: ref.Inputs = .{ .p = p, .hist = null, .conv_w = cw, .a_log = &.{ -1, -0.5 }, .dt_bias = &.{ 0.1, -0.2 }, .norm_w = &weights, .init = null, .eps = 1e-6 };
    var rng = std.Random.DefaultPrng.init(1234);
    for ([_]bool{ false, true }) |silu| {
        in.silu = silu;
        const fd = try ref.finiteCheck(a, d, in, dout, rng.random(), 120);
        try std.testing.expect(fd.worst < 1e-5);
    }
    const f = try ref.forward(a, d, in, false);
    const right = try ref.backward(a, d, in, f.act, f.beta, f.y, dout, false);
    in.silu = false;
    const wrong = try ref.backward(a, d, in, f.act, f.beta, f.y, dout, false);
    var error2: f64 = 0;
    var norm2: f64 = 0;
    for (right.dy, wrong.dy) |r, w| {
        error2 += (r - w) * (r - w);
        norm2 += r * r;
    }
    try std.testing.expect(@sqrt(error2 / norm2) > 0.15);
}

test "GDN gate: 27B adapter selects SiLU and Flash Next retains the default gate" {
    const dense = @embedFile("src/families/qwen3_5/cuda_train_load.zig");
    const flash = @embedFile("src/families/flashnext/cuda_train_buf.zig");
    try std.testing.expect(std.mem.indexOf(u8, dense, "lw.gdn = .{ .gate = .silu,") != null);
    try std.testing.expect(std.mem.indexOf(u8, flash, ".gate = .silu") == null);
}
