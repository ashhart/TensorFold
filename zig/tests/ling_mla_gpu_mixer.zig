//! Ling MLA mixer on Metal vs mla.zig f64 and bf16-point.
const std = @import("std");
const mtl = @import("metal");
const mla = @import("mla");
const c = @import("ling_mla_gpu_common.zig");

fn checkMixer(rows: u32) !void {
    const a = std.testing.allocator;
    const s = mla.Shape{ .heads = 16, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const in = try c.Inputs.make(a, s, rows, 0x6d6c61);
    defer in.free(a);
    const n = rows * s.outWidth();
    const want64 = try a.alloc(f64, n);
    defer a.free(want64);
    const want_bf = try a.alloc(f64, n);
    defer a.free(want_bf);
    try mla.mixF64(a, want64, in.q, in.kv, in.w, in.kvb, in.g, s, rows, 0);
    try mla.mixBf16(a, want_bf, in.q, in.kv, in.w, in.kvb, in.g, s, rows, 0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try c.Gpu.init();
    defer g.deinit();
    var l = try c.Launch.open(g, s, rows, in.q, in.kv, in.w, in.kvb, in.g);
    defer l.deinit();
    try l.dispatchMixer(g, 0, rows);
    var buf: [64]u8 = undefined;
    const tag = try std.fmt.bufPrint(&buf, "mla {d} rows", .{rows});
    try c.gateNamed(tag, l.y(), want64, want_bf);
}

test "MLA mixer Metal vs f64 and bf16-point on 2 rows from pos 0" {
    try checkMixer(2);
}

test "MLA mixer Metal vs f64 and bf16-point on 4 rows from pos 0" {
    try checkMixer(4);
}
