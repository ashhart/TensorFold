//! The proposal projection's quantizer, packing and GPU arithmetic against an independent FP64 reference.
const std = @import("std");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const k = 2560;
const n = 137;
const most = 7;

fn half(bytes: []const u8, i: usize) f64 {
    return rig.bf(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rng = std.Random.DefaultPrng.init(0xaff14e);
    var r = try rig.Rig.init(gpu, a, rng.random());
    defer r.deinit();
    const source = r.bfs(n * k, 0.125);
    const words = try a.alloc(u8, n * k / 2);
    const scales = try a.alloc(u8, n * k / 16);
    const biases = try a.alloc(u8, scales.len);
    for (0..n) |i| try flash.draft_affine.row(std.mem.sliceAsBytes(source[i * k ..][0..k]), words[i * k / 2 ..][0 .. k / 2], scales[i * k / 16 ..][0 .. k / 16], biases[i * k / 16 ..][0 .. k / 16]);
    var layout = try flash.qmm.pack(a, words, scales, biases, n, k / 8);
    defer layout.deinit(a);
    const dw = try r.dev(layout.weight);
    const ds = try r.dev(layout.scales);
    const db = try r.dev(layout.biases);
    const x = r.bfs(most * k, 0.25);
    const sums = try a.alloc(f32, most * k / 32);
    for (sums, 0..) |*v, g| {
        v.* = 0;
        for (x[g * 32 ..][0..32]) |h| v.* += @floatCast(rig.bf(h));
    }
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dsum = try r.dev(std.mem.sliceAsBytes(sums));
    const out = try r.out(most * n * 2);
    var widest: []u16 = undefined;
    for ([_]usize{ most, 1, 3 }) |rows| {
        try flash.qmm.matmul(gpu.d, r.s, dx, dsum, dw, ds, db, out, rows, n, k);
        if (rows == most) widest = try r.back(u16, out, most * n) else try r.sameRows("affine proposal rows", u16, out, widest[0 .. rows * n]);
    }
    for (0..most) |row| for (0..n) |col| {
        var total: f64 = 0;
        var magnitude: f64 = 0;
        for (0..k) |i| {
            const at = col * k + i;
            const code = (words[at / 2] >> @as(u3, @intCast((i % 2) * 4))) & 15;
            const g = col * (k / 32) + i / 32;
            const weight = @as(f64, @floatFromInt(code)) * half(scales, g) + half(biases, g);
            const term = rig.bf(x[row * k + i]) * weight;
            total += term;
            magnitude += @abs(term);
        }
        const want = rig.rbf(total);
        try r.within("affine proposal FP64", rig.bf(widest[row * n + col]), want, rig.ulps(want, 1) + rig.sumTol(magnitude, k));
    };
    r.report("group-32 proposal quantization and affine projection, every column; rows 1/3/7 equal");
}
