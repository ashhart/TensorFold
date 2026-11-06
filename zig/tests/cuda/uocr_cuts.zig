//! tf-cuda-test uocr-cuts: the decoder's prompt products cut anywhere, each row's bits equal to the uncut product's.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const uo = @import("uocr_cuda");
const cfg = uo.config;

const H = cfg.hidden;
const rows = uo.gemm.reference_rows;

/// Every (n, k) the decoder's prompt pass multiplies by: attention, router, output, the dense, shared and routed FFNs.
const shapes = [_][2]usize{
    .{ 3 * H, H },             .{ 2 * cfg.router_dim, H }, .{ H, H },
    .{ 2 * cfg.dense_ffn, H }, .{ H, cfg.dense_ffn },      .{ 2 * cfg.shared_ffn, H },
    .{ H, cfg.shared_ffn },    .{ cfg.expert_ffn, H },     .{ H, cfg.expert_ffn },
};

/// Cuts: one row, odd sizes below the prompt pass's rows, and a seam one row before its end.
const cuts = [_]usize{ 1, 7, 333, 1111, rows - 1 };

fn random(a: std.mem.Allocator, n: usize, rng: std.Random, scale: f32) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*o| o.* = uo.resize.bf16((rng.float(f32) * 2 - 1) * scale);
    return out;
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x0c12);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try uo.gemm.Gemms.init(gpu.gpa, gpu.d);
    defer g.deinit();
    for (shapes) |sh| {
        const n = sh[0];
        const k = sh[1];
        var x = try cuda.DeviceBuffer.alloc(gpu.d, rows * k * 2);
        defer x.free();
        var w = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
        defer w.free();
        var whole = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
        defer whole.free();
        var cut = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
        defer cut.free();
        try x.upload(0, std.mem.sliceAsBytes(try random(a, rows * k, prng.random(), 1)));
        try w.upload(0, std.mem.sliceAsBytes(try random(a, n * k, prng.random(), 1 / @sqrt(@as(f32, @floatFromInt(k))))));
        try g.run(s, x.ptr, w.ptr, whole.ptr, rows, n, k);
        try s.synchronize();
        const want = try a.alloc(u8, rows * n * 2);
        try whole.download(0, want);
        const got = try a.alloc(u8, rows * n * 2);
        for (cuts) |c| {
            var r0: usize = 0;
            while (r0 < rows) : (r0 += c) {
                const r = @min(c, rows - r0);
                try g.run(s, x.ptr + r0 * k * 2, w.ptr, cut.ptr + r0 * n * 2, r, n, k);
            }
            try s.synchronize();
            try cut.download(0, got);
            try check.sameBytes(try std.fmt.allocPrint(a, "uocr-cuts (n {d}, k {d}) in pieces of {d} rows", .{ n, k, c }), got, want);
        }
        check.pass("uocr-cuts (n {d}, k {d}): {d} rows in pieces of 1, 7, 333, 1111 and {d}, every row's bits the same", .{ n, k, rows, rows - 1 });
    }
}
