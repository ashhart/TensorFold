//! Times the release projections at the world-4 served shapes: GB/s of weight and activation bytes, and TFLOPS.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const kr = dsv41.release;
const split = @import("dsv41_kernels").release.split;

const Shape = struct { name: []const u8, fp8: bool, k: usize, n: usize, f32_out: bool = false };
const shapes = [_]Shape{
    .{ .name = "wq_a", .fp8 = true, .k = 5120, .n = 1280 },
    .{ .name = "wkv", .fp8 = true, .k = 5120, .n = 512 },
    .{ .name = "wq_b", .fp8 = true, .k = 1280, .n = 8192 },
    .{ .name = "ix_wq_b", .fp8 = true, .k = 1280, .n = 4096 },
    .{ .name = "wo_a group", .fp8 = true, .k = 4096, .n = 1024 },
    .{ .name = "wo_b", .fp8 = true, .k = 2048, .n = 5120 },
    .{ .name = "shared w1/w3", .fp8 = true, .k = 5120, .n = 576 },
    .{ .name = "shared w2", .fp8 = true, .k = 576, .n = 5120 },
    .{ .name = "engram wkv", .fp8 = true, .k = 6144, .n = 25600 },
    .{ .name = "main_proj", .fp8 = true, .k = 15360, .n = 5120 },
    .{ .name = "head", .fp8 = false, .k = 5120, .n = 32384, .f32_out = true },
    .{ .name = "compressor r2", .fp8 = false, .k = 5120, .n = 512, .f32_out = true },
    .{ .name = "ix_wk", .fp8 = false, .k = 512, .n = 128 },
};
const rows = [_]usize{ 1, 4, 16, 64, 512, 2048 };
/// Weight copies cycle through at least this many bytes, so a decode launch reads its weight from memory, not L2.
const cold_bytes = 256 << 20;

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

/// `release-bench [reps] [name]`: every served shape (or those whose name holds `name`) at 1 to 2048 rows.
pub fn run(gpu: check.Gpu, k: *dsv41.Kernels, rest: []const [:0]const u8) !void {
    try k.loadRelease(gpu.ctx);
    const reps: usize = if (rest.len > 0) try std.fmt.parseInt(usize, rest[0], 10) else 20;
    const only: ?[]const u8 = if (rest.len > 1) rest[1] else null;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const ops = k.releaseOps(s);
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    for ([_]kr.Form{ .fp8, .bf16 }) |form| for ([_]kr.Size{ .decode, .mid, .prompt }) |size| {
        const t = kr.tile(form, size);
        const per = try k.rel.?.f.staged[@backingInt(form)][@backingInt(size)].occupancy(t.threads, t.smem);
        std.debug.print("INFO {t} {t} tile {d}x{d}, {d} threads, {d} shared bytes: {d} CTAs an SM\n", .{ form, size, t.rows, t.cols, t.threads, t.smem, per });
    };
    std.debug.print("RESULT shape,K,N,S,rows,us,GB/s,TFLOPS\n", .{});
    for (shapes) |sh| {
        if (only) |w| if (std.mem.indexOf(u8, sh.name, w) == null) continue;
        const es: usize = if (sh.fp8) 1 else 2;
        const wbytes = sh.n * sh.k * es + if (sh.fp8) sh.n * sh.k / 1024 else 0;
        const copies = @max(1, cold_bytes / wbytes);
        var w = try cuda.DeviceBuffer.alloc(gpu.d, copies * wbytes);
        defer w.free();
        try w.fill8(0x22, s.handle);
        const top = rows[rows.len - 1];
        var a = try cuda.DeviceBuffer.alloc(gpu.d, top * sh.k * es + top * sh.k / 32);
        defer a.free();
        try a.fill8(0x31, s.handle);
        var y = try cuda.DeviceBuffer.alloc(gpu.d, top * sh.n * 4);
        defer y.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(4, kr.partBytes(top, sh.n, sh.k)));
        defer part.free();
        for (rows) |m| {
            var ms: [64]f32 = undefined;
            const n = @min(reps, ms.len);
            var i: usize = 0;
            while (i < n + 2) : (i += 1) {
                const wc = w.ptr + (i % copies) * wbytes;
                var p: kr.Proj = .{ .a = a.ptr, .a_ld = @intCast(sh.k), .sa = a.ptr + top * sh.k * es, .sa_ld = @intCast(sh.k / 32), .w = wc, .ws = wc + sh.n * sh.k, .y = y.ptr, .y_ld = @intCast(sh.n), .part = part.ptr, .m = @intCast(m), .n = @intCast(sh.n), .k = @intCast(sh.k), .f32 = @intFromBool(sh.f32_out) };
                if (!sh.fp8) p.sa = 0;
                try t0.record(s);
                try if (sh.fp8) ops.fp8(p) else ops.bf16(p);
                try t1.record(s);
                try t1.synchronize();
                if (i >= 2) ms[i - 2] = try cuda.Event.elapsedMs(t0, t1);
            }
            const t = median(ms[0..n]) * 1e-3;
            const bytes: f64 = @floatFromInt(wbytes + m * sh.k * es + m * sh.n * @as(usize, if (sh.f32_out) 4 else 2));
            const flops: f64 = @floatFromInt(2 * m * sh.n * sh.k);
            std.debug.print("RESULT {s},{d},{d},{d},{d},{d:.1},{d:.1},{d:.2}\n", .{ sh.name, sh.k, sh.n, split(sh.k, sh.n), m, t * 1e6, bytes / t * 1e-9, flops / t * 1e-12 });
        }
    }
}
