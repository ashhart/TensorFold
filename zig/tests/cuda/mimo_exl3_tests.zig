//! MiMo's EXL3 kernels on one checkpoint tensor: the decode bit for bit against core/exl3_format.zig, linear vs fp64.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Gpu = check.Gpu;
const fmt = core.exl3_format;
const exl3 = cuda.exl3;

pub fn run(gpu: Gpu, file_path: []const u8, prefix: []const u8) !void {
    const a = gpu.gpa;
    var file = try core.safetensors.File.open(a, gpu.io, file_path);
    defer file.close(gpu.io);
    var nb: [256]u8 = undefined;
    const t = file.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix})) orelse return error.NoTrellis;
    try check.expect(t.dtype == .i16 and t.rank == 3 and t.shape[2] % 8 == 0, "{s}.trellis must be int16 [K/16, N/16, 8 * K2]", .{prefix});
    const tk, const tn, const k2: u32 = .{ t.shape[0], t.shape[1], @intCast(t.shape[2] / 8) };
    const k, const n = .{ tk * 16, tn * 16 };
    const width: fmt.Width = .{ .halves = @intCast(k2) };
    const trellis = try a.alloc(i16, t.numel());
    defer a.free(trellis);
    @memcpy(std.mem.sliceAsBytes(trellis), t.bytes);

    // reference decode (rotated domain, fp16)
    const want = try a.alloc(f16, k * n);
    defer a.free(want);
    fmt.unpack(width, .mul1, trellis, tk, tn, want);

    var kern = try exl3.Kernels.load(gpu.d);
    defer kern.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    // stored layout through dequant_kernel
    var dev_t = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(trellis));
    defer dev_t.free();
    var dev_w = try cuda.DeviceBuffer.alloc(gpu.d, k * n * 2);
    defer dev_w.free();
    try kern.dequant(s, dev_t.ptr, dev_w.ptr, k, n, k2);
    try s.synchronize();
    const got = try check.download(gpu, dev_w);
    defer a.free(got);
    try check.sameBytes("dequant_kernel (stored layout) against exl3_format.unpack", got, std.mem.sliceAsBytes(want));
    check.pass("{s}: dequant_kernel K2 {d}, [{d}, {d}]: bit-equal to the reference decode", .{ prefix, k2, k, n });

    // strips layout through unpack_kernel: [K/16, N/16, W] int32 words -> [N/128, K/16, 8, W] (linear.strips)
    const w32 = 4 * @as(usize, k2);
    const words: []align(2) const u32 = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(trellis));
    const strips = try a.alloc(u32, words.len);
    defer a.free(strips);
    for (0..tk) |kt| for (0..tn / 8) |nb8| for (0..8) |j| {
        const src = ((kt * tn) + nb8 * 8 + j) * w32;
        const dst = ((nb8 * tk + kt) * 8 + j) * w32;
        @memcpy(strips[dst..][0..w32], words[src..][0..w32]);
    };
    var dev_s = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(strips));
    defer dev_s.free();
    var counters = try cuda.DeviceBuffer.alloc(gpu.d, 8 * (n / 128) * 4);
    defer counters.free();
    try counters.fill32(0, null);
    const suh_h, const svh_h = .{ try scales(a, &file, prefix, "suh", k), try scales(a, &file, prefix, "svh", n) };
    defer a.free(suh_h);
    defer a.free(svh_h);
    var suh = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(suh_h));
    defer suh.free();
    var svh = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(svh_h));
    defer svh.free();
    const lin: exl3.Linear = .{ .words = dev_s.ptr, .suh = suh.ptr, .svh = svh.ptr, .counters = counters.ptr, .k = k, .n = n, .k2 = k2, .split = exl3.plan(k, n) };
    try dev_w.fill8(0, null);
    try kern.unpackWeight(s, lin, dev_w.ptr);
    try s.synchronize();
    const got_s = try check.download(gpu, dev_w);
    defer a.free(got_s);
    try check.sameBytes("unpack_kernel (strips layout) against exl3_format.unpack", got_s, std.mem.sliceAsBytes(want));
    check.pass("{s}: unpack_kernel strips: bit-equal to the reference decode", .{prefix});

    if (k * n > 64 << 20) {
        check.pass("{s}: linear check skipped ({d} x {d} is past the fp64 reference's budget)", .{ prefix, k, n });
        return;
    }
    // linear: y = x @ W, W = diag(suh) H W_q H diag(svh), against fp64
    const ref_w = try a.alloc(f64, k * n);
    defer a.free(ref_w);
    const scratch = try a.alloc(f64, 2 * k * n);
    defer a.free(scratch);
    fmt.dequantize(width, .mul1, trellis, tk, tn, suh_h, svh_h, ref_w, scratch);
    var prng = std.Random.DefaultPrng.init(0x6d696d6f);
    const rnd = prng.random();
    for ([_]usize{ 1, 4, 8, 16 }) |m| {
        const x = try a.alloc(u16, m * k); // bf16 bits
        defer a.free(x);
        for (x) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.floatNorm(f32))) >> 16);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var xh = try cuda.DeviceBuffer.alloc(gpu.d, m * k * 2);
        defer xh.free();
        var y = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 4);
        defer y.free();
        var z = try cuda.DeviceBuffer.alloc(gpu.d, @max(1, lin.split[0]) * m * n * 4);
        defer z.free();
        try kern.linear(s, lin, dx.ptr, .bf16, xh.ptr, y.ptr, .f32, if (lin.split[0] > 1) z.ptr else 0, m);
        try s.synchronize();
        const yb = try check.download(gpu, y);
        defer a.free(yb);
        const yg: []align(1) const f32 = std.mem.bytesAsSlice(f32, yb);
        var num: f64 = 0;
        var den: f64 = 0;
        for (0..m) |r| for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |i| acc += @as(f64, @as(f32, @bitCast(@as(u32, x[r * k + i]) << 16))) * ref_w[i * n + c];
            const d = @as(f64, yg[r * n + c]) - acc;
            num += d * d;
            den += acc * acc;
        };
        const rel = @sqrt(num / @max(den, 1e-30));
        try check.expect(rel < 2e-2, "{s}: linear M={d}: relative L2 error {e} against fp64", .{ prefix, m, rel });
        check.pass("{s}: linear M={d} (SK {d}, WK {d}): relative L2 error {e} against the fp64 reference", .{ prefix, m, lin.split[0], lin.split[1], rel });
    }
}

/// fp16 [size] scales, or int16 sign words [size / 16] unpacked (Exl3Linear.from_tensors' scales()).
fn scales(a: std.mem.Allocator, file: *const core.safetensors.File, prefix: []const u8, part: []const u8, size: usize) ![]f16 {
    var nb: [256]u8 = undefined;
    const t = file.get(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, part })) orelse return error.NoScales;
    const out = try a.alloc(f16, size);
    errdefer a.free(out);
    if (t.dtype == .f16 and t.numel() == size) {
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    } else if (t.dtype == .i16 and t.numel() * 16 == size) {
        const w = try a.alloc(i16, t.numel());
        defer a.free(w);
        @memcpy(std.mem.sliceAsBytes(w), t.bytes);
        fmt.unpackSigns(w, out);
    } else return error.BadScales;
    return out;
}
