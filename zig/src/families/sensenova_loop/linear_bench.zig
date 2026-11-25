//! Standalone microbenchmark comparing BF16 and AWQ Metal linear kernels across SenseNova shapes.
const std = @import("std");
const mtl = @import("metal");
const awq = @import("awq.zig");
const linear = @import("linear_gpu.zig");

const LinearGpu = linear.LinearGpu;
const AwqMatrix = linear.AwqMatrix;
const Bf16Matrix = linear.Bf16Matrix;

const BenchJob = struct {
    x: mtl.Buffer,
    y: mtl.Buffer,
    bf: []const mtl.Buffer,
    aw: []const mtl.Buffer,
    sb: []const mtl.Buffer,
    n: u32,
    k: u32,
};

fn benchRun(gpu: *LinearGpu, queue: mtl.Queue, j: BenchJob, m: u32, sk: ?u32, is_awq: bool, prompt: bool) !void {
    const reps: usize = 8;
    const m5 = gpu.tensor_units;
    const layout: awq.Layout = if (m5) .m5_tile else .m3_row;
    const awq_bytes = awq.weightWords(j.n, j.k, layout) * 4 + awq.metaCount(j.n, j.k) * 8;
    const w_bytes = if (is_awq) awq_bytes else @as(usize, j.n) * j.k * 2;
    var best_s: f64 = 1e9;
    for (0..3) |_| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..reps) |r| {
            const c = r % j.bf.len;
            if (is_awq) {
                const mat = AwqMatrix{ .w = j.aw[c], .sb = j.sb[c], .n = j.n, .k = j.k, .m5_layout = m5 };
                try gpu.encodeAwq(enc, mat, .{ .buf = j.x }, null, .{ .buf = j.y }, m, j.n, prompt, sk);
            } else {
                const mat = Bf16Matrix{ .buf = j.bf[c], .n = j.n, .k = j.k };
                try gpu.encodeBf16(enc, mat, .{ .buf = j.x }, .{ .buf = j.y }, m, j.n, prompt, sk);
            }
        }
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("bench gpu err: {s}\n", .{why});
            return error.GpuFailed;
        }
        const per = cb.gpuSeconds() / @as(f64, @floatFromInt(reps));
        if (per > 0 and per < best_s) best_s = per;
    }
    const bytes: f64 = @floatFromInt(w_bytes + @as(usize, m) * (j.k + j.n) * 2);
    const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(j.n)) * @as(f64, @floatFromInt(j.k));
    const tag = if (is_awq) "AWQ " else "BF16";
    std.debug.print(
        "  {s} ({d},{d}) M={d} {s}={d}: {d:.3} ms, {d:.1} GB/s, {d:.2} TF\n",
        .{ tag, j.n, j.k, m, if (sk != null) "SK" else "pf", sk orelse 0, best_s * 1e3, bytes / best_s / 1e9, flops / best_s / 1e12 },
    );
}

fn allocCopies(device: mtl.Device, a: std.mem.Allocator, cnt: usize, bytes: usize, fill: u8) ![]mtl.Buffer {
    const list = try a.alloc(mtl.Buffer, cnt);
    var done: usize = 0;
    errdefer {
        for (list[0..done]) |b| b.deinit();
        a.free(list);
    }
    for (0..cnt) |i| {
        list[i] = try device.buffer(bytes, mtl.ResourceOptions.shared);
        @memset(list[i].contents()[0..bytes], fill);
        done += 1;
    }
    return list;
}

fn freeCopies(a: std.mem.Allocator, list: []const mtl.Buffer) void {
    for (list) |b| b.deinit();
    a.free(list);
}

/// Sweeps SK in {1,2,4,8} for decode (M=1,2,4,8,16,32) and prefill (M=128,512) on all SenseNova shapes.
pub fn bench(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
    var gpu = LinearGpu.init(a, device);
    defer gpu.deinit();
    try gpu.reserve(512, true);
    const shapes = [_][2]u32{
        .{ 248320, 4096 },
        .{ 24576, 4096 },
        .{ 4096, 12288 },
        .{ 12288, 4096 },
        .{ 10240, 4096 },
        .{ 8192, 4096 },
        .{ 4096, 4096 },
        .{ 2048, 4096 },
    };
    const copies: usize = 3;
    const opts = mtl.ResourceOptions.shared;
    std.debug.print("SenseNova microbench on {s} (tensor_units={}):\n", .{ device.name(), gpu.tensor_units });
    for (shapes) |sh| {
        const n = sh[0];
        const k = sh[1];
        const layout: awq.Layout = if (gpu.tensor_units) .m5_tile else .m3_row;
        const x_buf = try device.buffer(@as(usize, 512) * k * 2, opts);
        const y_buf = try device.buffer(@as(usize, 512) * n * 2, opts);
        defer x_buf.deinit();
        defer y_buf.deinit();
        @memset(x_buf.contents()[0 .. @as(usize, 512) * k * 2], 0x3f);

        const bf_copies = try allocCopies(device, a, copies, @as(usize, n) * k * 2, 0x3c);
        defer freeCopies(a, bf_copies);
        const awq_w = try allocCopies(device, a, copies, awq.weightWords(n, k, layout) * 4, 0x55);
        defer freeCopies(a, awq_w);
        const awq_sb = try allocCopies(device, a, copies, awq.metaCount(n, k) * 8, 0x3f);
        defer freeCopies(a, awq_sb);
        const job = BenchJob{ .x = x_buf, .y = y_buf, .bf = bf_copies, .aw = awq_w, .sb = awq_sb, .n = n, .k = k };
        for ([_]u32{ 1, 2, 4, 8, 16, 32 }) |m| {
            for ([_]u32{ 1, 2, 4, 8 }) |sk| {
                for ([_]bool{ false, true }) |is_awq| {
                    try benchRun(&gpu, queue, job, m, sk, is_awq, false);
                }
            }
        }
        for ([_]u32{ 1, 4, 8, 32, 128, 512 }) |m| {
            for ([_]bool{ false, true }) |is_awq| {
                try benchRun(&gpu, queue, job, m, null, is_awq, m > 32);
            }
        }
    }
}
