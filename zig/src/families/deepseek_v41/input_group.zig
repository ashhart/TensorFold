//! Concurrent wq_a/wkv and owner-layer compressor wkv/wgate products read x and write separate outputs.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("../../core/exl3_projection.zig");

pub const Matrices = struct {
    wq_a: exl3.Matrix,
    wkv: exl3.Matrix,
    /// Owner layers only; wgate only where the compress ratio is above one.
    compressor_wkv: ?exl3.Matrix = null,
    compressor_wgate: ?exl3.Matrix = null,
};
/// qr/raw: BF16; values: BF16 at ratio 1, otherwise FP32; gates: FP32, matching individual products.
pub const Outputs = struct { qr: exl3.View, raw: exl3.View, values: exl3.View, gates: exl3.View };
pub const Dims = struct { hidden: u32, q_lora_rank: u32, head_dim: u32 };

/// Each product uses a separate scratch slice; their combined size must fit scratch_bytes.
pub fn encode(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, m: Matrices, out: Outputs, x: exl3.View, scratch: exl3.View, scratch_bytes: usize, rows: u32, d: Dims, ratio: u32) !void {
    var jobs: [4]exl3.Job = undefined;
    jobs[0] = .{ .weights = .{ .matrix = m.wq_a }, .x = x, .y = out.qr, .scratch = undefined, .rows = .{ .count = rows, .x_stride = d.hidden, .y_stride = d.q_lora_rank } };
    jobs[1] = .{ .weights = .{ .matrix = m.wkv }, .x = x, .y = out.raw, .scratch = undefined, .rows = .{ .count = rows, .x_stride = d.hidden, .y_stride = d.head_dim } };
    var count: usize = 2;
    if (m.compressor_wkv) |wkv| {
        jobs[count] = .{ .weights = .{ .matrix = wkv }, .x = x, .y = out.values, .scratch = undefined, .rows = .{ .count = rows, .x_stride = d.hidden, .y_stride = d.head_dim, .output = if (ratio == 1) .bf16 else .f32 } };
        count += 1;
    }
    if (m.compressor_wgate) |gate| {
        jobs[count] = .{ .weights = .{ .matrix = gate }, .x = x, .y = out.gates, .scratch = undefined, .rows = .{ .count = rows, .x_stride = d.hidden, .y_stride = d.head_dim, .output = .f32 } };
        count += 1;
    }
    var used: usize = 0;
    for (jobs[0..count]) |*job| {
        job.scratch = .{ .buffer = scratch.buffer, .offset = scratch.offset + used };
        used += std.mem.alignForward(usize, try exl3.scratchBytes(job.weights.matrix.k, job.weights.matrix.n, rows), 16);
    }
    if (used > scratch_bytes) return error.ProjectionScratchTooSmall;
    try exl3.encodeJobs(ep, e, jobs[0..count]);
}

fn testMatrix(d: mtl.Device, held: *[12]mtl.Buffer, n_held: *usize, k: usize, n: usize, halves: u16, seed: u64) !exl3.Matrix {
    const words = @as(usize, halves) * 8 * (k / 16) * (n / 16);
    const t = try d.buffer(words * 2, mtl.ResourceOptions.shared);
    held[n_held.*] = t;
    n_held.* += 1;
    const su = try d.buffer(k * 2, mtl.ResourceOptions.shared);
    held[n_held.*] = su;
    n_held.* += 1;
    const sv = try d.buffer(n * 2, mtl.ResourceOptions.shared);
    held[n_held.*] = sv;
    n_held.* += 1;
    var prng = std.Random.DefaultPrng.init(seed);
    for (t.slice(i16, words)) |*v| v.* = prng.random().int(i16);
    for (su.slice(f16, k), 0..) |*v, i| v.* = if ((i + seed) % 3 == 0) -0.75 else 1.125;
    for (sv.slice(f16, n), 0..) |*v, i| v.* = if ((i + seed) % 5 == 0) -0.25 else 0.875;
    return .{ .trellis = .{ .buffer = t }, .suh = .{ .buffer = su }, .svh = .{ .buffer = sv }, .k = k, .n = n, .width = .{ .halves = halves } };
}

test "the input product group writes what the one-by-one products write, at the released shapes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var ep = try exl3.Pipelines.load(device);
    defer ep.deinit();
    var held: [12]mtl.Buffer = undefined;
    var n_held: usize = 0;
    defer for (held[0..n_held]) |b| b.deinit();
    const wq_a = try testMatrix(device, &held, &n_held, 5120, 1280, 5, 1);
    const wkv = try testMatrix(device, &held, &n_held, 5120, 512, 6, 2);
    const cwkv = try testMatrix(device, &held, &n_held, 5120, 512, 4, 3);
    const cgate = try testMatrix(device, &held, &n_held, 5120, 512, 3, 4);
    const rows = 3;
    const d: Dims = .{ .hidden = 5120, .q_lora_rank = 1280, .head_dim = 512 };
    const x = try device.buffer(rows * 5120 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(9);
    for (x.slice(u16, rows * 5120)) |*v| v.* = @truncate(@as(u32, @bitCast((prng.random().float(f32) - 0.5) * 4)) >> 16);
    const scratch_bytes = try exl3.scratchBytes(5120, 129280, rows);
    const scratch = try device.buffer(scratch_bytes, mtl.ResourceOptions.shared);
    defer scratch.deinit();
    // qr | raw | values | gates per side, values and gates FP32-sized.
    const sizes = [_]usize{ rows * 1280 * 2, rows * 512 * 2, rows * 512 * 4, rows * 512 * 4 };
    var outs: [2][4]mtl.Buffer = undefined;
    var made: usize = 0;
    defer for (0..made) |i| outs[i / 4][i % 4].deinit();
    for (0..2) |side| for (sizes, 0..) |size, i| {
        outs[side][i] = try device.buffer(size, mtl.ResourceOptions.shared);
        made += 1;
    };
    // A plain layer (two products), a ratio-two owner (four) and the ratio-one owner (three).
    const cases = [_]struct { compressor: bool, ratio: u32 }{ .{ .compressor = false, .ratio = 0 }, .{ .compressor = true, .ratio = 2 }, .{ .compressor = true, .ratio = 1 } };
    for (cases) |case| {
        for (outs) |side| for (side, sizes) |b, size| @memset(b.contents()[0..size], 0);
        const m: Matrices = .{ .wq_a = wq_a, .wkv = wkv, .compressor_wkv = if (case.compressor) cwkv else null, .compressor_wgate = if (case.compressor and case.ratio > 1) cgate else null };
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        for (held[0..n_held]) |b| e.useResource(b, mtl.ResourceUsage.read);
        try encode(&ep, e, m, .{ .qr = .{ .buffer = outs[0][0] }, .raw = .{ .buffer = outs[0][1] }, .values = .{ .buffer = outs[0][2] }, .gates = .{ .buffer = outs[0][3] } }, .{ .buffer = x }, .{ .buffer = scratch }, scratch_bytes, rows, d, case.ratio);
        const s: exl3.View = .{ .buffer = scratch };
        try exl3.encode(&ep, e, wq_a, .{ .buffer = x }, .{ .buffer = outs[1][0] }, s, .{ .count = rows, .x_stride = 5120, .y_stride = 1280 });
        try exl3.encode(&ep, e, wkv, .{ .buffer = x }, .{ .buffer = outs[1][1] }, s, .{ .count = rows, .x_stride = 5120, .y_stride = 512 });
        if (case.compressor) {
            try exl3.encode(&ep, e, cwkv, .{ .buffer = x }, .{ .buffer = outs[1][2] }, s, .{ .count = rows, .x_stride = 5120, .y_stride = 512, .output = if (case.ratio == 1) .bf16 else .f32 });
            if (case.ratio > 1) try exl3.encode(&ep, e, cgate, .{ .buffer = x }, .{ .buffer = outs[1][3] }, s, .{ .count = rows, .x_stride = 5120, .y_stride = 512, .output = .f32 });
        }
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        for (outs[0], outs[1], sizes) |grouped, single, size|
            try std.testing.expectEqualSlices(u8, single.contents()[0..size], grouped.contents()[0..size]);
    }
}
