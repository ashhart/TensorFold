//! Times the release experts (top-6 of 384, every expert held, width 576): gate/up and down at 1 to 2048 tokens.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");

const kx = dsv41.release.ex;
const hidden = 5120;
const width = 576;
const topk = 6;
const nslots = 384;
const tokens = [_]usize{ 1, 4, 16, 64, 512, 2048 };

fn median(xs: []f32) f32 {
    std.mem.sort(f32, xs, {}, std.sort.asc(f32));
    return xs[xs.len / 2];
}

fn alloc(d: *const cuda.Driver, s: cuda.Stream, len: usize, byte: u8) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, len);
    try b.fill8(byte, s.handle);
    return b;
}

/// New random routes each repetition, so the experts a launch reads come from memory, not L2.
pub fn run(gpu: check.Gpu, ops: kx.Ops, reps: usize) !void {
    const d = gpu.d;
    const s = ops.s;
    const most = tokens[tokens.len - 1] * topk;
    var bufs: [14]cuda.DeviceBuffer = undefined;
    const sizes = [_]struct { usize, u8 }{
        .{ nslots * width * hidden / 2, 0x22 },  .{ nslots * width * hidden / 32, 0x70 },
        .{ nslots * width * hidden / 2, 0x33 },  .{ nslots * width * hidden / 32, 0x70 },
        .{ nslots * hidden * width / 2, 0x44 },  .{ nslots * hidden * width / 32, 0x70 },
        .{ most / topk * hidden, 0x31 },         .{ most / topk * hidden / 32, 0x70 },
        .{ 4 * most, 0 },                        .{ 4 * most, 0x3f },
        .{ 4 * (2 * nslots + 2 + 2 * most), 0 }, .{ most * width, 0 },
        .{ most * width / 32, 0 },               .{ 2 * most * hidden, 0 },
    };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, &bufs) |sz, *b| {
        b.* = try alloc(d, s, sz[0], sz[1]);
        made += 1;
    }
    var slot_of: [nslots]i32 = undefined;
    for (&slot_of, 0..) |*v, i| v.* = @intCast(i);
    var so = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&slot_of));
    defer so.free();
    var na = try alloc(d, s, 4, 0);
    defer na.free();
    const ids = try gpu.gpa.alloc(i32, most);
    defer gpu.gpa.free(ids);
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var t0 = try cuda.Event.init(d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(d, true);
    defer t1.deinit();
    const g = bufs[10].ptr;
    std.debug.print("RESULT experts,tokens,pairs,experts read,gate/up us,GB/s,TFLOPS,down us,GB/s,TFLOPS\n", .{});
    for (tokens) |m| {
        const pairs = m * topk;
        const words = [_]u64{ g, g + 4 * (nslots + 1), g + 4 * (nslots + 1 + most), g + 4 * (nslots + 1 + 2 * most) };
        var ms: [2][64]f32 = undefined;
        var used: f64 = 0;
        const n = @min(reps, 64);
        for (0..n + 2) |i| {
            var seen: [nslots]bool = @splat(false);
            for (0..m) |r| {
                var k: usize = 0;
                while (k < topk) {
                    const id: i32 = @intCast(rng.uintLessThan(usize, nslots));
                    if (std.mem.indexOfScalar(i32, ids[r * topk ..][0..k], id) != null) continue;
                    ids[r * topk + k] = id;
                    seen[@intCast(id)] = true;
                    k += 1;
                }
            }
            try bufs[8].upload(0, std.mem.sliceAsBytes(ids[0..pairs]));
            try ops.group(.{ .ids = bufs[8].ptr, .slot_of = so.ptr, .offs = words[0], .pair_tok = words[1], .pair_of = words[2], .active = words[3], .nactive = na.ptr, .m = @intCast(m), .topk = topk, .nslots = nslots, .nids = nslots });
            try s.synchronize();
            try t0.record(s);
            try ops.gateUp(.{ .x = bufs[6].ptr, .x_ld = hidden, .xs = bufs[7].ptr, .xs_ld = hidden / 32, .w1 = bufs[0].ptr, .s1 = bufs[1].ptr, .w3 = bufs[2].ptr, .s3 = bufs[3].ptr, .wts = bufs[9].ptr, .pair_tok = words[1], .offs = words[0], .active = words[3], .nactive = na.ptr, .codes = bufs[11].ptr, .scales = bufs[12].ptr, .n = width, .k = hidden, .topk = topk }, m, nslots, .served);
            try t1.record(s);
            try t1.synchronize();
            if (i >= 2) ms[0][i - 2] = try cuda.Event.elapsedMs(t0, t1);
            try t0.record(s);
            try ops.down(.{ .x = bufs[11].ptr, .xs = bufs[12].ptr, .w2 = bufs[4].ptr, .s2 = bufs[5].ptr, .offs = words[0], .active = words[3], .nactive = na.ptr, .part = bufs[13].ptr, .n = hidden, .k = width }, m, topk, nslots, .served);
            try t1.record(s);
            try t1.synchronize();
            if (i >= 2) ms[1][i - 2] = try cuda.Event.elapsedMs(t0, t1);
            var e: f64 = 0;
            for (seen) |v| e += @floatFromInt(@intFromBool(v));
            if (i >= 2) used += e / @as(f64, @floatFromInt(n));
        }
        const p: f64 = @floatFromInt(pairs);
        const gu = median(ms[0][0..n]) * 1e-3;
        const dn = median(ms[1][0..n]) * 1e-3;
        const gu_bytes = used * (2 * width * hidden / 2 + 2 * width * hidden / 32) + p * (hidden + hidden / 32 + width + width / 32);
        const dn_bytes = used * (hidden * width / 2 + hidden * width / 32) + p * (width + width / 32 + 2 * hidden);
        const gu_flops = 2 * p * width * hidden * 2;
        const dn_flops = 2 * p * hidden * width;
        std.debug.print("RESULT experts,{d},{d},{d:.0},{d:.1},{d:.1},{d:.2},{d:.1},{d:.1},{d:.2}\n", .{ m, pairs, used, gu * 1e6, gu_bytes / gu * 1e-9, gu_flops / gu * 1e-12, dn * 1e6, dn_bytes / dn * 1e-9, dn_flops / dn * 1e-12 });
    }
}
