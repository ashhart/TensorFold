//! The 27B's embedding gather at oQ2's 8-bit and a 3-bit width: every column the float64 decode rounded once to bf16.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mlx = tf.quant.mlx;
const q27 = tf.qwen27;

const opts = mtl.ResourceOptions.shared;
const a = std.heap.page_allocator;
const group = 64;
const hidden: usize = 5120;
const vocab = 248320;
const rows = 64;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// One rounding of a float64 to bf16, ties to even (normal range; the gathered values are).
fn bf64(v: f64) u16 {
    const u: u64 = @bitCast(v);
    const r: f64 = @bitCast((u +% 0xfff_ffff_ffff +% ((u >> 45) & 1)) & ~@as(u64, 0x1fff_ffff_ffff));
    return @truncate(@as(u32, @bitCast(@as(f32, @floatCast(r)))) >> 16);
}

/// Token ids: both ends of the vocabulary, a repeat, and random rows.
fn ids(seed: u64) [rows]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    var out: [rows]u32 = undefined;
    for (&out) |*v| v.* = prng.random().uintLessThan(u32, vocab);
    out[0] = 0;
    out[1] = vocab - 1;
    out[2] = 1;
    out[3] = vocab - 2;
    out[4] = out[7];
    return out;
}

/// A [vocab, hidden] table at `bits`, the gathered rows filled with random codes, bf16 scales and biases.
const Table = struct {
    w: mtl.Buffer,
    s: mtl.Buffer,
    b: mtl.Buffer,
    bits: u8,

    fn init(device: mtl.Device, bits: u8, tokens: []const u32, seed: u64) !Table {
        const wb = hidden * bits / 8;
        const g = hidden / group;
        const t = Table{ .w = try device.buffer(vocab * wb, opts), .s = try device.buffer(vocab * g * 2, opts), .b = try device.buffer(vocab * g * 2, opts), .bits = bits };
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        for (tokens) |id| {
            r.bytes(t.w.slice(u8, vocab * wb)[id * wb ..][0..wb]);
            for (t.s.slice(u16, vocab * g)[id * g ..][0..g], t.b.slice(u16, vocab * g)[id * g ..][0..g]) |*s, *b| {
                s.* = toBf((r.float(f32) + 0.5) / 64 / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(bits - 2))));
                b.* = toBf(-(r.float(f32) + 0.5) * 0.15);
            }
        }
        return t;
    }

    fn deinit(t: Table) void {
        for ([_]mtl.Buffer{ t.w, t.s, t.b }) |x| x.deinit();
    }

    fn tensor(buffer: mtl.Buffer, dtype: tf.checkpoint.DType, c: usize, size: usize) tf.checkpoint.Tensor {
        return .{ .buffer = buffer, .offset = 0, .bytes = vocab * c * size, .dtype = dtype, .shape = .{ vocab, c, 1, 1, 1 }, .rank = 2 };
    }

    /// What gpu_weights hands the embedding.
    fn triple(t: Table) q27.projection.Triple {
        const g = hidden / group;
        return .{ .weight = tensor(t.w, .u32, hidden * t.bits / 32, 4), .scales = tensor(t.s, .bf16, g, 2), .biases = tensor(t.b, .bf16, g, 2), .bits = t.bits, .group = group };
    }

    /// Row `id` decoded in float64 by core.quant.mlx, its scales times `scale` (1 for the reference).
    fn reference(t: Table, id: usize, scale: f64, out: []f64) !void {
        const g = hidden / group;
        const wb = hidden * t.bits / 8;
        var s: [g * 4]u8 = undefined; // the control's scales, fp32 so 1% off is exactly that
        for (0..g) |i| {
            const v: f64 = @as(f32, @bitCast(@as(u32, t.s.slice(u16, vocab * g)[id * g + i]) << 16));
            std.mem.writeInt(u32, s[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v * scale))), .little);
        }
        const shape: [5]usize = .{ 1, g, 1, 1, 1 };
        try mlx.reference(.{
            .words = .{ .dtype = .u32, .rank = 2, .shape = .{ 1, wb / 4, 1, 1, 1 }, .bytes = t.w.slice(u8, vocab * wb)[id * wb ..][0..wb] },
            .scales = if (scale == 1) .{ .dtype = .bf16, .rank = 2, .shape = shape, .bytes = t.s.slice(u8, vocab * g * 2)[id * g * 2 ..][0 .. g * 2] } else .{ .dtype = .f32, .rank = 2, .shape = shape, .bytes = &s },
            .biases = .{ .dtype = .bf16, .rank = 2, .shape = shape, .bytes = t.b.slice(u8, vocab * g * 2)[id * g * 2 ..][0 .. g * 2] },
            .bits = t.bits,
            .group = group,
        }, out);
    }
};

fn commit(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }
}

/// The family's gather of `rows` tokens in calls of `m` rows, into a NaN-filled bf16 output.
fn gather(device: mtl.Device, queue: mtl.Queue, k: q27.glue_gpu.Kernels, p: q27.projection.Triple, id: mtl.Buffer, m: usize) !mtl.Buffer {
    const c = q27.config.Config{ .hidden = hidden, .intermediate = 17408, .layers = 64, .vocab = vocab, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
    const y = try device.buffer(rows * hidden * 2, opts);
    @memset(y.slice(u8, rows * hidden * 2), 0xff);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var start: usize = 0;
    while (start < rows) : (start += m) try k.embedding(e, c, p, .{ .buffer = id, .offset = start * 4 }, .{ .buffer = y, .offset = start * hidden * 2 }, @intCast(m));
    try commit(cb, e);
    return y;
}

/// Each width: gathered bits against float64 rounded once, one-row calls against one call, scales 1% off as control.
pub fn check(device: mtl.Device, queue: mtl.Queue) !usize {
    const k = try q27.glue_gpu.Kernels.init(device);
    defer k.deinit();
    var failed: usize = 0;
    const deq = try a.alloc(f64, hidden);
    defer a.free(deq);
    for ([_]u8{ 8, 3 }, 0..) |bits, at| {
        const tokens = ids(0xe27 + at);
        const t = try Table.init(device, bits, &tokens, 0x27e + at);
        defer t.deinit();
        const id = try device.buffer(rows * 4, opts);
        defer id.deinit();
        @memcpy(id.slice(u32, rows), &tokens);
        const all = try gather(device, queue, k, t.triple(), id, rows);
        defer all.deinit();
        const one = try gather(device, queue, k, t.triple(), id, 1);
        defer one.deinit();
        const got = all.slice(u16, rows * hidden);
        var differ: usize = 0;
        var ctl: usize = 0;
        for (tokens, 0..) |token, r| {
            try t.reference(token, 1, deq);
            for (deq, got[r * hidden ..][0..hidden]) |d, g| differ += @intFromBool(g != bf64(d));
            try t.reference(token, 1.01, deq);
            for (deq, got[r * hidden ..][0..hidden]) |d, g| ctl += @intFromBool(g != bf64(d));
        }
        const same = std.mem.eql(u16, got, one.slice(u16, rows * hidden));
        const ok = differ == 0 and same and ctl * 10 > rows * hidden;
        if (!ok) failed += 1;
        std.debug.print("RESULT embedding [{d}, {d}] q{d} g64, {d} tokens: {d} of {d} values differ from float64 rounded once to bf16; " ++
            "one-row calls {s}; scales 1% off: {d} differ{s}\n", .{ vocab, hidden, bits, rows, differ, rows * hidden, if (same) "equal bit for bit" else "DIFFER", ctl, if (ok) "" else " FAIL" });
    }
    return failed;
}
