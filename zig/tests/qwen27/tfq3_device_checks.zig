//! Device-only sign reset and unchanged BF16 operation checks.
const std = @import("std");
const mtl = @import("metal");
const qwen27 = @import("qwen27");
const dispatch = qwen27.tfq3_dispatch;
const glue_gpu = qwen27.glue_gpu;
const projection = qwen27.projection;

fn f32ToBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    const lsb = (bits >> 16) & 1;
    return @truncate((bits +% 0x7fff +% lsb) >> 16);
}

fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn checkSignRows(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
    const base = try dispatch.buildSource(a, device.tensorUnits());
    defer a.free(base);
    const src = try std.mem.concat(a, u8, &.{
        base,
        \\[[kernel]] void q27_test_sr0(const device uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
        \\  out[gid] = q27_pw_sr0(in[gid * 3], in[gid * 3 + 1], in[gid * 3 + 2]);
        \\}
    });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "q27_test_sr0", false);
    defer pipe.deinit();
    const cases = [_][4]u32{
        .{ 8191, 16640, 0, 8191 },    .{ 8192, 16640, 0, 0 },
        .{ 16383, 16640, 0, 8191 },   .{ 16384, 16640, 0, 0 },
        .{ 8192, 16384, 0, 8192 },    .{ 245760, 248320, 0, 0 },
        .{ 248319, 248320, 0, 2559 }, .{ 34816, 52224, 17408, 0 },
    };
    var input: [cases.len * 3]u32 = undefined;
    for (cases, 0..) |case, i| {
        input[i * 3] = case[0];
        input[i * 3 + 1] = case[1];
        input[i * 3 + 2] = case[2];
    }
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const in_buf = try device.buffer(@sizeOf(@TypeOf(input)), opts);
    defer in_buf.deinit();
    @memcpy(in_buf.contents()[0..@sizeOf(@TypeOf(input))], std.mem.asBytes(&input));
    const out_buf = try device.buffer(cases.len * 4, opts);
    defer out_buf.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(in_buf, 0, 0);
    e.setBuffer(out_buf, 0, 1);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(cases.len, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
    for (cases, out_buf.slice(u32, cases.len)) |case, got| if (case[3] != got) return error.SignRowMismatch;
}

pub fn checkBf16Ops(device: mtl.Device, queue: mtl.Queue, pipes: *const dispatch.Pipes, rng: std.Random) !void {
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const glue = try glue_gpu.Kernels.init(device);
    defer glue.deinit();

    const vocab: u32 = 16;
    const width: u32 = 256;
    const rows: u32 = 5;
    const table_buf = try device.buffer(@as(usize, vocab) * width * 2, opts);
    defer table_buf.deinit();
    const table = table_buf.slice(u16, @as(usize, vocab) * width);
    for (table) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 4.0);
    const ids_buf = try device.buffer(rows * 4, opts);
    defer ids_buf.deinit();
    const ids = ids_buf.slice(u32, rows);
    for (ids, [_]u32{ 3, 0, 15, 7, 3 }) |*d, v| d.* = v;
    const emb_out = try device.buffer(@as(usize, rows) * width * 2, opts);
    defer emb_out.deinit();
    const emb_t = qwen27.gpu_weights.Tensor{ .buffer = table_buf, .offset = 0, .bytes = table_buf.length(), .dtype = .bf16, .rank = 2, .shape = .{ vocab, width, 0, 0, 0 } };
    const emb_triple: projection.Triple = .{ .weight = emb_t, .scales = emb_t, .biases = emb_t, .bits = 16, .group = 0 };
    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try glue.embeddingShape(e, width, vocab, emb_triple, .{ .buffer = ids_buf }, .{ .buffer = emb_out }, rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
    }
    const got_emb = emb_out.slice(u16, @as(usize, rows) * width);
    for (0..rows) |r| {
        const want = table[ids[r] * width ..][0..width];
        if (!std.mem.eql(u16, want, got_emb[r * width ..][0..width])) return error.EmbedBf16Mismatch;
    }

    const n: u32 = 96;
    const k: u32 = 512;
    const ldo: u32 = 128;
    const m_rows: u32 = 7;
    const w_buf = try device.buffer(@as(usize, n) * k * 2, opts);
    defer w_buf.deinit();
    const x_buf = try device.buffer(@as(usize, m_rows) * k * 2, opts);
    defer x_buf.deinit();
    const y1_buf = try device.buffer(@as(usize, ldo) * 2, opts);
    defer y1_buf.deinit();
    const ym_buf = try device.buffer(@as(usize, m_rows) * ldo * 2, opts);
    defer ym_buf.deinit();
    const w_u16 = w_buf.slice(u16, @as(usize, n) * k);
    const x_u16 = x_buf.slice(u16, @as(usize, m_rows) * k);
    for (w_u16) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 0.4);
    for (x_u16) |*v| v.* = f32ToBf16((rng.float(f32) - 0.5) * 0.4);
    @memset(y1_buf.slice(u16, ldo), 0xCAFE);
    @memset(ym_buf.slice(u16, @as(usize, m_rows) * ldo), 0xCAFE);

    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try dispatch.encodeBf16(pipes, e, .{ .buffer = w_buf }, n, k, .{ .buffer = x_buf }, k, .{ .buffer = y1_buf }, ldo, 1);
        try dispatch.encodeBf16(pipes, e, .{ .buffer = w_buf }, n, k, .{ .buffer = x_buf }, k, .{ .buffer = ym_buf }, ldo, m_rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
    }
    const y1 = y1_buf.slice(u16, ldo);
    const ym = ym_buf.slice(u16, @as(usize, m_rows) * ldo);
    if (!std.mem.eql(u16, y1[0..n], ym[0..n])) return error.Bf16MvMmParity;
    for (0..m_rows) |m| {
        for (0..n) |r| {
            var acc: f64 = 0.0;
            for (0..k) |c| acc += @as(f64, bf16ToF32(w_u16[r * k + c])) * bf16ToF32(x_u16[m * k + c]);
            if (@abs(@as(f64, bf16ToF32(ym[m * ldo + r])) - acc) > 0.02) return error.Bf16ParityMismatch;
        }
        for (n..ldo) |c| if (ym[m * ldo + c] != 0xCAFE) return error.StrideClobber;
    }
}
