//! The codec's encoder on Metal for voice references: a clip through the encoder, downsampler, pre-transformer and RVQ.
const std = @import("std");
const codec = @import("codec.zig");

const Codec = codec.Codec;
const Stage = codec.Stage;
const D = codec.D;
const samples_per_frame = codec.samples_per_frame;

/// Buffers for clips up to `frames` frames, kept on the codec so the next clip that fits reuses them.
fn buffers(c: *Codec, nf: usize) !Codec.EncBufs {
    if (c.enc_bufs) |b| if (b.frames >= nf) return b;
    const frames = std.mem.alignForward(usize, nf, 64);
    const L0 = frames * samples_per_frame;
    const big = L0 * 64 + 64 * 1024;
    var b: Codec.EncBufs = .{ .frames = frames, .sc = undefined, .stbuf = undefined, .kv = undefined, .codes = undefined };
    for (&b.sc) |*p| p.* = try c.g.alloc(big * 4);
    b.stbuf = try c.g.alloc(big * 4);
    // the transformers' caches: the most rows any of them takes (an encoder block's output, or the frames)
    var kv_rows = frames;
    var rows = L0;
    for (c.enc) |B| {
        rows /= B.s;
        if (B.tf != null) kv_rows = @max(kv_rows, rows);
    }
    for (&b.kv) |*p| p.* = .{ try c.g.alloc(kv_rows * D * 4), try c.g.alloc(kv_rows * D * 4) };
    b.codes = try c.g.alloc(frames * 10 * 4);
    c.enc_bufs = b;
    return b;
}

/// A stage over `buf` with its carried rows zeroed (after the GPU work before it).
fn stage(c: *Codec, buf: u64, margin: usize, cols: usize) !Stage {
    try c.g.flush();
    const st: Stage = .{ .buf = buf, .margin = margin, .cols = cols };
    c.zeroMargin(st);
    return st;
}

fn probe(c: *Codec, name: []const u8, ptr: u64, rows: usize, cols: usize) !void {
    if (c.probe) |p| {
        try c.g.flush();
        try p.f(p.ctx, name, ptr, rows, cols);
    }
}

/// Codes [frames, 10] of a 44.1 kHz mono clip (padded to whole frames), as DAC.encode makes them.
pub fn encode(c: *Codec, audio: []const f32) ![][10]i32 {
    try @import("../model.zig").reference(audio, 44100);
    const nf = (audio.len + samples_per_frame - 1) / samples_per_frame;
    if (nf == 0) return error.EmptyAudio;
    const L0 = nf * samples_per_frame;
    const bufs = try buffers(c, nf);
    const sc = bufs.sc;
    const stbuf = bufs.stbuf;
    // audio in, padded with zeros to whole frames
    try c.g.flush();
    const in = c.g.host(sc[0], L0 * 4);
    @memset(in, 0);
    @memcpy(in[0 .. audio.len * 4], std.mem.sliceAsBytes(audio));
    var rows = L0;
    var st = try stage(c, stbuf, 6, 1);
    c.stageIn(st, sc[0], rows, 0);
    c.conv(st, rows, c.enc_conv0, 7, 1, 64, sc[3], c.enc_conv0_b, false);
    var cur = sc[3];
    try probe(c, "enc.block0", cur, rows, 64);
    for (0..4) |bi| {
        const B = c.enc[bi];
        for (0..3) |r| {
            const dil = ([3]usize{ 1, 3, 9 })[r];
            st = try stage(c, stbuf, 6 * dil, B.cin);
            c.residualUnit(B.ru[r], st, dil, cur, rows, .{ sc[0], sc[1] });
        }
        st = try stage(c, stbuf, B.s, B.cin);
        c.stageIn(st, cur, rows, B.alpha);
        const out_rows = rows / B.s;
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        // out[t] = Wa x[block t - 1] + Wb x[block t]: rows of s * cin floats
        c.sgemm(B.wb, B.dim, B.s * B.cin, Codec.rowsAt(st, B.s), B.s * B.cin, nxt, B.dim, out_rows, 0);
        c.sgemm(B.wa, B.dim, B.s * B.cin, Codec.rowsAt(st, 0), B.s * B.cin, nxt, B.dim, out_rows, 1);
        c.bias(nxt, B.b, out_rows, B.dim, false);
        rows = out_rows;
        cur = nxt;
        if (B.tf) |t| c.transformer(t, cur, rows, 0, 512, bufs.kv[0..t.layers.len], rows, .{ sc[0], sc[1], if (cur == sc[2]) sc[3] else sc[2] });
        var nb: [32]u8 = undefined;
        try probe(c, try std.fmt.bufPrint(&nb, "enc.block{d}", .{bi + 1}), cur, rows, B.dim);
    }
    st = try stage(c, stbuf, 2, D);
    c.stageIn(st, cur, rows, c.enc_final_alpha);
    const z = if (cur == sc[3]) sc[2] else sc[3];
    c.conv(st, rows, c.enc_final, 3, 1, D, z, c.enc_final_b, false);
    cur = z;
    try probe(c, "enc.block6", cur, rows, D);
    // the quantizer: downsample x2 x2 (kernel = stride 2, then ConvNeXt), pre-transformer, RVQ
    for (0..2) |i| {
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        c.sgemm(c.down_w[i], D, 2 * D, cur, 2 * D, nxt, D, rows / 2, 0);
        c.bias(nxt, c.down_b[i], rows / 2, D, false);
        rows /= 2;
        st = try stage(c, stbuf, 6, D);
        c.convnext(c.down_cn[i], st, nxt, rows, .{ sc[0], sc[1] });
        cur = nxt;
    }
    c.transformer(c.pre, cur, rows, 0, 128, &bufs.kv, rows, .{ sc[0], sc[1], if (cur == sc[2]) sc[3] else sc[2] });
    std.debug.assert(rows == nf);
    c.g.go(.c_vq_encode, .{ nf, 1, 1 }, 256, &.{ cur, c.vq_in_w, c.vq_in_b, c.vq_cbn, c.vq_table, bufs.codes }, @as(u32, D));
    try c.g.flush();
    const out = try c.gpa.alloc([10]i32, nf);
    @memcpy(std.mem.sliceAsBytes(out), c.g.host(bufs.codes, nf * 40));
    return out;
}
