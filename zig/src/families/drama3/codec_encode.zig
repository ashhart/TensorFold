//! The codec's encoder for voice references: a whole clip through the encoder, downsampler, pre-transformer and RVQ.
const std = @import("std");
const cuda = @import("cuda");
const codec = @import("codec.zig");

const Codec = codec.Codec;
const Stage = codec.Stage;
const D = codec.D;
const samples_per_frame = codec.samples_per_frame;

/// Codes [frames, 10] of a 44.1 kHz mono clip (padded to whole frames), as DAC.encode makes them.
pub fn encode(c: *Codec, s: cuda.Stream, audio: []const f32) ![][10]i32 {
    const gpa = c.gpa;
    const nf = (audio.len + samples_per_frame - 1) / samples_per_frame;
    if (nf == 0) return error.EmptyAudio;
    const L0 = nf * samples_per_frame;
    // buffers for the whole clip: the widest activation is L0 x 64 (and the stages' margins)
    const big = L0 * 64 + 64 * 1024;
    var tmp: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (tmp.items) |*b| b.free();
        tmp.deinit(gpa);
    }
    const A = struct {
        fn get(d: *const cuda.Driver, list: *std.ArrayList(cuda.DeviceBuffer), g: std.mem.Allocator, bytes: usize) !u64 {
            const b = try cuda.DeviceBuffer.alloc(d, @max(bytes, 256));
            try list.append(g, b);
            return b.ptr;
        }
    };
    var sc: [4]u64 = undefined;
    for (&sc) |*p| p.* = try A.get(c.d, &tmp, gpa, big * 4);
    const stbuf = try A.get(c.d, &tmp, gpa, (L0 * 64 + 64 * 1024) * 4);
    const mk = struct {
        fn st(buf: u64, margin: usize, cols: usize, d: *const cuda.Driver, strm: cuda.Stream) !Stage {
            var b: cuda.DeviceBuffer = .{ .d = d, .ptr = buf, .len = margin * cols * 4, .borrowed = true };
            try b.fill8(0, strm.handle);
            return .{ .buf = buf, .margin = margin, .cols = cols };
        }
    };
    // audio in
    {
        var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = sc[0], .len = L0 * 4, .borrowed = true };
        try b.fill8(0, s.handle);
        try b.uploadAsync(0, std.mem.sliceAsBytes(audio), s.handle);
    }
    var rows = L0;
    var st = try mk.st(stbuf, 6, 1, c.d, s);
    try c.stageIn(s, st, sc[0], rows, 0);
    try c.conv(s, st, rows, c.enc_conv0, 7, 1, 64, sc[3], c.enc_conv0_b, false);
    var cur = sc[3];
    if (c.probe) |p| {
        try s.synchronize();
        try p.f(p.ctx, "enc.block0", cur, rows, 64);
    }
    for (0..4) |bi| {
        const B = c.enc[bi];
        for (0..3) |r| {
            const dil = ([3]usize{ 1, 3, 9 })[r];
            st = try mk.st(stbuf, 6 * dil, B.cin, c.d, s);
            try c.residualUnit(s, B.ru[r], st, dil, cur, rows, .{ sc[0], sc[1] });
        }
        st = try mk.st(stbuf, B.s, B.cin, c.d, s);
        try c.stageIn(s, st, cur, rows, B.alpha);
        const out_rows = rows / B.s;
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        // out[t] = Wa x[block t - 1] + Wb x[block t]: rows of s * cin floats
        try c.sgemm(s, B.wb, B.dim, B.s * B.cin, Codec.rowsAt(st, B.s), B.s * B.cin, nxt, B.dim, out_rows, 0);
        try c.sgemm(s, B.wa, B.dim, B.s * B.cin, Codec.rowsAt(st, 0), B.s * B.cin, nxt, B.dim, out_rows, 1);
        try c.bias(s, nxt, B.b, out_rows, B.dim, false);
        rows = out_rows;
        cur = nxt;
        if (B.tf) |t| {
            const kv = try gpa.alloc([2]u64, t.layers.len);
            defer gpa.free(kv);
            for (kv) |*p| p.* = .{ try A.get(c.d, &tmp, gpa, rows * D * 4), try A.get(c.d, &tmp, gpa, rows * D * 4) };
            try c.transformer(s, t, cur, rows, 0, 512, kv, rows, .{ sc[0], sc[1], if (cur == sc[2]) sc[3] else sc[2] });
        }
        if (c.probe) |p| {
            try s.synchronize();
            var nb: [32]u8 = undefined;
            try p.f(p.ctx, try std.fmt.bufPrint(&nb, "enc.block{d}", .{bi + 1}), cur, rows, B.dim);
        }
    }
    st = try mk.st(stbuf, 2, D, c.d, s);
    try c.stageIn(s, st, cur, rows, c.enc_final_alpha);
    const z = if (cur == sc[3]) sc[2] else sc[3];
    try c.conv(s, st, rows, c.enc_final, 3, 1, D, z, c.enc_final_b, false);
    cur = z;
    if (c.probe) |p| {
        try s.synchronize();
        try p.f(p.ctx, "enc.block6", cur, rows, D);
    }
    // the quantizer: downsample x2 x2 (kernel = stride 2, then ConvNeXt), pre-transformer, RVQ
    for (0..2) |i| {
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        try c.sgemm(s, c.down_w[i], D, 2 * D, cur, 2 * D, nxt, D, rows / 2, 0);
        try c.bias(s, nxt, c.down_b[i], rows / 2, D, false);
        rows /= 2;
        st = try mk.st(stbuf, 6, D, c.d, s);
        try c.convnext(s, c.down_cn[i], st, nxt, rows, .{ sc[0], sc[1] });
        cur = nxt;
    }
    {
        const kv = try gpa.alloc([2]u64, 8);
        defer gpa.free(kv);
        for (kv) |*p| p.* = .{ try A.get(c.d, &tmp, gpa, rows * D * 4), try A.get(c.d, &tmp, gpa, rows * D * 4) };
        try c.transformer(s, c.pre, cur, rows, 0, 128, kv, rows, .{ sc[0], sc[1], if (cur == sc[2]) sc[3] else sc[2] });
    }
    std.debug.assert(rows == nf);
    const codes_dev = try A.get(c.d, &tmp, gpa, nf * 10 * 4);
    var q: cuda.Args = .{};
    q.add(cur);
    q.add(@as(i32, D));
    q.add(c.vq_in_w);
    q.add(c.vq_in_b);
    q.add(c.vq_cbn);
    q.add(c.vq_table);
    q.add(codes_dev);
    try c.go(.vq_encode, s, nf, 1, 256, &q);
    const out = try gpa.alloc([10]i32, nf);
    errdefer gpa.free(out);
    var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = codes_dev, .len = nf * 40, .borrowed = true };
    try b.downloadAsync(0, std.mem.sliceAsBytes(out), s.handle);
    try s.synchronize();
    return out;
}
