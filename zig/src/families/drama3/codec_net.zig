//! The codec's weights by name, decoder state and decode/encode schedules for both backends; `r` binds the kernels.
const std = @import("std");
const source = @import("codec_load.zig");

pub const samples_per_frame = 2048;
pub const sample_rate = 44100;
pub const D = 1024;

/// A causal convolution's input buffer: `margin` carried rows, then a chunk's rows, `cols` floats a row.
pub const Stage = struct { buf: u64, margin: usize, cols: usize };

pub const TfLayer = struct {
    attn_norm: u64,
    wqkv: u64,
    wo: u64,
    ffn_norm: u64,
    w13: u64,
    w2: u64,
    g_attn: u64,
    g_ffn: u64,
};
pub const Tf = struct { layers: []TfLayer, norm: u64 };
pub const ConvNext = struct {
    dw: u64,
    dw_b: u64,
    ln_w: u64,
    ln_b: u64,
    pw1: u64,
    pw1_b: u64,
    pw2: u64,
    pw2_b: u64,
    gamma: u64,
};
pub const Ru = struct { a1: u64, c7: u64, c7_b: u64, a2: u64, c1: u64, c1_b: u64 };
pub const DecBlock = struct { alpha: u64, wa: u64, wb: u64, b: u64, ru: [3]Ru, cin: usize, cout: usize, s: usize };
pub const EncBlock = struct {
    ru: [3]Ru,
    alpha: u64,
    wa: u64,
    wb: u64,
    b: u64,
    cin: usize,
    dim: usize,
    s: usize,
    tf: ?Tf,
};

/// Diagnostics: a device fp32 tensor [rows, cols] by its codec.pth name.
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void,
};

pub const Options = struct { max_chunk: usize = 32, ring: usize = 512 };

const dilations = [3]usize{ 1, 3, 9 };

/// Every codec weight as a device address, each uploaded by the backend's `up.put(bytes)` in the order read.
pub const Weights = struct {
    host_layers: std.ArrayList([]TfLayer) = .empty,
    bytes: usize = 0,
    vq_table: u64,
    vq_cbn: u64,
    vq_in_w: u64,
    vq_in_b: u64,
    post: Tf,
    pre: Tf,
    up_wa: [2]u64,
    up_b: [2]u64,
    up_cn: [2]ConvNext,
    down_w: [2]u64,
    down_b: [2]u64,
    down_cn: [2]ConvNext,
    conv0: u64,
    conv0_b: u64,
    dec: [4]DecBlock,
    final_alpha: u64,
    final: u64,
    final_b: u64,
    enc_conv0: u64,
    enc_conv0_b: u64,
    enc: [4]EncBlock,
    enc_final_alpha: u64,
    enc_final: u64,
    enc_final_b: u64,

    /// Frees the transformers' host layer tables; the device buffers stay with the backend.
    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.host_layers.items) |layers| gpa.free(layers);
        w.host_layers.deinit(gpa);
    }

    pub fn load(gpa: std.mem.Allocator, file: *const source.Source, up: anytype) !Weights {
        var w: Weights = undefined;
        w.host_layers = .empty;
        w.bytes = 0;
        errdefer w.deinit(gpa);
        var l: Loader(@TypeOf(up)) = .{ .gpa = gpa, .file = file, .up = up, .w = &w };
        w.vq_table = try l.t("vq.table");
        w.vq_cbn = try l.t("vq.cbn");
        w.vq_in_w = try l.t("vq.in_w");
        w.vq_in_b = try l.t("vq.in_b");
        w.post = try l.tf("post", 8);
        w.pre = try l.tf("pre", 8);
        for (0..2) |i| {
            w.up_wa[i] = try l.f("up.{d}.wa", .{i});
            w.up_b[i] = try l.f("up.{d}.b", .{i});
            w.up_cn[i] = try l.convnext("up.{d}.cn", i);
            w.down_w[i] = try l.f("down.{d}.w", .{i});
            w.down_b[i] = try l.f("down.{d}.b", .{i});
            w.down_cn[i] = try l.convnext("down.{d}.cn", i);
        }
        w.conv0 = try l.t("dec.conv0");
        w.conv0_b = try l.t("dec.conv0_b");
        const strides = [4]usize{ 8, 8, 4, 2 };
        for (0..4) |bi| {
            const B = &w.dec[bi];
            B.cin = @as(usize, 1536) >> @intCast(bi);
            B.cout = B.cin / 2;
            B.s = strides[bi];
            B.alpha = try l.f("dec.{d}.alpha", .{bi});
            B.wa = try l.f("dec.{d}.wa", .{bi});
            B.wb = try l.f("dec.{d}.wb", .{bi});
            B.b = try l.f("dec.{d}.b", .{bi});
            for (0..3) |r| B.ru[r] = try l.ru("dec", bi, r);
        }
        w.final_alpha = try l.t("dec.final_alpha");
        w.final = try l.t("dec.final");
        w.final_b = try l.t("dec.final_b");
        w.enc_conv0 = try l.t("enc.conv0");
        w.enc_conv0_b = try l.t("enc.conv0_b");
        const enc_strides = [4]usize{ 2, 4, 8, 8 };
        for (0..4) |bi| {
            const B = &w.enc[bi];
            B.dim = @as(usize, 128) << @intCast(bi);
            B.cin = B.dim / 2;
            B.s = enc_strides[bi];
            for (0..3) |r| B.ru[r] = try l.ru("enc", bi, r);
            B.alpha = try l.f("enc.{d}.alpha", .{bi});
            B.wa = try l.f("enc.{d}.wa", .{bi});
            B.wb = try l.f("enc.{d}.wb", .{bi});
            B.b = try l.f("enc.{d}.b", .{bi});
            var nb: [32]u8 = undefined;
            const tfp = try std.fmt.bufPrint(&nb, "enc.{d}.tf", .{bi});
            B.tf = if (file.has(try std.fmt.bufPrint(&l.nb2, "{s}.0.wqkv", .{tfp}))) try l.tf(tfp, 4) else null;
        }
        w.enc_final_alpha = try l.t("enc.final_alpha");
        w.enc_final = try l.t("enc.final");
        w.enc_final_b = try l.t("enc.final_b");
        return w;
    }
};

fn Loader(comptime Up: type) type {
    return struct {
        gpa: std.mem.Allocator,
        file: *const source.Source,
        up: Up,
        w: *Weights,
        nb: [96]u8 = undefined,
        nb2: [96]u8 = undefined,

        const L = @This();

        fn t(l: *L, name: []const u8) !u64 {
            const values = try l.file.read(name);
            defer l.gpa.free(values);
            const bytes = std.mem.sliceAsBytes(values);
            const p = try l.up.put(bytes);
            l.w.bytes += bytes.len;
            return p;
        }

        fn f(l: *L, comptime fmt: []const u8, args: anytype) !u64 {
            return l.t(try std.fmt.bufPrint(&l.nb, fmt, args));
        }

        fn tf(l: *L, p: []const u8, n: usize) !Tf {
            const layers = try l.gpa.alloc(TfLayer, n);
            errdefer l.gpa.free(layers);
            for (layers, 0..) |*y, i| y.* = .{
                .attn_norm = try l.f("{s}.{d}.attn_norm", .{ p, i }),
                .wqkv = try l.f("{s}.{d}.wqkv", .{ p, i }),
                .wo = try l.f("{s}.{d}.wo", .{ p, i }),
                .ffn_norm = try l.f("{s}.{d}.ffn_norm", .{ p, i }),
                .w13 = try l.f("{s}.{d}.w13", .{ p, i }),
                .w2 = try l.f("{s}.{d}.w2", .{ p, i }),
                .g_attn = try l.f("{s}.{d}.g_attn", .{ p, i }),
                .g_ffn = try l.f("{s}.{d}.g_ffn", .{ p, i }),
            };
            const norm = try l.f("{s}.norm", .{p});
            try l.w.host_layers.append(l.gpa, layers);
            return .{ .layers = layers, .norm = norm };
        }

        fn convnext(l: *L, comptime p: []const u8, i: usize) !ConvNext {
            return .{
                .dw = try l.f(p ++ ".dw", .{i}),
                .dw_b = try l.f(p ++ ".dw_b", .{i}),
                .ln_w = try l.f(p ++ ".ln_w", .{i}),
                .ln_b = try l.f(p ++ ".ln_b", .{i}),
                .pw1 = try l.f(p ++ ".pw1", .{i}),
                .pw1_b = try l.f(p ++ ".pw1_b", .{i}),
                .pw2 = try l.f(p ++ ".pw2", .{i}),
                .pw2_b = try l.f(p ++ ".pw2_b", .{i}),
                .gamma = try l.f(p ++ ".gamma", .{i}),
            };
        }

        fn ru(l: *L, comptime side: []const u8, bi: usize, r: usize) !Ru {
            return .{
                .a1 = try l.f(side ++ ".{d}.ru{d}.a1", .{ bi, r }),
                .c7 = try l.f(side ++ ".{d}.ru{d}.c7", .{ bi, r }),
                .c7_b = try l.f(side ++ ".{d}.ru{d}.c7_b", .{ bi, r }),
                .a2 = try l.f(side ++ ".{d}.ru{d}.a2", .{ bi, r }),
                .c1 = try l.f(side ++ ".{d}.ru{d}.c1", .{ bi, r }),
                .c1_b = try l.f(side ++ ".{d}.ru{d}.c1_b", .{ bi, r }),
            };
        }
    };
}

/// A decoding stream's buffers: the post transformer's rings, every causal stage's carried rows, the scratch.
pub const Decoder = struct {
    pos: usize = 0, // frames decoded since reset
    kv: [8][2]u64, // post transformer ring caches [ring, 1024]
    st_up: [2]Stage,
    st_conv0: Stage,
    st_tconv: [4]Stage,
    st_ru: [4][3]Stage,
    st_final: Stage,
    scratch: [4]u64, // activations, max rows x cols a chunk
    scratch_floats: usize,
    carry_tmp: u64,

    /// The buffers for chunks of up to `o.max_chunk` frames, each from the backend's `a.alloc(bytes)`.
    pub fn alloc(w: *const Weights, o: Options, a: anytype) !Decoder {
        var s: Decoder = undefined;
        s.pos = 0;
        const fr = o.max_chunk;
        for (0..8) |i| s.kv[i] = .{ try a.alloc(o.ring * D * 4), try a.alloc(o.ring * D * 4) };
        for (0..2) |i| s.st_up[i] = try stage(a, 6, D, fr * (@as(usize, 2) << @intCast(i)));
        s.st_conv0 = try stage(a, 6, D, fr * 4);
        var rows = fr * 4;
        for (0..4) |bi| {
            const B = w.dec[bi];
            s.st_tconv[bi] = try stage(a, 1, B.cin, rows);
            rows *= B.s;
            for (0..3) |r| s.st_ru[bi][r] = try stage(a, 6 * dilations[r], B.cout, rows);
        }
        s.st_final = try stage(a, 6, 96, rows);
        // the largest activation of a chunk: 2048 rows x 96 a frame (and 4 x 4096 for ConvNeXt, 6144 for qkv)
        s.scratch_floats = fr * 2048 * 96;
        for (&s.scratch) |*b| b.* = try a.alloc(s.scratch_floats * 4);
        s.carry_tmp = try a.alloc(54 * 1536 * 4 + 6 * D * 4);
        return s;
    }

    fn stage(a: anytype, margin: usize, cols: usize, rows: usize) !Stage {
        return .{ .buf = try a.alloc((margin + rows) * cols * 4), .margin = margin, .cols = cols };
    }

    /// Every stage with carried rows, which a reset zeroes.
    pub fn stages(s: *const Decoder) [20]Stage {
        var out: [20]Stage = undefined;
        const head = [_]Stage{ s.st_up[0], s.st_up[1], s.st_conv0, s.st_final } ++ s.st_tconv;
        @memcpy(out[0..8], &head);
        for (s.st_ru, 0..) |row, i| @memcpy(out[8 + 3 * i ..][0..3], &row);
        return out;
    }
};

pub fn rowsAt(st: Stage, row: usize) u64 {
    return st.buf + row * st.cols * 4;
}

/// The chunk's input into the stage (snake applied when alpha != 0), after its carried rows.
pub fn stageIn(r: anytype, st: Stage, src: u64, rows: usize, alpha: u64) !void {
    try r.snake(src, rowsAt(st, st.margin), rows, st.cols, alpha);
}

/// The chunk's last `margin` rows become the next chunk's carried rows, through `r.tmp`.
fn carry(r: anytype, st: Stage, rows: usize) !void {
    if (st.margin == 0) return;
    const n = st.margin * st.cols;
    try r.copy(rowsAt(st, rows), r.tmp, n);
    try r.copy(r.tmp, st.buf, n);
}

/// A window-limited causal transformer over x [rows, 1024] (in place) at positions pos0.., caches kv (ring rows).
pub fn transformer(
    r: anytype,
    t: Tf,
    x: u64,
    rows: usize,
    pos0: usize,
    window: usize,
    kv: []const [2]u64,
    ring: usize,
    sc: [3]u64,
) !void {
    for (t.layers, 0..) |ly, li| {
        try r.rmsnorm(x, ly.attn_norm, sc[0], rows);
        try r.sgemm(ly.wqkv, 3 * D, D, sc[0], D, sc[1], 3 * D, rows, 0);
        try r.ropeKv(sc[1], kv[li], pos0, ring, rows);
        try r.attn(sc[1], kv[li], sc[0], pos0, window, ring, rows);
        try r.sgemm(ly.wo, D, D, sc[0], D, sc[2], D, rows, 0);
        try r.addGamma(x, sc[2], ly.g_attn, rows, D);
        try r.rmsnorm(x, ly.ffn_norm, sc[0], rows);
        try r.sgemm(ly.w13, 6 * D, D, sc[0], D, sc[1], 6 * D, rows, 0);
        try r.siluMul(sc[1], sc[2], rows);
        try r.sgemm(ly.w2, D, 3 * D, sc[2], 3 * D, sc[0], D, rows, 0);
        try r.addGamma(x, sc[0], ly.g_ffn, rows, D);
    }
    try r.rmsnorm(x, t.norm, x, rows);
}

/// ConvNeXt block over x [rows, 1024] in place, its causal depthwise conv from stage st (carried).
pub fn convnext(r: anytype, cn: ConvNext, st: Stage, x: u64, rows: usize, sc: [2]u64) !void {
    try stageIn(r, st, x, rows, 0);
    try r.dwconv7(st.buf, cn.dw, cn.dw_b, sc[0], rows);
    try r.layernorm(sc[0], cn.ln_w, cn.ln_b, sc[0], rows);
    try r.sgemm(cn.pw1, 4 * D, D, sc[0], D, sc[1], 4 * D, rows, 0);
    try r.bias(sc[1], cn.pw1_b, rows, 4 * D, false);
    try r.gelu(sc[1], rows * 4 * D);
    try r.sgemm(cn.pw2, D, 4 * D, sc[1], 4 * D, sc[0], D, rows, 0);
    try r.bias(sc[0], cn.pw2_b, rows, D, false);
    try r.addGamma(x, sc[0], cn.gamma, rows, D);
    try carry(r, st, rows);
}

/// A residual unit on x [rows, C] in place: x += conv1(snake(conv7_dil(snake(x)))).
pub fn residualUnit(r: anytype, u: Ru, st: Stage, dil: usize, x: u64, rows: usize, sc: [2]u64) !void {
    const C = st.cols;
    try stageIn(r, st, x, rows, u.a1);
    try r.conv(st, rows, u.c7, 7, dil, C, sc[0], u.c7_b, false);
    try carry(r, st, rows);
    try r.snake(sc[0], sc[1], rows, C, u.a2);
    try r.sgemm(u.c1, C, C, sc[1], C, sc[0], C, rows, 0);
    try r.bias(sc[0], u.c1_b, rows, C, false);
    try r.addGamma(x, sc[0], 0, rows, C);
}

/// Decode `nf` frames (device int32 rows [nf, 11], codes in columns 1..10) into nf * 2048 fp32 samples at `audio`.
pub fn decode(r: anytype, w: *const Weights, s: *Decoder, o: Options, frames: u64, nf: usize, audio: u64) !void {
    var done: usize = 0;
    while (done < nf) {
        const n = @min(o.max_chunk, nf - done);
        try decodeChunk(r, w, s, o.ring, frames + done * 11 * 4, n, audio + done * samples_per_frame * 4);
        done += n;
    }
}

fn decodeChunk(r: anytype, w: *const Weights, s: *Decoder, ring: usize, frames: u64, n: usize, audio: u64) !void {
    const sc = s.scratch;
    const x = sc[3];
    // codes -> z [n, 1024]
    try r.vqGather(frames, w.vq_table, x, n);
    try transformer(r, w.post, x, n, s.pos, 128, &s.kv, ring, .{ sc[0], sc[1], sc[2] });
    // upsample x2 x2 (transposed conv kernel = stride 2, then ConvNeXt)
    var rows = n;
    var cur = x;
    for (0..2) |i| {
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        try r.sgemm(w.up_wa[i], 2 * D, D, cur, D, nxt, 2 * D, rows, 0);
        try r.bias(nxt, w.up_b[i], rows, 2 * D, false);
        rows *= 2;
        try convnext(r, w.up_cn[i], s.st_up[i], nxt, rows, .{ sc[0], sc[1] });
        cur = nxt;
    }
    // decoder: conv0 1024 -> 1536
    try stageIn(r, s.st_conv0, cur, rows, 0);
    const y0 = if (cur == sc[3]) sc[2] else sc[3];
    try r.conv(s.st_conv0, rows, w.conv0, 7, 1, 1536, y0, w.conv0_b, false);
    try carry(r, s.st_conv0, rows);
    cur = y0;
    for (0..4) |bi| {
        const B = w.dec[bi];
        const st = s.st_tconv[bi];
        try stageIn(r, st, cur, rows, B.alpha);
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        try r.sgemm(B.wa, B.s * B.cout, B.cin, rowsAt(st, 1), B.cin, nxt, B.s * B.cout, rows, 0);
        try r.sgemm(B.wb, B.s * B.cout, B.cin, rowsAt(st, 0), B.cin, nxt, B.s * B.cout, rows, 1);
        try r.bias(nxt, B.b, rows, B.s * B.cout, false);
        try carry(r, st, rows);
        rows *= B.s;
        cur = nxt;
        for (0..3) |i| try residualUnit(r, B.ru[i], s.st_ru[bi][i], dilations[i], cur, rows, .{ sc[0], sc[1] });
    }
    try stageIn(r, s.st_final, cur, rows, w.final_alpha);
    try r.conv(s.st_final, rows, w.final, 7, 1, 1, audio, w.final_b, true);
    try carry(r, s.st_final, rows);
    s.pos += n;
}

/// nf * 2048 samples at sc[0] through the encoder, downsampler, pre-transformer and RVQ, into `e`'s clip buffers.
pub fn encode(r: anytype, e: anytype, w: *const Weights, sc: [4]u64, stbuf: u64, nf: usize) !void {
    var rows = nf * samples_per_frame;
    var st = try e.stage(stbuf, 6, 1);
    try stageIn(r, st, sc[0], rows, 0);
    try r.conv(st, rows, w.enc_conv0, 7, 1, 64, sc[3], w.enc_conv0_b, false);
    var cur = sc[3];
    try e.probe("enc.block0", cur, rows, 64);
    for (0..4) |bi| {
        const B = w.enc[bi];
        for (0..3) |i| {
            st = try e.stage(stbuf, 6 * dilations[i], B.cin);
            try residualUnit(r, B.ru[i], st, dilations[i], cur, rows, .{ sc[0], sc[1] });
        }
        st = try e.stage(stbuf, B.s, B.cin);
        try stageIn(r, st, cur, rows, B.alpha);
        const out_rows = rows / B.s;
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        // out[t] = Wa x[block t - 1] + Wb x[block t]: rows of s * cin floats
        try r.sgemm(B.wb, B.dim, B.s * B.cin, rowsAt(st, B.s), B.s * B.cin, nxt, B.dim, out_rows, 0);
        try r.sgemm(B.wa, B.dim, B.s * B.cin, rowsAt(st, 0), B.s * B.cin, nxt, B.dim, out_rows, 1);
        try r.bias(nxt, B.b, out_rows, B.dim, false);
        rows = out_rows;
        cur = nxt;
        if (B.tf) |t| {
            const kv = try e.kv(t.layers.len, rows);
            try transformer(r, t, cur, rows, 0, 512, kv, rows, .{ sc[0], sc[1], if (cur == sc[2]) sc[3] else sc[2] });
        }
        var nb: [32]u8 = undefined;
        try e.probe(try std.fmt.bufPrint(&nb, "enc.block{d}", .{bi + 1}), cur, rows, B.dim);
    }
    st = try e.stage(stbuf, 2, D);
    try stageIn(r, st, cur, rows, w.enc_final_alpha);
    const z = if (cur == sc[3]) sc[2] else sc[3];
    try r.conv(st, rows, w.enc_final, 3, 1, D, z, w.enc_final_b, false);
    cur = z;
    try e.probe("enc.block6", cur, rows, D);
    // the quantizer: downsample x2 x2 (kernel = stride 2, then ConvNeXt), pre-transformer, RVQ
    for (0..2) |i| {
        const nxt = if (cur == sc[3]) sc[2] else sc[3];
        try r.sgemm(w.down_w[i], D, 2 * D, cur, 2 * D, nxt, D, rows / 2, 0);
        try r.bias(nxt, w.down_b[i], rows / 2, D, false);
        rows /= 2;
        st = try e.stage(stbuf, 6, D);
        try convnext(r, w.down_cn[i], st, nxt, rows, .{ sc[0], sc[1] });
        cur = nxt;
    }
    const kv = try e.kv(8, rows);
    try transformer(r, w.pre, cur, rows, 0, 128, kv, rows, .{ sc[0], sc[1], if (cur == sc[2]) sc[3] else sc[2] });
    std.debug.assert(rows == nf);
    try r.vqEncode(cur, w, try e.codes(nf), nf);
}

test "a decoder's stages are its 20 carried buffers in reset order" {
    var w: Weights = undefined;
    for (&w.dec, 0..) |*B, bi| {
        B.cin = @as(usize, 1536) >> @intCast(bi);
        B.cout = B.cin / 2;
        B.s = ([4]usize{ 8, 8, 4, 2 })[bi];
    }
    const Bump = struct {
        next: u64 = 1 << 20,
        fn alloc(b: *@This(), bytes: usize) !u64 {
            defer b.next += bytes;
            return b.next;
        }
    };
    var bump: Bump = .{};
    const s = try Decoder.alloc(&w, .{ .max_chunk = 2, .ring = 4 }, &bump);
    const all = s.stages();
    try std.testing.expectEqual(s.st_final, all[3]);
    try std.testing.expectEqual(s.st_tconv[3], all[7]);
    try std.testing.expectEqual(s.st_ru[3][2], all[19]);
    // the last decoder block's residual stages: 2 frames x 4 x 8 x 8 x 4 x 2 rows of 96 floats after 54 carried
    try std.testing.expectEqual(@as(usize, 54), s.st_ru[3][2].margin);
    try std.testing.expectEqual(@as(usize, 96), s.st_ru[3][2].cols);
}
