//! Drama 3's codec (Fish S2 modded DAC) on Metal in fp32, causal so it decodes in carried chunks (as codec.zig).
const std = @import("std");
const source = @import("../codec_load.zig");
const gpu = @import("gpu.zig");

const Gpu = gpu.Gpu;

pub const samples_per_frame = 2048;
pub const sample_rate = 44100;
pub const D = 1024;

/// A causal convolution's input buffer: `margin` carried rows, then a chunk's rows, `cols` floats a row.
pub const Stage = struct { buf: u64, margin: usize, cols: usize };

const TfLayer = struct { attn_norm: u64, wqkv: u64, wo: u64, ffn_norm: u64, w13: u64, w2: u64, g_attn: u64, g_ffn: u64 };
pub const Tf = struct { layers: []TfLayer, norm: u64 };
const ConvNext = struct { dw: u64, dw_b: u64, ln_w: u64, ln_b: u64, pw1: u64, pw1_b: u64, pw2: u64, pw2_b: u64, gamma: u64 };
const Ru = struct { a1: u64, c7: u64, c7_b: u64, a2: u64, c1: u64, c1_b: u64 };
const DecBlock = struct { alpha: u64, wa: u64, wb: u64, b: u64, ru: [3]Ru, cin: usize, cout: usize, s: usize };
const EncBlock = struct { ru: [3]Ru, alpha: u64, wa: u64, wb: u64, b: u64, cin: usize, dim: usize, s: usize, tf: ?Tf };

/// Diagnostics: a device fp32 tensor [rows, cols] by name (the codec_convert.py --ref names).
pub const Probe = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void };

pub const Options = struct { max_chunk: usize = 32, ring: usize = 512 };

fn u(x: usize) u32 {
    return @intCast(x);
}

const ElemArgs = extern struct { total: u32, n: u32, flag: u32 };

pub const Codec = struct {
    gpa: std.mem.Allocator,
    owns_weights: bool,
    state_start: usize,
    state_end: usize,
    host_layers: std.ArrayList([]TfLayer),
    g: *Gpu,
    o: Options,
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
    weight_bytes: usize,
    // decoder state
    pos: usize, // frames decoded since reset
    kv: [8][2]u64, // post transformer ring caches [ring, 1024]
    st_up: [2]Stage,
    st_conv0: Stage,
    st_tconv: [4]Stage,
    st_ru: [4][3]Stage,
    st_final: Stage,
    scratch: [4]u64, // activations, max rows x cols a chunk
    scratch_floats: usize,
    carry_tmp: u64,
    probe: ?Probe = null,
    enc_bufs: ?EncBufs = null, // the encoder's buffers, kept for the next clip that fits

    pub const EncBufs = struct { frames: usize, sc: [4]u64, stbuf: u64, kv: [8][2]u64, codes: u64 };

    pub fn load(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, path: []const u8, o: Options) !*Codec {
        const c = try gpa.create(Codec);
        errdefer gpa.destroy(c);
        c.* = undefined;
        c.gpa = gpa;
        c.owns_weights = true;
        c.host_layers = .empty;
        errdefer {
            for (c.host_layers.items) |layers| gpa.free(layers);
            c.host_layers.deinit(gpa);
        }
        c.g = g;
        const first_buffer = g.bufs.items.len;
        errdefer g.releaseFrom(first_buffer);
        c.o = o;
        c.weight_bytes = 0;
        c.pos = 0;
        c.probe = null;
        c.enc_bufs = null;
        var file = try source.Source.open(gpa, io, path);
        defer file.close();
        var l: L = .{ .c = c, .file = &file };
        c.vq_table = try l.t("vq.table");
        c.vq_cbn = try l.t("vq.cbn");
        c.vq_in_w = try l.t("vq.in_w");
        c.vq_in_b = try l.t("vq.in_b");
        c.post = try l.tf("post", 8);
        c.pre = try l.tf("pre", 8);
        for (0..2) |i| {
            c.up_wa[i] = try l.f("up.{d}.wa", .{i});
            c.up_b[i] = try l.f("up.{d}.b", .{i});
            c.up_cn[i] = try l.convnext("up.{d}.cn", i);
            c.down_w[i] = try l.f("down.{d}.w", .{i});
            c.down_b[i] = try l.f("down.{d}.b", .{i});
            c.down_cn[i] = try l.convnext("down.{d}.cn", i);
        }
        c.conv0 = try l.t("dec.conv0");
        c.conv0_b = try l.t("dec.conv0_b");
        const strides = [4]usize{ 8, 8, 4, 2 };
        for (0..4) |bi| {
            const B = &c.dec[bi];
            B.cin = @as(usize, 1536) >> @intCast(bi);
            B.cout = B.cin / 2;
            B.s = strides[bi];
            B.alpha = try l.f("dec.{d}.alpha", .{bi});
            B.wa = try l.f("dec.{d}.wa", .{bi});
            B.wb = try l.f("dec.{d}.wb", .{bi});
            B.b = try l.f("dec.{d}.b", .{bi});
            for (0..3) |r| B.ru[r] = try l.ru("dec", bi, r);
        }
        c.final_alpha = try l.t("dec.final_alpha");
        c.final = try l.t("dec.final");
        c.final_b = try l.t("dec.final_b");
        c.enc_conv0 = try l.t("enc.conv0");
        c.enc_conv0_b = try l.t("enc.conv0_b");
        const enc_strides = [4]usize{ 2, 4, 8, 8 };
        for (0..4) |bi| {
            const B = &c.enc[bi];
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
        c.enc_final_alpha = try l.t("enc.final_alpha");
        c.enc_final = try l.t("enc.final");
        c.enc_final_b = try l.t("enc.final_b");
        c.state_start = c.g.bufs.items.len;
        errdefer c.g.releaseFrom(c.state_start);
        try c.allocDecoder();
        c.state_end = c.g.bufs.items.len;
        return c;
    }

    /// The buffers stay with the Gpu (it frees them at close); this frees the host side.
    pub fn fork(base: *Codec) !*Codec {
        const c = try base.gpa.create(Codec);
        errdefer base.gpa.destroy(c);
        c.* = base.*;
        c.owns_weights = false;
        c.enc_bufs = null;
        c.state_start = c.g.bufs.items.len;
        errdefer c.g.releaseFrom(c.state_start);
        try c.allocDecoder();
        c.state_end = c.g.bufs.items.len;
        return c;
    }

    pub fn poison(c: *Codec) !void {
        try c.g.flush();
        for (c.g.bufs.items[c.state_start..c.state_end]) |buffer| @memset(buffer.contents()[0..buffer.length()], 0xa5);
    }

    pub fn destroy(c: *Codec) void {
        if (c.owns_weights) {
            for (c.host_layers.items) |layers| c.gpa.free(layers);
            c.host_layers.deinit(c.gpa);
        }
        c.gpa.destroy(c);
    }

    const L = struct {
        c: *Codec,
        file: *const source.Source,
        nb: [96]u8 = undefined,
        nb2: [96]u8 = undefined,

        fn t(l: *L, name: []const u8) !u64 {
            const values = try l.file.read(name);
            defer l.c.gpa.free(values);
            const bytes = std.mem.sliceAsBytes(values);
            const p = try l.c.g.alloc(bytes.len);
            @memcpy(l.c.g.host(p, bytes.len), bytes);
            l.c.weight_bytes += bytes.len;
            return p;
        }

        fn f(l: *L, comptime fmt: []const u8, args: anytype) !u64 {
            return l.t(try std.fmt.bufPrint(&l.nb, fmt, args));
        }

        fn tf(l: *L, p: []const u8, n: usize) !Tf {
            const layers = try l.c.gpa.alloc(TfLayer, n);
            errdefer l.c.gpa.free(layers);
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
            try l.c.host_layers.append(l.c.gpa, layers);
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

    fn stage(c: *Codec, margin: usize, cols: usize, rows: usize) !Stage {
        return .{ .buf = try c.g.alloc((margin + rows) * cols * 4), .margin = margin, .cols = cols };
    }

    fn allocDecoder(c: *Codec) !void {
        const fr = c.o.max_chunk;
        for (0..8) |i| c.kv[i] = .{ try c.g.alloc(c.o.ring * D * 4), try c.g.alloc(c.o.ring * D * 4) };
        for (0..2) |i| c.st_up[i] = try c.stage(6, D, fr * (@as(usize, 2) << @intCast(i)));
        c.st_conv0 = try c.stage(6, D, fr * 4);
        var rows = fr * 4;
        for (0..4) |bi| {
            const B = c.dec[bi];
            c.st_tconv[bi] = try c.stage(1, B.cin, rows);
            rows *= B.s;
            for (0..3) |r| c.st_ru[bi][r] = try c.stage(6 * ([3]usize{ 1, 3, 9 })[r], B.cout, rows);
        }
        c.st_final = try c.stage(6, 96, rows);
        // the largest activation of a chunk: 2048 rows x 96 a frame (and 4 x 4096 for ConvNeXt, 6144 for qkv)
        c.scratch_floats = fr * 2048 * 96;
        for (&c.scratch) |*b| b.* = try c.g.alloc(c.scratch_floats * 4);
        c.carry_tmp = try c.g.alloc(54 * 1536 * 4 + 6 * D * 4);
        try c.reset();
    }

    /// Forget the stream: zero every carried row (the transformer ring is rewritten before it is read).
    pub fn reset(c: *Codec) !void {
        try c.g.flush();
        c.pos = 0;
        const all = [_]Stage{ c.st_up[0], c.st_up[1], c.st_conv0, c.st_final } ++ c.st_tconv;
        for (all) |st| c.zeroMargin(st);
        for (c.st_ru) |row| for (row) |st| c.zeroMargin(st);
    }

    /// Host write: the caller has flushed the GPU work that reads this stage.
    pub fn zeroMargin(c: *Codec, st: Stage) void {
        if (st.margin == 0) return;
        @memset(c.g.host(st.buf, st.margin * st.cols * 4), 0);
    }

    // ---------------------------------------------------------------------------------------------- primitives

    /// fp32 Y [t, o] (rows ldy apart) = X [t, cin] (rows ldx apart) W [o, cin]^T + beta Y.
    pub fn sgemm(c: *Codec, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32) void {
        c.gemmTaps(w, o, cin, x, ldx, y, ldy, t, beta, 1, 0, 0);
    }

    /// sgemm summed over `taps` weight blocks [o, cin], tap j reading X rows (taps - 1 - j) * dil earlier.
    fn gemmTaps(c: *Codec, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32, taps: usize, dil: usize, base: usize) void {
        const A = extern struct { rows: u32, o: u32, cin: u32, ldx: u32, ldy: u32, beta: f32, taps: u32, dil: u32, base: u32 };
        const a: A = .{ .rows = u(t), .o = u(o), .cin = u(cin), .ldx = u(ldx), .ldy = u(ldy), .beta = beta, .taps = u(taps), .dil = u(dil), .base = u(base) };
        c.g.go(.c_gemm, .{ (o + 63) / 64, (t + 63) / 64, 1 }, 256, &.{ x, w, y }, a);
    }

    fn elem(c: *Codec, p: gpu.P, bufs: []const u64, total: usize, n: usize, flag: bool) void {
        c.g.go(p, .{ (total + 255) / 256, 1, 1 }, 256, bufs, ElemArgs{ .total = u(total), .n = u(n), .flag = @intFromBool(flag) });
    }

    pub fn bias(c: *Codec, y: u64, b: u64, rows: usize, n: usize, tanh: bool) void {
        c.elem(.c_bias, &.{ y, b }, rows * n, n, tanh);
    }

    fn snake(c: *Codec, x: u64, y: u64, rows: usize, cols: usize, alpha: u64) void {
        c.elem(.c_snake, &.{ x, y, if (alpha != 0) alpha else x }, rows * cols, cols, alpha != 0);
    }

    pub fn rowsAt(st: Stage, row: usize) u64 {
        return st.buf + row * st.cols * 4;
    }

    /// The chunk's input into the stage (snake applied when alpha != 0), after its carried rows.
    pub fn stageIn(c: *Codec, st: Stage, src: u64, rows: usize, alpha: u64) void {
        c.snake(src, rowsAt(st, st.margin), rows, st.cols, alpha);
    }

    /// The chunk's last `margin` rows become the next chunk's carried rows.
    fn carry(c: *Codec, st: Stage, rows: usize) void {
        if (st.margin == 0) return;
        const n = st.margin * st.cols * 4;
        c.g.copy(rowsAt(st, rows), c.carry_tmp, n);
        c.g.copy(c.carry_tmp, st.buf, n);
    }

    /// Causal convolution from a stage: out [rows, o] = b + sum_j W_j x[t - (K - 1 - j) dil].
    pub fn conv(c: *Codec, st: Stage, rows: usize, taps: u64, kk: usize, dil: usize, o: usize, out: u64, b: u64, tanh: bool) void {
        std.debug.assert((kk - 1) * dil <= st.margin);
        c.gemmTaps(taps, o, st.cols, st.buf, st.cols, out, o, rows, 0, kk, dil, st.margin);
        c.bias(out, b, rows, o, tanh);
    }

    fn rmsnorm(c: *Codec, x: u64, w: u64, y: u64, rows: usize) void {
        const A = extern struct { D: u32, eps: f32 };
        c.g.go(.c_rmsnorm, .{ rows, 1, 1 }, 256, &.{ x, w, y }, A{ .D = D, .eps = 1e-5 });
    }

    fn addGamma(c: *Codec, x: u64, y: u64, gamma: u64, rows: usize, n: usize) void {
        c.elem(.c_add_gamma, &.{ x, y, if (gamma != 0) gamma else y }, rows * n, n, gamma != 0);
    }

    /// A window-limited causal transformer over x [rows, 1024] (in place) at positions pos0.., caches kv (ring rows).
    pub fn transformer(c: *Codec, t: Tf, x: u64, rows: usize, pos0: usize, window: usize, kv: []const [2]u64, ring: usize, sc: [3]u64) void {
        const g = c.g;
        for (t.layers, 0..) |ly, li| {
            c.rmsnorm(x, ly.attn_norm, sc[0], rows);
            c.sgemm(ly.wqkv, 3 * D, D, sc[0], D, sc[1], 3 * D, rows, 0);
            const R = extern struct { pos0: u32, ring: u32 };
            g.go(.c_rope_kv, .{ rows, 1, 1 }, 512, &.{ sc[1], kv[li][0], kv[li][1] }, R{ .pos0 = u(pos0), .ring = u(ring) });
            const A = extern struct { pos0: u32, window: u32, ring: u32 };
            g.go(.c_attn, .{ rows, 2, 1 }, 256, &.{ sc[1], kv[li][0], kv[li][1], sc[0] }, A{ .pos0 = u(pos0), .window = u(window), .ring = u(ring) });
            c.sgemm(ly.wo, D, D, sc[0], D, sc[2], D, rows, 0);
            c.addGamma(x, sc[2], ly.g_attn, rows, D);
            c.rmsnorm(x, ly.ffn_norm, sc[0], rows);
            c.sgemm(ly.w13, 6 * D, D, sc[0], D, sc[1], 6 * D, rows, 0);
            g.go(.c_silu_mul, .{ (3 * D + 255) / 256, rows, 1 }, 256, &.{ sc[1], sc[2] }, @as(u32, 3 * D));
            c.sgemm(ly.w2, D, 3 * D, sc[2], 3 * D, sc[0], D, rows, 0);
            c.addGamma(x, sc[0], ly.g_ffn, rows, D);
        }
        c.rmsnorm(x, t.norm, x, rows);
    }

    /// ConvNeXt block over x [rows, 1024] in place, its causal depthwise conv from stage st (carried).
    pub fn convnext(c: *Codec, cn: ConvNext, st: Stage, x: u64, rows: usize, sc: [2]u64) void {
        const g = c.g;
        c.stageIn(st, x, rows, 0);
        g.go(.c_dwconv7, .{ (D + 255) / 256, rows, 1 }, 256, &.{ st.buf, cn.dw, cn.dw_b, sc[0] }, @as(u32, D));
        const A = extern struct { D: u32, eps: f32 };
        g.go(.c_layernorm, .{ rows, 1, 1 }, 256, &.{ sc[0], cn.ln_w, cn.ln_b, sc[0] }, A{ .D = D, .eps = 1e-6 });
        c.sgemm(cn.pw1, 4 * D, D, sc[0], D, sc[1], 4 * D, rows, 0);
        c.bias(sc[1], cn.pw1_b, rows, 4 * D, false);
        const n: u32 = u(rows * 4 * D);
        g.go(.c_gelu, .{ (n + 255) / 256, 1, 1 }, 256, &.{sc[1]}, n);
        c.sgemm(cn.pw2, D, 4 * D, sc[1], 4 * D, sc[0], D, rows, 0);
        c.bias(sc[0], cn.pw2_b, rows, D, false);
        c.addGamma(x, sc[0], cn.gamma, rows, D);
        c.carry(st, rows);
    }

    /// A residual unit on x [rows, C] in place: x += conv1(snake(conv7_dil(snake(x)))).
    pub fn residualUnit(c: *Codec, ru_: Ru, st: Stage, dil: usize, x: u64, rows: usize, sc: [2]u64) void {
        const C = st.cols;
        c.stageIn(st, x, rows, ru_.a1);
        c.conv(st, rows, ru_.c7, 7, dil, C, sc[0], ru_.c7_b, false);
        c.carry(st, rows);
        c.snake(sc[0], sc[1], rows, C, ru_.a2);
        c.sgemm(ru_.c1, C, C, sc[1], C, sc[0], C, rows, 0);
        c.bias(sc[0], ru_.c1_b, rows, C, false);
        c.addGamma(x, sc[0], 0, rows, C);
    }

    // ---------------------------------------------------------------------------------------------- decode

    /// Enqueue the decode of `nf` frames (device int32 rows [nf, 11], codes in columns 1..10) into nf * 2048 samples.
    pub fn decode(c: *Codec, frames: u64, nf: usize, audio: u64) !void {
        var done: usize = 0;
        while (done < nf) {
            const n = @min(c.o.max_chunk, nf - done);
            c.decodeChunk(frames + done * 11 * 4, n, audio + done * samples_per_frame * 4);
            done += n;
        }
    }

    fn decodeChunk(c: *Codec, frames: u64, n: usize, audio: u64) void {
        const sc = c.scratch;
        const x = sc[3];
        const G = extern struct { fs: u32, D: u32 };
        c.g.go(.c_vq_gather, .{ n, 1, 1 }, 256, &.{ frames, c.vq_table, x }, G{ .fs = 11, .D = D });
        c.transformer(c.post, x, n, c.pos, 128, &c.kv, c.o.ring, .{ sc[0], sc[1], sc[2] });
        // upsample x2 x2 (transposed conv kernel = stride 2, then ConvNeXt)
        var rows = n;
        var cur = x;
        for (0..2) |i| {
            const nxt = if (cur == sc[3]) sc[2] else sc[3];
            c.sgemm(c.up_wa[i], 2 * D, D, cur, D, nxt, 2 * D, rows, 0);
            c.bias(nxt, c.up_b[i], rows, 2 * D, false);
            rows *= 2;
            c.convnext(c.up_cn[i], c.st_up[i], nxt, rows, .{ sc[0], sc[1] });
            cur = nxt;
        }
        // decoder: conv0 1024 -> 1536
        c.stageIn(c.st_conv0, cur, rows, 0);
        const y0 = if (cur == sc[3]) sc[2] else sc[3];
        c.conv(c.st_conv0, rows, c.conv0, 7, 1, 1536, y0, c.conv0_b, false);
        c.carry(c.st_conv0, rows);
        cur = y0;
        for (0..4) |bi| {
            const B = c.dec[bi];
            const st = c.st_tconv[bi];
            c.stageIn(st, cur, rows, B.alpha);
            const nxt = if (cur == sc[3]) sc[2] else sc[3];
            c.sgemm(B.wa, B.s * B.cout, B.cin, rowsAt(st, 1), B.cin, nxt, B.s * B.cout, rows, 0);
            c.sgemm(B.wb, B.s * B.cout, B.cin, rowsAt(st, 0), B.cin, nxt, B.s * B.cout, rows, 1);
            c.bias(nxt, B.b, rows, B.s * B.cout, false);
            c.carry(st, rows);
            rows *= B.s;
            cur = nxt;
            for (0..3) |r| c.residualUnit(B.ru[r], c.st_ru[bi][r], ([3]usize{ 1, 3, 9 })[r], cur, rows, .{ sc[0], sc[1] });
        }
        c.stageIn(c.st_final, cur, rows, c.final_alpha);
        c.conv(c.st_final, rows, c.final, 7, 1, 1, audio, c.final_b, true);
        c.carry(c.st_final, rows);
        c.pos += n;
    }

    /// Codes [frames, 10] of a 44.1 kHz mono clip (padded to whole frames), as DAC.encode makes them.
    pub fn encode(c: *Codec, audio: []const f32) ![][10]i32 {
        return @import("codec_encode.zig").encode(c, audio);
    }
};
