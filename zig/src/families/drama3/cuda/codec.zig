//! Drama 3's codec (Fish S2 modded DAC) on CUDA in fp32 with cuBLAS SGEMM, causal so it decodes in carried chunks.
const std = @import("std");
const cuda = @import("cuda");
const source = @import("../codec_load.zig");
const kern = @import("kernels.zig");

pub const samples_per_frame = 2048;
pub const sample_rate = 44100;
pub const D = 1024;

const names = .{ "c_vq_gather", "c_vq_encode", "c_rmsnorm", "c_layernorm", "c_rope_kv", "c_attn", "c_add_gamma", "c_silu_mul", "c_gelu", "c_bias", "c_snake", "c_dwconv7", "c_copy" };
const F = enum(u8) { vq_gather, vq_encode, rmsnorm, layernorm, rope_kv, attn, add_gamma, silu_mul, gelu, bias, snake, dwconv7, copy };

/// A causal convolution's input buffer: `margin` carried rows, then a chunk's rows, `cols` floats a row.
pub const Stage = struct { buf: u64, margin: usize, cols: usize };

const TfLayer = struct { attn_norm: u64, wqkv: u64, wo: u64, ffn_norm: u64, w13: u64, w2: u64, g_attn: u64, g_ffn: u64 };
pub const Tf = struct { layers: []TfLayer, norm: u64 };
const ConvNext = struct { dw: u64, dw_b: u64, ln_w: u64, ln_b: u64, pw1: u64, pw1_b: u64, pw2: u64, pw2_b: u64, gamma: u64 };
const Ru = struct { a1: u64, c7: u64, c7_b: u64, a2: u64, c1: u64, c1_b: u64 };
const DecBlock = struct { alpha: u64, wa: u64, wb: u64, b: u64, ru: [3]Ru, cin: usize, cout: usize, s: usize };
const EncBlock = struct { ru: [3]Ru, alpha: u64, wa: u64, wb: u64, b: u64, cin: usize, dim: usize, s: usize, tf: ?Tf };

/// Diagnostics: a device fp32 tensor [rows, cols] by its codec.pth name.
pub const Probe = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void };

pub const Options = struct { max_chunk: usize = 32, ring: usize = 512 };

pub const Codec = struct {
    gpa: std.mem.Allocator,
    owns_weights: bool,
    root: ?*Codec = null,
    refs: std.atomic.Value(usize) = .init(1),
    state_start: usize,
    state_end: usize,
    host_layers: std.ArrayList([]TfLayer),
    d: *const cuda.Driver,
    k: *const kern.Kernels,
    mod: cuda.Module,
    fns: [names.len]cuda.Function,
    bufs: std.ArrayList(cuda.DeviceBuffer),
    o: Options,
    // weights
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

    fn f(c: *const Codec, which: F) cuda.Function {
        return c.fns[@backingInt(which)];
    }

    pub fn go(c: *const Codec, which: F, s: cuda.Stream, gx: usize, gy: usize, block: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(c.f(which), .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) } }, s, a);
    }

    fn alloc(c: *Codec, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(c.d, @max(bytes, 256));
        errdefer b.free();
        try c.bufs.append(c.gpa, b);
        return b.ptr;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const kern.Kernels, path: []const u8, o: Options) !*Codec {
        const c = try gpa.create(Codec);
        errdefer gpa.destroy(c);
        c.* = undefined;
        c.gpa = gpa;
        c.owns_weights = true;
        c.root = null;
        c.refs = .init(1);
        c.host_layers = .empty;
        errdefer {
            for (c.host_layers.items) |layers| gpa.free(layers);
            c.host_layers.deinit(gpa);
        }
        c.d = d;
        c.k = k;
        c.o = o;
        c.bufs = .empty;
        errdefer {
            for (c.bufs.items) |*buf| buf.free();
            c.bufs.deinit(gpa);
        }
        c.weight_bytes = 0;
        c.pos = 0;
        c.probe = null;
        c.mod = try cuda.Module.load(d, cuda.kernels.drama3_codec);
        errdefer c.mod.unload();
        inline for (names, 0..) |name, j| c.fns[j] = try c.mod.function(name);
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
        c.state_start = c.bufs.items.len;
        try c.allocDecoder();
        c.state_end = c.bufs.items.len;
        return c;
    }

    pub fn fork(base: *Codec) !*Codec {
        const c = try base.gpa.create(Codec);
        errdefer base.gpa.destroy(c);
        c.* = base.*;
        c.owns_weights = false;
        c.root = base.root orelse base;
        c.refs = .init(0);
        c.bufs = .empty;
        errdefer {
            for (c.bufs.items) |*b| b.free();
            c.bufs.deinit(c.gpa);
        }
        c.state_start = c.bufs.items.len;
        try c.allocDecoder();
        c.state_end = c.bufs.items.len;
        _ = c.root.?.refs.fetchAdd(1, .monotonic);
        return c;
    }

    pub fn poison(c: *Codec) !void {
        for (c.bufs.items[c.state_start..c.state_end]) |*buffer| try buffer.fill8(0xa5, null);
    }

    pub fn destroy(c: *Codec) void {
        const root = c.root orelse c;
        if (c.root != null) c.releaseOwned();
        const prior = root.refs.fetchSub(1, .acq_rel);
        std.debug.assert(prior > 0);
        // A suspended or quarantined fork retains its root, including host descriptors and the CUDA module.
        if (prior == 1) root.releaseOwned();
    }

    fn releaseOwned(c: *Codec) void {
        for (c.bufs.items) |*b| b.free();
        c.bufs.deinit(c.gpa);
        if (c.owns_weights) {
            for (c.host_layers.items) |layers| c.gpa.free(layers);
            c.host_layers.deinit(c.gpa);
            c.mod.unload();
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
            const p = try l.c.alloc(bytes.len);
            var b: cuda.DeviceBuffer = .{ .d = l.c.d, .ptr = p, .len = bytes.len, .borrowed = true };
            try b.upload(0, bytes);
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
        return .{ .buf = try c.alloc((margin + rows) * cols * 4), .margin = margin, .cols = cols };
    }

    fn allocDecoder(c: *Codec) !void {
        const fr = c.o.max_chunk;
        for (0..8) |i| c.kv[i] = .{ try c.alloc(c.o.ring * D * 4), try c.alloc(c.o.ring * D * 4) };
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
        for (&c.scratch) |*b| b.* = try c.alloc(c.scratch_floats * 4);
        c.carry_tmp = try c.alloc(54 * 1536 * 4 + 6 * D * 4);
        try c.reset(null);
    }

    /// Forget the stream: zero every carried row (and the transformer ring is rewritten before it is read).
    pub fn reset(c: *Codec, s: ?cuda.Stream) !void {
        c.pos = 0;
        const all = [_]Stage{ c.st_up[0], c.st_up[1], c.st_conv0, c.st_final } ++ c.st_tconv;
        for (all) |st| try c.zeroMargin(st, s);
        for (c.st_ru) |row| for (row) |st| try c.zeroMargin(st, s);
    }

    fn zeroMargin(c: *Codec, st: Stage, s: ?cuda.Stream) !void {
        if (st.margin == 0) return;
        var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = st.buf, .len = st.margin * st.cols * 4, .borrowed = true };
        try b.fill8(0, if (s) |x| x.handle else null);
    }

    // ---------------------------------------------------------------------------------------------- primitives

    pub fn sgemm(c: *Codec, s: cuda.Stream, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32) !void {
        try c.k.sgemm(s, w, o, cin, x, ldx, y, ldy, t, beta);
    }

    pub fn bias(c: *Codec, s: cuda.Stream, y: u64, b: u64, rows: usize, n: usize, tanh: bool) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(b);
        a.add(rows);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intFromBool(tanh)));
        try c.go(.bias, s, @min((rows * n + 255) / 256, 8192), 1, 256, &a);
    }

    fn snake(c: *Codec, s: cuda.Stream, x: u64, y: u64, rows: usize, cols: usize, alpha: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(rows);
        a.add(@as(i32, @intCast(cols)));
        a.add(alpha);
        try c.go(.snake, s, @min((rows * cols + 255) / 256, 8192), 1, 256, &a);
    }

    fn copy(c: *Codec, s: cuda.Stream, src: u64, dst: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(n);
        try c.go(.copy, s, @min((n + 255) / 256, 4096), 1, 256, &a);
    }

    pub fn rowsAt(st: Stage, row: usize) u64 {
        return st.buf + row * st.cols * 4;
    }

    /// The chunk's input into the stage (snake applied when alpha != 0), after its carried rows.
    pub fn stageIn(c: *Codec, s: cuda.Stream, st: Stage, src: u64, rows: usize, alpha: u64) !void {
        try c.snake(s, src, rowsAt(st, st.margin), rows, st.cols, alpha);
    }

    /// The chunk's last `margin` rows become the next chunk's carried rows.
    fn carry(c: *Codec, s: cuda.Stream, st: Stage, rows: usize) !void {
        if (st.margin == 0) return;
        const n = st.margin * st.cols;
        try c.copy(s, rowsAt(st, rows), c.carry_tmp, n);
        try c.copy(s, c.carry_tmp, st.buf, n);
    }

    /// Causal convolution from a stage: out [rows, o] = b + sum_j W_j x[t - (K - 1 - j) dil].
    pub fn conv(c: *Codec, s: cuda.Stream, st: Stage, rows: usize, taps: u64, kk: usize, dil: usize, o: usize, out: u64, b: u64, tanh: bool) !void {
        const cin = st.cols;
        for (0..kk) |j| {
            const x = rowsAt(st, st.margin - (kk - 1 - j) * dil);
            try c.sgemm(s, taps + j * o * cin * 4, o, cin, x, cin, out, o, rows, if (j == 0) 0 else 1);
        }
        try c.bias(s, out, b, rows, o, tanh);
    }

    fn rmsnorm(c: *Codec, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(y);
        a.add(@as(i32, D));
        a.add(@as(f32, 1e-5));
        try c.go(.rmsnorm, s, rows, 1, 256, &a);
    }

    fn addGamma(c: *Codec, s: cuda.Stream, x: u64, y: u64, gamma: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(gamma);
        a.add(rows * n);
        a.add(@as(i32, @intCast(n)));
        try c.go(.add_gamma, s, @min((rows * n + 255) / 256, 8192), 1, 256, &a);
    }

    /// A window-limited causal transformer over x [rows, 1024] (in place) at positions pos0.., caches kv (ring rows).
    pub fn transformer(c: *Codec, s: cuda.Stream, t: Tf, x: u64, rows: usize, pos0: usize, window: usize, kv: []const [2]u64, ring: usize, sc: [3]u64) !void {
        for (t.layers, 0..) |ly, li| {
            try c.rmsnorm(s, x, ly.attn_norm, sc[0], rows);
            try c.sgemm(s, ly.wqkv, 3 * D, D, sc[0], D, sc[1], 3 * D, rows, 0);
            var a: cuda.Args = .{};
            a.add(sc[1]);
            a.add(@as(i32, @intCast(pos0)));
            a.add(kv[li][0]);
            a.add(kv[li][1]);
            a.add(@as(i32, @intCast(ring)));
            try c.go(.rope_kv, s, rows, 1, 512, &a);
            var b: cuda.Args = .{};
            b.add(sc[1]);
            b.add(kv[li][0]);
            b.add(kv[li][1]);
            b.add(sc[0]);
            b.add(@as(i32, @intCast(pos0)));
            b.add(@as(i32, @intCast(window)));
            b.add(@as(i32, @intCast(ring)));
            try c.go(.attn, s, rows, 2, 256, &b);
            try c.sgemm(s, ly.wo, D, D, sc[0], D, sc[2], D, rows, 0);
            try c.addGamma(s, x, sc[2], ly.g_attn, rows, D);
            try c.rmsnorm(s, x, ly.ffn_norm, sc[0], rows);
            try c.sgemm(s, ly.w13, 6 * D, D, sc[0], D, sc[1], 6 * D, rows, 0);
            var m: cuda.Args = .{};
            m.add(sc[1]);
            m.add(sc[2]);
            m.add(@as(i32, 3 * D));
            try c.go(.silu_mul, s, (3 * D + 255) / 256, rows, 256, &m);
            try c.sgemm(s, ly.w2, D, 3 * D, sc[2], 3 * D, sc[0], D, rows, 0);
            try c.addGamma(s, x, sc[0], ly.g_ffn, rows, D);
        }
        try c.rmsnorm(s, x, t.norm, x, rows);
    }

    /// ConvNeXt block over x [rows, 1024] in place, its causal depthwise conv from stage st (carried).
    pub fn convnext(c: *Codec, s: cuda.Stream, cn: ConvNext, st: Stage, x: u64, rows: usize, sc: [2]u64) !void {
        try c.stageIn(s, st, x, rows, 0);
        var a: cuda.Args = .{};
        a.add(st.buf);
        a.add(cn.dw);
        a.add(cn.dw_b);
        a.add(sc[0]);
        a.add(@as(i32, D));
        try c.go(.dwconv7, s, (D + 255) / 256, rows, 256, &a);
        var b: cuda.Args = .{};
        b.add(sc[0]);
        b.add(cn.ln_w);
        b.add(cn.ln_b);
        b.add(sc[0]);
        b.add(@as(i32, D));
        b.add(@as(f32, 1e-6));
        try c.go(.layernorm, s, rows, 1, 256, &b);
        try c.sgemm(s, cn.pw1, 4 * D, D, sc[0], D, sc[1], 4 * D, rows, 0);
        try c.bias(s, sc[1], cn.pw1_b, rows, 4 * D, false);
        var g: cuda.Args = .{};
        g.add(sc[1]);
        g.add(rows * 4 * D);
        try c.go(.gelu, s, @min((rows * 4 * D + 255) / 256, 8192), 1, 256, &g);
        try c.sgemm(s, cn.pw2, D, 4 * D, sc[1], 4 * D, sc[0], D, rows, 0);
        try c.bias(s, sc[0], cn.pw2_b, rows, D, false);
        try c.addGamma(s, x, sc[0], cn.gamma, rows, D);
        try c.carry(s, st, rows);
    }

    /// A residual unit on x [rows, C] in place: x += conv1(snake(conv7_dil(snake(x)))).
    pub fn residualUnit(c: *Codec, s: cuda.Stream, u: Ru, st: Stage, dil: usize, x: u64, rows: usize, sc: [2]u64) !void {
        const C = st.cols;
        try c.stageIn(s, st, x, rows, u.a1);
        try c.conv(s, st, rows, u.c7, 7, dil, C, sc[0], u.c7_b, false);
        try c.carry(s, st, rows);
        try c.snake(s, sc[0], sc[1], rows, C, u.a2);
        try c.sgemm(s, u.c1, C, C, sc[1], C, sc[0], C, rows, 0);
        try c.bias(s, sc[0], u.c1_b, rows, C, false);
        try c.addGamma(s, x, sc[0], 0, rows, C);
    }

    // ---------------------------------------------------------------------------------------------- decode

    /// Decode `nf` frames (device int32 rows [nf, 11], codes in columns 1..10) into nf * 2048 fp32 samples at `audio`.
    pub fn decode(c: *Codec, s: cuda.Stream, frames: u64, nf: usize, audio: u64) !void {
        var done: usize = 0;
        while (done < nf) {
            const n = @min(c.o.max_chunk, nf - done);
            try c.decodeChunk(s, frames + done * 11 * 4, n, audio + done * samples_per_frame * 4);
            done += n;
        }
    }

    fn decodeChunk(c: *Codec, s: cuda.Stream, frames: u64, n: usize, audio: u64) !void {
        const sc = c.scratch;
        const x = sc[3];
        // codes -> z [n, 1024]
        var a: cuda.Args = .{};
        a.add(frames);
        a.add(@as(i32, 11));
        a.add(c.vq_table);
        a.add(x);
        a.add(@as(i32, D));
        try c.go(.vq_gather, s, n, 1, 256, &a);
        try c.transformer(s, c.post, x, n, c.pos, 128, &c.kv, c.o.ring, .{ sc[0], sc[1], sc[2] });
        // upsample x2 x2 (transposed conv kernel = stride 2, then ConvNeXt)
        var rows = n;
        var cur = x;
        for (0..2) |i| {
            const nxt = if (cur == sc[3]) sc[2] else sc[3];
            try c.sgemm(s, c.up_wa[i], 2 * D, D, cur, D, nxt, 2 * D, rows, 0);
            try c.bias(s, nxt, c.up_b[i], rows, 2 * D, false);
            rows *= 2;
            try c.convnext(s, c.up_cn[i], c.st_up[i], nxt, rows, .{ sc[0], sc[1] });
            cur = nxt;
        }
        // decoder: conv0 1024 -> 1536
        try c.stageIn(s, c.st_conv0, cur, rows, 0);
        const y0 = if (cur == sc[3]) sc[2] else sc[3];
        try c.conv(s, c.st_conv0, rows, c.conv0, 7, 1, 1536, y0, c.conv0_b, false);
        try c.carry(s, c.st_conv0, rows);
        cur = y0;
        for (0..4) |bi| {
            const B = c.dec[bi];
            const st = c.st_tconv[bi];
            try c.stageIn(s, st, cur, rows, B.alpha);
            const nxt = if (cur == sc[3]) sc[2] else sc[3];
            try c.sgemm(s, B.wa, B.s * B.cout, B.cin, rowsAt(st, 1), B.cin, nxt, B.s * B.cout, rows, 0);
            try c.sgemm(s, B.wb, B.s * B.cout, B.cin, rowsAt(st, 0), B.cin, nxt, B.s * B.cout, rows, 1);
            try c.bias(s, nxt, B.b, rows, B.s * B.cout, false);
            try c.carry(s, st, rows);
            rows *= B.s;
            cur = nxt;
            for (0..3) |r| try c.residualUnit(s, B.ru[r], c.st_ru[bi][r], ([3]usize{ 1, 3, 9 })[r], cur, rows, .{ sc[0], sc[1] });
        }
        try c.stageIn(s, c.st_final, cur, rows, c.final_alpha);
        try c.conv(s, c.st_final, rows, c.final, 7, 1, 1, audio, c.final_b, true);
        try c.carry(s, c.st_final, rows);
        c.pos += n;
    }

    /// Codes [frames, 10] of a 44.1 kHz mono clip (padded to whole frames), as DAC.encode makes them.
    pub fn encode(c: *Codec, s: cuda.Stream, audio: []const f32) ![][10]i32 {
        return @import("codec_encode.zig").encode(c, s, audio);
    }
};
