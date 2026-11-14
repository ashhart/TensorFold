//! Drama 3's codec (Fish S2 modded DAC) on CUDA in fp32 with cuBLAS SGEMM: codec_net.zig's schedules on its kernels.
const std = @import("std");
const cuda = @import("cuda");
const net = @import("../codec_net.zig");
const source = @import("../codec_load.zig");
const kern = @import("kernels.zig");

pub const samples_per_frame = net.samples_per_frame;
pub const sample_rate = net.sample_rate;
pub const D = net.D;
pub const Stage = net.Stage;
pub const Tf = net.Tf;
pub const Probe = net.Probe;
pub const Options = net.Options;

const names = .{ "c_vq_gather", "c_vq_encode", "c_rmsnorm", "c_layernorm", "c_rope_kv", "c_attn", "c_add_gamma", "c_silu_mul", "c_gelu", "c_bias", "c_snake", "c_dwconv7", "c_copy" };
const F = enum(u8) { vq_gather, vq_encode, rmsnorm, layernorm, rope_kv, attn, add_gamma, silu_mul, gelu, bias, snake, dwconv7, copy };

pub const Codec = struct {
    gpa: std.mem.Allocator,
    owns_weights: bool,
    root: ?*Codec = null,
    refs: std.atomic.Value(usize) = .init(1),
    state_start: usize,
    state_end: usize,
    d: *const cuda.Driver,
    k: *const kern.Kernels,
    mod: cuda.Module,
    fns: [names.len]cuda.Function,
    bufs: std.ArrayList(cuda.DeviceBuffer),
    o: Options,
    w: net.Weights,
    state: net.Decoder,
    probe: ?Probe = null,

    fn f(c: *const Codec, which: F) cuda.Function {
        return c.fns[@backingInt(which)];
    }

    pub fn go(c: *const Codec, which: F, s: cuda.Stream, gx: usize, gy: usize, block: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(c.f(which), .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) } }, s, a);
    }

    pub fn alloc(c: *Codec, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(c.d, @max(bytes, 256));
        errdefer b.free();
        try c.bufs.append(c.gpa, b);
        return b.ptr;
    }

    /// A weight's device copy, owned by the codec.
    pub fn put(c: *Codec, bytes: []const u8) !u64 {
        const p = try c.alloc(bytes.len);
        var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = p, .len = bytes.len, .borrowed = true };
        try b.upload(0, bytes);
        return p;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const kern.Kernels, path: []const u8, o: Options) !*Codec {
        const c = try gpa.create(Codec);
        errdefer gpa.destroy(c);
        c.* = undefined;
        c.gpa = gpa;
        c.owns_weights = true;
        c.root = null;
        c.refs = .init(1);
        c.d = d;
        c.k = k;
        c.o = o;
        c.bufs = .empty;
        errdefer {
            for (c.bufs.items) |*buf| buf.free();
            c.bufs.deinit(gpa);
        }
        c.probe = null;
        c.mod = try cuda.Module.load(d, cuda.kernels.drama3_codec);
        errdefer c.mod.unload();
        inline for (names, 0..) |name, j| c.fns[j] = try c.mod.function(name);
        var file = try source.Source.open(gpa, io, path);
        defer file.close();
        c.w = try net.Weights.load(gpa, &file, c);
        errdefer c.w.deinit(gpa);
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
        // Reset and decode use non-blocking streams, so the default-stream fills must finish first.
        try c.d.synchronize();
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
            c.w.deinit(c.gpa);
            c.mod.unload();
        }
        c.gpa.destroy(c);
    }

    fn allocDecoder(c: *Codec) !void {
        c.state = try net.Decoder.alloc(&c.w, c.o, c);
        try c.reset(null);
    }

    /// Forget the stream: zero every carried row (and the transformer ring is rewritten before it is read).
    pub fn reset(c: *Codec, s: ?cuda.Stream) !void {
        c.state.pos = 0;
        for (c.state.stages()) |st| try c.zeroMargin(st, s);
    }

    fn zeroMargin(c: *Codec, st: Stage, s: ?cuda.Stream) !void {
        if (st.margin == 0) return;
        var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = st.buf, .len = st.margin * st.cols * 4, .borrowed = true };
        try b.fill8(0, if (s) |x| x.handle else null);
    }

    fn run(c: *Codec, s: cuda.Stream) Run {
        return .{ .c = c, .s = s, .tmp = c.state.carry_tmp };
    }

    /// Decode `nf` frames (device int32 rows [nf, 11], codes in columns 1..10) into nf * 2048 fp32 samples at `audio`.
    pub fn decode(c: *Codec, s: cuda.Stream, frames: u64, nf: usize, audio: u64) !void {
        try net.decode(c.run(s), &c.w, &c.state, c.o, frames, nf, audio);
    }

    /// Codes [frames, 10] of a 44.1 kHz mono clip (padded to whole frames), as DAC.encode makes them.
    pub fn encode(c: *Codec, s: cuda.Stream, audio: []const f32) ![][10]i32 {
        try @import("../model.zig").reference(audio, 44100);
        const gpa = c.gpa;
        const nf = (audio.len + samples_per_frame - 1) / samples_per_frame;
        if (nf == 0) return error.EmptyAudio;
        const L0 = nf * samples_per_frame;
        // buffers for the whole clip: the widest activation is L0 x 64 (and the stages' margins)
        const big = L0 * 64 + 64 * 1024;
        var e: Clip = .{ .c = c, .s = s };
        defer {
            for (e.tmp.items) |*b| b.free();
            e.tmp.deinit(gpa);
        }
        var sc: [4]u64 = undefined;
        for (&sc) |*p| p.* = try e.get(big * 4);
        const stbuf = try e.get((L0 * 64 + 64 * 1024) * 4);
        // audio in
        {
            var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = sc[0], .len = L0 * 4, .borrowed = true };
            try b.fill8(0, s.handle);
            try b.uploadAsync(0, std.mem.sliceAsBytes(audio), s.handle);
        }
        try net.encode(c.run(s), &e, &c.w, sc, stbuf, nf);
        const out = try gpa.alloc([10]i32, nf);
        errdefer gpa.free(out);
        var b: cuda.DeviceBuffer = .{ .d = c.d, .ptr = e.codes_dev, .len = nf * 40, .borrowed = true };
        try b.downloadAsync(0, std.mem.sliceAsBytes(out), s.handle);
        try s.synchronize();
        return out;
    }
};

/// A clip's encoder buffers on CUDA, freed when the encode returns.
const Clip = struct {
    c: *Codec,
    s: cuda.Stream,
    tmp: std.ArrayList(cuda.DeviceBuffer) = .empty,
    caches: [8][2]u64 = undefined,
    codes_dev: u64 = 0,

    fn get(e: *Clip, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(e.c.d, @max(bytes, 256));
        try e.tmp.append(e.c.gpa, b);
        return b.ptr;
    }

    pub fn stage(e: *Clip, buf: u64, margin: usize, cols: usize) !Stage {
        var b: cuda.DeviceBuffer = .{ .d = e.c.d, .ptr = buf, .len = margin * cols * 4, .borrowed = true };
        try b.fill8(0, e.s.handle);
        return .{ .buf = buf, .margin = margin, .cols = cols };
    }

    pub fn kv(e: *Clip, layers: usize, rows: usize) ![]const [2]u64 {
        for (e.caches[0..layers]) |*p| p.* = .{ try e.get(rows * D * 4), try e.get(rows * D * 4) };
        return e.caches[0..layers];
    }

    pub fn codes(e: *Clip, nf: usize) !u64 {
        e.codes_dev = try e.get(nf * 10 * 4);
        return e.codes_dev;
    }

    pub fn probe(e: *Clip, name: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        if (e.c.probe) |p| {
            try e.s.synchronize();
            try p.f(p.ctx, name, ptr, rows, cols);
        }
    }
};

/// The codec's kernels on one stream, as codec_net.zig's schedules call them.
const Run = struct {
    c: *Codec,
    s: cuda.Stream,
    tmp: u64,

    pub fn sgemm(r: Run, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32) !void {
        try r.c.k.sgemm(r.s, w, o, cin, x, ldx, y, ldy, t, beta);
    }

    pub fn bias(r: Run, y: u64, b: u64, rows: usize, n: usize, tanh: bool) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(b);
        a.add(rows);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intFromBool(tanh)));
        try r.c.go(.bias, r.s, @min((rows * n + 255) / 256, 8192), 1, 256, &a);
    }

    pub fn snake(r: Run, x: u64, y: u64, rows: usize, cols: usize, alpha: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(rows);
        a.add(@as(i32, @intCast(cols)));
        a.add(alpha);
        try r.c.go(.snake, r.s, @min((rows * cols + 255) / 256, 8192), 1, 256, &a);
    }

    /// `n` floats.
    pub fn copy(r: Run, src: u64, dst: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(n);
        try r.c.go(.copy, r.s, @min((n + 255) / 256, 4096), 1, 256, &a);
    }

    /// Causal convolution from a stage: out [rows, o] = b + sum_j W_j x[t - (K - 1 - j) dil].
    pub fn conv(r: Run, st: Stage, rows: usize, taps: u64, kk: usize, dil: usize, o: usize, out: u64, b: u64, tanh: bool) !void {
        const cin = st.cols;
        for (0..kk) |j| {
            const x = net.rowsAt(st, st.margin - (kk - 1 - j) * dil);
            try r.sgemm(taps + j * o * cin * 4, o, cin, x, cin, out, o, rows, if (j == 0) 0 else 1);
        }
        try r.bias(out, b, rows, o, tanh);
    }

    pub fn rmsnorm(r: Run, x: u64, w: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(y);
        a.add(@as(i32, D));
        a.add(@as(f32, 1e-5));
        try r.c.go(.rmsnorm, r.s, rows, 1, 256, &a);
    }

    pub fn addGamma(r: Run, x: u64, y: u64, gamma: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(gamma);
        a.add(rows * n);
        a.add(@as(i32, @intCast(n)));
        try r.c.go(.add_gamma, r.s, @min((rows * n + 255) / 256, 8192), 1, 256, &a);
    }

    pub fn ropeKv(r: Run, qkv: u64, kv: [2]u64, pos0: usize, ring: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(@as(i32, @intCast(pos0)));
        a.add(kv[0]);
        a.add(kv[1]);
        a.add(@as(i32, @intCast(ring)));
        try r.c.go(.rope_kv, r.s, rows, 1, 512, &a);
    }

    pub fn attn(r: Run, qkv: u64, kv: [2]u64, out: u64, pos0: usize, window: usize, ring: usize, rows: usize) !void {
        var b: cuda.Args = .{};
        b.add(qkv);
        b.add(kv[0]);
        b.add(kv[1]);
        b.add(out);
        b.add(@as(i32, @intCast(pos0)));
        b.add(@as(i32, @intCast(window)));
        b.add(@as(i32, @intCast(ring)));
        try r.c.go(.attn, r.s, rows, 2, 256, &b);
    }

    pub fn siluMul(r: Run, x: u64, y: u64, rows: usize) !void {
        var m: cuda.Args = .{};
        m.add(x);
        m.add(y);
        m.add(@as(i32, 3 * D));
        try r.c.go(.silu_mul, r.s, (3 * D + 255) / 256, rows, 256, &m);
    }

    pub fn dwconv7(r: Run, src: u64, dw: u64, dw_b: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dw);
        a.add(dw_b);
        a.add(out);
        a.add(@as(i32, D));
        try r.c.go(.dwconv7, r.s, (D + 255) / 256, rows, 256, &a);
    }

    pub fn layernorm(r: Run, x: u64, w: u64, b: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(@as(i32, D));
        a.add(@as(f32, 1e-6));
        try r.c.go(.layernorm, r.s, rows, 1, 256, &a);
    }

    pub fn gelu(r: Run, x: u64, n: usize) !void {
        var g: cuda.Args = .{};
        g.add(x);
        g.add(n);
        try r.c.go(.gelu, r.s, @min((n + 255) / 256, 8192), 1, 256, &g);
    }

    pub fn vqGather(r: Run, frames: u64, table: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(frames);
        a.add(@as(i32, 11));
        a.add(table);
        a.add(out);
        a.add(@as(i32, D));
        try r.c.go(.vq_gather, r.s, n, 1, 256, &a);
    }

    pub fn vqEncode(r: Run, x: u64, w: *const net.Weights, codes: u64, nf: usize) !void {
        var q: cuda.Args = .{};
        q.add(x);
        q.add(@as(i32, D));
        q.add(w.vq_in_w);
        q.add(w.vq_in_b);
        q.add(w.vq_cbn);
        q.add(w.vq_table);
        q.add(codes);
        try r.c.go(.vq_encode, r.s, nf, 1, 256, &q);
    }
};
