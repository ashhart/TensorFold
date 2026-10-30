//! One Liquid stack on Metal (gated short conv or attention layers, then SwiGLU) with its caches.
const std = @import("std");
const st = @import("core").safetensors;
const kern = @import("kernels.zig");
const gpu = @import("gpu.zig");
const spec = @import("../spec.zig");

pub const Kind = spec.Kind;
pub const Rope = spec.Rope;
pub const Spec = spec.Spec;

pub const Layer = struct {
    kind: Kind,
    op_norm: u64,
    ffn_norm: u64,
    post_op_norm: u64 = 0,
    post_ffn_norm: u64 = 0,
    // conv
    in_proj: u64 = 0, // [3D, D]
    conv_w: u64 = 0, // [D, K]
    out_proj: u64 = 0, // [D, D]
    state: u64 = 0, // [D, K-1]
    // attention
    qkvg: u64 = 0, // [q | k | v | gate, D]
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    o_proj: u64 = 0, // [D, H * hd]
    kc: u64 = 0, // [max_ctx, Hkv * hd]
    vc: u64 = 0,
    // feed-forward
    w13: u64 = 0, // [2F, D]
    w2: u64 = 0, // [D, F]
};

pub const Stack = struct {
    s: Spec,
    layers: []Layer,
    final_norm: u64,
    inv_full: u64,
    inv_sliding: u64,
    // scratch
    xn: u64,
    proj: u64,
    y: u64,
    tmp: u64,
    states: u64,
    states_bytes: usize,
    /// Rows already in the caches (the next row's position).
    pos: usize = 0,
    g: *gpu.Gpu,

    pub fn nq(x: *const Stack) usize {
        return x.s.heads * x.s.head_dim;
    }
    pub fn nk(x: *const Stack) usize {
        return x.s.kv_heads * x.s.head_dim;
    }
    fn qkvgCols(x: *const Stack) usize {
        return x.nq() + 2 * x.nk() + if (x.s.lfm3) x.s.heads else 0;
    }

    /// Clears the conv states and the cache position (in order with the dispatches; the caches' rows are overwritten).
    pub fn reset(x: *Stack) void {
        x.pos = 0;
        if (x.states_bytes > 0) x.g.zero(x.states, x.states_bytes);
    }

    /// Runs `t` rows of the residual stream `x` in place at the cache position, then the final norm into `out`.
    pub fn forward(x_: *Stack, k: *const kern.Kernels, x: u64, t: usize, out: u64) !void {
        const sp = x_.s;
        if (t == 0) return;
        if (t > sp.max_rows) return error.TooManyRows;
        if (x_.pos + t > sp.max_ctx) return error.ContextFull;
        if (t <= kern.gemv_rows and kern.Kernels.few(t, sp.d, sp.d, x)) try x_.forwardFew(k, x, t, out) else try x_.forwardRows(k, x, t, out);
        x_.pos += t;
    }

    fn attend(x_: *const Stack, k: *const kern.Kernels, l: *const Layer, t: usize) !void {
        const sp = x_.s;
        const nq_ = x_.nq();
        const nk_ = x_.nk();
        const cols = x_.qkvgCols();
        const windowed = l.kind == .sliding or (sp.window_all and sp.window > 0);
        try k.attention(.{ .q = x_.proj, .ldq = cols, .kc = l.kc, .vc = l.vc, .ldkv = nk_, .o = x_.y, .ldo = nq_, .t = t, .heads = sp.heads, .kv_heads = sp.kv_heads, .hd = sp.head_dim, .pos0 = x_.pos, .kv_len = x_.pos + t, .window = if (windowed) sp.window else 0, .causal = sp.causal, .gate = if (sp.lfm3) x_.proj + (nq_ + 2 * nk_) * 2 else 0, .ldg = cols });
    }

    /// Up to 16 rows, fused: heads' norm, rotary and cache write; the gate in attention; SwiGLU in the w1 | w3 product.
    fn forwardFew(x_: *Stack, k: *const kern.Kernels, x: u64, t: usize, out: u64) !void {
        const sp = x_.s;
        const D = sp.d;
        const nq_ = x_.nq();
        const cols = x_.qkvgCols();
        var pending: u64 = 0; // the last block's output, added to the residual by the next norm
        var pending_pw: u64 = 0; // its post norm (sandwich layers)
        for (x_.layers) |*l| {
            k.rmsnormPost(x, pending, pending_pw, l.op_norm, 0, x_.xn, D, D, D, t, sp.eps, .lfm);
            switch (l.kind) {
                .conv => {
                    k.product(l.in_proj, x_.xn, D, D, 3 * D, t, x_.proj, 3 * D, 0, 1, 0, 0, 0, .plain);
                    k.shortConv(x_.proj, 3 * D, l.conv_w, l.state, x_.y, D, t, D, sp.conv_k);
                    k.linear(l.out_proj, x_.y, D, D, D, x_.tmp, D, t);
                },
                .full, .sliding => {
                    k.product(l.qkvg, x_.xn, D, D, cols, t, x_.proj, cols, 0, 1, 0, 0, 0, .plain);
                    const inv = if (l.kind == .sliding) x_.inv_sliding else x_.inv_full;
                    k.qkPrep(x_.proj, l.q_norm, l.k_norm, inv, l.kc, l.vc, t, sp.heads, sp.kv_heads, sp.head_dim, cols, x_.pos, sp.rope_mode, x_.nk(), sp.eps);
                    try x_.attend(k, l, t);
                    k.linear(l.o_proj, x_.y, nq_, nq_, D, x_.tmp, D, t);
                },
            }
            k.rmsnormPost(x, x_.tmp, if (sp.sandwich) l.post_op_norm else 0, l.ffn_norm, 0, x_.xn, D, D, D, t, sp.eps, .lfm);
            k.product(l.w13, x_.xn, D, D, 2 * sp.ffn, t, x_.y, sp.ffn, 0, 1, 0, 0, 0, .swiglu);
            k.linear(l.w2, x_.y, sp.ffn, sp.ffn, D, x_.tmp, D, t);
            pending = x_.tmp;
            pending_pw = if (sp.sandwich) l.post_ffn_norm else 0;
        }
        k.rmsnormPost(x, pending, pending_pw, x_.final_norm, 0, out, D, D, D, t, sp.eps, .lfm);
    }

    /// Many rows (prompt chunks, the speech encoder, codec chunks): one op a dispatch.
    fn forwardRows(x_: *Stack, k: *const kern.Kernels, x: u64, t: usize, out: u64) !void {
        const sp = x_.s;
        const D = sp.d;
        const nq_ = x_.nq();
        const nk_ = x_.nk();
        const cols = x_.qkvgCols();
        var pending: u64 = 0; // the last block's output, added to the residual by the next norm
        for (x_.layers) |*l| {
            k.rmsnorm(x, pending, l.op_norm, x_.xn, D, D, D, t, sp.eps, .lfm);
            switch (l.kind) {
                .conv => {
                    k.linear(l.in_proj, x_.xn, D, D, 3 * D, x_.proj, 3 * D, t);
                    k.shortConv(x_.proj, 3 * D, l.conv_w, l.state, x_.y, D, t, D, sp.conv_k);
                    k.linear(l.out_proj, x_.y, D, D, D, x_.tmp, D, t);
                },
                .full, .sliding => {
                    k.linear(l.qkvg, x_.xn, D, D, cols, x_.proj, cols, t);
                    k.headNorm(x_.proj, l.q_norm, t, sp.heads, sp.head_dim, cols, sp.eps);
                    k.headNorm(x_.proj + nq_ * 2, l.k_norm, t, sp.kv_heads, sp.head_dim, cols, sp.eps);
                    const inv = if (l.kind == .sliding) x_.inv_sliding else x_.inv_full;
                    k.rope(x_.proj, inv, t, sp.heads, sp.head_dim, cols, x_.pos, sp.rope_mode);
                    k.rope(x_.proj + nq_ * 2, inv, t, sp.kv_heads, sp.head_dim, cols, x_.pos, sp.rope_mode);
                    k.copyRows(x_.proj + nq_ * 2, l.kc + x_.pos * nk_ * 2, t, nk_, cols, nk_);
                    k.copyRows(x_.proj + (nq_ + nk_) * 2, l.vc + x_.pos * nk_ * 2, t, nk_, cols, nk_);
                    try x_.attend(k, l, t);
                    k.linear(l.o_proj, x_.y, nq_, nq_, D, x_.tmp, D, t);
                },
            }
            if (sp.sandwich) k.rmsnorm(x_.tmp, 0, l.post_op_norm, x_.tmp, D, D, D, t, sp.eps, .lfm);
            k.rmsnorm(x, x_.tmp, l.ffn_norm, x_.xn, D, D, D, t, sp.eps, .lfm);
            k.linear(l.w13, x_.xn, D, D, 2 * sp.ffn, x_.proj, 2 * sp.ffn, t);
            k.swiglu(x_.proj, x_.y, t, sp.ffn, true);
            k.linear(l.w2, x_.y, sp.ffn, sp.ffn, D, x_.tmp, D, t);
            if (sp.sandwich) k.rmsnorm(x_.tmp, 0, l.post_ffn_norm, x_.tmp, D, D, D, t, sp.eps, .lfm);
            pending = x_.tmp;
        }
        k.rmsnorm(x, pending, x_.final_norm, out, D, D, D, t, sp.eps, .lfm);
    }
};

/// Device buffers a loader hands out (freed with the device).
pub const Arena = struct {
    g: *gpu.Gpu,

    pub fn alloc(a: *Arena, n: usize) !u64 {
        return a.g.alloc(n);
    }

    pub fn zeroed(a: *Arena, n: usize) !u64 {
        return a.g.alloc(n);
    }

    pub fn upload(a: *Arena, bytes: []const u8) !u64 {
        const p = try a.g.alloc(bytes.len);
        @memcpy(a.g.host(p, bytes.len), bytes);
        return p;
    }

    /// Bytes at `off` of a buffer made here (before any GPU work reads it).
    pub fn write(a: *Arena, p: u64, off: usize, bytes: []const u8) void {
        @memcpy(a.g.host(p + off, bytes.len), bytes);
    }
};

/// A bf16 tensor of the given shape, or a named refusal.
pub fn tensor(f: *const st.File, name: []const u8, shape: []const usize) ![]const u8 {
    const t = f.get(name) orelse {
        std.log.err("lfm3_omni: {s} is missing", .{name});
        return error.MissingTensor;
    };
    if (!t.is(.bf16, shape)) {
        std.log.err("lfm3_omni: {s} is {s} {any}, expected bf16 {any}", .{ name, @tagName(t.dtype), t.shape[0..t.rank], shape });
        return error.BadTensor;
    }
    return t.bytes;
}

fn one(a: *Arena, f: *const st.File, name: []const u8, shape: []const usize) !u64 {
    return a.upload(try tensor(f, name, shape));
}

/// Tensors of `k` columns stacked by rows into one matrix.
fn stacked(a: *Arena, f: *const st.File, names: []const []const u8, rows: []const usize, k: usize) !u64 {
    var total: usize = 0;
    for (rows) |r| total += r;
    const p = try a.alloc(total * k * 2);
    var at: usize = 0;
    for (names, rows) |n, r| {
        a.write(p, at * k * 2, try tensor(f, n, &.{ r, k }));
        at += r;
    }
    return p;
}

/// Two [n, k] tensors as one [2n, k] matrix of row pairs (w1[j], w3[j]): SwiGLU's operands side by side.
fn interleaved(a: *Arena, f: *const st.File, n1: []const u8, n3: []const u8, n: usize, k: usize) !u64 {
    const p = try a.alloc(2 * n * k * 2);
    const t1 = try tensor(f, n1, &.{ n, k });
    const t3 = try tensor(f, n3, &.{ n, k });
    for (0..n) |j| {
        a.write(p, (2 * j) * k * 2, t1[j * k * 2 ..][0 .. k * 2]);
        a.write(p, (2 * j + 1) * k * 2, t3[j * k * 2 ..][0 .. k * 2]);
    }
    return p;
}

fn invFreq(gpa: std.mem.Allocator, a: *Arena, r: Rope, hd: usize) !u64 {
    const v = try gpa.alloc(f32, hd / 2);
    defer gpa.free(v);
    spec.invFreq(v, r, hd);
    return a.upload(std.mem.sliceAsBytes(v));
}

/// Loads a stack's weights from `f` and allocates its caches and scratch in `a`.
pub fn load(gpa: std.mem.Allocator, a: *Arena, f: *const st.File, sp: Spec) !Stack {
    const D = sp.d;
    const hd = sp.head_dim;
    var x: Stack = undefined;
    x.s = sp;
    x.g = a.g;
    x.pos = 0;
    x.layers = try gpa.alloc(Layer, sp.kinds.len);
    errdefer gpa.free(x.layers);
    var nbuf: [5][256]u8 = undefined;
    const P = sp.prefix;
    var conv_layers: usize = 0;
    for (sp.kinds) |kd| conv_layers += @intFromBool(kd == .conv);
    x.states_bytes = conv_layers * D * (sp.conv_k - 1) * 2;
    x.states = if (x.states_bytes > 0) try a.zeroed(x.states_bytes) else 0;
    var conv_i: usize = 0;
    for (sp.kinds, 0..) |kd, li| {
        var l: Layer = .{ .kind = kd, .op_norm = 0, .ffn_norm = 0 };
        const n0 = try std.fmt.bufPrint(&nbuf[0], "{s}layers.{d}.", .{ P, li });
        l.op_norm = try one(a, f, try cat(&nbuf[1], n0, "operator_norm.weight"), &.{D});
        l.ffn_norm = try one(a, f, try cat(&nbuf[1], n0, "ffn_norm.weight"), &.{D});
        if (sp.sandwich) {
            l.post_op_norm = try one(a, f, try cat(&nbuf[1], n0, "post_operator_norm.weight"), &.{D});
            l.post_ffn_norm = try one(a, f, try cat(&nbuf[1], n0, "post_ffn_norm.weight"), &.{D});
        }
        switch (kd) {
            .conv => {
                l.in_proj = try one(a, f, try cat(&nbuf[1], n0, "conv.in_proj.weight"), &.{ 3 * D, D });
                l.conv_w = try one(a, f, try cat(&nbuf[1], n0, "conv.conv.weight"), &.{ D, 1, sp.conv_k });
                l.out_proj = try one(a, f, try cat(&nbuf[1], n0, "conv.out_proj.weight"), &.{ D, D });
                l.state = x.states + conv_i * D * (sp.conv_k - 1) * 2;
                conv_i += 1;
            },
            .full, .sliding => {
                const nq_ = sp.heads * hd;
                const nk_ = sp.kv_heads * hd;
                const q = try cat(&nbuf[1], n0, "self_attn.q_proj.weight");
                const kk = try cat(&nbuf[2], n0, "self_attn.k_proj.weight");
                const v = try cat(&nbuf[3], n0, "self_attn.v_proj.weight");
                if (sp.lfm3) {
                    const g = try cat(&nbuf[4], n0, "self_attn.gate_proj.weight");
                    l.qkvg = try stacked(a, f, &.{ q, kk, v, g }, &.{ nq_, nk_, nk_, sp.heads }, D);
                    l.q_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.q_norm.weight"), &.{hd});
                    l.k_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.k_norm.weight"), &.{hd});
                    l.o_proj = try one(a, f, try cat(&nbuf[1], n0, "self_attn.o_proj.weight"), &.{ D, nq_ });
                } else {
                    l.qkvg = try stacked(a, f, &.{ q, kk, v }, &.{ nq_, nk_, nk_ }, D);
                    l.q_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.q_layernorm.weight"), &.{hd});
                    l.k_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.k_layernorm.weight"), &.{hd});
                    l.o_proj = try one(a, f, try cat(&nbuf[1], n0, "self_attn.out_proj.weight"), &.{ D, nq_ });
                }
                l.kc = try a.alloc(sp.max_ctx * nk_ * 2);
                l.vc = try a.alloc(sp.max_ctx * nk_ * 2);
            },
        }
        const w1 = try cat(&nbuf[1], n0, "feed_forward.w1.weight");
        const w3 = try cat(&nbuf[2], n0, "feed_forward.w3.weight");
        l.w13 = try interleaved(a, f, w1, w3, sp.ffn, D);
        l.w2 = try one(a, f, try cat(&nbuf[1], n0, "feed_forward.w2.weight"), &.{ D, sp.ffn });
        x.layers[li] = l;
    }
    x.final_norm = try one(a, f, try cat(&nbuf[1], P, "embedding_norm.weight"), &.{D});
    x.inv_full = try invFreq(gpa, a, sp.rope_full, hd);
    x.inv_sliding = try invFreq(gpa, a, sp.rope_sliding, hd);
    const R = sp.max_rows;
    const cols = @max(3 * D, @max(sp.heads * hd + 2 * sp.kv_heads * hd + sp.heads, 2 * sp.ffn));
    x.xn = try a.alloc(R * D * 2);
    x.proj = try a.alloc(R * cols * 2);
    x.y = try a.alloc(R * @max(D, @max(sp.ffn, sp.heads * hd)) * 2);
    x.tmp = try a.alloc(R * D * 2);
    return x;
}

fn cat(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}
