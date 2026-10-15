//! Ling-3.1-flash's TP=1 layer graph on CUDA: the kernels it launches, the launches its blocks share, derived weights.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const weights = @import("weights.zig");

pub const glue = glm.glue;
pub const kda = glm.kda;
pub const latent = glm.latent;
const Proj = weights.Proj;
const Config = @import("config.zig").Config;

/// Every module the graph launches from: GLM's glue, KDA and ops, the lane formats, the grouped experts, our glue.
pub const Modules = struct {
    mods: [5]cuda.Module, // glm_glue, glm_kda, glm_ops, experts (the plan kernels), ling_glue
    glue: glue.Kernels,
    kda: kda.Kernels,
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    x8: cuda.experts.Fp8Experts,
    x4: cuda.experts.Nvfp4Experts,
    router: cuda.grouped.Router,
    copy_rows: cuda.Function,
    add_i32: cuda.Function,
    add: cuda.Function,
    gate: cuda.Function,
    combine: cuda.Function,
    swiglu: [2]cuda.Function, // glm_glue's 512 and 64 SwiGLU in Ling's order, min(silu(g), L) * clip(u, -L, L)

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        const k = cuda.kernels;
        var m: Modules = undefined;
        var n: usize = 0;
        errdefer for (m.mods[0..n]) |*x| x.unload();
        for (&m.mods, [_][]const u8{ k.glm_glue, k.glm_kda, k.glm_ops, k.experts, k.ling_glue }) |*x, image| {
            x.* = try cuda.Module.load(d, image);
            n += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        m.glue = try glue.Kernels.load(m.mods[0]);
        m.kda = try kda.Kernels.load(m.mods[1]);
        m.copy_rows = try m.mods[2].function("tf_glm_copy_rows");
        m.add_i32 = try m.mods[2].function("tf_glm_add_i32");
        m.router = try cuda.grouped.Router.resolve(m.mods[3]);
        m.add = try m.mods[4].function("tf_ling_add");
        m.gate = try m.mods[4].function("tf_ling_gate");
        m.combine = try m.mods[4].function("tf_ling_combine");
        m.swiglu = .{ try m.mods[0].function("tf_ling_swiglu_512"), try m.mods[0].function("tf_ling_swiglu_64") };
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer m.prompt.unload();
        m.x8 = try cuda.experts.Fp8Experts.load(d, sms);
        errdefer m.x8.unload();
        m.x4 = try cuda.experts.Nvfp4Experts.load(d, sms);
        return m;
    }

    pub fn deinit(m: *Modules) void {
        for (&m.mods) |*x| x.unload();
        m.lane.unload();
        m.prompt.unload();
        m.x8.unload();
        m.x4.unload();
        m.* = undefined;
    }

    pub fn set(m: *const Modules) glue.Set {
        return .{ .glue = &m.glue };
    }

    /// Every lane format's decode rows; nvfp4 prompt rows on the prompt GEMM.
    pub fn linear(m: *const Modules) cuda.qlinear.Linear {
        return .{ .lane = &m.lane, .nvfp4_prompt = &m.prompt };
    }

    pub fn grouped(m: *const Modules) cuda.experts.Grouped {
        return .{ .fp8g = &m.x8, .nvfp4 = &m.x4 };
    }
};

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// One forward's launches: the modules, the stream, prompt or decode kernels, the lane's split-K partials.
pub const Ops = struct {
    m: *const Modules,
    s: cuda.Stream,
    prompt: bool,
    part: u64,
    kda_wide: bool = true, // decode windows on KDA's three-kernel chain (bit-equal to the fused one, faster)

    pub fn set(o: Ops) glue.Set {
        return o.m.set();
    }

    /// out [rows][n] bf16 = x rows (`ldx` apart) through `w` in whatever format it loaded.
    pub fn mm(o: Ops, w: Proj, x: u64, ldx: usize, rows: usize, out: u64) !void {
        const in: cuda.qlinear.Rows = .{ .x = x, .ldx = ldx, .part = o.part };
        if (o.prompt) return o.m.linear().prompt(o.s, w, in, out, rows);
        return o.m.linear().decode(o.s, w, in, out, rows);
    }

    /// Bytes [rows][row_bytes] from rows `src_pitch` apart into rows `dst_pitch` apart, as they are.
    pub fn copy(o: Ops, src: u64, src_pitch: usize, dst: u64, dst_pitch: usize, rows: usize, row_bytes: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ src, @as(c_longlong, @intCast(src_pitch)), dst, @as(c_longlong, @intCast(dst_pitch)), @as(c_int, @intCast(rows)), @as(c_int, @intCast(row_bytes)) }) |v| a.add(v);
        try cuda.launch.launch(o.m.copy_rows, .{ .grid = .{ .x = cdiv(rows * row_bytes, 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// A projection into columns col0 .. col0 + n of rows `stride` wide (bf16), through `tmp`.
    pub fn mmCols(o: Ops, w: Proj, x: u64, ldx: usize, rows: usize, tmp: u64, dst: u64, stride: usize, col0: usize) !void {
        const n = w.outputs();
        try o.mm(w, x, ldx, rows, tmp);
        try o.copy(tmp, n * 2, dst + col0 * 2, stride * 2, rows, n * 2);
    }

    /// out = bf16(x + y), n bf16 elements; out may alias either.
    pub fn add(o: Ops, x: u64, y: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ x, y, out, @as(i64, @intCast(n)) }) |v| a.add(v);
        try cuda.launch.launch(o.m.add, .{ .grid = .{ .x = cdiv(n, 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// The head-wise gate: v [rows][heads][dv] times bf16 sigmoid of g [rows][g_stride] column h.
    pub fn gate(o: Ops, v: u64, g: u64, g_stride: usize, rows: usize, heads: usize, dv: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ v, @as(i64, @intCast(heads)), g, @as(i64, @intCast(g_stride)) }) |x| a.add(x);
        try cuda.launch.launch(o.m.gate, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = @intCast(dv) } }, o.s, &a);
    }

    /// The device i32 at `pos` plus `r` into `out`.
    pub fn addI32(o: Ops, pos: u64, r: usize, out: u64) !void {
        var a: cuda.Args = .{};
        inline for (.{ pos, @as(c_int, @intCast(r)), out }) |v| a.add(v);
        try cuda.launch.launch(o.m.add_i32, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, o.s, &a);
    }
};

/// An MLA layer's kv_b as the absorbed path reads it, bf16: keys wk [heads][nope][lw] and values wv [heads][v][lw].
pub const Absorb = struct { wk: u64, wv: u64 };

/// What the graph derives from the loaded weights: absorbed kv_b, each layer's slot, the rope table, a zero bias.
pub const Graph = struct {
    w: *const weights.Weights,
    store: weights.Store,
    slot: []u32, // loaded layer j's index among the loaded KDA layers, or among the MLA layers
    absorb: []Absorb, // by MLA slot
    kda_layers: usize,
    mla_layers: usize,
    freq: u64, // fp32 [32]: the rope's inv_freq
    zero_bias: u64, // fp32 [experts]: select's bias without moe_router_enable_expert_bias

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *const Modules, s: cuda.Stream, w: *const weights.Weights) !Graph {
        const c = &w.config;
        var g: Graph = .{ .w = w, .store = .{ .gpa = gpa, .d = d }, .slot = try gpa.alloc(u32, w.layers.len), .absorb = &.{}, .kda_layers = 0, .mla_layers = 0, .freq = 0, .zero_bias = 0 };
        errdefer g.deinit();
        for (w.layers, g.slot) |l, *x| switch (l.mixer) {
            .kda => {
                x.* = @intCast(g.kda_layers);
                g.kda_layers += 1;
            },
            .mla => {
                x.* = @intCast(g.mla_layers);
                g.mla_layers += 1;
            },
        };
        if (c.rope_dim != latent.rope_width or c.rope_style != .gptj) return error.RopeNotBuilt;
        const f = latent.ropeFreqs(c.rope_theta);
        g.freq = try g.store.keep(std.mem.sliceAsBytes(&f), "ling.rope.inv_freq");
        const zero = try gpa.alloc(u8, @as(usize, c.experts) * 4);
        defer gpa.free(zero);
        @memset(zero, 0);
        g.zero_bias = try g.store.keep(zero, "ling.zero_bias");
        g.absorb = try gpa.alloc(Absorb, g.mla_layers);
        for (w.layers, g.slot) |l, x| if (l.mixer == .mla) {
            g.absorb[x] = try absorbOf(gpa, &g.store, m, s, c, l.mixer.mla.kv_b, l.index);
        };
        return g;
    }

    pub fn deinit(g: *Graph) void {
        const gpa = g.store.gpa;
        gpa.free(g.slot);
        gpa.free(g.absorb);
        g.store.deinit();
        g.* = undefined;
    }
};

/// kv_b in its loaded format through the lane on identity rows: W^T [lw][heads (nope + v)], regrouped by head.
fn absorbOf(gpa: std.mem.Allocator, store: *weights.Store, m: *const Modules, s: cuda.Stream, c: *const Config, kv_b: Proj, index: u32) !Absorb {
    const lw: usize = c.kv_lora;
    const h: usize = c.mla_heads;
    const nope: usize = c.nope;
    const dv: usize = c.v_dim;
    const per = nope + dv;
    const n = h * per;
    if (kv_b.outputs() != n or kv_b.inputs() != lw) return error.UnexpectedTensor;
    const eye = try gpa.alloc(u16, lw * lw);
    defer gpa.free(eye);
    @memset(eye, 0);
    for (0..lw) |i| eye[i * lw + i] = 0x3f80; // bf16 1.0
    var x = try cuda.DeviceBuffer.fromHost(store.d, std.mem.sliceAsBytes(eye));
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(store.d, lw * n * 2);
    defer y.free();
    var part = try cuda.DeviceBuffer.alloc(store.d, @max(16, m.lane.partBytes(lw, switch (kv_b) {
        .fp8g, .nvfp4, .fp8, .bf16 => |q| q,
        else => return error.FormatNotBuilt,
    })));
    defer part.free();
    try m.linear().decode(s, kv_b, .{ .x = x.ptr, .ldx = lw, .part = part.ptr }, y.ptr, lw);
    try s.synchronize();
    const t = try gpa.alloc(u16, lw * n);
    defer gpa.free(t);
    try y.download(0, std.mem.sliceAsBytes(t));
    const wk = try gpa.alloc(u16, h * nope * lw);
    defer gpa.free(wk);
    const wv = try gpa.alloc(u16, h * dv * lw);
    defer gpa.free(wv);
    for (0..h) |hd| for (0..per) |j| for (0..lw) |col| {
        const v = t[col * n + hd * per + j];
        if (j < nope) wk[(hd * nope + j) * lw + col] = v else wv[(hd * dv + j - nope) * lw + col] = v;
    };
    var b: [64]u8 = undefined;
    return .{
        .wk = try store.keep(std.mem.sliceAsBytes(wk), try std.fmt.bufPrint(&b, "ling.{d}.absorb.wk", .{index})),
        .wv = try store.keep(std.mem.sliceAsBytes(wv), try std.fmt.bufPrint(&b, "ling.{d}.absorb.wv", .{index})),
    };
}
