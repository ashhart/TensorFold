//! MOSS-Image's prompt encoder on CUDA: the chat-formatted prompt through Qwen3.5's text model to its final-norm rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("encoder_config.zig");
const host = @import("host.zig");
const ks = @import("cuda_encoder_kernels.zig");
const family = @import("cuda_kernels.zig");

const H = cfg.hidden;
const F = cfg.ffn;
const D = cfg.head_dim;

/// The pipeline's chat-formatted prompt: one user turn, then the assistant's turn opened with thinking on.
pub fn format(a: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.mem.concat(a, u8, &.{ "<|im_start|>user\n", prompt, "<|im_end|>\n<|im_start|>assistant\n<think>\n" });
}

/// The prompt's ids by `tok` (the checkpoint's tokenizer.json) as the pipeline makes them, cut at 4,096; caller frees.
pub fn ids(tok: anytype, a: std.mem.Allocator, prompt: []const u8) ![]u32 {
    const text = try format(a, prompt);
    defer a.free(text);
    const all = try tok.encode(a, text);
    if (all.len <= cfg.max_rows) return all;
    defer a.free(all);
    return a.dupe(u32, all[0..cfg.max_rows]);
}

/// The rotary tables [rows, 32] as the text model computes them: fp32 angles, their cos and sin, rounded to bf16.
pub fn rotary(cos: []f32, sin: []f32, rows: usize) void {
    const half = cfg.rotary / 2;
    for (0..half) |i| {
        const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, cfg.rotary);
        const freq: f32 = 1.0 / std.math.pow(f32, @floatCast(cfg.theta), exponent);
        for (0..rows) |r| {
            const angle: f32 = @as(f32, @floatFromInt(r)) * freq;
            cos[r * half + i] = host.round(@as(f32, @floatCast(@cos(@as(f64, angle)))));
            sin[r * half + i] = host.round(@as(f32, @floatCast(@sin(@as(f64, angle)))));
        }
    }
}

/// A Gated DeltaNet layer's mixer on the device: the q k v projection and its conv, z, b and a, then norm and out.
pub const Delta = struct { qkv: u64, z: u64, b: u64, a: u64, conv: u64, a_log: u64, dt_bias: u64, norm: u64, out: u64 };

/// A full-attention layer's mixer: q (with its gate), k, v, o and the head norms.
pub const Attention = struct { q: u64, k: u64, v: u64, o: u64, q_norm: u64, k_norm: u64 };

pub const Layer = struct {
    norm1: u64,
    norm2: u64,
    gate: u64,
    up: u64,
    down: u64,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

/// The checkpoint's language-model tensors: each on the device as stored, the embedding table mapped on the host.
pub const Weights = struct {
    gpa: std.mem.Allocator,
    ck: core.Checkpoint,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    layers: [cfg.layers]Layer = undefined,
    /// The embedding table [vocab, hidden] on the host (mapped from the checkpoint).
    embed: []const u16 = &.{},
    norm: u64 = 0,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.ck.close();
    }

    /// The bf16 tensor `name` of exactly `shape`, copied onto the device.
    fn take(w: *Weights, d: *const cuda.Driver, name: []const u8, shape: []const usize) !u64 {
        const t = try w.ck.get(name);
        if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("moss image encoder: {s} is {t} {any}; the encoder reads bf16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        const b = try cuda.DeviceBuffer.fromHost(d, t.bytes);
        errdefer {
            var owned = b;
            owned.free();
        }
        try w.buffers.append(w.gpa, b);
        w.bytes += b.len;
        return b.ptr;
    }
};

const prefix = "language_model.";

/// The text_encoder folder `dir` on the first GPU: its config checked, then every language-model tensor.
fn loadWeights(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = .{ .gpa = gpa, .ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &.{prefix}) };
    errdefer w.deinit();
    const h = cfg.hidden;
    const vd = cfg.v_heads * cfg.v_dim;
    var name: [96]u8 = undefined;
    for (&w.layers, 0..) |*l, i| {
        const at = struct {
            fn n(buf: []u8, layer: usize, field: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, prefix ++ "layers.{d}.{s}", .{ layer, field });
            }
        };
        l.norm1 = try w.take(d, try at.n(&name, i, "input_layernorm.weight"), &.{h});
        l.norm2 = try w.take(d, try at.n(&name, i, "post_attention_layernorm.weight"), &.{h});
        l.gate = try w.take(d, try at.n(&name, i, "mlp.gate_proj.weight"), &.{ cfg.ffn, h });
        l.up = try w.take(d, try at.n(&name, i, "mlp.up_proj.weight"), &.{ cfg.ffn, h });
        l.down = try w.take(d, try at.n(&name, i, "mlp.down_proj.weight"), &.{ h, cfg.ffn });
        if (cfg.isFull(i)) {
            const d_ = cfg.head_dim;
            l.mixer = .{ .attention = .{
                .q = try w.take(d, try at.n(&name, i, "self_attn.q_proj.weight"), &.{ 2 * cfg.heads * d_, h }),
                .k = try w.take(d, try at.n(&name, i, "self_attn.k_proj.weight"), &.{ cfg.kv_heads * d_, h }),
                .v = try w.take(d, try at.n(&name, i, "self_attn.v_proj.weight"), &.{ cfg.kv_heads * d_, h }),
                .o = try w.take(d, try at.n(&name, i, "self_attn.o_proj.weight"), &.{ h, cfg.heads * d_ }),
                .q_norm = try w.take(d, try at.n(&name, i, "self_attn.q_norm.weight"), &.{d_}),
                .k_norm = try w.take(d, try at.n(&name, i, "self_attn.k_norm.weight"), &.{d_}),
            } };
        } else {
            l.mixer = .{ .delta = .{
                .qkv = try w.take(d, try at.n(&name, i, "linear_attn.in_proj_qkv.weight"), &.{ cfg.mix, h }),
                .z = try w.take(d, try at.n(&name, i, "linear_attn.in_proj_z.weight"), &.{ vd, h }),
                .b = try w.take(d, try at.n(&name, i, "linear_attn.in_proj_b.weight"), &.{ cfg.v_heads, h }),
                .a = try w.take(d, try at.n(&name, i, "linear_attn.in_proj_a.weight"), &.{ cfg.v_heads, h }),
                .conv = try w.take(d, try at.n(&name, i, "linear_attn.conv1d.weight"), &.{ cfg.mix, 1, cfg.conv_taps }),
                .a_log = try w.take(d, try at.n(&name, i, "linear_attn.A_log"), &.{cfg.v_heads}),
                .dt_bias = try w.take(d, try at.n(&name, i, "linear_attn.dt_bias"), &.{cfg.v_heads}),
                .norm = try w.take(d, try at.n(&name, i, "linear_attn.norm.weight"), &.{cfg.v_dim}),
                .out = try w.take(d, try at.n(&name, i, "linear_attn.out_proj.weight"), &.{ h, vd }),
            } };
        }
    }
    const embed = try w.ck.get(prefix ++ "embed_tokens.weight");
    if (embed.dtype != .bf16 or embed.rank != 2 or embed.shape[0] != cfg.vocab or embed.shape[1] != h) return error.UnexpectedTensor;
    w.embed = @alignCast(std.mem.bytesAsSlice(u16, embed.bytes));
    w.norm = try w.take(d, prefix ++ "norm.weight", &.{h});
    const full = cfg.layers / cfg.full_every;
    const read = 5 * cfg.layers + 6 * full + 9 * (cfg.layers - full) + 2;
    if (w.buffers.items.len + 1 != read or w.ck.unused() != 0) {
        std.log.err("moss image encoder: the checkpoint has {d} language-model tensors; the encoder reads {d}", .{ w.buffers.items.len + 1 + w.ck.unused(), read });
        return error.UnexpectedTensors;
    }
    return w;
}

/// A prompt of `T` rows' scratch: Metal's parts, each 256-aligned, in its order.
pub const Plan = struct {
    pub const Part = enum { x, xn, y, mix, conv, z, a, b, qn, kn, g, beta, core, mixed, kp, vp, q, k, v, att, gate, up, act, cos, sin, ones, out };
    const n = std.enums.values(Part).len;

    offs: [n]usize,
    total: usize,

    fn size(p: Part, T: usize) usize {
        return switch (p) {
            .x, .xn, .y, .z, .core, .mixed, .q, .att, .out => T * H * 2,
            .mix, .conv => T * cfg.mix * 2,
            .a, .b => T * cfg.v_heads * 2,
            .qn, .kn => T * cfg.k_heads * cfg.k_dim * 4,
            .g, .beta => T * cfg.v_heads * 4,
            .kp, .vp, .k, .v => T * cfg.kv_heads * D * 2,
            .gate, .up, .act => T * F * 2,
            .cos, .sin => T * (cfg.rotary / 2) * 4,
            .ones => H * 2,
        };
    }

    pub fn of(T: usize) !Plan {
        if (T == 0 or T > cfg.max_rows) return error.PromptRows;
        var p: Plan = .{ .offs = undefined, .total = 0 };
        for (std.enums.values(Part), 0..) |part, i| {
            p.offs[i] = p.total;
            p.total += std.mem.alignForward(usize, size(part, T), 256);
        }
        return p;
    }

    pub fn bytes(p: Plan) usize {
        return p.total;
    }
};

/// One prompt's scratch seen as the buffers a layer's steps take.
pub const Views = struct {
    base: u64,
    offs: [std.enums.values(Plan.Part).len]usize,

    pub fn of(base: u64, plan: Plan) Views {
        return .{ .base = base, .offs = plan.offs };
    }

    pub fn at(v: Views, part: Plan.Part) u64 {
        return v.base + v.offs[@backingInt(part)];
    }

    /// The offset of a part inside the scratch buffer, for a copy.
    pub fn off(v: Views, part: Plan.Part) usize {
        return v.offs[@backingInt(part)];
    }
};

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: ks.Kernels,
    fk: family.Kernels,
    gemms: cuda.vit.Gemms,
    w: *Weights,

    /// The checkpoint folder `dir`: its text_encoder/, on the first GPU (sm_120 or sm_121 only).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Encoder {
        const t = try gpa.create(Encoder);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        t.driver = try cuda.Driver.open();
        errdefer t.driver.close();
        t.ctx = try cuda.Context.init(&t.driver, 0);
        errdefer t.ctx.deinit();
        const sm = try t.ctx.capability();
        if (sm != 120 and sm != 121) {
            std.log.err("moss image encoder: sm_{d} is not a GPU this family was checked on (sm_120, sm_121)", .{sm});
            return error.UncheckedGpu;
        }
        t.stream = try cuda.Stream.init(&t.driver, false);
        errdefer t.stream.deinit();
        t.k = try ks.Kernels.load(&t.driver);
        errdefer t.k.unload();
        t.fk = try family.Kernels.load(&t.driver);
        errdefer t.fk.unload();
        t.gemms = try cuda.vit.Gemms.init(gpa, &t.driver);
        errdefer t.gemms.deinit();
        const text = try std.fs.path.join(gpa, &.{ dir, "text_encoder" });
        defer gpa.free(text);
        t.w = try loadWeights(gpa, io, &t.driver, text);
        errdefer {
            t.w.deinit();
            gpa.destroy(t.w);
        }
        return t;
    }

    pub fn deinit(t: *Encoder) void {
        t.w.deinit();
        t.gpa.destroy(t.w);
        t.gemms.deinit();
        t.fk.unload();
        t.k.unload();
        t.stream.deinit();
        t.ctx.deinit();
        t.driver.close();
        t.gpa.destroy(t);
    }

    /// The GPU this encoder's context is on, for a tool's receipt.
    pub fn device(t: *const Encoder, buf: []u8) []const u8 {
        return t.ctx.name(buf) catch "a CUDA GPU";
    }

    /// The final-norm rows [tokens, 4096] bf16 into `out`; with `kept`, each layer's rows [layers, tokens, 4096].
    pub fn rows(t: *Encoder, tokens: []const u32, out: []u16, kept: ?[]u16) !void {
        const T = tokens.len;
        if (T == 0 or T > cfg.max_rows) return error.PromptRows;
        if (out.len != T * H) return error.PromptRows;
        if (kept) |k| if (k.len != cfg.layers * T * H) return error.PromptRows;
        for (tokens) |id| if (id >= cfg.vocab) return error.TokenOutOfRange;
        var run = try t.begin(T);
        defer run.buf.free();
        const gathered = try t.gpa.alloc(u16, T * H);
        defer t.gpa.free(gathered);
        for (tokens, 0..) |id, r| @memcpy(gathered[r * H ..][0..H], t.w.embed[id * H ..][0..H]);
        try run.buf.upload(run.v.off(.x), std.mem.sliceAsBytes(gathered));
        for (t.w.layers, 0..) |l, i| {
            try t.layer(&run, l, T);
            if (kept) |k| try run.take(t, .x, k[i * T * H ..][0 .. T * H]);
        }
        try ks.norm(&t.k, t.stream, run.v.at(.x), t.w.norm, run.v.at(.out), T, H, H, H, cfg.eps);
        try run.take(t, .out, out);
    }

    /// For checks, layer i on block i of `inputs` [layers + 1, rows, 4096], the final norm on the last.
    pub fn steps(t: *Encoder, T: usize, inputs: []const u16, out: []u16) !void {
        const n = (cfg.layers + 1) * T * H;
        if (T == 0 or T > cfg.max_rows) return error.PromptRows;
        if (inputs.len != n or out.len != n) return error.PromptRows;
        var run = try t.begin(T);
        defer run.buf.free();
        for (0..cfg.layers + 1) |i| {
            try run.buf.upload(run.v.off(.x), std.mem.sliceAsBytes(inputs[i * T * H ..][0 .. T * H]));
            const last = i == cfg.layers;
            if (last) {
                try ks.norm(&t.k, t.stream, run.v.at(.x), t.w.norm, run.v.at(.out), T, H, H, H, cfg.eps);
            } else try t.layer(&run, t.w.layers[i], T);
            try run.take(t, if (last) .out else .x, out[i * T * H ..][0 .. T * H]);
        }
    }

    fn begin(t: *Encoder, T: usize) !Run {
        const plan = try Plan.of(T);
        var buf = try cuda.DeviceBuffer.alloc(&t.driver, plan.bytes());
        errdefer buf.free();
        const v = Views.of(buf.ptr, plan);
        const tables = try t.gpa.alloc(f32, 2 * T * (cfg.rotary / 2));
        defer t.gpa.free(tables);
        const half = T * (cfg.rotary / 2);
        rotary(tables[0..half], tables[half..], T);
        try buf.upload(v.off(.cos), std.mem.sliceAsBytes(tables[0..half]));
        try buf.upload(v.off(.sin), std.mem.sliceAsBytes(tables[half..]));
        var ones: [H]u16 = undefined;
        @memset(ones[0..], 0x3f80);
        try buf.upload(v.off(.ones), std.mem.sliceAsBytes(&ones));
        return .{ .buf = buf, .v = v };
    }

    /// One layer's rows in place on `.x`: both norms, the mixer, then the MLP, each residual added gated by ones.
    fn layer(t: *Encoder, run: *Run, l: Layer, T: usize) !void {
        const v = run.v;
        try ks.norm(&t.k, t.stream, v.at(.x), l.norm1, v.at(.xn), T, H, H, H, cfg.eps);
        switch (l.mixer) {
            .delta => |m| try t.delta(v, m, T),
            .attention => |m| try t.attention(v, m, T),
        }
        try family.gateAdd(&t.fk, t.stream, v.at(.x), v.at(.y), v.at(.ones), T, H);
        try ks.norm(&t.k, t.stream, v.at(.x), l.norm2, v.at(.xn), T, H, H, H, cfg.eps);
        try t.linear(v.at(.xn), l.gate, v.at(.gate), T, F, H);
        try t.linear(v.at(.xn), l.up, v.at(.up), T, F, H);
        try ks.siluMul(&t.k, t.stream, v.at(.gate), v.at(.up), v.at(.act), T, F);
        try t.linear(v.at(.act), l.down, v.at(.y), T, H, F);
        try family.gateAdd(&t.fk, t.stream, v.at(.x), v.at(.y), v.at(.ones), T, H);
    }

    fn delta(t: *Encoder, v: Views, m: Delta, T: usize) !void {
        try t.linear(v.at(.xn), m.qkv, v.at(.mix), T, cfg.mix, H);
        try t.linear(v.at(.xn), m.z, v.at(.z), T, H, H);
        try t.linear(v.at(.xn), m.b, v.at(.b), T, cfg.v_heads, H);
        try t.linear(v.at(.xn), m.a, v.at(.a), T, cfg.v_heads, H);
        try ks.conv(&t.k, t.stream, v.at(.mix), m.conv, v.at(.conv), T, cfg.mix);
        const g: ks.Gdn = .{ .rows = T, .kheads = cfg.k_heads, .vheads = cfg.v_heads, .mix_stride = cfg.mix, .scale = 1.0 / @sqrt(@as(f32, cfg.k_dim)) };
        try ks.gdnPrep(&t.k, t.stream, v.at(.conv), v.at(.a), v.at(.b), m.a_log, m.dt_bias, v.at(.qn), v.at(.kn), v.at(.g), v.at(.beta), g);
        try ks.gdn(&t.k, t.stream, v.at(.qn), v.at(.kn), v.at(.conv), v.at(.g), v.at(.beta), v.at(.core), g);
        try ks.gatedNorm(&t.k, t.stream, v.at(.core), v.at(.z), m.norm, v.at(.mixed), T, H, cfg.eps);
        try t.linear(v.at(.mixed), m.out, v.at(.y), T, H, H);
    }

    fn attention(t: *Encoder, v: Views, m: Attention, T: usize) !void {
        const kv = cfg.kv_heads * D;
        try t.linear(v.at(.xn), m.q, v.at(.mix), T, 2 * cfg.heads * D, H);
        try t.linear(v.at(.xn), m.k, v.at(.kp), T, kv, H);
        try t.linear(v.at(.xn), m.v, v.at(.vp), T, kv, H);
        try ks.head(&t.k, t.stream, v.at(.mix), m.q_norm, v.at(.cos), v.at(.sin), v.at(.q), T, cfg.heads, 2 * cfg.heads * D, 2 * D, cfg.eps, true);
        try ks.head(&t.k, t.stream, v.at(.kp), m.k_norm, v.at(.cos), v.at(.sin), v.at(.k), T, cfg.kv_heads, kv, D, cfg.eps, true);
        try ks.head(&t.k, t.stream, v.at(.vp), m.k_norm, v.at(.cos), v.at(.sin), v.at(.v), T, cfg.kv_heads, kv, D, cfg.eps, false);
        try ks.attention(&t.k, t.stream, v.at(.q), v.at(.k), v.at(.v), v.at(.att), T, cfg.heads, cfg.kv_heads, 1.0 / @sqrt(@as(f32, D)));
        try ks.gate(&t.k, t.stream, v.at(.att), v.at(.mix), v.at(.mixed), T, H);
        try t.linear(v.at(.mixed), m.o, v.at(.y), T, H, H);
    }

    /// y [m, n] bf16 = x [m, k] . w^T, fp32 sums rounded once.
    fn linear(t: *Encoder, x: u64, w: u64, y: u64, m: usize, n: usize, k: usize) !void {
        try t.gemms.run(t.stream, x, w, y, m, n, k, .bf16);
    }
};

/// One call's scratch and its views.
const Run = struct {
    buf: cuda.DeviceBuffer,
    v: Views,

    /// Downloads a part's rows into `out`, the GPU's steps before it complete.
    fn take(run: *Run, t: *Encoder, part: Plan.Part, out: []u16) !void {
        try t.stream.synchronize();
        try run.buf.download(run.v.off(part), std.mem.sliceAsBytes(out));
    }
};

test "the pipeline's chat format, and the rotary tables' bf16 bits" {
    const a = std.testing.allocator;
    const text = try format(a, "A red fox");
    defer a.free(text);
    try std.testing.expectEqualStrings("<|im_start|>user\nA red fox<|im_end|>\n<|im_start|>assistant\n<think>\n", text);
    var cos: [3 * 32]f32 = undefined;
    var sin: [3 * 32]f32 = undefined;
    rotary(&cos, &sin, 3);
    for (cos, sin) |c, s| {
        try std.testing.expectEqual(c, host.round(c));
        try std.testing.expectEqual(s, host.round(s));
    }
    try std.testing.expectEqual(@as(f32, 1), @as(f32, cos[0]));
    try std.testing.expectEqual(@as(f32, 0), @as(f32, sin[0]));
    // row r's first angle is r: cos(r) and sin(r), rounded to bf16
    for (0..3) |r| {
        try std.testing.expectEqual(host.round(@floatCast(@cos(@as(f64, @floatFromInt(r))))), cos[r * 32]);
        try std.testing.expectEqual(host.round(@floatCast(@sin(@as(f64, @floatFromInt(r))))), sin[r * 32]);
    }
}

test "scratch: Metal's parts, each 256-aligned, and the rows it refuses" {
    const p = try Plan.of(33);
    try std.testing.expectEqual(@as(usize, 33 * H * 2), Plan.size(.x, 33));
    try std.testing.expectEqual(@as(usize, 33 * cfg.mix * 2), Plan.size(.mix, 33));
    try std.testing.expectEqual(@as(usize, 33 * cfg.k_heads * cfg.k_dim * 4), Plan.size(.qn, 33));
    try std.testing.expectEqual(@as(usize, 33 * 32 * 4), Plan.size(.cos, 33));
    try std.testing.expectEqual(@as(usize, H * 2), Plan.size(.ones, 33));
    var at: usize = 0;
    for (std.enums.values(Plan.Part), 0..) |part, i| {
        try std.testing.expectEqual(at, p.offs[i]);
        try std.testing.expectEqual(@as(usize, 0), p.offs[i] % 256);
        at += std.mem.alignForward(usize, Plan.size(part, 33), 256);
    }
    try std.testing.expectEqual(p.bytes(), at);
    const wide = try Plan.of(cfg.max_rows);
    try std.testing.expect(wide.bytes() > p.bytes());
    try std.testing.expectError(error.PromptRows, Plan.of(0));
    try std.testing.expectError(error.PromptRows, Plan.of(cfg.max_rows + 1));
    const v = Views.of(1 << 20, p);
    try std.testing.expectEqual(@as(u64, (1 << 20) + p.offs[0]), v.at(.x));
    try std.testing.expectEqual(@as(u64, (1 << 20) + p.offs[p.offs.len - 1]), v.at(.out));
}

test "ids: the chat format out of the tokenizer, cut at the pipeline's longest prompt" {
    const a = std.testing.allocator;
    const Fake = struct {
        seen: std.ArrayList(u8) = .empty,
        n: usize,
        fn encode(t: *@This(), al: std.mem.Allocator, text: []const u8) ![]u32 {
            try t.seen.appendSlice(al, text);
            const out = try al.alloc(u32, t.n);
            for (out, 0..) |*v, i| v.* = @intCast(i % cfg.vocab);
            return out;
        }
    };
    var short: Fake = .{ .n = 5 };
    defer short.seen.deinit(a);
    const few = try ids(&short, a, "A red fox");
    defer a.free(few);
    try std.testing.expectEqual(@as(usize, 5), few.len);
    try std.testing.expectEqualStrings("<|im_start|>user\nA red fox<|im_end|>\n<|im_start|>assistant\n<think>\n", short.seen.items);
    var long: Fake = .{ .n = cfg.max_rows + 7 };
    defer long.seen.deinit(a);
    const cut = try ids(&long, a, "A red fox");
    defer a.free(cut);
    try std.testing.expectEqual(@as(usize, cfg.max_rows), cut.len);
    try std.testing.expectEqual(@as(u32, cfg.max_rows - 1), cut[cut.len - 1]);
}
