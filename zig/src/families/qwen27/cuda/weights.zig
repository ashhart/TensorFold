//! The 27B's weights on the GPU in the Python engine's layouts: tiled projections, the stored embedding, fp32 gates.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const source = @import("nemotron").source;
const c = @import("shape.zig");
const config_eos = @import("../config.zig").max_eos;
const kern = @import("kernels.zig");

pub const QLinear = kern.QLinear;
const Tensor = core.checkpoint.Tensor;
const prefix = "language_model.";

/// The token table as stored (MLX words, scales, biases): a row lookup, not a matmul.
pub const Embed = struct { w: u64, s: u64, b: u64 };
pub const Delta = struct { qkv: QLinear, z: QLinear, b: QLinear, a: QLinear, out: QLinear, conv: u64, a_log: u64, dt_bias: u64, norm: u64 };
pub const Attention = struct { q: QLinear, k: QLinear, v: QLinear, o: QLinear, q_norm: u64, k_norm: u64 };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };
pub const Layer = struct { input_norm: u64, post_norm: u64, mixer: Mixer, gate: QLinear, up: QLinear, down: QLinear };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    g: c.Geometry,
    embed: Embed = undefined,
    layers: []Layer = &.{},
    norm: u64 = 0,
    head: QLinear = undefined,
    inv_freq: u64 = 0, // (rotary_dim / 2,) fp32
    eos: [config_eos]u32 = undefined, // config.json's end tokens, then generation_config.json's
    eos_count: usize = 0,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    pub fn isEos(w: *const Weights, token: u32) bool {
        for (w.eos[0..w.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// Device bytes the weights hold.
    pub fn bytes(w: *const Weights) usize {
        var n: usize = 0;
        for (w.buffers.items) |b| n += b.len;
        return n;
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    w: *Weights,
    src: *source.Source,
    scratch: cuda.DeviceBuffer,
    host: std.ArrayList(u8) = .empty,

    fn alloc(L: *Loader, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(L.ops.k.d, len);
        try L.w.buffers.append(L.gpa, b);
        return b.ptr;
    }

    /// Device scratch of at least `len` bytes; growing waits for the stream so pending packs keep their inputs.
    fn tmp(L: *Loader, len: usize) !u64 {
        if (L.scratch.len < len) {
            try L.src.flush();
            try L.ops.s.synchronize();
            L.scratch.free();
            L.scratch = try cuda.DeviceBuffer.alloc(L.ops.k.d, len);
        }
        return L.scratch.ptr;
    }

    /// The host staging buffer, once the stream has read its last contents.
    fn staging(L: *Loader, len: usize) ![]u8 {
        try L.src.flush();
        try L.ops.s.synchronize();
        try L.host.resize(L.gpa, len);
        return L.host.items;
    }

    fn raw(L: *Loader, t: Tensor) !u64 {
        const ptr = try L.alloc(t.bytes.len);
        try L.src.upload(ptr, t.bytes);
        return ptr;
    }

    /// `.float()` of a bf16 tensor (or an f32 one as stored): exact.
    fn widen(L: *Loader, t: Tensor) !u64 {
        if (t.dtype == .f32) return L.raw(t);
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const n = t.bytes.len / 2;
        const out = std.mem.bytesAsSlice(u32, try L.staging(n * 4));
        const in = try L.src.view(t.bytes);
        for (out, 0..) |*o, i| o.* = @as(u32, std.mem.readInt(u16, in[2 * i ..][0..2], .little)) << 16;
        const ptr = try L.alloc(n * 4);
        try L.ops.upload(ptr, L.host.items);
        return ptr;
    }

    /// qmm_fast.tile: MLX words, scales and biases packed into the lane matmul's tiled layout.
    fn tile(L: *Loader, ck: *core.Checkpoint, name: []const u8, n: usize, k: usize) !QLinear {
        var parts: [3]Tensor = undefined;
        var buf: [192]u8 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |s, i| parts[i] = try ck.get(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ prefix, name, s }));
        const k8 = k / 8;
        const kg = k / 64;
        if (parts[0].dtype != .u32 or parts[1].dtype != .bf16 or parts[2].dtype != .bf16) return error.UnsupportedQuantization;
        if (!parts[0].is(.u32, &.{ n, k8 }) or !parts[1].is(.bf16, &.{ n, kg }) or !parts[2].is(.bf16, &.{ n, kg })) {
            std.log.err("{s}: not a 4-bit group-64 ({d}, {d}) projection", .{ name, n, k });
            return error.UnexpectedTensor;
        }
        const npad = (n + 127) / 128 * 128;
        const sizes = [3]usize{ n * k8 * 4, n * kg * 2, n * kg * 2 };
        const base = try L.tmp(sizes[0] + sizes[1] + sizes[2]);
        const at = [3]usize{ 0, sizes[0], sizes[0] + sizes[1] };
        for (0..3) |j| try L.src.upload(base + at[j], parts[j].bytes);
        const q: QLinear = .{ .w = try L.alloc(npad * k / 2), .s = try L.alloc(kg * npad * 2), .b = try L.alloc(kg * npad * 2), .n = n, .k = k, .npad = npad };
        const total: u64 = @as(u64, npad / 64) * kg * 512;
        try L.src.flush();
        var a: cuda.Args = .{};
        a.add(base);
        for ([_]usize{ n, k8, kg }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(q.w);
        a.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(L.ops.k.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.ops.s, &a);
        for ([_]u64{ q.s, q.b }, [_]usize{ at[1], at[2] }) |out, off| {
            var t: cuda.Args = .{};
            t.add(base + off);
            for ([_]usize{ n, kg, npad }) |v| t.add(@as(c_int, @intCast(v)));
            t.add(out);
            const cells = @as(u64, kg) * npad;
            try cuda.launch.launch(L.ops.k.transpose16, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, L.ops.s, &t);
        }
        return q;
    }

    fn vector(L: *Loader, ck: *core.Checkpoint, name: []const u8, len: usize) !u64 {
        var buf: [192]u8 = undefined;
        return L.raw(try ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }), .bf16, &.{len}));
    }

    fn delta(L: *Loader, ck: *core.Checkpoint, p: []const u8, g: c.Geometry) !Delta {
        var b1: [160]u8 = undefined;
        var b2: [192]u8 = undefined;
        const at = struct {
            fn f(buf: []u8, pre: []const u8, name: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}linear_attn.{s}", .{ pre, name });
            }
        }.f;
        var d: Delta = undefined;
        d.qkv = try L.tile(ck, try at(&b1, p, "in_proj_qkv"), g.convDim(), g.hidden);
        d.z = try L.tile(ck, try at(&b1, p, "in_proj_z"), g.vInner(), g.hidden);
        d.b = try L.tile(ck, try at(&b1, p, "in_proj_b"), g.linear_v_heads, g.hidden);
        d.a = try L.tile(ck, try at(&b1, p, "in_proj_a"), g.linear_v_heads, g.hidden);
        d.out = try L.tile(ck, try at(&b1, p, "out_proj"), g.hidden, g.vInner());
        // conv1d.weight (C, 4, 1) is (C, 4) as stored: the reshape moves no bytes
        d.conv = try L.raw(try ck.expect(try std.fmt.bufPrint(&b2, "{s}{s}", .{ prefix, try at(&b1, p, "conv1d.weight") }), .bf16, &.{ g.convDim(), c.conv_taps, 1 }));
        d.a_log = try L.widen(try ck.get(try std.fmt.bufPrint(&b2, "{s}{s}", .{ prefix, try at(&b1, p, "A_log") })));
        d.dt_bias = try L.widen(try ck.get(try std.fmt.bufPrint(&b2, "{s}{s}", .{ prefix, try at(&b1, p, "dt_bias") })));
        d.norm = try L.vector(ck, try at(&b1, p, "norm.weight"), c.linear_dim);
        return d;
    }

    fn attention(L: *Loader, ck: *core.Checkpoint, p: []const u8, g: c.Geometry) !Attention {
        var b1: [160]u8 = undefined;
        const at = struct {
            fn f(buf: []u8, pre: []const u8, name: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}self_attn.{s}", .{ pre, name });
            }
        }.f;
        return .{
            .q = try L.tile(ck, try at(&b1, p, "q_proj"), 2 * g.qInner(), g.hidden),
            .k = try L.tile(ck, try at(&b1, p, "k_proj"), g.kvInner(), g.hidden),
            .v = try L.tile(ck, try at(&b1, p, "v_proj"), g.kvInner(), g.hidden),
            .o = try L.tile(ck, try at(&b1, p, "o_proj"), g.hidden, g.qInner()),
            .q_norm = try L.vector(ck, try at(&b1, p, "q_norm.weight"), c.head_dim),
            .k_norm = try L.vector(ck, try at(&b1, p, "k_norm.weight"), c.head_dim),
        };
    }
};

/// weights.inv_freq: theta ** (-i / half) in float64, rounded to fp32.
pub fn invFreq(out: []f32) void {
    const half: f64 = @floatFromInt(out.len);
    for (out, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, c.theta, -@as(f64, @floatFromInt(i)) / half));
}

/// weights.load (tiled, one GPU): the model folder's text tensors on the GPU, leftovers refused.
pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, cfg: c.Config) !Weights {
    const g = cfg.g;
    var w: Weights = .{ .gpa = gpa, .g = g, .eos = cfg.eos, .eos_count = cfg.eos_count };
    errdefer w.deinit();
    var src = try source.Source.init(gpa, .{ .d = ops.k.d, .s = ops.s, .discrete = ops.k.discrete });
    defer src.deinit();
    var L: Loader = .{ .gpa = gpa, .ops = ops, .w = &w, .src = &src, .scratch = try cuda.DeviceBuffer.alloc(ops.k.d, 1 << 20) };
    defer {
        src.flush() catch {};
        ops.s.synchronize() catch {};
        L.scratch.free();
        L.host.deinit(gpa);
    }
    // the vision tower and the MTP head stay on disk, as weights._Tensors skips them
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    try src.add(&ck);
    w.layers = try gpa.alloc(Layer, g.layers);
    for (w.layers, 0..) |*layer, i| {
        var pb: [64]u8 = undefined;
        var nb: [160]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        const join = struct {
            fn f(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
            }
        }.f;
        layer.input_norm = try L.vector(&ck, try join(&nb, p, "input_layernorm.weight"), g.hidden);
        layer.mixer = if (c.linear(i)) .{ .delta = try L.delta(&ck, p, g) } else .{ .attention = try L.attention(&ck, p, g) };
        layer.gate = try L.tile(&ck, try join(&nb, p, "mlp.gate_proj"), g.intermediate, g.hidden);
        layer.up = try L.tile(&ck, try join(&nb, p, "mlp.up_proj"), g.intermediate, g.hidden);
        layer.down = try L.tile(&ck, try join(&nb, p, "mlp.down_proj"), g.hidden, g.intermediate);
        layer.post_norm = try L.vector(&ck, try join(&nb, p, "post_attention_layernorm.weight"), g.hidden);
    }
    w.embed = .{
        .w = try L.raw(try ck.expect(prefix ++ "model.embed_tokens.weight", .u32, &.{ c.vocab, g.hidden / 8 })),
        .s = try L.raw(try ck.expect(prefix ++ "model.embed_tokens.scales", .bf16, &.{ c.vocab, g.hidden / 64 })),
        .b = try L.raw(try ck.expect(prefix ++ "model.embed_tokens.biases", .bf16, &.{ c.vocab, g.hidden / 64 })),
    };
    w.norm = try L.vector(&ck, "model.norm.weight", g.hidden);
    w.head = try L.tile(&ck, "lm_head", c.vocab, g.hidden);
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    var inv: [c.rotary_dim / 2]f32 = undefined;
    invFreq(&inv);
    w.inv_freq = try L.alloc(@sizeOf(@TypeOf(inv)));
    try ops.upload(w.inv_freq, std.mem.asBytes(&inv));
    try src.flush();
    try ops.s.synchronize();
    return w;
}

test "inv_freq: theta ** (-i / 32) rounded to fp32" {
    var inv: [32]f32 = undefined;
    invFreq(&inv);
    try std.testing.expectEqual(@as(f32, 1.0), inv[0]);
    try std.testing.expectEqual(@as(f32, 0.6042963862419128), inv[1]);
    try std.testing.expectEqual(@as(f32, 0.22067341208457947), inv[3]);
}
