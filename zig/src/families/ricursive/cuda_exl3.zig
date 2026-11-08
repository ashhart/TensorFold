//! Ripple's EXL3 packs on CUDA: every projection and the head through main's exl3.Dense, bf16 in and out.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("config.zig");

const Config = cfg.Config;
const Checkpoint = core.checkpoint.Checkpoint;
const Tensor = core.checkpoint.Tensor;
const Weight = cuda.qmmf.Weight;
pub const X3 = cuda.exl3.Weight;
const Stream = cuda.Stream;

/// exllamav3's mul1 codebook marker, stored as a scalar `<name>.mul1` beside each trellis.
const mul1_magic: u32 = 0x83DCD12D;

/// A projection: the lane and prompt matmuls' packed weight (the NVFP4 release), or an EXL3 trellis (a pack).
pub const Proj = union(enum) {
    lane: Weight,
    exl3: X3,
};

/// q, k and g: one fused lane projection (rows q, k, g), or an EXL3 pack's three, each its own output.
pub const Qkg = union(enum) {
    lane: Weight,
    exl3: [3]X3,
};

/// The fused lane projection, or an EXL3 pack's q (a stand-in where only the lane's weights count).
pub fn first(p: Qkg) Proj {
    return switch (p) {
        .lane => |w| .{ .lane = w },
        .exl3 => |ws| .{ .exl3 = ws[0] },
    };
}

/// Where a forward of `n` rows finds q, k and g in `qkg`: fused rows (stride q + kv + q), or three planar blocks.
pub const QkgRows = struct { q: u64, q_ld: usize, k: u64, k_ld: usize, g: u64, g_ld: usize };

pub fn qkgRows(p: Qkg, base: u64, n: usize, qw: usize, kvw: usize) QkgRows {
    return switch (p) {
        .lane => .{ .q = base, .q_ld = 2 * qw + kvw, .k = base + qw * 2, .k_ld = 2 * qw + kvw, .g = base + (qw + kvw) * 2, .g_ld = 2 * qw + kvw },
        .exl3 => .{ .q = base, .q_ld = qw, .k = base + n * qw * 2, .k_ld = kvw, .g = base + n * (qw + kvw) * 2, .g_ld = qw },
    };
}

/// One stored EXL3 projection: its trellis, Hadamard sign scales and bits (`halves` = 2 * bits).
pub const Stored = struct { trellis: Tensor, suh: Tensor, svh: Tensor, n: usize, k: usize, halves: usize };

/// An n-by-k EXL3 projection `base` (trellis [k/16][n/16][8 * halves] i16, mul1, fp16 suh [k] and svh [n]).
pub fn read(ck: *Checkpoint, base: []const u8, n: usize, k: usize) !Stored {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) return error.UnexpectedTensor;
    const marker = try ck.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{base}));
    if (marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedRicursivePrecision;
    const suh = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.suh", .{base}), .f16, &.{k});
    const svh = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.svh", .{base}), .f16, &.{n});
    return .{ .trellis = t, .suh = suh, .svh = svh, .n = n, .k = k, .halves = t.dim(2) / 8 };
}

/// A pack's layer: q, k, g, v, o, gate, up, down; the four norms, q_norm and k_norm; value_res_lambda's bf16 bits.
pub const LayerMap = struct { proj: [8]Stored, norms: [6]Tensor, lambda: u16 };

/// Every tensor an EXL3 pack holds, checked by dtype and shape; a missing, mistyped or leftover one refuses it.
pub const Map = struct {
    gpa: std.mem.Allocator,
    embed: Tensor,
    norm: Tensor,
    head: Stored,
    layers: []LayerMap,

    pub fn deinit(m: *Map) void {
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

pub fn map(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    const h = c.hidden;
    const q = c.queryWidth();
    const kv = c.kvWidth();
    const f = c.intermediate;
    var m: Map = .{ .gpa = gpa, .embed = undefined, .norm = undefined, .head = undefined, .layers = try gpa.alloc(LayerMap, c.layers) };
    errdefer gpa.free(m.layers);
    m.embed = try ck.expect("model.embed_tokens.weight", .bf16, &.{ c.vocab, h });
    m.norm = try ck.expect("model.norm.weight", .bf16, &.{h});
    m.head = try read(ck, "lm_head", c.vocab, h);
    const shapes = [_]struct { []const u8, usize, usize }{ .{ "self_attn.q_proj", q, h }, .{ "self_attn.k_proj", kv, h }, .{ "self_attn.g_proj", q, h }, .{ "self_attn.v_proj", kv, h }, .{ "self_attn.o_proj", h, q }, .{ "mlp.gate_proj", f, h }, .{ "mlp.up_proj", f, h }, .{ "mlp.down_proj", h, f } };
    const norms = [_][]const u8{ "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm" };
    var name: [160]u8 = undefined;
    for (m.layers, 0..) |*l, i| {
        for (shapes, &l.proj) |sh, *x| x.* = try read(ck, try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ i, sh[0] }), sh[1], sh[2]);
        for (norms, &l.norms, 0..) |nn, *x, j| x.* = try ck.expect(try std.fmt.bufPrint(&name, "model.layers.{d}.{s}.weight", .{ i, nn }), .bf16, &.{if (j < 4) h else cfg.head_dim});
        l.lambda = try lambdaBits(ck, try std.fmt.bufPrint(&name, "model.layers.{d}.self_attn.value_res_lambda", .{i}));
    }
    if (ck.unused() != 0) return error.UnusedTensors;
    return m;
}

/// The device weight over uploaded trellis, suh and svh, refused when no exl3_linear variant takes its bits.
pub fn weight(trellis: u64, suh: u64, svh: u64, n: usize, k: usize, halves: usize) !X3 {
    const st = cuda.exl3.strides(k, n, halves, false);
    const w: X3 = .{ .trellis = trellis, .suh = suh, .svh = svh, .k = @intCast(k), .n = @intCast(n), .halves = @intCast(halves), .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    if (cuda.exl3.variantIndex(w.halves, w.codebook, cuda.exl3.planOf(w).wk) == null) return error.UnsupportedRicursivePrecision;
    return w;
}

/// value_res_lambda as a pack stores it (fp32 [1]): the release's bf16 value, refused when it is not one.
pub fn lambdaBits(ck: *Checkpoint, name: []const u8) !u16 {
    const t = try ck.get(name);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    const bits = std.mem.readInt(u32, t.bytes[0..4], .little);
    if (bits & 0xffff != 0) return error.UnsupportedRicursivePrecision;
    return @intCast(bits >> 16);
}

/// The scratch exl3.Dense grows to for 128 rows of `w`: rotated inputs, split partials and counters.
pub fn scratchBytes(w: X3) usize {
    const z = if (cuda.exl3.planOf(w).sk > 1) cuda.exl3.Linear.zBytes(w, 128) else 0;
    return 128 * @as(usize, w.k) * 2 + z + cuda.exl3.Linear.counterBytes(w);
}

/// Every EXL3 projection of a layer list and the head: what a forward's calls run over.
pub fn each(head: Proj, layers: anytype, ctx: anytype, comptime f: fn (@TypeOf(ctx), X3) anyerror!void) !void {
    if (head == .exl3) try f(ctx, head.exl3);
    for (layers) |l| {
        if (l.qkg == .exl3) for (l.qkg.exl3) |x| try f(ctx, x);
        for ([_]Proj{ l.v, l.o, l.gate, l.up, l.down }) |p| if (p == .exl3) try f(ctx, p.exl3);
    }
}

/// The scratch Dense grows to over every projection (the most any one needs, as it keeps one set).
pub fn denseBytes(head: Proj, layers: anytype) usize {
    var most: usize = 0;
    const Most = struct {
        fn take(m: *usize, w: X3) anyerror!void {
            m.* = @max(m.*, scratchBytes(w));
        }
    };
    each(head, layers, &most, Most.take) catch unreachable;
    return most;
}

/// main's dense EXL3 kernels for a forward, scratch grown to every projection's (no allocation in a graph capture).
pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, s: Stream, head: Proj, layers: anytype) !*cuda.exl3.Dense {
    const de = try gpa.create(cuda.exl3.Dense);
    errdefer gpa.destroy(de);
    de.* = try cuda.exl3.Dense.load(d);
    errdefer de.unload();
    try reserve(de, s, head, layers);
    return de;
}

pub fn close(gpa: std.mem.Allocator, de: *cuda.exl3.Dense) void {
    de.unload();
    gpa.destroy(de);
}

/// Dense with its scratch grown to every projection's on `s`, so a call inside a graph capture allocates nothing.
pub fn reserve(de: *cuda.exl3.Dense, s: Stream, head: Proj, layers: anytype) !void {
    const Ctx = struct { de: *cuda.exl3.Dense, s: Stream };
    const R = struct {
        fn grow(c: Ctx, w: X3) anyerror!void {
            try c.de.reserve(c.s, w);
        }
    };
    try each(head, layers, Ctx{ .de = de, .s = s }, R.grow);
}

const testing = std.testing;

test "ricursive cuda exl3: q, k and g rows fused or planar, each at its own stride" {
    const fused = qkgRows(.{ .lane = undefined }, 0x1000, 7, 10240, 1024);
    try testing.expectEqual(@as(usize, 21504), fused.q_ld);
    try testing.expectEqual(@as(u64, 0x1000 + 10240 * 2), fused.k);
    try testing.expectEqual(@as(u64, 0x1000 + 11264 * 2), fused.g);
    const planar = qkgRows(.{ .exl3 = undefined }, 0x1000, 7, 10240, 1024);
    try testing.expectEqual(@as(usize, 10240), planar.q_ld);
    try testing.expectEqual(@as(usize, 1024), planar.k_ld);
    try testing.expectEqual(@as(u64, 0x1000 + 7 * 10240 * 2), planar.k);
    try testing.expectEqual(@as(u64, 0x1000 + 7 * 11264 * 2), planar.g);
    try testing.expectEqual(@as(usize, 10240), planar.g_ld); // gate reads g's rows contiguous: no copy
}

test "ricursive cuda exl3: a 5-bit mul1 projection's strides and plan; an unbuilt bit width refused" {
    const w = try weight(0, 0, 0, 1024, 5120, 10);
    try testing.expectEqual(@as(u32, 5120), w.k);
    try testing.expectEqual(cuda.exl3.Codebook.mul1, w.codebook);
    try testing.expect(scratchBytes(w) >= 128 * 5120 * 2);
    try testing.expectError(error.UnsupportedRicursivePrecision, weight(0, 0, 0, 1024, 5120, 40));
}
