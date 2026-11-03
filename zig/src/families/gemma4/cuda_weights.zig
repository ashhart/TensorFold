//! Gemma 4's weights on CUDA: MLX affine-4 projections packed into qlinear's tiles, norms as stored, the tied head.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("cuda_config.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;

/// A projection at its MLX width: affine-4 tiles, or the tiles of every other width.
pub const Proj = union(enum) {
    a4: Affine4,
    an: AffineN,

    pub fn n(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.n,
        };
    }

    pub fn k(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.k,
        };
    }
};

/// A layer's tensors: each projection as qlinear tiles at its width, bf16 vectors as stored (norm weights with no 1 +).
pub const Layer = struct {
    q: Proj,
    k: ?Proj, // null on a layer that reads another's cache (E-series shared KV)
    v: ?Proj, // none on such a layer, nor on the 12B's and 31B's global layers: v is k_proj's output
    o: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
    q_norm: u64, // bf16 [head dim]
    k_norm: u64, // 0 on a shared layer
    input_norm: u64, // bf16 [hidden]
    post_attention: u64,
    pre_feedforward: u64,
    post_feedforward: u64,
    scalar: u64, // bf16 [1]: the residual's layer scalar
    /// The E-series' per-layer block, null without.
    ple: ?Ple = null,
};

/// A layer's per-layer input block: gate [width, hidden], projection [hidden, width], post norm bf16 [hidden].
pub const Ple = struct { gate: Proj, project: Proj, post_norm: u64 };

/// The E-series' per-layer inputs: the token table as MLX stores it, the model projection, the 256-wide norm.
pub const PleModel = struct { table_w: u64, table_s: u64, table_b: u64, table_bits: u4, project: Proj, norm: u64 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    /// The embedding table as MLX stores it: words u32 [V, H bits / 32], scales and biases [V, H / 64].
    embed_w: u64 = 0,
    embed_bits: u8 = 4,
    embed_s: u64 = 0,
    embed_b: u64 = 0,
    /// The same table as the tied head's tiles.
    head: Proj = undefined,
    final_norm: u64 = 0,
    layers: []Layer = &.{},
    /// The E-series' per-layer inputs, null without.
    ple: ?PleModel = null,
    /// The 12B's image embedder (--vision), null without.
    vision: ?Vision = null,
    /// The 31B's image tower (--vision), null without.
    tower: ?Tower = null,
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        const b = try DeviceBuffer.fromHost(d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }
};

/// The 12B's image embedder: LayerNorms, patch_dense bf16, the position table, the projection at its width (or bf16).
pub const Vision = struct {
    ln1_w: u64,
    ln1_b: u64,
    dense_w: u64,
    dense_b: u64,
    ln2_w: u64,
    ln2_b: u64,
    table: u64,
    pos_w: u64,
    pos_b: u64,
    proj: union(enum) { q: Proj, bf16: u64 },
};

/// The 31B's image tower on bf16 weights (q|k|v stacked), its patch rows and pooling, the projection at its width.
pub const Tower = struct {
    input_proj: u64, // bf16 [1152, 768]
    table: u64, // bf16 [2, 10240, 1152]
    std_bias: u64,
    std_scale: u64,
    layers: [tower_depth]TowerLayer,
    proj: Proj,
};

pub const TowerLayer = struct { in_norm: u64, post_attn: u64, pre_ff: u64, post_ff: u64, qkv: u64, o: u64, q_norm: u64, k_norm: u64, gate: u64, up: u64, down: u64 };

/// The tower's sizes: width, heads of 72, depth, MLP width, each axis's position rows, a 16-px patch's values.
pub const tower_width = 1152;
pub const tower_heads = 16;
pub const tower_head_dim = 72;
pub const tower_depth = 27;
pub const tower_inter = 4304;
pub const tower_table_rows = 10240;
pub const patch_width = 768;

/// The embedder's sizes: a merged 48-px patch's values, the output width, each axis's position rows.
pub const merged_width = 6912;
pub const vision_width = 3840;
pub const table_rows = 1120;

const Ctx = struct {
    w: *Weights,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const kern.Kernels,
    ck: *core.checkpoint.Checkpoint,
    stage: DeviceBuffer,

    fn tensor(x: *Ctx, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        return x.w.upload(x.d, (try x.ck.expect(name, dtype, shape)).bytes);
    }

    /// An n-by-k MLX affine projection of `bits` codes (words, scales, biases) staged and packed into qlinear's tiles.
    fn linear(x: *Ctx, prefix: []const u8, n: usize, k: usize, bits: u8) !Proj {
        var buf: [192]u8 = undefined;
        const matrix = @import("living").dense_matrix;
        const role = matrix.role(prefix);
        const extra = if (role != null) x.w.config.learned else 0;
        if (extra != 0 and bits != 4) return error.DensePocketFormat;
        const stored_n = n + if (role != null and role.? != .down) extra else @as(usize, 0);
        const stored_k = k + if (role != null and role.? == .down) extra else @as(usize, 0);
        const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, stored_k * bits / 32 }, .{ "scales", .bf16, stored_k / 64 }, .{ "biases", .bf16, stored_k / 64 } };
        var raw: [3]u64 = undefined;
        var at: usize = 0;
        var temporary: std.heap.ArenaAllocator = .init(x.w.gpa);
        defer temporary.deinit();
        for (specs, &raw, 0..) |sp, *r, index| {
            const t = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, sp[0] }), sp[1], &.{ stored_n, sp[2] });
            const bytes = if (extra == 0) t.bytes else try matrix.base(temporary.allocator(), x.w.config.geo.hidden, x.w.config.geo.intermediate, extra, role.?, @fromBackingInt(@intCast(index)), matrix.View.of(t));
            if (at + bytes.len > x.stage.len) return error.StageTooSmall;
            try x.stage.upload(at, bytes);
            r.* = x.stage.ptr + at;
            at += bytes.len;
        }
        const served = cuda.qlinear.servedBits(bits);
        const out: Proj = if (served == 4) blk: {
            const l = Affine4.layout(n, k);
            const q: Affine4 = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad };
            try x.k.affine.pack(x.s, raw[0], raw[1], raw[2], q);
            break :blk .{ .a4 = q };
        } else blk: {
            const l = AffineN.layout(n, k, served);
            const q: AffineN = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad, .bits = served };
            try x.k.wide.packWords(x.s, raw[0], bits, n, k, q.w);
            try x.k.wide.packTable(x.s, raw[1], n, k, q.npad, q.s);
            try x.k.wide.packTable(x.s, raw[2], n, k, q.npad, q.b);
            break :blk .{ .an = q };
        };
        try x.s.synchronize(); // the next projection's staging overwrites this one's
        return out;
    }
};

const lm = "language_model.model.";

/// Admission as Metal's (MLX affine 2-6 and 8 bits, groups of 64), every tensor checked. `layers` caps them (checks).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, layers: ?usize, images: bool) !Weights {
    const c = try cfg.Config.read(gpa, io, dir);
    if (c.geo.layer_input != 0) try requireAffineInputs(c) else try c.requireDense(); // the E-series (#1222)
    if (c.geo.layer_input != 0 and images) return error.GemmaESeriesImagesNotServed;
    return loadConfig(gpa, io, d, s, k, dir, c, layers, images);
}

/// The CUDA E-series reads its per-layer linears and table at an MLX affine width only (a BF16 one: dense_rows, next).
pub fn requireAffineInputs(c: cfg.Config) !void {
    if (c.geo.layer_input == 0) return;
    const ws = &c.widths;
    if (ws.model_project == null or ws.layer_table == null) return error.GemmaLayerInputBf16NotServed;
    for (ws.layer_gate[0..c.geo.layers], ws.layer_project[0..c.geo.layers]) |g, p|
        if (g == null or p == null) return error.GemmaLayerInputBf16NotServed;
}

fn loadConfig(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, c: cfg.Config, layers: ?usize, images: bool) !Weights {
    if (c.learned != 0) try @import("learning_profile.zig").check(c);
    const geo = c.geo;
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    const h = geo.hidden;
    // the largest projection as stored: the tied head's table
    var stage = try DeviceBuffer.alloc(d, cfg.vocab * h * c.widths.embed.bits / 8 + 2 * cfg.vocab * (h / 64) * 2);
    defer stage.free();
    var x: Ctx = .{ .w = &w, .d = d, .s = s, .k = k, .ck = &ck, .stage = stage };
    w.embed_bits = c.widths.embed.bits;
    w.embed_w = try x.tensor(lm ++ "embed_tokens.weight", .u32, &.{ cfg.vocab, h * w.embed_bits / 32 });
    w.embed_s = try x.tensor(lm ++ "embed_tokens.scales", .bf16, &.{ cfg.vocab, h / 64 });
    w.embed_b = try x.tensor(lm ++ "embed_tokens.biases", .bf16, &.{ cfg.vocab, h / 64 });
    w.head = try x.linear(lm ++ "embed_tokens", cfg.vocab, h, w.embed_bits);
    w.final_norm = try x.tensor(lm ++ "norm.weight", .bf16, &.{h});
    if (geo.layer_input != 0) {
        const stacked = geo.layers * geo.layer_input;
        const tb: u4 = @intCast(c.widths.layer_table.?.bits);
        w.ple = .{
            .table_w = try x.tensor(lm ++ "embed_tokens_per_layer.weight", .u32, &.{ cfg.vocab, stacked * tb / 32 }),
            .table_s = try x.tensor(lm ++ "embed_tokens_per_layer.scales", .bf16, &.{ cfg.vocab, stacked / 64 }),
            .table_b = try x.tensor(lm ++ "embed_tokens_per_layer.biases", .bf16, &.{ cfg.vocab, stacked / 64 }),
            .table_bits = tb,
            .project = try x.linear(lm ++ "per_layer_model_projection", stacked, h, c.widths.model_project.?.bits),
            .norm = try x.tensor(lm ++ "per_layer_projection_norm.weight", .bf16, &.{geo.layer_input}),
        };
    }
    const n = @min(layers orelse geo.layers, geo.layers);
    w.layers = try gpa.alloc(Layer, n);
    var buf: [192]u8 = undefined;
    const readers = try core.checkpoint.loadReaders();
    for (w.layers, 0..) |*l, i| {
        const dh = cfg.headDim(i);
        const kvh = geo.kvHeads(i);
        const q_out = geo.query_heads * dh;
        const kv_out = kvh * dh;
        const p = try std.fmt.allocPrint(gpa, lm ++ "layers.{d}.", .{i});
        defer gpa.free(p);
        defer if (readers > 0) x.ck.release(); // a read-ahead layer is on the device: its pages leave the cache (none read ahead: main's one release at close)
        var reading = x.ck.prefetch(p, readers); // read ahead by threads of their own while the layer packs
        defer _ = reading.join();
        const name = struct {
            fn of(b: []u8, pre: []const u8, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}{s}", .{ pre, suffix });
            }
        }.of;
        l.q = try x.linear(try name(&buf, p, "self_attn.q_proj"), q_out, h, c.widths.of(i, .q).bits);
        const owns = i < geo.cacheLayers(); // a shared layer's k_proj, v_proj and k_norm are left unread
        l.k = if (owns) try x.linear(try name(&buf, p, "self_attn.k_proj"), kv_out, h, c.widths.of(i, .k).bits) else null;
        const has_v = owns and (!cfg.global(i) or !geo.keys_are_values);
        l.v = if (has_v) try x.linear(try name(&buf, p, "self_attn.v_proj"), kv_out, h, c.widths.of(i, .v).bits) else null;
        l.o = try x.linear(try name(&buf, p, "self_attn.o_proj"), h, q_out, c.widths.of(i, .o).bits);
        l.gate = try x.linear(try name(&buf, p, "mlp.gate_proj"), geo.intermediate, h, c.widths.of(i, .gate).bits);
        l.up = try x.linear(try name(&buf, p, "mlp.up_proj"), geo.intermediate, h, c.widths.of(i, .up).bits);
        l.down = try x.linear(try name(&buf, p, "mlp.down_proj"), h, geo.intermediate, c.widths.of(i, .down).bits);
        l.q_norm = try x.tensor(try name(&buf, p, "self_attn.q_norm.weight"), .bf16, &.{dh});
        l.k_norm = if (owns) try x.tensor(try name(&buf, p, "self_attn.k_norm.weight"), .bf16, &.{dh}) else 0;
        l.input_norm = try x.tensor(try name(&buf, p, "input_layernorm.weight"), .bf16, &.{h});
        l.post_attention = try x.tensor(try name(&buf, p, "post_attention_layernorm.weight"), .bf16, &.{h});
        l.pre_feedforward = try x.tensor(try name(&buf, p, "pre_feedforward_layernorm.weight"), .bf16, &.{h});
        l.post_feedforward = try x.tensor(try name(&buf, p, "post_feedforward_layernorm.weight"), .bf16, &.{h});
        l.scalar = try x.tensor(try name(&buf, p, "layer_scalar"), .bf16, &.{1});
        l.ple = if (geo.layer_input == 0) null else .{
            .gate = try x.linear(try name(&buf, p, "per_layer_input_gate"), geo.layer_input, h, c.widths.layer_gate[i].?.bits),
            .project = try x.linear(try name(&buf, p, "per_layer_projection"), h, geo.layer_input, c.widths.layer_project[i].?.bits),
            .post_norm = try x.tensor(try name(&buf, p, "post_per_layer_input_norm.weight"), .bf16, &.{h}),
        };
    }
    if (images) {
        if (c.geo.unified()) w.vision = try vision(&x, c) else w.tower = try tower(&x, c);
    }
    return w;
}

/// The 31B's tower tensors, bf16 as the checkpoint keeps them (q, k and v stacked for one GEMM).
fn tower(x: *Ctx, c: cfg.Config) !Tower {
    if (!x.ck.has("vision_tower.patch_embedder.input_proj.weight")) return error.NoGemmaVision;
    const w = tower_width;
    const pe = "vision_tower.patch_embedder.";
    var t: Tower = .{
        .input_proj = try x.tensor(pe ++ "input_proj.weight", .bf16, &.{ w, patch_width }),
        .table = try x.tensor(pe ++ "position_embedding_table", .bf16, &.{ 2, tower_table_rows, w }),
        .std_bias = try x.tensor("vision_tower.std_bias", .bf16, &.{w}),
        .std_scale = try x.tensor("vision_tower.std_scale", .bf16, &.{w}),
        .layers = undefined,
        .proj = undefined,
    };
    var buf: [160]u8 = undefined;
    for (&t.layers, 0..) |*l, i| {
        const name = struct {
            fn of(b: []u8, layer: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "vision_tower.encoder.layers.{d}.{s}", .{ layer, suffix });
            }
        }.of;
        l.in_norm = try x.tensor(try name(&buf, i, "input_layernorm.weight"), .bf16, &.{w});
        l.post_attn = try x.tensor(try name(&buf, i, "post_attention_layernorm.weight"), .bf16, &.{w});
        l.pre_ff = try x.tensor(try name(&buf, i, "pre_feedforward_layernorm.weight"), .bf16, &.{w});
        l.post_ff = try x.tensor(try name(&buf, i, "post_feedforward_layernorm.weight"), .bf16, &.{w});
        l.q_norm = try x.tensor(try name(&buf, i, "self_attn.q_norm.weight"), .bf16, &.{tower_head_dim});
        l.k_norm = try x.tensor(try name(&buf, i, "self_attn.k_norm.weight"), .bf16, &.{tower_head_dim});
        l.o = try x.tensor(try name(&buf, i, "self_attn.o_proj.linear.weight"), .bf16, &.{ w, w });
        l.gate = try x.tensor(try name(&buf, i, "mlp.gate_proj.linear.weight"), .bf16, &.{ tower_inter, w });
        l.up = try x.tensor(try name(&buf, i, "mlp.up_proj.linear.weight"), .bf16, &.{ tower_inter, w });
        l.down = try x.tensor(try name(&buf, i, "mlp.down_proj.linear.weight"), .bf16, &.{ w, tower_inter });
        l.qkv = try x.w.alloc(x.d, 3 * w * w * 2);
        for ([_][]const u8{ "self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight" }, 0..) |part, j| {
            const bytes = (try x.ck.expect(try name(&buf, i, part), .bf16, &.{ w, w })).bytes;
            try (DeviceBuffer{ .d = x.d, .ptr = l.qkv + j * w * w * 2, .len = w * w * 2, .borrowed = true }).upload(0, bytes);
        }
    }
    t.proj = try x.linear("embed_vision.embedding_projection", c.geo.hidden, w, c.widths.vision.bits);
    return t;
}

/// The image embedder's tensors; a quantized patch_dense (the bf16 GEMM's input) is refused by name.
fn vision(x: *Ctx, c: cfg.Config) !Vision {
    if (!x.ck.has("vision_embedder.patch_dense.weight")) return error.NoGemmaVision;
    if (x.ck.has("vision_embedder.patch_dense.scales")) return error.GemmaVisionDenseQuantized;
    const e = "vision_embedder.";
    const mw = merged_width;
    const vw = vision_width;
    var v: Vision = .{
        .ln1_w = try x.tensor(e ++ "patch_ln1.weight", .bf16, &.{mw}),
        .ln1_b = try x.tensor(e ++ "patch_ln1.bias", .bf16, &.{mw}),
        .dense_w = try x.tensor(e ++ "patch_dense.weight", .bf16, &.{ vw, mw }),
        .dense_b = try x.tensor(e ++ "patch_dense.bias", .bf16, &.{vw}),
        .ln2_w = try x.tensor(e ++ "patch_ln2.weight", .bf16, &.{vw}),
        .ln2_b = try x.tensor(e ++ "patch_ln2.bias", .bf16, &.{vw}),
        .table = try x.tensor(e ++ "pos_embedding", .bf16, &.{ table_rows, 2, vw }),
        .pos_w = try x.tensor(e ++ "pos_norm.weight", .bf16, &.{vw}),
        .pos_b = try x.tensor(e ++ "pos_norm.bias", .bf16, &.{vw}),
        .proj = undefined,
    };
    const pp = "embed_vision.embedding_projection";
    v.proj = if (x.ck.has(pp ++ ".scales")) .{ .q = try x.linear(pp, c.geo.hidden, vw, c.widths.vision.bits) } else .{ .bf16 = try x.tensor(pp ++ ".weight", .bf16, &.{ c.geo.hidden, vw }) };
    return v;
}

test "an E-series checkpoint with BF16 per-layer linears is refused before a weight is read or the device is touched" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bf16 = try std.mem.replaceOwned(u8, a, @embedFile("fixtures/config_e4b.json"), "\"mode\": \"affine\"", "\"mode\": \"affine\", \"layers.3.per_layer_input_gate\": false");
    defer a.free(bf16);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = bf16 });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(path);
    var d: cuda.Driver = undefined;
    const s: cuda.Stream = undefined;
    var k: kern.Kernels = undefined;
    try std.testing.expectError(error.GemmaLayerInputBf16NotServed, load(a, std.testing.io, &d, s, &k, path, null, false));
}

test "the CUDA E-series takes affine per-layer linears and refuses a BF16 one by name" {
    var c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config_e4b.json"));
    try requireAffineInputs(c);
    c.widths.layer_gate[7] = null;
    try std.testing.expectError(error.GemmaLayerInputBf16NotServed, requireAffineInputs(c));
    c.widths.layer_gate[7] = cfg.widths.fast;
    c.widths.model_project = null;
    try std.testing.expectError(error.GemmaLayerInputBf16NotServed, requireAffineInputs(c));
}
