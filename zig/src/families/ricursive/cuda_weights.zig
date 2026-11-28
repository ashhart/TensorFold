//! Ripple's weights on CUDA: NVFP4 MLPs for the lane matmul, bf16 attention (q, k, g as one) and head; gate refusals.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const glue = @import("cuda_kernels.zig");
const attn = @import("cuda_attn.zig");
const x3 = @import("cuda_exl3.zig");

const Config = cfg.Config;
const Checkpoint = core.checkpoint.Checkpoint;
const Tensor = core.checkpoint.Tensor;
const Weight = cuda.qmmf.Weight;
const DeviceBuffer = cuda.DeviceBuffer;

/// Bytes a small tensor's device copy starts on: tf_rc_add_norm reads 16-byte chunks.
const small_align = 256;

/// What CUDA serves: the NVFP4 release's MLPs or an EXL3 pack, and shapes the kernels and packers take.
pub fn gate(c: *const Config) !void {
    if (!c.fp4_mlp and !c.exl3) return error.RicursiveCudaNeedsNvfp4;
    const g = c.group();
    // tf_window_decode_tc holds a key head's queries in one tile (at most 16); the prompt kernel pairs them
    if (g % attn.heads_a_block != 0 or g > 16) return error.UnsupportedRicursiveCudaShape;
    if (c.hidden % 8 != 0 or c.hidden > glue.most_width) return error.UnsupportedRicursiveCudaShape;
    // packBf16 and packWords take k in whole 64s; every output a whole number of 128-row blocks (no padding)
    for ([_]usize{ c.hidden, c.queryWidth(), c.intermediate }) |k| if (k % cuda.qmmf.group != 0) return error.UnsupportedRicursiveCudaShape;
    for ([_]usize{ c.queryWidth(), c.kvWidth(), c.hidden, c.intermediate, c.vocab }) |n| if (n % 128 != 0) return error.UnsupportedRicursiveCudaShape;
}

/// The fused projection's rows: q, then k, then g; v stays apart (its input is the value-residual blend).
pub fn qkgWidth(c: Config) usize {
    return 2 * c.queryWidth() + c.kvWidth();
}

/// An NVFP4 projection as ModelOpt stores it: nibbles [n, k / 2], E4M3 scales [n, k / 16], the fp32 global.
pub const Fp4 = struct { w: Tensor, scales: Tensor, global: f32, n: usize, k: usize };

pub const LayerMap = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    lambda: u16, // value_res_lambda's bf16 bits
    input_norm: Tensor,
    post_attn: Tensor,
    pre_ffn: Tensor,
    post_ffn: Tensor,
    gate: Fp4,
    up: Fp4,
    down: Fp4,
};

/// Every tensor the CUDA engine reads, checked by dtype and shape; input_scale is taken and unread, as on Metal.
pub const Map = struct {
    gpa: std.mem.Allocator,
    embed: Tensor,
    head: Tensor,
    norm: Tensor,
    layers: []LayerMap,

    pub fn deinit(m: *Map) void {
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

const Names = struct {
    ck: *Checkpoint,
    buf: [160]u8 = undefined,

    fn bf16(n: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return n.ck.expect(try std.fmt.bufPrint(&n.buf, fmt, args), .bf16, shape);
    }

    fn fp4(n: *Names, rows: usize, k: usize, comptime fmt: []const u8, args: anytype) !Fp4 {
        const w = try n.ck.expect(try std.fmt.bufPrint(&n.buf, fmt ++ ".weight", args), .u8, &.{ rows, k / 2 });
        const s = try n.ck.expect(try std.fmt.bufPrint(&n.buf, fmt ++ ".weight_scale", args), .f8_e4m3, &.{ rows, k / 16 });
        const t = try n.ck.expect(try std.fmt.bufPrint(&n.buf, fmt ++ ".weight_scale_2", args), .f32, &.{});
        const v: f32 = @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
        if (!(v > 0) or !std.math.isFinite(v)) {
            std.log.err("ricursive: " ++ fmt ++ ".weight_scale_2 {e} is not a positive finite scale", args ++ .{v});
            return error.UnexpectedTensor;
        }
        const input = try std.fmt.bufPrint(&n.buf, fmt ++ ".input_scale", args);
        if (n.ck.has(input)) _ = try n.ck.get(input); // the activations stay bf16 (W4A16): taken, never read
        return .{ .w = w, .scales = s, .global = v, .n = rows, .k = k };
    }
};

/// The release's names (PR 2231's weights.zig); a missing, mistyped or leftover tensor refuses the checkpoint.
pub fn map(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    var nm: Names = .{ .ck = ck };
    const h = c.hidden;
    const q = c.queryWidth();
    const kv = c.kvWidth();
    const f = c.intermediate;
    var m: Map = .{ .gpa = gpa, .embed = undefined, .head = undefined, .norm = undefined, .layers = try gpa.alloc(LayerMap, c.layers) };
    errdefer gpa.free(m.layers);
    m.embed = try nm.bf16(&.{ c.vocab, h }, "model.embed_tokens.weight", .{});
    m.head = try nm.bf16(&.{ c.vocab, h }, "lm_head.weight", .{});
    m.norm = try nm.bf16(&.{h}, "model.norm.weight", .{});
    for (m.layers, 0..) |*l, i| {
        const p = "model.layers.{d}.";
        const a = p ++ "self_attn.";
        const lambda = try nm.bf16(&.{1}, a ++ "value_res_lambda", .{i});
        l.* = .{
            .q = try nm.bf16(&.{ q, h }, a ++ "q_proj.weight", .{i}),
            .k = try nm.bf16(&.{ kv, h }, a ++ "k_proj.weight", .{i}),
            .v = try nm.bf16(&.{ kv, h }, a ++ "v_proj.weight", .{i}),
            .g = try nm.bf16(&.{ q, h }, a ++ "g_proj.weight", .{i}),
            .o = try nm.bf16(&.{ h, q }, a ++ "o_proj.weight", .{i}),
            .q_norm = try nm.bf16(&.{cfg.head_dim}, a ++ "q_norm.weight", .{i}),
            .k_norm = try nm.bf16(&.{cfg.head_dim}, a ++ "k_norm.weight", .{i}),
            .lambda = std.mem.readInt(u16, lambda.bytes[0..2], .little),
            .input_norm = try nm.bf16(&.{h}, p ++ "input_layernorm.weight", .{i}),
            .post_attn = try nm.bf16(&.{h}, p ++ "post_attention_layernorm.weight", .{i}),
            .pre_ffn = try nm.bf16(&.{h}, p ++ "pre_feedforward_layernorm.weight", .{i}),
            .post_ffn = try nm.bf16(&.{h}, p ++ "post_feedforward_layernorm.weight", .{i}),
            .gate = try nm.fp4(f, h, p ++ "mlp.gate_proj", .{i}),
            .up = try nm.fp4(f, h, p ++ "mlp.up_proj", .{i}),
            .down = try nm.fp4(h, f, p ++ "mlp.down_proj", .{i}),
        };
    }
    if (ck.unused() != 0) return error.UnusedTensors;
    return m;
}

/// torch's bf16 sigmoid of a bf16 scalar: fp32 1 / (1 + exp(-x)) to bf16, nearest even (Metal's weights.zig's copy).
pub fn blendOf(lambda: u16) f32 {
    const x: f32 = @bitCast(@as(u32, lambda) << 16);
    const s: f32 = 1.0 / (1.0 + @exp(-x));
    const bits: u32 = @bitCast(s);
    const rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    return @bitCast(rounded);
}

pub const Layer = struct {
    qkg: x3.Qkg, // bf16 [q + kv + q, H]: q rows, k rows, g rows (a pack: q, k and g)
    v: x3.Proj, // bf16 [kv, H]
    o: x3.Proj, // bf16 [H, q]
    gate: x3.Proj, // NVFP4 [I, H]
    up: x3.Proj,
    down: x3.Proj, // NVFP4 [H, I]
    input_norm: u64, // bf16 [H]
    post_attn: u64,
    pre_ffn: u64,
    post_ffn: u64,
    q_norm: u64, // bf16 [128]
    k_norm: u64,
    blend: f32, // bf16(sigmoid(value_res_lambda))
};

/// Device bytes `deviceBytes` counts beside the small tensors: every projection, the table and the head.
pub fn deviceBytes(c: Config) usize {
    const h = c.hidden;
    const fp4 = c.intermediate * h / 2 + c.intermediate * h / 16; // words and scale tiles (no padding: gate)
    const per_layer = (qkgWidth(c) + c.kvWidth()) * h * 2 + h * c.queryWidth() * 2 + 3 * fp4 + smallLayerBytes(c);
    return 2 * c.vocab * h * 2 + c.layers * per_layer + std.mem.alignForward(usize, h * 2, small_align);
}

/// A layer's norm weights, each at its own 256-byte boundary.
fn smallLayerBytes(c: Config) usize {
    return 4 * std.mem.alignForward(usize, c.hidden * 2, small_align) + 2 * std.mem.alignForward(usize, cfg.head_dim * 2, small_align);
}

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    embed: u64 = 0, // bf16 [V, H] as stored (tf_rc_embed gathers rows)
    head: x3.Proj = undefined, // bf16 [V, H] packed (a pack: EXL3)
    norm: u64 = 0, // bf16 [H]
    layers: []Layer = &.{},
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(d, bytes);
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        w.bytes += bytes.len;
        return b.ptr;
    }

    /// bf16 matrices one under another ([sum n, k]), packed as one projection for the lane and prompt matmuls.
    fn dense(w: *Weights, d: *const cuda.Driver, parts: []const Tensor, k: usize) !Weight {
        var n: usize = 0;
        for (parts) |p| n += p.bytes.len / 2 / k;
        const src = try w.gpa.alloc(u16, n * k);
        defer w.gpa.free(src);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(std.mem.sliceAsBytes(src)[at..][0..p.bytes.len], p.bytes);
            at += p.bytes.len;
        }
        const out = try w.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer w.gpa.free(out);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try w.upload(d, std.mem.sliceAsBytes(out)), n, k);
    }

    fn fp4(w: *Weights, d: *const cuda.Driver, p: Fp4) !Weight {
        const words = try w.gpa.alloc(u32, cuda.nvfp4.padded(p.n) / 64 * (p.k / 64) * 512);
        defer w.gpa.free(words);
        const tiles = try w.gpa.alloc(u8, cuda.nvfp4.padded(p.n) * (p.k / 16));
        defer w.gpa.free(tiles);
        cuda.nvfp4.packWords(words, p.w.bytes, p.n, p.k);
        cuda.nvfp4.packScales(tiles, p.scales.bytes, p.n, p.k);
        const wp = try w.upload(d, std.mem.sliceAsBytes(words));
        const tp = try w.upload(d, tiles);
        return cuda.nvfp4.weight(wp, tp, p.global, p.n, p.k);
    }
};

/// The small tensors in one host image, each at a 256-byte boundary; their offsets into it.
const Small = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn put(s: *Small, gpa: std.mem.Allocator, t: Tensor) !usize {
        const at = std.mem.alignForward(usize, s.bytes.items.len, small_align);
        try s.bytes.appendNTimes(gpa, 0, at - s.bytes.items.len);
        try s.bytes.appendSlice(gpa, t.bytes);
        return at;
    }
};

/// config.json through the CUDA gate, every tensor mapped and checked, then layer by layer onto the GPU.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    const c = try Config.read(gpa, io, dir);
    try gate(&c);
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    if (c.exl3) return loadExl3(gpa, d, &ck, c);
    var m = try map(gpa, &ck, c);
    defer m.deinit();
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    const h = c.hidden;
    w.layers = try gpa.alloc(Layer, c.layers);
    w.embed = try w.upload(d, m.embed.bytes);
    w.head = .{ .lane = try w.dense(d, &.{m.head}, h) };
    var small: Small = .{};
    defer small.bytes.deinit(gpa);
    const at_norm = try small.put(gpa, m.norm);
    const offsets = try gpa.alloc([6]usize, c.layers);
    defer gpa.free(offsets);
    for (w.layers, m.layers, offsets) |*l, t, *o| {
        l.* = .{
            .qkg = .{ .lane = try w.dense(d, &.{ t.q, t.k, t.g }, h) },
            .v = .{ .lane = try w.dense(d, &.{t.v}, h) },
            .o = .{ .lane = try w.dense(d, &.{t.o}, c.queryWidth()) },
            .gate = .{ .lane = try w.fp4(d, t.gate) },
            .up = .{ .lane = try w.fp4(d, t.up) },
            .down = .{ .lane = try w.fp4(d, t.down) },
            .input_norm = 0,
            .post_attn = 0,
            .pre_ffn = 0,
            .post_ffn = 0,
            .q_norm = 0,
            .k_norm = 0,
            .blend = blendOf(t.lambda),
        };
        for ([_]Tensor{ t.input_norm, t.post_attn, t.pre_ffn, t.post_ffn, t.q_norm, t.k_norm }, o) |x, *at| at.* = try small.put(gpa, x);
    }
    const base = try w.upload(d, small.bytes.items);
    w.norm = base + at_norm;
    for (w.layers, offsets) |*l, o| {
        l.input_norm = base + o[0];
        l.post_attn = base + o[1];
        l.pre_ffn = base + o[2];
        l.post_ffn = base + o[3];
        l.q_norm = base + o[4];
        l.k_norm = base + o[5];
    }
    return w;
}

/// An EXL3 pack's projection onto the device as stored (trellis, suh, svh).
fn exl3Proj(w: *Weights, d: *const cuda.Driver, t: x3.Stored) !x3.X3 {
    return x3.weight(try w.upload(d, t.trellis.bytes), try w.upload(d, t.suh.bytes), try w.upload(d, t.svh.bytes), t.n, t.k, t.halves);
}

/// An EXL3 pack (exllamav3, mul1) mapped, then onto the GPU: trellises for the projections and head, the rest bf16.
fn loadExl3(gpa: std.mem.Allocator, d: *const cuda.Driver, ck: *Checkpoint, c: Config) !Weights {
    var m = try x3.map(gpa, ck, c);
    defer m.deinit();
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    w.layers = try gpa.alloc(Layer, c.layers);
    w.embed = try w.upload(d, m.embed.bytes);
    w.head = .{ .exl3 = try exl3Proj(&w, d, m.head) };
    var small: Small = .{};
    defer small.bytes.deinit(gpa);
    const at_norm = try small.put(gpa, m.norm);
    const offsets = try gpa.alloc([6]usize, c.layers);
    defer gpa.free(offsets);
    for (w.layers, m.layers, offsets) |*l, t, *o| {
        var p: [8]x3.X3 = undefined;
        for (t.proj, &p) |sp, *x| x.* = try exl3Proj(&w, d, sp);
        l.* = .{ .qkg = .{ .exl3 = p[0..3].* }, .v = .{ .exl3 = p[3] }, .o = .{ .exl3 = p[4] }, .gate = .{ .exl3 = p[5] }, .up = .{ .exl3 = p[6] }, .down = .{ .exl3 = p[7] }, .input_norm = 0, .post_attn = 0, .pre_ffn = 0, .post_ffn = 0, .q_norm = 0, .k_norm = 0, .blend = blendOf(t.lambda) };
        for (t.norms, o) |x, *at| at.* = try small.put(gpa, x);
    }
    const base = try w.upload(d, small.bytes.items);
    w.norm = base + at_norm;
    for (w.layers, offsets) |*l, o| {
        l.input_norm = base + o[0];
        l.post_attn = base + o[1];
        l.pre_ffn = base + o[2];
        l.post_ffn = base + o[3];
        l.q_norm = base + o[4];
        l.k_norm = base + o[5];
    }
    return w;
}

const testing = std.testing;
const tio = testing.io;

/// The release's config (the fixture), as its NVFP4 quantization_config sets it.
fn release() !Config {
    var c = try cfg.parse(testing.allocator, @embedFile("fixtures/config.json"));
    c.fp4_mlp = true;
    return c;
}

test "ricursive cuda gate: the NVFP4 release passes, the BF16 checkpoint and unbuilt shapes are refused" {
    var c = try release();
    try gate(&c);
    try testing.expectEqual(@as(usize, 21504), qkgWidth(c));
    c.fp4_mlp = false;
    try testing.expectError(error.RicursiveCudaNeedsNvfp4, gate(&c));
    c.exl3 = true; // an EXL3 pack of the BF16 checkpoint
    try gate(&c);
    c = try release();
    c.kv_heads = 16; // group 5: the prompt kernel pairs query heads
    try testing.expectError(error.UnsupportedRicursiveCudaShape, gate(&c));
    c = try release();
    c.hidden = 10240; // past tf_rc_add_norm's 8,192
    try testing.expectError(error.UnsupportedRicursiveCudaShape, gate(&c));
}

test "ricursive cuda weights: the release's device bytes, about 57 GB, the planned 53.3 GiB" {
    const c = try release();
    const b = deviceBytes(c);
    try testing.expect(b > 57_000_000_000 and b < 57_400_000_000);
}

test "ricursive cuda blendOf: torch's bf16 sigmoid" {
    try testing.expectEqual(@as(f32, 0.5), blendOf(0x0000));
    try testing.expectEqual(@as(f32, 0.73046875), blendOf(0x3f80));
    try testing.expectEqual(@as(f32, 0.26953125), blendOf(0xbf80));
}

test "ricursive cuda fused q, k and g: one packing of the rows equals each part packed in its place" {
    const gpa = testing.allocator;
    const k = 128;
    const ns = [_]usize{ 256, 128, 256 };
    var total: usize = 0;
    for (ns) |n| total += n;
    const src = try gpa.alloc(u16, total * k);
    defer gpa.free(src);
    for (src, 0..) |*v, i| v.* = @truncate(i *% 2654435761);
    const whole = try gpa.alloc(u16, total * k);
    defer gpa.free(whole);
    cuda.qmmf.packBf16(whole, src, total, k);
    var at: usize = 0;
    const part = try gpa.alloc(u16, 256 * k);
    defer gpa.free(part);
    for (ns) |n| {
        cuda.qmmf.packBf16(part[0 .. n * k], src[at..][0 .. n * k], n, k);
        try testing.expectEqualSlices(u16, whole[at..][0 .. n * k], part[0 .. n * k]);
        at += n * k;
    }
}

/// A tiny Ripple: 5 layers (the fifth global), 4 query heads over 2 key heads.
fn tiny() Config {
    var c: Config = .{ .hidden = 128, .intermediate = 256, .layers = 5, .heads = 4, .kv_heads = 2, .vocab = 128, .eps = 1e-6, .theta = 10000, .window = 16, .embed_scale = 1, .clamp = 20, .context = 64, .fp4_mlp = true };
    c.global[4] = true;
    return c;
}

/// A safetensors file of every tensor `c` names (zeros, each global scale 1; `pack`: an EXL3 pack's, 5 bits).
fn writeFixture(a: std.mem.Allocator, dir: std.Io.Dir, c: Config, input_scales: bool, pack: bool) !void {
    const E = struct { name: []const u8, dtype: []const u8, shape: []const usize };
    var list: std.ArrayList(E) = .empty;
    const h = c.hidden;
    const Add = struct {
        fn bf(l: *std.ArrayList(E), al: std.mem.Allocator, name: []const u8, shape: []const usize) !void {
            try l.append(al, .{ .name = name, .dtype = "BF16", .shape = try al.dupe(usize, shape) });
        }
        fn fp4(l: *std.ArrayList(E), al: std.mem.Allocator, pre: []const u8, n: usize, k: usize, input: bool) !void {
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight", .{pre}), .dtype = "U8", .shape = try al.dupe(usize, &.{ n, k / 2 }) });
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale", .{pre}), .dtype = "F8_E4M3", .shape = try al.dupe(usize, &.{ n, k / 16 }) });
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale_2", .{pre}), .dtype = "F32", .shape = &.{} });
            if (input) try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.input_scale", .{pre}), .dtype = "F32", .shape = &.{} });
        }
        fn trellis(l: *std.ArrayList(E), al: std.mem.Allocator, pre: []const u8, n: usize, k: usize) !void {
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.trellis", .{pre}), .dtype = "I16", .shape = try al.dupe(usize, &.{ k / 16, n / 16, 80 }) });
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.suh", .{pre}), .dtype = "F16", .shape = try al.dupe(usize, &.{k}) });
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.svh", .{pre}), .dtype = "F16", .shape = try al.dupe(usize, &.{n}) });
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.mul1", .{pre}), .dtype = "I32", .shape = &.{} });
        }
        fn proj(l: *std.ArrayList(E), al: std.mem.Allocator, pre: []const u8, n: usize, k: usize, x: bool) !void {
            if (x) return trellis(l, al, pre, n, k);
            try bf(l, al, try std.fmt.allocPrint(al, "{s}.weight", .{pre}), &.{ n, k });
        }
    };
    try Add.bf(&list, a, "model.embed_tokens.weight", &.{ c.vocab, h });
    try Add.proj(&list, a, "lm_head", c.vocab, h, pack);
    try Add.bf(&list, a, "model.norm.weight", &.{h});
    for (0..c.layers) |i| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" }) |n| try Add.bf(&list, a, try std.fmt.allocPrint(a, "{s}{s}.weight", .{ p, n }), &.{h});
        for ([_][]const u8{ "q_proj", "g_proj" }) |n| try Add.proj(&list, a, try std.fmt.allocPrint(a, "{s}self_attn.{s}", .{ p, n }), c.queryWidth(), h, pack);
        for ([_][]const u8{ "k_proj", "v_proj" }) |n| try Add.proj(&list, a, try std.fmt.allocPrint(a, "{s}self_attn.{s}", .{ p, n }), c.kvWidth(), h, pack);
        try Add.proj(&list, a, try std.fmt.allocPrint(a, "{s}self_attn.o_proj", .{p}), h, c.queryWidth(), pack);
        for ([_][]const u8{ "q_norm", "k_norm" }) |n| try Add.bf(&list, a, try std.fmt.allocPrint(a, "{s}self_attn.{s}.weight", .{ p, n }), &.{cfg.head_dim});
        const lambda = try std.fmt.allocPrint(a, "{s}self_attn.value_res_lambda", .{p});
        if (pack) try list.append(a, .{ .name = lambda, .dtype = "F32", .shape = try a.dupe(usize, &.{1}) }) else try Add.bf(&list, a, lambda, &.{1});
        const mlp = [_]struct { []const u8, usize, usize }{ .{ "gate_proj", c.intermediate, h }, .{ "up_proj", c.intermediate, h }, .{ "down_proj", h, c.intermediate } };
        for (mlp) |x| {
            const pre = try std.fmt.allocPrint(a, "{s}mlp.{s}", .{ p, x[0] });
            if (pack) try Add.trellis(&list, a, pre, x[1], x[2]) else try Add.fp4(&list, a, pre, x[1], x[2], input_scales);
        }
    }
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{");
    var at: usize = 0;
    const sizes = try a.alloc(usize, list.items.len);
    for (list.items, sizes, 0..) |e, *size, j| {
        var n: usize = if (e.dtype[0] == 'B' or std.mem.endsWith(u8, e.dtype, "16")) 2 else if (std.mem.endsWith(u8, e.dtype, "32")) 4 else 1;
        for (e.shape) |x| n *= x;
        size.* = n;
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (j == 0) "" else ",", e.name, e.dtype });
        for (e.shape, 0..) |x, i| try header.writer.print("{s}{d}", .{ if (i == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        at += n;
    }
    try header.writer.writeAll("}");
    const hd = header.written();
    const file = try a.alloc(u8, 8 + hd.len + at);
    @memset(file, 0);
    std.mem.writeInt(u64, file[0..8], hd.len, .little);
    @memcpy(file[8..][0..hd.len], hd);
    var off: usize = 0;
    for (list.items, sizes) |e, n| {
        if (std.mem.endsWith(u8, e.name, "weight_scale_2")) std.mem.writeInt(u32, file[8 + hd.len + off ..][0..4], @bitCast(@as(f32, 1)), .little);
        if (std.mem.endsWith(u8, e.name, ".mul1")) std.mem.writeInt(u32, file[8 + hd.len + off ..][0..4], 0x83DCD12D, .little);
        off += n;
    }
    try dir.writeFile(tio, .{ .sub_path = "model.safetensors", .data = file });
}

fn tmpPath(a: std.mem.Allocator, tmp: testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(tio, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "ricursive cuda names: every tensor of a tiny NVFP4 release maps, input_scale taken and unread" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = tiny();
    for ([_]bool{ true, false }) |input| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeFixture(a, tmp.dir, c, input, false);
        var ck = try Checkpoint.openModel(testing.allocator, tio, try tmpPath(a, tmp));
        defer ck.close();
        var m = try map(testing.allocator, &ck, c);
        defer m.deinit();
        try testing.expectEqual(@as(usize, 5), m.layers.len);
        try testing.expectEqual(@as(f32, 1), m.layers[3].down.global);
        try testing.expectEqual(@as(usize, 128), m.layers[0].gate.k);
        try testing.expectEqual(@as(f32, 0.5), blendOf(m.layers[4].lambda));
    }
}

test "ricursive cuda exl3 names: every tensor of a tiny EXL3 pack maps, 5-bit trellises, lambda as bf16 bits" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = tiny();
    c.fp4_mlp = false;
    c.exl3 = true;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeFixture(a, tmp.dir, c, false, true);
    var ck = try Checkpoint.openModel(testing.allocator, tio, try tmpPath(a, tmp));
    defer ck.close();
    var m = try x3.map(testing.allocator, &ck, c);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 5), m.layers.len);
    try testing.expectEqual(@as(usize, 10), m.head.halves); // 80 i16 a 16x16 tile: 5 bits
    try testing.expectEqual(@as(usize, 512), m.layers[2].proj[0].n); // q
    try testing.expectEqual(@as(usize, 256), m.layers[2].proj[7].k); // down's input
    try testing.expectEqual(@as(f32, 0.5), blendOf(m.layers[4].lambda));
}
