//! z-lab's DFlash drafter for Gemma 4 31B: config admitted, bf16 linears quantized to the 4-bit lane layout at load.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A packed 4-bit lane linear, or (GEMMA_DRAFT_BF16, a check of the quantizer) the bf16 rows as stored, `sb` unused.
pub const Linear = struct { w: mtl.Buffer, sb: mtl.Buffer, n: usize, k: usize, bf16: bool = false };
pub const Ref = weights.Ref;

/// The stock drafter's geometry (z-lab/gemma-4-31B-it-DFlash); any other config is refused with its reason.
pub const Config = struct {
    taps: [8]u32 = @splat(0),
    tap_count: u32,
    layers: u32,
    hidden: u32,
    intermediate: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block: u32,
    mask: u32,
    window: u32,
    eps: f32,
    theta: f32,
    /// Each layer's kind: true for a sliding layer (causal within the block), false for a full one.
    sliding: [8]bool = @splat(true),

    /// `geo`: the target's sizes, which the drafter's width and taps must fit.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, geo: cfg.Geometry) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes, geo);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, geo: cfg.Geometry) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        const o = if (parsed.value == .object) parsed.value.object else return error.BadDraftConfig;
        const archs = o.get("architectures") orelse return error.BadDraftConfig;
        if (archs != .array or archs.array.items.len != 1 or archs.array.items[0] != .string or !std.mem.eql(u8, archs.array.items[0].string, "DFlashDraftModel")) return error.NotDFlashV1;
        const d = if (o.get("dflash_config")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        var c: Config = .{ .tap_count = 0, .layers = @intCast(try integer(o, "num_hidden_layers")), .hidden = @intCast(try integer(o, "hidden_size")), .intermediate = @intCast(try integer(o, "intermediate_size")), .heads = @intCast(try integer(o, "num_attention_heads")), .kv_heads = @intCast(try integer(o, "num_key_value_heads")), .head_dim = @intCast(try integer(o, "head_dim")), .block = @intCast(if (d.get("block_size") != null) try integer(d, "block_size") else try integer(o, "block_size")), .mask = @intCast(try integer(d, "mask_token_id")), .window = @intCast(try integer(o, "sliding_window")), .eps = @floatCast(try number(o, "rms_norm_eps")), .theta = @floatCast(try number(o, "rope_theta")) };
        if (c.hidden != geo.hidden) return error.DraftForAnotherTarget;
        const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > 8) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0) return error.BadDraftConfig;
            if (t.integer >= geo.layers) return error.DraftForAnotherTarget;
            c.taps[i] = @intCast(t.integer);
        }
        c.tap_count = @intCast(taps.array.items.len);
        if (c.layers == 0 or c.layers > 8) return error.BadDraftConfig;
        if (o.get("layer_types")) |kinds| {
            if (kinds != .array or kinds.array.items.len != c.layers) return error.BadDraftConfig;
            for (kinds.array.items, 0..) |k, i| c.sliding[i] = k == .string and std.mem.eql(u8, k.string, "sliding_attention");
        }
        if (o.get("is_causal") != null) return error.UnsupportedDraftMask; // the vendored default decides the block's mask
        if (c.head_dim != 128 or c.heads % c.kv_heads != 0 or c.heads > 128 or c.block != 16 or c.window < 32) return error.UnsupportedDraftShape;
        if (c.hidden % 64 != 0 or c.intermediate % 64 != 0 or (c.heads * c.head_dim) % 32 != 0 or (c.kv_heads * c.head_dim) % 64 != 0) return error.UnsupportedDraftShape;
        if (c.mask >= cfg.vocab) return error.BadDraftConfig;
        return c;
    }
};

fn integer(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return if (v == .integer) v.integer else error.BadDraftConfig;
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

pub const Layer = struct {
    /// q, k and v stacked (the block's rows), and k and v alone (context rows at absorb).
    qkv: Linear,
    kv: Linear,
    o: Linear,
    gate_up: Linear,
    down: Linear,
    input_norm: Ref,
    post_attention: Ref,
    q_norm: Ref,
    k_norm: Ref,
};

pub const Weights = struct {
    fc: Linear,
    hidden_norm: Ref,
    norm: Ref,
    layers: [8]Layer,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }
};

/// MLX's affine 4-bit codes of 64 bf16 values: bias the larger edge, scale |edge| / round(|edge| / ((max - min) / 15)).
fn quantizeGroup(x: []const u16, words: []u32, scale: *u16, bias: *u16) void {
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    for (x) |v| {
        const f = bf16ToF32(v);
        lo = @min(lo, f);
        hi = @max(hi, f);
    }
    const bins: f32 = 15;
    const scale0 = @max(hi - lo, 1e-7) / bins;
    const from_min = @abs(lo) > @abs(hi);
    const edge = if (from_min) lo else hi;
    const q0 = @round(@abs(edge) / scale0);
    const s_f: f32 = if (q0 == 0) scale0 else (if (from_min) @abs(edge) else -@abs(edge)) / q0;
    scale.* = f32ToBf16(s_f);
    bias.* = f32ToBf16(if (q0 == 0) lo else edge);
    const s = bf16ToF32(scale.*);
    const b = bf16ToF32(bias.*);
    @memset(words, 0);
    for (x, 0..) |v, i| {
        const q = std.math.clamp(@round((bf16ToF32(v) - b) / s), 0, bins);
        words[i / 8] |= @as(u32, @intFromFloat(q)) << @intCast((i % 8) * 4);
    }
}

fn bf16ToF32(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Round to nearest even, as MLX casts.
fn f32ToBf16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    if ((bits & 0x7f80_0000) == 0x7f80_0000) return @truncate(bits >> 16);
    const round = 0x7fff + ((bits >> 16) & 1);
    return @truncate((bits + round) >> 16);
}

/// A thread's share of a tensor's rows quantized.
const Job = struct {
    src: []const u16,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,
    fn run(j: Job, from: usize, to: usize) void {
        const groups = j.k / 64;
        for (from..to) |r| for (0..groups) |g| {
            quantizeGroup(j.src[r * j.k + g * 64 ..][0..64], j.words[(r * groups + g) * 8 ..][0..8], &j.scales[r * groups + g], &j.biases[r * groups + g]);
        };
    }
};

/// The members' bf16 rows stacked along N, quantized to 4-bit groups of 64 on every core, packed for the reg kernel.
fn quantizeStack(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8) !Linear {
    _ = try stackShape(c, prefix, members);
    const bf16_env = std.c.getenv("GEMMA_DRAFT_BF16");
    if (bf16_env != null and (std.mem.eql(u8, std.mem.span(bf16_env.?), "all") or std.mem.eql(u8, std.mem.span(bf16_env.?), members[0]))) return bf16Stack(gpa, device, c, owned, prefix, members);
    return quantized(gpa, device, c, owned, geo, prefix, members);
}

/// The stack's rows and columns: bf16 matrices of one width, n % 32 and k % 64 for the lane layout.
fn stackShape(c: *const ckpt.Checkpoint, prefix: []const u8, members: []const []const u8) ![2]usize {
    var name: [160]u8 = undefined;
    var n: usize = 0;
    var k: usize = 0;
    for (members) |m| {
        const t = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }));
        if (t.dtype != .bf16 or t.rank != 2) return error.BadDraftWeight;
        if (k != 0 and k != t.shape[1]) return error.BadDraftWeight;
        n += t.shape[0];
        k = t.shape[1];
    }
    if (k % 64 != 0 or n % 32 != 0) return error.BadDraftWeight;
    return .{ n, k };
}

/// The members' bf16 rows stacked along N, always 4-bit: DFlash's by default, the MTP assistant's linears and head.
pub fn quantized(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), geo: cfg.Geometry, prefix: []const u8, members: []const []const u8) !Linear {
    var name: [160]u8 = undefined;
    const n, const k = try stackShape(c, prefix, members);
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, n * k / 64);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * k / 64);
    defer gpa.free(biases);
    var row: usize = 0;
    for (members) |m| {
        const t = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }));
        const rows = t.shape[0];
        const job = Job{ .src = t.host(u16)[0 .. rows * k], .k = k, .words = words[row * k / 8 ..][0 .. rows * k / 8], .scales = scales[row * k / 64 ..][0 .. rows * k / 64], .biases = biases[row * k / 64 ..][0 .. rows * k / 64] };
        try parallel(gpa, job, rows);
        row += rows;
    }
    const lin = try weights.packArrays(gpa, device, owned, geo, words, scales, biases, n, k);
    return .{ .w = lin.w, .sb = lin.sb, .n = n, .k = k };
}

/// The members' bf16 rows stacked along N as stored (the slow per-row kernel reads them).
fn bf16Stack(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), prefix: []const u8, members: []const []const u8) !Linear {
    var name: [160]u8 = undefined;
    var n: usize = 0;
    var k: usize = 0;
    for (members) |m| {
        const t = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }));
        n += t.shape[0];
        k = t.shape[1];
    }
    const buf = try device.buffer(n * k * 2, shared);
    errdefer buf.deinit();
    var at: usize = 0;
    for (members) |m| {
        const t = try c.get(try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, m }));
        @memcpy(buf.slice(u16, n * k)[at..][0 .. t.shape[0] * k], t.host(u16)[0 .. t.shape[0] * k]);
        at += t.shape[0] * k;
    }
    try owned.append(gpa, buf);
    return .{ .w = buf, .sb = buf, .n = n, .k = k, .bf16 = true };
}

/// `job` over its rows on every CPU.
fn parallel(gpa: std.mem.Allocator, job: Job, rows: usize) !void {
    const threads = @min(std.Thread.getCpuCount() catch 1, 16);
    const per = (rows + threads - 1) / threads;
    const handles = try gpa.alloc(std.Thread, threads);
    defer gpa.free(handles);
    var spawned: usize = 0;
    var at: usize = 0;
    while (at < rows) : (at += per) {
        handles[spawned] = try std.Thread.spawn(.{}, Job.run, .{ job, at, @min(rows, at + per) });
        spawned += 1;
    }
    for (handles[0..spawned]) |h| h.join();
}

/// A bf16 vector copied into its own buffer (the checkpoint's buffers are released after load).
pub fn vector(gpa: std.mem.Allocator, device: mtl.Device, c: *const ckpt.Checkpoint, owned: *std.ArrayList(mtl.Buffer), name: []const u8, len: usize) !Ref {
    const t = try c.get(name);
    if (t.dtype != .bf16 or t.count() != len) return error.BadDraftWeight;
    const buf = try device.buffer(len * 2, shared);
    errdefer buf.deinit();
    @memcpy(buf.slice(u16, len), t.host(u16)[0..len]);
    try owned.append(gpa, buf);
    return .{ .buf = buf };
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, io: std.Io, dir: []const u8, c: *const Config, geo: cfg.Geometry) !Weights {
    var check = ckpt.Checkpoint.init(gpa);
    defer check.deinit();
    const files = try shards.shardFiles(gpa, io, dir);
    defer shards.freeShardFiles(gpa, files);
    for (files) |path| try check.addFile(device, path, "");
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    var p: [64]u8 = undefined;
    var name: [160]u8 = undefined;
    w.fc = try quantizeStack(gpa, device, &check, &w.owned, geo, "", &.{"fc"});
    if (w.fc.k != c.tap_count * c.hidden or w.fc.n != c.hidden) return error.BadDraftWeight;
    w.hidden_norm = try vector(gpa, device, &check, &w.owned, "hidden_norm.weight", c.hidden);
    w.norm = try vector(gpa, device, &check, &w.owned, "norm.weight", c.hidden);
    for (w.layers[0..c.layers], 0..) |*l, i| {
        const pre = try std.fmt.bufPrint(&p, "layers.{d}.", .{i});
        l.qkv = try quantizeStack(gpa, device, &check, &w.owned, geo, pre, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" });
        l.kv = try quantizeStack(gpa, device, &check, &w.owned, geo, pre, &.{ "self_attn.k_proj", "self_attn.v_proj" });
        l.o = try quantizeStack(gpa, device, &check, &w.owned, geo, pre, &.{"self_attn.o_proj"});
        l.gate_up = try quantizeStack(gpa, device, &check, &w.owned, geo, pre, &.{ "mlp.gate_proj", "mlp.up_proj" });
        l.down = try quantizeStack(gpa, device, &check, &w.owned, geo, pre, &.{"mlp.down_proj"});
        if (l.qkv.n != (c.heads + 2 * c.kv_heads) * c.head_dim or l.o.k != c.heads * c.head_dim or l.gate_up.n != 2 * c.intermediate or l.down.k != c.intermediate) return error.BadDraftWeight;
        inline for (.{ "input_norm", "post_attention", "q_norm", "k_norm" }, .{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight" }, .{ c.hidden, c.hidden, c.head_dim, c.head_dim }) |field, t, len|
            @field(l, field) = try vector(gpa, device, &check, &w.owned, try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, t }), len);
    }
    return w;
}

test "the stock drafter config is admitted and others refused with a reason" {
    const stock =
        \\{"architectures":["DFlashDraftModel"],"dflash_config":{"mask_token_id":4,"target_layer_ids":[1,12,23,35,46,57]},"block_size":16,
        \\"hidden_size":5376,"intermediate_size":10752,"num_hidden_layers":5,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,
        \\"sliding_window":2048,"rms_norm_eps":1e-6,"rope_theta":1000000,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]}
    ;
    const c = try Config.parse(std.testing.allocator, stock, cfg.dense31b);
    try std.testing.expectEqual(@as(u32, 6), c.tap_count);
    try std.testing.expectEqual(@as(u32, 57), c.taps[5]);
    try std.testing.expect(c.sliding[3] and !c.sliding[4]);
    const two = try std.mem.replaceOwned(u8, std.testing.allocator, stock, "DFlashDraftModel", "DFlash2DraftModel");
    defer std.testing.allocator.free(two);
    try std.testing.expectError(error.NotDFlashV1, Config.parse(std.testing.allocator, two, cfg.dense31b));
    const causal = try std.mem.replaceOwned(u8, std.testing.allocator, stock, "\"hidden_size\"", "\"is_causal\":false,\"hidden_size\"");
    defer std.testing.allocator.free(causal);
    try std.testing.expectError(error.UnsupportedDraftMask, Config.parse(std.testing.allocator, causal, cfg.dense31b));
    try std.testing.expectError(error.DraftForAnotherTarget, Config.parse(std.testing.allocator, stock, cfg.unified12b));
}

test "4-bit quantization keeps the larger edge exact and zero within the scale's bf16 rounding, as MLX does" {
    var x: [64]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = f32ToBf16(@as(f32, @floatFromInt(i)) * 0.125 - 2.0); // -2.0 .. 5.875: the max is the edge
    var words: [8]u32 = undefined;
    var s: u16 = undefined;
    var b: u16 = undefined;
    quantizeGroup(&x, &words, &s, &b);
    const scale = bf16ToF32(s);
    const bias = bf16ToF32(b);
    try std.testing.expect(scale < 0 and bias == 5.875); // codes descend from the max
    const q63: u32 = (words[7] >> 28) & 0xf;
    try std.testing.expectEqual(@as(u32, 0), q63);
    const q16: u32 = words[2] & 0xf; // x[16] = 0.0 lands on a code, off by the bf16 rounding of the scale times the code
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scale * @as(f32, @floatFromInt(q16)) + bias, 0.05);
    try std.testing.expect(@abs(scale * @as(f32, @floatFromInt(q16)) + bias) < @abs(scale) / 2);
}
