//! The DSpark drafter's weights: the released bf16 linears as stored, or on request quantized to 8-bit affine codes.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const dc = @import("dspark_config.zig");
const quant = @import("dspark_quant.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const tile = @import("../../core/tile_qmm.zig");

const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// An [n, k] bf16 matrix as released (rows only reordered for half-split RoPE where the checkpoint is read).
pub const Bf16 = struct { w: mtl.Buffer, n: usize, k: usize };

/// A linear: bf16 as stored (bits 0), or MLX affine codes at the loader's width.
pub const Linear = union(enum) { bf16: Bf16, codes: tile.Matrix };

/// A bf16 vector in its own buffer (norm gains; RoPE-paired ones in the half-split order the kernels rotate).
pub const Vector = mtl.Buffer;

pub const Layer = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    input_norm: Vector,
    post_norm: Vector,
    q_norm: Vector,
    k_norm: Vector,
};

pub const Weights = struct {
    fc: Linear,
    hidden_norm: Vector,
    norm: Vector,
    layers: [8]Layer,
    /// The Markov head: W1 [vocab, rank] bf16 as stored (rows gathered by token), W2 [vocab, rank] as the linears.
    w1: mtl.Buffer,
    w2: Linear,
    /// The confidence head: its [hidden + rank] bf16 weights and fp32 bias (null: the checkpoint has none).
    confidence: ?mtl.Buffer = null,
    confidence_bias: f32 = 0,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }

    /// Every buffer the weights hold, for a residency set.
    pub fn buffers(w: *const Weights) []const mtl.Buffer {
        return w.owned.items;
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    check: *const ckpt.Checkpoint,
    stage: *ckpt.Stage, // a matrix read from the file, one at a time, into one reused buffer
    owned: *std.ArrayList(mtl.Buffer),
    bits: u8,

    fn buffer(l: Loader, bytes: usize) !mtl.Buffer {
        const b = try l.device.buffer(bytes, shared);
        errdefer b.deinit();
        try l.owned.append(l.gpa, b);
        return b;
    }

    /// Check deferred metadata before allocating or reading a matrix's payload.
    fn validate(l: Loader, name: []const u8, shape: []const usize) !void {
        const valid = if (l.check.deferred.get(name)) |t| matches(t, shape) else matches(try l.check.get(name), shape);
        if (!valid) {
            std.log.err("{s}: unexpected DSpark tensor dtype or shape", .{name});
            return error.BadDraftWeight;
        }
    }

    fn matches(t: anytype, shape: []const usize) bool {
        return t.dtype == .bf16 and t.rank == shape.len and std.mem.eql(usize, t.shape[0..t.rank], shape);
    }

    /// A vector as loaded, or one staged matrix, valid until the next matrix read.
    fn tensor(l: Loader, name: []const u8, shape: []const usize) !ckpt.Tensor {
        try l.validate(name, shape);
        l.stage.reset();
        if (shape.len == 2) try l.stage.reserve(l.check, &.{name});
        return if (shape.len == 2) l.stage.read(l.check, name) else l.check.get(name);
    }

    /// A bf16 vector copied out, its pairs reordered for half-split RoPE when `paired`.
    fn vector(l: Loader, name: []const u8, len: usize, paired: bool) !Vector {
        const src = (try l.tensor(name, &.{len})).host(u16);
        const b = try l.buffer(len * 2);
        const dst = b.slice(u16, len);
        for (dst, 0..) |*v, i| v.* = src[if (paired) quant.halfSplit(i, len) else i];
        return b;
    }

    /// Unpaired BF16 reads into its final backing; paired rows copy from the one reusable stage.
    fn bf16(l: Loader, name: []const u8, n: usize, k: usize, paired: bool) !Bf16 {
        try l.validate(name, &.{ n, k });
        const b = try l.buffer(n * k * 2);
        if (paired) {
            l.stage.reset();
            try l.stage.reserve(l.check, &.{name});
            const rows = (try l.stage.read(l.check, name)).host(u16);
            const dst = b.slice(u16, n * k);
            for (0..n) |r| @memcpy(dst[r * k ..][0..k], rows[quant.halfSplit(r, c.head_dim) * k ..][0..k]);
        } else {
            try l.check.read(name, 0, b.contents()[0 .. n * k * 2]);
        }
        return .{ .w = b, .n = n, .k = k };
    }

    /// An [n, k] bf16 matrix as it is or as affine codes; rows reordered for half-split RoPE when `paired`.
    fn matrix(l: Loader, name: []const u8, n: usize, k: usize, paired: bool) !Linear {
        if (l.bits == 0) return .{ .bf16 = try l.bf16(name, n, k, paired) };
        const rows = (try l.tensor(name, &.{ n, k })).host(u16);
        const codes = try l.buffer(n * k * l.bits / 8);
        const scales = try l.buffer(n * k / 64 * 2);
        const biases = try l.buffer(n * k / 64 * 2);
        const job: quant.Job = .{ .src = rows, .k = k, .bits = l.bits, .pair_dim = if (paired) c.head_dim else 0, .codes = codes.slice(u8, n * k * l.bits / 8), .scales = scales.slice(u16, n * k / 64), .biases = biases.slice(u16, n * k / 64) };
        try job.all(n);
        return .{ .codes = .{ .w = codes, .scales = scales, .biases = biases, .n = n, .k = k, .bits = l.bits } };
    }
};

fn part(buf: []u8, layer: usize, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "layers.{d}.{s}.weight", .{ layer, name });
}

/// Matrices stay in the file until the loader reads them; vectors load with the header.
fn matrices(_: []const u8, rank: usize) bool {
    return rank == 2;
}

/// The checkpoint read as released (`bits` 0) or quantized to 8-bit codes, a matrix at a time; buffers freed after.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, io: std.Io, dir: []const u8, cfg: *const dc.Config, bits: u8) !Weights {
    if (bits != 0 and bits != 8) return error.UnsupportedDraftBits;
    var check = ckpt.Checkpoint.init(gpa);
    defer check.deinit();
    var stage: ckpt.Stage = .{ .device = device };
    defer stage.deinit();
    const files = try shards.shardFiles(gpa, io, dir);
    defer shards.freeShardFiles(gpa, files);
    for (files) |path| try check.addFileSplit(device, path, "", null, matrices);
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    const l: Loader = .{ .gpa = gpa, .device = device, .check = &check, .stage = &stage, .owned = &w.owned, .bits = bits };
    const qw = c.query_heads * c.head_dim;
    const kw = c.kv_heads * c.head_dim;
    w.fc = try l.matrix("fc.weight", c.hidden, cfg.tap_count * c.hidden, false);
    w.hidden_norm = try l.vector("hidden_norm.weight", c.hidden, false);
    w.norm = try l.vector("norm.weight", c.hidden, false);
    var name: [128]u8 = undefined;
    for (w.layers[0..cfg.layers], 0..) |*ly, i| {
        ly.q = try l.matrix(try part(&name, i, "self_attn.q_proj"), qw, c.hidden, true);
        ly.k = try l.matrix(try part(&name, i, "self_attn.k_proj"), kw, c.hidden, true);
        ly.v = try l.matrix(try part(&name, i, "self_attn.v_proj"), kw, c.hidden, false);
        ly.o = try l.matrix(try part(&name, i, "self_attn.o_proj"), c.hidden, qw, false);
        ly.gate = try l.matrix(try part(&name, i, "mlp.gate_proj"), cfg.intermediate, c.hidden, false);
        ly.up = try l.matrix(try part(&name, i, "mlp.up_proj"), cfg.intermediate, c.hidden, false);
        ly.down = try l.matrix(try part(&name, i, "mlp.down_proj"), c.hidden, cfg.intermediate, false);
        ly.input_norm = try l.vector(try part(&name, i, "input_layernorm"), c.hidden, false);
        ly.post_norm = try l.vector(try part(&name, i, "post_attention_layernorm"), c.hidden, false);
        ly.q_norm = try l.vector(try part(&name, i, "self_attn.q_norm"), c.head_dim, true);
        ly.k_norm = try l.vector(try part(&name, i, "self_attn.k_norm"), c.head_dim, true);
    }
    w.w1 = (try l.bf16("markov_head.markov_w1.weight", c.vocab, cfg.rank, false)).w;
    w.w2 = try l.matrix("markov_head.markov_w2.weight", c.vocab, cfg.rank, false);
    w.confidence = null;
    w.confidence_bias = 0;
    if (check.has("confidence_head.proj.weight")) {
        const width = c.hidden + cfg.rank; // the final norm's row, then W1 of the token before the draft
        w.confidence = (try l.bf16("confidence_head.proj.weight", 1, width, false)).w;
        w.confidence_bias = quant.toF32((try l.tensor("confidence_head.proj.bias", &.{1})).host(u16)[0]);
    }
    return w;
}

fn fixture(gpa: std.mem.Allocator, tmp: *std.testing.TmpDir) ![:0]u8 {
    const header =
        \\{"matrix":{"dtype":"BF16","shape":[128,128],"data_offsets":[0,32768]},
        \\ "gain":{"dtype":"BF16","shape":[64],"data_offsets":[32768,32896]}}
    ;
    var data: [8 + header.len + 32896]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    for (0..32896 / 2) |i| {
        const value = quant.toBf16(@as(f32, @floatFromInt((i * 71 + i / 128 * 19) % 257)) * 0.0625 - 5);
        std.mem.writeInt(u16, data[8 + header.len + i * 2 ..][0..2], value, .little);
    }
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "weights.safetensors", .data = &data });
    return std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/weights.safetensors", .{tmp.sub_path}, 0);
}

test "lfm2 dspark: deferred BF16 and codes keep released rows and paired gains" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(gpa, &tmp);
    defer gpa.free(path);
    var check = ckpt.Checkpoint.init(gpa);
    defer check.deinit();
    try check.addFileSplit(device, path, "", null, matrices);
    var stage: ckpt.Stage = .{ .device = device };
    defer stage.deinit();
    var w: Weights = undefined;
    w.owned = .empty;
    defer w.deinit(gpa);
    var l: Loader = .{ .gpa = gpa, .device = device, .check = &check, .stage = &stage, .owned = &w.owned, .bits = 0 };
    var source: [128 * 128]u16 = undefined;
    try check.read("matrix", 0, std.mem.sliceAsBytes(&source));
    for ([_]u8{ 0, 8 }) |bits| {
        l.bits = bits;
        for ([_]bool{ false, true }) |paired| {
            const loaded = try l.matrix("matrix", 128, 128, paired);
            const gain = try l.vector("gain", c.head_dim, paired);
            const gain_src = (try check.get("gain")).host(u16);
            for (0..128) |r| {
                const from = if (paired) quant.halfSplit(r, c.head_dim) else r;
                const row = source[from * 128 ..][0..128];
                switch (loaded) {
                    .bf16 => |bf| try std.testing.expectEqualSlices(u16, row, bf.w.slice(u16, 128 * 128)[r * 128 ..][0..128]),
                    .codes => |q| for (0..2) |g| {
                        var codes: [64]u8 = undefined;
                        var scale: u16 = undefined;
                        var bias: u16 = undefined;
                        quant.group(row[g * 64 ..][0..64], bits, &codes, &scale, &bias);
                        try std.testing.expectEqualSlices(u8, &codes, q.w.slice(u8, 128 * 128)[r * 128 + g * 64 ..][0..64]);
                        try std.testing.expectEqual(scale, q.scales.slice(u16, 256)[r * 2 + g]);
                        try std.testing.expectEqual(bias, q.biases.slice(u16, 256)[r * 2 + g]);
                    },
                }
                if (r < c.head_dim) try std.testing.expectEqual(gain_src[if (paired) quant.halfSplit(r, c.head_dim) else r], gain.slice(u16, c.head_dim)[r]);
            }
        }
        // W1 and the confidence projection stay BF16 even when other matrices use codes.
        const raw = try l.bf16("matrix", 128, 128, false);
        try std.testing.expectEqualSlices(u16, &source, raw.w.slice(u16, source.len));
    }
}

test "lfm2 dspark: tensor admission checks deferred dtype and shape before payload access" {
    var t: ckpt.Deferred = .{ .file = 0, .at = 0, .bytes = 128 * 128 * 2, .dtype = .bf16, .rank = 2 };
    t.shape[0] = 128;
    t.shape[1] = 128;
    try std.testing.expect(Loader.matches(t, &.{ 128, 128 }));
    try std.testing.expect(!Loader.matches(t, &.{ 64, 256 }));
    try std.testing.expect(!Loader.matches(t, &.{128}));
    t.dtype = .f16;
    try std.testing.expect(!Loader.matches(t, &.{ 128, 128 }));
}

test "lfm2 dspark: failed ownership and short deferred reads unwind final buffers" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try fixture(gpa, &tmp);
    defer gpa.free(path);
    var check = ckpt.Checkpoint.init(gpa);
    defer check.deinit();
    try check.addFileSplit(device, path, "", null, matrices);
    var stage: ckpt.Stage = .{ .device = device };
    defer stage.deinit();
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer owned.deinit(failing.allocator());
    const l: Loader = .{ .gpa = failing.allocator(), .device = device, .check = &check, .stage = &stage, .owned = &owned, .bits = 0 };
    try std.testing.expectError(error.OutOfMemory, l.matrix("matrix", 128, 128, false));
    try std.testing.expectEqual(@as(usize, 0), owned.items.len);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "weights.safetensors", .data = "" });
    for ([_]u8{ 0, 8 }) |bits| for ([_]bool{ false, true }) |paired| {
        var w: Weights = undefined;
        w.owned = .empty;
        defer w.deinit(gpa);
        const short: Loader = .{ .gpa = gpa, .device = device, .check = &check, .stage = &stage, .owned = &w.owned, .bits = bits };
        try std.testing.expectError(error.ShortRead, short.matrix("matrix", 128, 128, paired));
    };
}
