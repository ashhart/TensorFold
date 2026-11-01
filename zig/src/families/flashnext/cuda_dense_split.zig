//! Whether each of two ranks keeps half of the BF16 dense projections, and what a rank's half holds on the device.
const std = @import("std");
const cuda = @import("cuda");
const safetensors = @import("core").safetensors;
const Tensor = safetensors.Tensor;
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const Axis = @import("cuda_projection_load.zig").Axis;
const Cut = @import("cuda_modelopt_experts.zig").Cut;
const Heads = @import("cuda_heads.zig").Heads;
const heads_cut = @import("heads_cut.zig");
const Entry = safetensors.Entry;

comptime {
    // DeltaNet's out_proj and attention's o_proj take the same input width, so one cut serves both
    std.debug.assert(gdn.value_dim == wt.out_k);
}

/// The tensors the split is decided from: layer 0's DeltaNet out projection and its shared expert's down projection.
const deciders = [_][]const u8{
    "language_model.model.layers.0.linear_attn.out_proj.weight",
    "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
};

/// TF_FLASHNEXT_SPLIT_DENSE: unset or 1 splits, 0 keeps every dense projection whole.
pub fn setting(value: ?[]const u8) !bool {
    const v = value orelse return true;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDenseSplit;
}

pub fn enabled() !bool {
    return setting(if (std.c.getenv("TF_FLASHNEXT_SPLIT_DENSE")) |v| std.mem.span(v) else null);
}

/// At two ranks with the setting on, a checkpoint whose deciders are BF16 splits; world 1 never reads the setting.
pub fn applies(store: anytype, world: usize) !bool {
    if (world != 2 or !try enabled()) return false;
    for (deciders) |name| if (!store.has(name)) return false;
    for (deciders) |name| if ((try store.tensor(name)).dtype != .bf16) return false;
    return true;
}

/// The heads a rank computes of every model layer: half of each under the split, else the whole layer.
pub fn heads(store: anytype, world: usize) !Heads {
    return if (try applies(store, world)) .half else .full;
}

pub fn appliesDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize) !bool {
    if (world != 2) return false;
    var store = try wt.Store.open(gpa, io, dir);
    defer store.deinit();
    return applies(&store, world);
}

/// How the split cuts a checkpoint projection by its name; null: kept whole on every rank.
pub fn axisOf(name: []const u8) ?Axis {
    for ([_][]const u8{ ".linear_attn.out_proj.weight", ".self_attn.o_proj.weight", ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight", ".mlp.shared_expert.down_proj.weight" }) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return .rows;
    }
    return null;
}

/// Device bytes of one rank's cut of a BF16 [n, k] projection, its rows padded as packBf16 pads them.
pub fn residentBytes(n: usize, k: usize, axis: Axis, world: usize) !u64 {
    const part = (try Cut.init(if (axis == .rows) n else k, world, 0)).width();
    const rows = if (axis == .rows) part else n;
    const cols = if (axis == .rows) k else part;
    return try std.math.mul(u64, try std.math.mul(u64, cuda.nvfp4.padded(rows), cols), 2);
}

/// One rank's device bytes of a tensor the head split cuts (in-projection parts, conv1d, A_log, dt_bias); null: none.
pub fn headBytes(name: []const u8, e: Entry, world: usize) !?u64 {
    const h = try Heads.split(world);
    const bytes: u64 = e.end - e.begin;
    inline for (.{ .{ ".linear_attn", heads_cut.delta_parts }, .{ ".self_attn", heads_cut.attention_parts } }) |class| {
        inline for (class[1], 0..) |part, i| if (std.mem.endsWith(u8, name, class[0] ++ part ++ ".weight")) {
            if (e.dtype != .bf16 or e.rank != 2) return null;
            const s = if (comptime std.mem.eql(u8, class[0], ".linear_attn")) try heads_cut.delta(h, 0) else try heads_cut.attention(h, 0);
            const k: u64 = e.dim(1);
            // the stitched weight packs as one: its rows past the last part's padded once, counted with the first
            const pad: u64 = if (i == 0) cuda.nvfp4.padded(s.n) - s.n else 0;
            return (s.ranges[i].rows() + pad) * k * 2;
        };
    }
    if (std.mem.endsWith(u8, name, ".linear_attn.conv1d.weight")) {
        if (e.rank == 0 or e.dim(0) != Heads.full.convDim() or bytes % e.dim(0) != 0) return error.UnexpectedTensor;
        return bytes / e.dim(0) * h.convDim();
    }
    for ([_][]const u8{ ".linear_attn.A_log", ".linear_attn.dt_bias" }) |suffix| if (std.mem.endsWith(u8, name, suffix)) {
        if (e.rank != 1 or e.dim(0) != Heads.full.nv or bytes % e.dim(0) != 0) return error.UnexpectedTensor;
        return bytes / e.dim(0) * h.nv;
    };
    return null;
}

/// A checkpoint of two tensors, each of one dtype: what `applies` reads.
const Fake = struct {
    dtypes: [2]safetensors.DType,
    present: bool = true,

    pub fn has(f: *const Fake, name: []const u8) bool {
        _ = name;
        return f.present;
    }

    pub fn tensor(f: *const Fake, name: []const u8) !Tensor {
        const i: usize = if (std.mem.eql(u8, name, deciders[0])) 0 else 1;
        return .{ .dtype = f.dtypes[i], .rank = 2, .shape = .{ 2560, 64, 1, 1, 1 }, .bytes = &.{} };
    }
};

test "the dense split setting is on unless 0 and refuses any other value" {
    try std.testing.expect(try setting(null));
    try std.testing.expect(try setting("1"));
    try std.testing.expect(!try setting("0"));
    try std.testing.expectError(error.BadDenseSplit, setting("2"));
    try std.testing.expectError(error.BadDenseSplit, setting(""));
}

test "the dense split applies at two ranks to BF16 layers only, and world 1 never reads the setting" {
    const env = @import("sys").env;
    defer _ = env.unsetenv("TF_FLASHNEXT_SPLIT_DENSE");
    _ = env.unsetenv("TF_FLASHNEXT_SPLIT_DENSE");
    const bf16: Fake = .{ .dtypes = .{ .bf16, .bf16 } };
    try std.testing.expect(try applies(&bf16, 2));
    try std.testing.expect(!try applies(&bf16, 1));
    try std.testing.expect(!try applies(&Fake{ .dtypes = .{ .f8_e4m3, .f8_e4m3 } }, 2));
    try std.testing.expect(!try applies(&Fake{ .dtypes = .{ .bf16, .f8_e4m3 } }, 2));
    try std.testing.expect(!try applies(&Fake{ .dtypes = .{ .f8_e4m3, .bf16 } }, 2));
    try std.testing.expect(!try applies(&Fake{ .dtypes = .{ .bf16, .bf16 }, .present = false }, 2));
    _ = env.setenv("TF_FLASHNEXT_SPLIT_DENSE", "0", 1);
    try std.testing.expect(!try applies(&bf16, 2));
    _ = env.setenv("TF_FLASHNEXT_SPLIT_DENSE", "1", 1);
    try std.testing.expect(try applies(&bf16, 2));
    _ = env.setenv("TF_FLASHNEXT_SPLIT_DENSE", "on", 1);
    try std.testing.expectError(error.BadDenseSplit, applies(&bf16, 2));
    try std.testing.expect(!try applies(&bf16, 1));
    try std.testing.expect(!try appliesDir(std.testing.allocator, std.testing.io, "/nonexistent", 1));
}

test "every model layer's heads: whole at one rank (the MTP head's too), half each at two under the split, whole without it" {
    const env = @import("sys").env;
    defer _ = env.unsetenv("TF_FLASHNEXT_SPLIT_DENSE");
    _ = env.unsetenv("TF_FLASHNEXT_SPLIT_DENSE");
    const bf16: Fake = .{ .dtypes = .{ .bf16, .bf16 } };
    // the MTP head's layer loads at world 1 (cuda_prompt_load.zig loadLayerAt), so its heads are the whole layer's
    try std.testing.expect((try heads(&bf16, 1)).eql(.full));
    try std.testing.expect((try heads(&bf16, 2)).eql(.half));
    try std.testing.expect((try heads(&Fake{ .dtypes = .{ .f8_e4m3, .f8_e4m3 } }, 2)).eql(.full));
    _ = env.setenv("TF_FLASHNEXT_SPLIT_DENSE", "0", 1);
    try std.testing.expect((try heads(&bf16, 2)).eql(.full));
    try std.testing.expect((try heads(&bf16, 1)).eql(.full));
}

test "the head split's per-rank bytes: stitched parts as packed once, half the conv channels, half the gate values" {
    const k = 2560;
    const bf = struct {
        fn entry(n: usize, cols: usize) Entry {
            return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, cols, 1, 1, 1 }, .begin = 0, .end = n * cols * 2 };
        }
    };
    const root = "model.language_model.layers.0.linear_attn";
    // 5,120 + 3,072 + 24 + 24 = 8,240 rows, padded once to 8,320: the 80 pad rows go with in_proj_qkv's 5,120
    try std.testing.expectEqual(@as(?u64, (5120 + 80) * k * 2), try headBytes(root ++ ".in_proj_qkv.weight", bf.entry(10240, k), 2));
    try std.testing.expectEqual(@as(?u64, 3072 * k * 2), try headBytes(root ++ ".in_proj_z.weight", bf.entry(6144, k), 2));
    try std.testing.expectEqual(@as(?u64, 24 * k * 2), try headBytes(root ++ ".in_proj_b.weight", bf.entry(48, k), 2));
    try std.testing.expectEqual(@as(?u64, 24 * k * 2), try headBytes(root ++ ".in_proj_a.weight", bf.entry(48, k), 2));
    const attn = "model.language_model.layers.3.self_attn";
    // 6,144 + 256 + 256 + 640 = 7,296 rows, a multiple of 128: no pad
    try std.testing.expectEqual(@as(?u64, 6144 * k * 2), try headBytes(attn ++ ".q_proj.weight", bf.entry(12288, k), 2));
    try std.testing.expectEqual(@as(?u64, 256 * k * 2), try headBytes(attn ++ ".k_proj.weight", bf.entry(512, k), 2));
    try std.testing.expectEqual(@as(?u64, 256 * k * 2), try headBytes(attn ++ ".v_proj.weight", bf.entry(512, k), 2));
    try std.testing.expectEqual(@as(?u64, 640 * k * 2), try headBytes(attn ++ ".indexer.index_qk_proj.weight", bf.entry(640, k), 2));
    const conv: Entry = .{ .dtype = .bf16, .rank = 3, .shape = .{ 10240, 1, 4, 1, 1 }, .begin = 0, .end = 10240 * 4 * 2 };
    try std.testing.expectEqual(@as(?u64, 5120 * 4 * 2), try headBytes(root ++ ".conv1d.weight", conv, 2));
    const gates: Entry = .{ .dtype = .f32, .rank = 1, .shape = .{ 48, 1, 1, 1, 1 }, .begin = 0, .end = 48 * 4 };
    try std.testing.expectEqual(@as(?u64, 24 * 4), try headBytes(root ++ ".A_log", gates, 2));
    try std.testing.expectEqual(@as(?u64, 24 * 4), try headBytes(root ++ ".dt_bias", gates, 2));
    // at one rank every share is the whole tensor
    try std.testing.expectEqual(@as(?u64, 10240 * 4 * 2), try headBytes(root ++ ".conv1d.weight", conv, 1));
    try std.testing.expectEqual(@as(?u64, 48 * 4), try headBytes(root ++ ".A_log", gates, 1));
    // whole on both ranks, or not BF16: not the head split's to price
    for ([_][]const u8{ root ++ ".norm.weight", attn ++ ".q_norm.weight", attn ++ ".indexer.k_layernorm.weight", attn ++ ".o_proj.weight", root ++ ".out_proj.weight" }) |name|
        try std.testing.expectEqual(@as(?u64, null), try headBytes(name, bf.entry(256, k), 2));
    var fp8 = bf.entry(10240, k);
    fp8.dtype = .f8_e4m3;
    try std.testing.expectEqual(@as(?u64, null), try headBytes(root ++ ".in_proj_qkv.weight", fp8, 2));
    try std.testing.expectError(error.UnexpectedTensor, headBytes(root ++ ".conv1d.weight", bf.entry(64, 4), 2));
    try std.testing.expectError(error.UnevenHeadSplit, headBytes(root ++ ".A_log", gates, 3));
}

test "the split cuts output rows for mixer and shared projections" {
    try std.testing.expectEqual(Axis.rows, axisOf("model.language_model.layers.0.linear_attn.out_proj.weight").?);
    try std.testing.expectEqual(Axis.rows, axisOf("model.language_model.layers.3.self_attn.o_proj.weight").?);
    try std.testing.expectEqual(@as(?Axis, null), axisOf("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    try std.testing.expectEqual(@as(?Axis, null), axisOf("model.language_model.layers.0.linear_attn.out_proj.weight_scale"));
    try std.testing.expectEqual(@as(?Axis, null), axisOf("model.language_model.layers.0.mlp.gate.weight"));
    try std.testing.expectEqual(@as(u64, 1280 * 6144 * 2), try residentBytes(2560, 6144, .rows, 2));
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |name| {
        var buf: [160]u8 = undefined;
        try std.testing.expectEqual(Axis.rows, axisOf(try std.fmt.bufPrint(&buf, "model.language_model.layers.0.mlp.shared_expert.{s}.weight", .{name})).?);
    }
    try std.testing.expectEqual(@as(u64, 384 * 2560 * 2), try residentBytes(640, 2560, .rows, 2));
    try std.testing.expectEqual(@as(u64, 2560 * 320 * 2), try residentBytes(2560, 640, .cols, 2));
    try std.testing.expectEqual(@as(u64, 2560 * 6144 * 2), try residentBytes(2560, 6144, .cols, 1));
    try std.testing.expectError(error.UnevenExpertSplit, residentBytes(640, 2560, .rows, 3));
    try std.testing.expectError(error.UnevenExpertSplit, residentBytes(2560, 6144, .cols, 0));
}
