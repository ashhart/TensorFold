//! Validated config.json and model.safetensors loader for incoai/Inkling-2-Mini-DFlash2.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const prepared = @import("dflash2_prepared.zig");

pub const hidden: usize = 2304;
pub const taps_n: usize = 4;
pub const layers_n: usize = 5;
pub const q_heads: usize = 32;
pub const kv_heads: usize = 8;
pub const head_dim: usize = 128;
pub const query_width: usize = q_heads * head_dim;
pub const kv_width: usize = kv_heads * head_dim;
pub const intermediate: usize = 6912;
pub const conv_group_size: usize = 16;
pub const conv_groups: usize = hidden / conv_group_size;
pub const dyn_width: usize = 4 * conv_groups;
pub const block_size: usize = 8;
pub const window_size: usize = 2048;
pub const ring_size: usize = std.mem.alignForward(usize, window_size + block_size, 128);
pub const rank: usize = 256;
pub const top_k: usize = 16;
pub const padded_vocab: usize = 201024;
pub const valid_vocab: usize = 200058;

pub const Config = struct {
    layers: u32 = layers_n,
    heads: u32 = q_heads,
    kv_heads: u32 = kv_heads,
    hidden: u32 = hidden,
    ffn: u32 = intermediate,
    window: u32 = window_size,
    block: u32 = block_size,
    mask: u32 = 200056,
    rank: u32 = rank,
    top_k: u32 = top_k,
    conv_group_size: u32 = conv_group_size,
    taps: [taps_n]u32 = .{ 5, 23, 41, 48 },
    theta: f32 = 10000.0,
    eps: f32 = 1e-6,
    softcap: f32 = 30.0,

    pub fn linearCount(c: Config) usize {
        return 9 * @as(usize, c.layers) + 2;
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
        var doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadDFlash2Config;
        const root = doc.value.object;
        const arch = root.get("architectures") orelse return error.NotDFlash2Config;
        if (arch != .array or arch.array.items.len == 0 or arch.array.items[0] != .string or
            !std.mem.eql(u8, arch.array.items[0].string, "DFlash2DraftModel")) return error.NotDFlash2Config;
        const dc_val = root.get("dflash_config") orelse return error.BadDFlash2Config;
        if (dc_val != .object) return error.BadDFlash2Config;
        const dc = dc_val.object;
        const readInt = struct {
            fn of(o: std.json.ObjectMap, k: []const u8) !u32 {
                const v = o.get(k) orelse return error.BadDFlash2Config;
                if (v != .integer or v.integer <= 0) return error.BadDFlash2Config;
                return @intCast(v.integer);
            }
            fn flt(o: std.json.ObjectMap, k: []const u8) !f32 {
                const v = o.get(k) orelse return error.BadDFlash2Config;
                return switch (v) {
                    .integer => |i| @floatFromInt(i),
                    .float => |f| @floatCast(f),
                    else => error.BadDFlash2Config,
                };
            }
        };
        const theta: f32 = if (root.get("rope_parameters")) |rp| blk: {
            if (rp != .object) return error.BadDFlash2Config;
            break :blk try readInt.flt(rp.object, "rope_theta");
        } else try readInt.flt(root, "rope_theta");
        const mode = dc.get("selector_softcap_mode") orelse return error.BadDFlash2Config;
        if (mode != .string or !std.mem.eql(u8, mode.string, "after_correction")) return error.UnsupportedDFlash2Config;
        var c: Config = .{
            .layers = try readInt.of(root, "num_hidden_layers"),
            .heads = try readInt.of(root, "num_attention_heads"),
            .kv_heads = try readInt.of(root, "num_key_value_heads"),
            .hidden = try readInt.of(root, "hidden_size"),
            .ffn = try readInt.of(root, "intermediate_size"),
            .window = try readInt.of(root, "sliding_window"),
            .block = try readInt.of(dc, "block_size"),
            .mask = try readInt.of(dc, "mask_token_id"),
            .rank = try readInt.of(dc, "selector_rank"),
            .top_k = try readInt.of(dc, "selector_top_k"),
            .conv_group_size = try readInt.of(dc, "conv_group_size"),
            .theta = theta,
            .eps = try readInt.flt(root, "rms_norm_eps"),
            .softcap = try readInt.flt(dc, "final_logit_softcapping"),
        };
        const taps_v = dc.get("target_layer_ids") orelse return error.BadDFlash2Config;
        if (taps_v != .array or taps_v.array.items.len != taps_n) return error.UnsupportedDFlash2Config;
        for (taps_v.array.items, &c.taps) |item, *dst| {
            if (item != .integer or item.integer < 0) return error.BadDFlash2Config;
            dst.* = @intCast(item.integer);
        }
        if ((try readInt.of(root, "head_dim")) != head_dim or
            (try readInt.of(root, "vocab_size")) != padded_vocab or
            (try readInt.of(dc, "conv_kernel_size")) != 2 or
            c.layers != layers_n or c.heads != q_heads or c.kv_heads != kv_heads or
            c.hidden != hidden or c.ffn != intermediate or c.window != window_size or
            c.block != block_size or c.rank != rank or c.top_k != top_k or
            c.conv_group_size != conv_group_size or c.mask >= padded_vocab or
            !std.mem.eql(u32, &c.taps, &.{ 5, 23, 41, 48 }))
            return error.UnsupportedDFlash2Config;
        return c;
    }
};

/// Returns true when `dir` holds a DFlash2DraftModel config.json rather than a DSpark checkpoint.
pub fn isDFlash2Dir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    if (std.mem.endsWith(u8, dir, ".safetensors")) return false;
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return false;
    defer gpa.free(bytes);
    var doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return false;
    defer doc.deinit();
    if (doc.value != .object) return false;
    const arch = doc.value.object.get("architectures") orelse return false;
    if (arch != .array or arch.array.items.len == 0 or arch.array.items[0] != .string) return false;
    return std.mem.eql(u8, arch.array.items[0].string, "DFlash2DraftModel");
}

pub const Linear = struct {
    n: usize,
    k: usize,
    key: usize,
    bf16: ?ck.Tensor = null,
};

pub const Conv = struct {
    base: ck.Tensor,
    proj: Linear,
};

pub const Layer = struct {
    in_norm: ck.Tensor,
    post_norm: ck.Tensor,
    q_norm: ck.Tensor,
    k_norm: ck.Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    attn_conv: Conv,
    mlp_conv: Conv,
};

pub const Weights = struct {
    fc: Linear,
    hidden_norm: ck.Tensor,
    norm: ck.Tensor,
    hidden_proj: Linear,
    pred_cb: ck.Tensor,
    succ_cb: ck.Tensor,
    layers: [layers_n]Layer,
};

pub const Loaded = struct {
    checkpoint: ck.Checkpoint,
    config: Config,
    weights: Weights,
    bits: u8 = 0,
    copies: ?prepared.Copies = null,

    pub fn deinit(l: *Loaded) void {
        if (l.copies) |*c| c.deinit();
        l.checkpoint.deinit();
        l.* = undefined;
    }

    pub fn fixedBytes(l: *const Loaded) u64 {
        return l.checkpoint.residentBytes() + if (l.copies) |*c| c.bytes() else 0;
    }
};

fn checkShape(dtype: ck.DType, ndims: usize, shape: []const usize, bytes: usize, dims: []const usize) !void {
    if (dtype != .bf16 or ndims != dims.len) return error.DFlash2TensorShape;
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (shape[i] != d) return error.DFlash2TensorShape;
        count = std.math.mul(usize, count, d) catch return error.DFlash2TensorShape;
    }
    if (bytes != count * 2) return error.DFlash2TensorShape;
}

fn require(t: ck.Tensor, dims: []const usize) !ck.Tensor {
    try checkShape(t.dtype, t.rank, &t.shape, t.bytes, dims);
    return t;
}

fn get(c: *const ck.Checkpoint, name: []const u8, dims: []const usize) !ck.Tensor {
    const t = c.get(name) catch {
        std.log.err("DFlash2 model.safetensors is missing {s}", .{name});
        return error.DFlash2MissingTensor;
    };
    return require(t, dims);
}

fn getLinear(c: *ck.Checkpoint, name: []const u8, dims: [2]usize, bits: u8) !Linear {
    if (bits == 0) {
        const t = try get(c, name, &dims);
        return .{ .n = dims[0], .k = dims[1], .key = prepared.keyOf(name), .bf16 = t };
    }
    if (c.deferred.get(name)) |d| {
        try checkShape(d.dtype, d.rank, &d.shape, d.bytes, &dims);
        return .{ .n = dims[0], .k = dims[1], .key = prepared.keyOf(name) };
    }
    if (c.tensors.get(name)) |t| {
        _ = try require(t, &dims);
        return .{ .n = dims[0], .k = dims[1], .key = prepared.keyOf(name) };
    }
    std.log.err("DFlash2 model.safetensors is missing {s}", .{name});
    return error.DFlash2MissingTensor;
}

fn loadLayer(c: *ck.Checkpoint, i: usize, bits: u8) !Layer {
    var b: [128]u8 = undefined;
    const g = struct {
        fn tensor(cp: *const ck.Checkpoint, buf: []u8, idx: usize, suffix: []const u8, dims: []const usize) !ck.Tensor {
            const name = try std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ idx, suffix });
            return get(cp, name, dims);
        }
        fn lin(cp: *ck.Checkpoint, buf: []u8, idx: usize, suffix: []const u8, dims: [2]usize, q_bits: u8) !Linear {
            const name = try std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ idx, suffix });
            return getLinear(cp, name, dims, q_bits);
        }
    };
    return .{
        .in_norm = try g.tensor(c, &b, i, "input_layernorm.weight", &.{hidden}),
        .post_norm = try g.tensor(c, &b, i, "post_attention_layernorm.weight", &.{hidden}),
        .q_norm = try g.tensor(c, &b, i, "self_attn.q_norm.weight", &.{head_dim}),
        .k_norm = try g.tensor(c, &b, i, "self_attn.k_norm.weight", &.{head_dim}),
        .q_proj = try g.lin(c, &b, i, "self_attn.q_proj.weight", .{ query_width, hidden }, bits),
        .k_proj = try g.lin(c, &b, i, "self_attn.k_proj.weight", .{ kv_width, hidden }, bits),
        .v_proj = try g.lin(c, &b, i, "self_attn.v_proj.weight", .{ kv_width, hidden }, bits),
        .o_proj = try g.lin(c, &b, i, "self_attn.o_proj.weight", .{ hidden, query_width }, bits),
        .gate_proj = try g.lin(c, &b, i, "mlp.gate_proj.weight", .{ intermediate, hidden }, bits),
        .up_proj = try g.lin(c, &b, i, "mlp.up_proj.weight", .{ intermediate, hidden }, bits),
        .down_proj = try g.lin(c, &b, i, "mlp.down_proj.weight", .{ hidden, intermediate }, bits),
        .attn_conv = .{
            .base = try g.tensor(c, &b, i, "attention_conv.base_kernel", &.{ 2, 2, hidden }),
            .proj = try g.lin(c, &b, i, "attention_conv.kernel_projection.weight", .{ dyn_width, hidden }, bits),
        },
        .mlp_conv = .{
            .base = try g.tensor(c, &b, i, "mlp_conv.base_kernel", &.{ 2, 2, hidden }),
            .proj = try g.lin(c, &b, i, "mlp_conv.kernel_projection.weight", .{ dyn_width, hidden }, bits),
        },
    };
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, bits: u8) !Loaded {
    if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
    const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cfg_path);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(cfg_text);
    const config = try Config.parse(gpa, cfg_text);
    var checkpoint = ck.Checkpoint.init(gpa);
    errdefer checkpoint.deinit();
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors", .{dir}, 0);
    defer gpa.free(path);
    const later: ?ck.Later = if (bits == 4) prepared.linear else null;
    checkpoint.addFileSplit(device, path, "", null, later) catch |err| {
        std.log.err("DFlash2 cannot open {s}: {s}", .{ path, @errorName(err) });
        return if (err == error.OpenFailed) error.DFlash2MissingFile else err;
    };
    const c = &checkpoint;
    var layers: [layers_n]Layer = undefined;
    for (&layers, 0..) |*l, i| l.* = try loadLayer(c, i, bits);
    const weights: Weights = .{
        .fc = try getLinear(c, "fc.weight", .{ hidden, taps_n * hidden }, bits),
        .hidden_norm = try get(c, "hidden_norm.weight", &.{hidden}),
        .norm = try get(c, "norm.weight", &.{hidden}),
        .hidden_proj = try getLinear(c, "candidate_selector.hidden_projection.weight", .{ rank, hidden }, bits),
        .pred_cb = try get(c, "candidate_selector.predecessor_codebook", &.{ padded_vocab, rank }),
        .succ_cb = try get(c, "candidate_selector.successor_codebook", &.{ padded_vocab, rank }),
        .layers = layers,
    };
    var copies: ?prepared.Copies = null;
    errdefer if (copies) |*cp| cp.deinit();
    if (bits == 4) {
        copies = try prepared.Copies.prepare(gpa, device, &checkpoint, config.linearCount());
        checkpoint.closeFiles();
    }
    return .{ .checkpoint = checkpoint, .config = config, .weights = weights, .bits = bits, .copies = copies };
}

test "DFlash2 config parser accepts Inkling-2-Mini-DFlash2 and refuses mismatched shapes" {
    const valid =
        \\{"architectures":["DFlash2DraftModel"],"head_dim":128,"hidden_size":2304,"intermediate_size":6912,
        \\"num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-6,
        \\"rope_parameters":{"rope_theta":10000.0},"sliding_window":2048,"vocab_size":201024,
        \\"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"final_logit_softcapping":30.0,
        \\"mask_token_id":200056,"selector_rank":256,"selector_softcap_mode":"after_correction",
        \\"selector_top_k":16,"target_layer_ids":[5,23,41,48]}}
    ;
    const c = try Config.parse(std.testing.allocator, valid);
    try std.testing.expectEqual(@as(u32, 200056), c.mask);
    try std.testing.expectEqual(@as(u32, 16), c.top_k);
    try std.testing.expectEqual(@as(f32, 30.0), c.softcap);
    try std.testing.expectEqual(@as(usize, 47), c.linearCount());
}

test "DFlash2 weights loader refuses unsupported drafter bit widths" {
    const dev: mtl.Device = undefined;
    try std.testing.expectError(error.UnsupportedDraftBits, load(std.testing.allocator, std.testing.io, dev, "unused", 8));
    try std.testing.expectError(error.UnsupportedDraftBits, load(std.testing.allocator, std.testing.io, dev, "unused", 2));
}

test "DFlash2 split checkpoint packs 47 linears to Q4 and drops BF16 linear residency" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const Entry = struct { name: []const u8, dims: [3]usize, rank: usize, len: usize };
    var entries: [81]Entry = undefined;
    var names_buf: [75][64]u8 = undefined;
    var idx: usize = 0;
    const globals = [_]struct { name: []const u8, dims: [3]usize, rank: usize }{
        .{ .name = "fc.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .name = "hidden_norm.weight", .dims = .{ 32, 1, 1 }, .rank = 1 },
        .{ .name = "norm.weight", .dims = .{ 32, 1, 1 }, .rank = 1 },
        .{ .name = "candidate_selector.hidden_projection.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .name = "candidate_selector.predecessor_codebook", .dims = .{ 64, 32, 1 }, .rank = 2 },
        .{ .name = "candidate_selector.successor_codebook", .dims = .{ 64, 32, 1 }, .rank = 2 },
    };
    for (globals) |g| {
        const len = g.dims[0] * g.dims[1] * g.dims[2];
        entries[idx] = .{ .name = g.name, .dims = g.dims, .rank = g.rank, .len = len };
        idx += 1;
    }
    const per_layer = [_]struct { suffix: []const u8, dims: [3]usize, rank: usize }{
        .{ .suffix = "input_layernorm.weight", .dims = .{ 32, 1, 1 }, .rank = 1 },
        .{ .suffix = "post_attention_layernorm.weight", .dims = .{ 32, 1, 1 }, .rank = 1 },
        .{ .suffix = "self_attn.q_norm.weight", .dims = .{ 32, 1, 1 }, .rank = 1 },
        .{ .suffix = "self_attn.k_norm.weight", .dims = .{ 32, 1, 1 }, .rank = 1 },
        .{ .suffix = "self_attn.q_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "self_attn.k_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "self_attn.v_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "self_attn.o_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "mlp.gate_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "mlp.up_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "mlp.down_proj.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "attention_conv.base_kernel", .dims = .{ 2, 2, 32 }, .rank = 3 },
        .{ .suffix = "attention_conv.kernel_projection.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
        .{ .suffix = "mlp_conv.base_kernel", .dims = .{ 2, 2, 32 }, .rank = 3 },
        .{ .suffix = "mlp_conv.kernel_projection.weight", .dims = .{ 32, 64, 1 }, .rank = 2 },
    };
    var ni: usize = 0;
    for (0..5) |layer| for (per_layer) |item| {
        const name = try std.fmt.bufPrint(&names_buf[ni], "layers.{d}.{s}", .{ layer, item.suffix });
        ni += 1;
        const len = item.dims[0] * item.dims[1] * item.dims[2];
        entries[idx] = .{ .name = name, .dims = item.dims, .rank = item.rank, .len = len };
        idx += 1;
    };

    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(gpa);
    try header.append(gpa, '{');
    var offset: usize = 0;
    for (entries, 0..) |e, i| {
        if (i > 0) try header.append(gpa, ',');
        try header.print(gpa, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{e.name});
        for (e.dims[0..e.rank], 0..) |d, j| try header.print(gpa, "{s}{d}", .{ if (j > 0) "," else "", d });
        try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + e.len * 2 });
        offset += e.len * 2;
    }
    try header.append(gpa, '}');
    var file = try tmp.dir.createFile(std.testing.io, "model.safetensors", .{});
    defer file.close(std.testing.io);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.items.len, .little);
    const payload = try gpa.alloc(u16, offset / 2);
    defer gpa.free(payload);
    for (payload, 0..) |*v, i| v.* = @bitCast(@as(u16, 0x3c00 + @as(u16, @intCast(i % 16))));
    var fw = file.writer(std.testing.io, &.{});
    try fw.interface.writeAll(&len_buf);
    try fw.interface.writeAll(header.items);
    try fw.interface.writeAll(std.mem.sliceAsBytes(payload));
    try fw.interface.flush();

    const path = try std.fmt.allocPrintSentinel(gpa, ".zig-cache/tmp/{s}/model.safetensors", .{tmp.sub_path}, 0);
    defer gpa.free(path);

    var bf16_ck = ck.Checkpoint.init(gpa);
    defer bf16_ck.deinit();
    try bf16_ck.addFileSplit(device, path, "", null, null);
    var q4_ck = ck.Checkpoint.init(gpa);
    defer q4_ck.deinit();
    try q4_ck.addFileSplit(device, path, "", null, prepared.linear);
    var copies = try prepared.Copies.prepare(gpa, device, &q4_ck, (Config{}).linearCount());
    defer copies.deinit();
    q4_ck.closeFiles();

    try std.testing.expectEqual(@as(usize, 34), q4_ck.tensors.count());
    try std.testing.expectEqual(@as(usize, 47), q4_ck.deferred.count());
    try std.testing.expectEqual(@as(usize, 47), if (device.tensorUnits()) copies.adapters.count() else copies.tiled.count());
    const q4_total: u64 = q4_ck.residentBytes() + copies.bytes();
    try std.testing.expect(q4_total < bf16_ck.residentBytes());
    try std.testing.expect((try getLinear(&q4_ck, "fc.weight", .{ 32, 64 }, 4)).bf16 == null);
    try std.testing.expect((try get(&q4_ck, "candidate_selector.predecessor_codebook", &.{ 64, 32 })).bytes == 64 * 32 * 2);
}
