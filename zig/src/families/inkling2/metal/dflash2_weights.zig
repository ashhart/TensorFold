//! Validated config.json and model.safetensors loader for incoai/Inkling-2-Mini-DFlash2.
const std = @import("std");
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;

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

pub const Conv = struct {
    base: ck.Tensor,
    proj: ck.Tensor,
};

pub const Layer = struct {
    in_norm: ck.Tensor,
    post_norm: ck.Tensor,
    q_norm: ck.Tensor,
    k_norm: ck.Tensor,
    q_proj: ck.Tensor,
    k_proj: ck.Tensor,
    v_proj: ck.Tensor,
    o_proj: ck.Tensor,
    gate_proj: ck.Tensor,
    up_proj: ck.Tensor,
    down_proj: ck.Tensor,
    attn_conv: Conv,
    mlp_conv: Conv,
};

pub const Weights = struct {
    fc: ck.Tensor,
    hidden_norm: ck.Tensor,
    norm: ck.Tensor,
    hidden_proj: ck.Tensor,
    pred_cb: ck.Tensor,
    succ_cb: ck.Tensor,
    layers: [layers_n]Layer,
};

pub const Loaded = struct {
    checkpoint: ck.Checkpoint,
    config: Config,
    weights: Weights,

    pub fn deinit(l: *Loaded) void {
        l.checkpoint.deinit();
        l.* = undefined;
    }

    pub fn fixedBytes(l: *const Loaded) u64 {
        return l.checkpoint.residentBytes();
    }
};

fn require(t: ck.Tensor, dims: []const usize) !ck.Tensor {
    if (t.dtype != .bf16 or t.rank != dims.len) return error.DFlash2TensorShape;
    var count: usize = 1;
    for (dims, 0..) |d, i| {
        if (t.shape[i] != d) return error.DFlash2TensorShape;
        count = std.math.mul(usize, count, d) catch return error.DFlash2TensorShape;
    }
    if (t.bytes != count * 2) return error.DFlash2TensorShape;
    return t;
}

fn get(c: *const ck.Checkpoint, name: []const u8, dims: []const usize) !ck.Tensor {
    const t = c.get(name) catch {
        std.log.err("DFlash2 model.safetensors is missing {s}", .{name});
        return error.DFlash2MissingTensor;
    };
    return require(t, dims);
}

fn loadLayer(c: *const ck.Checkpoint, i: usize) !Layer {
    var b: [128]u8 = undefined;
    const g = struct {
        fn tensor(cp: *const ck.Checkpoint, buf: []u8, idx: usize, suffix: []const u8, dims: []const usize) !ck.Tensor {
            const name = try std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ idx, suffix });
            return get(cp, name, dims);
        }
    }.tensor;
    return .{
        .in_norm = try g(c, &b, i, "input_layernorm.weight", &.{hidden}),
        .post_norm = try g(c, &b, i, "post_attention_layernorm.weight", &.{hidden}),
        .q_norm = try g(c, &b, i, "self_attn.q_norm.weight", &.{head_dim}),
        .k_norm = try g(c, &b, i, "self_attn.k_norm.weight", &.{head_dim}),
        .q_proj = try g(c, &b, i, "self_attn.q_proj.weight", &.{ query_width, hidden }),
        .k_proj = try g(c, &b, i, "self_attn.k_proj.weight", &.{ kv_width, hidden }),
        .v_proj = try g(c, &b, i, "self_attn.v_proj.weight", &.{ kv_width, hidden }),
        .o_proj = try g(c, &b, i, "self_attn.o_proj.weight", &.{ hidden, query_width }),
        .gate_proj = try g(c, &b, i, "mlp.gate_proj.weight", &.{ intermediate, hidden }),
        .up_proj = try g(c, &b, i, "mlp.up_proj.weight", &.{ intermediate, hidden }),
        .down_proj = try g(c, &b, i, "mlp.down_proj.weight", &.{ hidden, intermediate }),
        .attn_conv = .{
            .base = try g(c, &b, i, "attention_conv.base_kernel", &.{ 2, 2, hidden }),
            .proj = try g(c, &b, i, "attention_conv.kernel_projection.weight", &.{ dyn_width, hidden }),
        },
        .mlp_conv = .{
            .base = try g(c, &b, i, "mlp_conv.base_kernel", &.{ 2, 2, hidden }),
            .proj = try g(c, &b, i, "mlp_conv.kernel_projection.weight", &.{ dyn_width, hidden }),
        },
    };
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, target: *const Model, dir: []const u8) !Loaded {
    const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cfg_path);
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(cfg_text);
    const config = try Config.parse(gpa, cfg_text);
    var checkpoint = ck.Checkpoint.init(gpa);
    errdefer checkpoint.deinit();
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors", .{dir}, 0);
    defer gpa.free(path);
    checkpoint.addFile(target.device, path, "") catch |err| {
        std.log.err("DFlash2 cannot open {s}: {s}", .{ path, @errorName(err) });
        return error.DFlash2MissingFile;
    };
    const c = &checkpoint;
    var layers: [layers_n]Layer = undefined;
    for (&layers, 0..) |*l, i| l.* = try loadLayer(c, i);
    const weights: Weights = .{
        .fc = try get(c, "fc.weight", &.{ hidden, taps_n * hidden }),
        .hidden_norm = try get(c, "hidden_norm.weight", &.{hidden}),
        .norm = try get(c, "norm.weight", &.{hidden}),
        .hidden_proj = try get(c, "candidate_selector.hidden_projection.weight", &.{ rank, hidden }),
        .pred_cb = try get(c, "candidate_selector.predecessor_codebook", &.{ padded_vocab, rank }),
        .succ_cb = try get(c, "candidate_selector.successor_codebook", &.{ padded_vocab, rank }),
        .layers = layers,
    };
    return .{ .checkpoint = checkpoint, .config = config, .weights = weights };
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
}
