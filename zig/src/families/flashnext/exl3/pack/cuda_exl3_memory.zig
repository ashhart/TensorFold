//! Device bytes of an EXL3 Flash Next checkpoint as cuda_exl3_load.zig places it:

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const config = @import("../../cuda_config.zig");
const Entry = core.safetensors.Entry;

pub const Options = struct {
    /// Routed experts' trellis words live in pinned host memory (the offload pool's), not counted here.
    routed_on_host: bool = true,
};

/// TF_FLASHNEXT_EXL3_ROUTED=device counts every routed expert as resident; the default is the offload pool's layout.
pub fn options() Options {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_ROUTED") orelse return .{};
    return .{ .routed_on_host = !std.mem.eql(u8, std.mem.span(v), "device") };
}

/// Whether `dir`'s config.json admits as Mode.exl3 (any other mode, or a refusal: false).
pub fn isExl3(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    return (config.mode(gpa, io, dir) catch return false) == .exl3;
}

/// One tensor's device bytes (0: kept on the host).
pub fn tensorBytes(name: []const u8, e: Entry, o: Options) !u64 {
    const bytes: u64 = e.end - e.begin;
    if (wt.hostOnly(name)) return 0;
    // the BF16 embedding table is pinned host memory the gather reads in place (cuda_embed.Table.uploadDenseHost)
    if (e.dtype == .bf16 and std.mem.indexOf(u8, name, "embed_tokens") != null) return 0;
    // the codebook marker is checked at load and never uploaded
    if (std.mem.endsWith(u8, name, ".mul1") or std.mem.endsWith(u8, name, ".mcg")) return 0;
    if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) {
        if (std.mem.endsWith(u8, name, ".trellis")) return if (o.routed_on_host) 0 else bytes;
        return bytes; // stacked suh / svh, resident
    }
    if (std.mem.endsWith(u8, name, ".trellis") or std.mem.endsWith(u8, name, ".suh") or std.mem.endsWith(u8, name, ".svh")) return bytes;
    if (!std.mem.endsWith(u8, name, ".weight") and e.rank == 1) return try std.math.mul(u64, e.dim(0), 4); // A_log, dt_bias as fp32
    if ((e.dtype == .f16 or e.dtype == .bf16) and e.rank == 2 and std.mem.endsWith(u8, name, ".weight")) {
        // the router's rows and the embedding table go up as stored (BF16).
        if (std.mem.indexOf(u8, name, ".mlp.gate.") != null or std.mem.indexOf(u8, name, ".shared_expert_gate.") != null or std.mem.indexOf(u8, name, "embed_tokens") != null) return bytes;
        return try std.math.mul(u64, try std.math.mul(u64, cuda.nvfp4.padded(e.dim(0)), e.dim(1)), 2);
    }
    // norms the loaders scale go up as fp32 (cuda_rank_memory.zig's list), the GDN gated norm and convs as stored
    if (e.rank == 1 and centered(name)) return try std.math.mul(u64, e.dim(0), 4);
    return bytes;
}

fn centered(name: []const u8) bool {
    for ([_][]const u8{ ".hc_norm.weight", ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight" }) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return true;
    }
    return false;
}

/// Per layer: three 513-entry pointer tables (int64) and three width tables (int32) for the experts' grouped GEMMs.
const tables_per_layer: u64 = 3 * (wt.experts_n + 1) * 8 + 3 * (wt.experts_n + 1) * 4;

/// Every shard the index names (and the mixer patch, which Store.open adds), each read for its header alone.
pub fn deviceBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !u64 {
    var store = try wt.Store.open(gpa, io, dir);
    defer store.deinit();
    const wm = store.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    var it = wm.object.iterator();
    while (it.next()) |kv| try files.put(gpa, kv.value_ptr.string, {});
    var total: u64 = tables_per_layer * wt.layers_n;
    var fi = files.keyIterator();
    while (fi.next()) |rel| {
        const path = try std.fs.path.join(gpa, &.{ dir, rel.* });
        defer gpa.free(path);
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        var ni = file.names.iterator();
        while (ni.next()) |e| total = try std.math.add(u64, total, try tensorBytes(e.key_ptr.*, e.value_ptr.*, o));
    }
    return total;
}

/// The family's deviceWeightBytes for an EXL3 checkpoint; 0 if unread.
pub fn weightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    return deviceBytes(gpa, io, dir, options()) catch 0;
}

/// A rank's placement bytes: one rank holds the whole model (no MTP head: an EXL3 pack decodes plain).
pub fn placement(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize) !u64 {
    if (world != 1 or rank != 0) return error.BadRank;
    return deviceBytes(gpa, io, dir, options());
}

fn entry(dtype: core.safetensors.DType, shape: []const usize, bytes: u64) Entry {
    var e: Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = bytes };
    for (shape, 0..) |d, i| e.shape[i] = d;
    return e;
}

test "exl3: routed trellis stays on the host under offload, its scales and every other EXL3 matrix are resident" {
    const tr = entry(.i16, &.{ 160, 40, 24 }, 160 * 40 * 24 * 2);
    const routed = "model.language_model.layers.0.mlp.experts.7.gate_proj.trellis";
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes(routed, tr, .{}));
    try std.testing.expectEqual(tr.end, try tensorBytes(routed, tr, .{ .routed_on_host = false }));
    const suh = entry(.f16, &.{2560}, 5120);
    try std.testing.expectEqual(@as(u64, 5120), try tensorBytes("model.language_model.layers.0.mlp.experts.7.gate_proj.suh", suh, .{}));
    try std.testing.expectEqual(tr.end, try tensorBytes("model.language_model.layers.0.mlp.shared_expert.gate_proj.trellis", tr, .{}));
    try std.testing.expectEqual(tr.end, try tensorBytes("lm_head.trellis", tr, .{}));
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes("lm_head.mul1", entry(.i32, &.{}, 4), .{}));
}

test "exl3: dense F16 weights count as their BF16 device packing, norms as fp32" {
    const router = entry(.f16, &.{ 512, 2560 }, 512 * 2560 * 2);
    try std.testing.expectEqual(router.end, try tensorBytes("model.language_model.layers.3.mlp.gate.weight", router, .{}));
    const down = entry(.f16, &.{ 320, 10240 }, 320 * 10240 * 2);
    try std.testing.expectEqual(@as(u64, cuda.nvfp4.padded(320) * 10240 * 2), try tensorBytes("model.language_model.layers.3.attn_hyper_connection.input_mix_weight_down.weight", down, .{}));
    const norm = entry(.f16, &.{10240}, 20480);
    try std.testing.expectEqual(@as(u64, 40960), try tensorBytes("model.language_model.layers.3.attn_hyper_connection.hc_norm.weight", norm, .{}));
    const gnorm = entry(.bf16, &.{128}, 256);
    try std.testing.expectEqual(@as(u64, 256), try tensorBytes("model.language_model.layers.0.linear_attn.norm.weight", gnorm, .{}));
    try std.testing.expectEqual(@as(u64, 48 * 4), try tensorBytes("model.language_model.layers.0.linear_attn.A_log", entry(.bf16, &.{48}, 96), .{}));
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes("model.visual.blocks.0.attn.qkv.weight", entry(.bf16, &.{ 3456, 1152 }, 3456 * 1152 * 2), .{}));
}

test "exl3: placement names an EXL3 checkpoint and refuses any world but one" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/exl3_config.json") });
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    try std.testing.expect(isExl3(a, std.testing.io, dir));
    try std.testing.expectError(error.BadRank, placement(a, std.testing.io, dir, 2, 0));
    try std.testing.expectError(error.BadRank, placement(a, std.testing.io, dir, 1, 1));
}
