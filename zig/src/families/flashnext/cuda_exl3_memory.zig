//! Device bytes of an EXL3 Flash Next checkpoint as cuda_exl3_load.zig places it: EXL3 matrices in their stored
//! layout (trellis, suh, svh; the codebook marker stays on the host), dense weights widened to BF16 and packed as
//! projection.dense packs them, norms and GDN scalars as fp32, the router's 513 rows as stored. Routed experts' trellis
//! words are host-resident under offload (`routed_on_host`; the pool decides which lines reach the device), while their
//! stacked suh/svh scales and the per-expert pointer and width tables stay resident.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const config = @import("cuda_config.zig");
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
    // the codebook marker is checked at load and never uploaded
    if (std.mem.endsWith(u8, name, ".mul1") or std.mem.endsWith(u8, name, ".mcg")) return 0;
    if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) {
        if (std.mem.endsWith(u8, name, ".trellis")) return if (o.routed_on_host) 0 else bytes;
        return bytes; // stacked suh / svh, resident
    }
    if (std.mem.endsWith(u8, name, ".trellis") or std.mem.endsWith(u8, name, ".suh") or std.mem.endsWith(u8, name, ".svh")) return bytes;
    if (!std.mem.endsWith(u8, name, ".weight") and e.rank == 1) return try std.math.mul(u64, e.dim(0), 4); // A_log, dt_bias as fp32
    if ((e.dtype == .f16 or e.dtype == .bf16) and e.rank == 2 and std.mem.endsWith(u8, name, ".weight")) {
        // the router's rows and the embedding table go up as stored (BF16); every other dense matrix through projection.dense
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

/// The MTP head's device bytes: its `mtp.*` tensors placed as the main layers' are (its routed experts under the same
/// option), plus the draft vocabulary's bf16 rows dequantised from the EXL3 lm_head (cuda_mtp.zig's 79,591 ids).
pub fn headBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !u64 {
    var store = try wt.Store.open(gpa, io, dir);
    defer store.deinit();
    const wm = store.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    var it = wm.object.iterator();
    while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, "mtp.")) try files.put(gpa, kv.value_ptr.string, {});
    var total: u64 = tables_per_layer + draft_rows * wt.dims * 2;
    var fi = files.keyIterator();
    while (fi.next()) |rel| {
        const path = try std.fs.path.join(gpa, &.{ dir, rel.* });
        defer gpa.free(path);
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        var ni = file.names.iterator();
        while (ni.next()) |e| {
            const name = e.key_ptr.*;
            if (!std.mem.startsWith(u8, name, "mtp.")) continue;
            // the head's tensors as a decoder layer's: drop the `mtp.` prefix the host-only rule keys on
            var buf: [256]u8 = undefined;
            const as_layer = try std.fmt.bufPrint(&buf, "model.language_model.{s}", .{name["mtp.".len..]});
            total = try std.math.add(u64, total, try tensorBytes(as_layer, e.value_ptr.*, o));
        }
    }
    return total;
}

/// The draft head's vocabulary rows (cuda_mtp.zig headBytes' count).
const draft_rows: u64 = 79_591;

/// The family's deviceWeightBytes for an EXL3 checkpoint: the main model and the MTP head; 0 if unread.
pub fn weightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    const o = options();
    const main = deviceBytes(gpa, io, dir, o) catch return 0;
    return main + (headBytes(gpa, io, dir, o) catch return 0);
}

/// A rank's placement bytes; at world 2 the expert-parallel rank's (cuda_exl3_tp.zig, plan step 10).
pub fn placement(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    if (world != 1) return @import("cuda_exl3_tp.zig").placement(gpa, io, dir, world, rank, drafts);
    const o = options();
    return try deviceBytes(gpa, io, dir, o) + if (drafts) try headBytes(gpa, io, dir, o) else @as(u64, 0);
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
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes("mtp.layers.0.mlp.experts.3.up_proj.trellis", tr, .{ .routed_on_host = false }));
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

test "exl3: placement names an EXL3 checkpoint and refuses worlds past 2 (world 2: cuda_exl3_tp.zig's test)" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/exl3_config.json") });
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    try std.testing.expect(isExl3(a, std.testing.io, dir));
    try std.testing.expectError(error.BadRank, placement(a, std.testing.io, dir, 3, 0, false));
}
