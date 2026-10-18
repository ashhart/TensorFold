//! Conservative rank-local checkpoint bytes: replicated dense operators, partitioned routed weights, one shared branch.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const Entry = core.safetensors.Entry;

pub fn estimate(a: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize) !u64 {
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    var store = try wt.Store.open(a, io, dir);
    defer store.deinit();
    const wm = store.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(a);
    var wi = wm.object.iterator();
    while (wi.next()) |item| try files.put(a, item.value_ptr.string, {});
    var total: u64 = 0;
    var fi = files.keyIterator();
    while (fi.next()) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name.* });
        defer a.free(path);
        var file = try core.safetensors.File.open(a, io, path);
        defer file.close(io);
        var tensors = file.names.iterator();
        while (tensors.next()) |tensor| {
            total = try std.math.add(u64, total, try tensorBytes(tensor.key_ptr.*, tensor.value_ptr.*, world, rank));
        }
    }
    return total;
}

fn tensorBytes(name: []const u8, e: Entry, world: usize, rank: usize) !u64 {
    const bytes: u64 = e.end - e.begin;
    if (wt.hostOnly(name)) return 0;
    if (rank != 0 and std.mem.indexOf(u8, name, ".mlp.shared_expert.") != null) return 0;
    if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) {
        if (std.mem.endsWith(u8, name, ".weight_scale_2")) return bytes;
        if (!std.mem.endsWith(u8, name, ".weight")) return 0;
        if (e.dtype != .u8 or e.rank != 2 or bytes % (8 * world) != 0) return error.UnexpectedTensor;
        // NVFP4 device blocks hold 128 code words and 16 scale words; global scales stay replicated.
        return try std.math.mul(u64, bytes / (8 * world), 9);
    }
    if (std.mem.indexOf(u8, name, ".weight_scale") != null or std.mem.endsWith(u8, name, ".input_scale")) return 0;
    if (e.dtype == .f8_e4m3 and e.rank == 2 and std.mem.endsWith(u8, name, ".weight")) {
        const codes = try std.math.mul(u64, cuda.mxfp8.padded(e.dim(0)), e.dim(1));
        return try std.math.add(u64, codes, codes / 32);
    }
    if (e.dtype == .bf16 and e.rank == 2 and std.mem.endsWith(u8, name, ".weight")) {
        if (std.mem.indexOf(u8, name, ".mlp.gate.") != null or std.mem.indexOf(u8, name, ".shared_expert_gate.") != null) return bytes;
        return try std.math.mul(u64, try std.math.mul(u64, cuda.nvfp4.padded(e.dim(0)), e.dim(1)), 2);
    }
    if (e.dtype == .bf16 and centered(name)) return try std.math.mul(u64, bytes, 2);
    return bytes;
}

fn centered(name: []const u8) bool {
    for ([_][]const u8{ ".hc_norm.weight", ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight" }) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return true;
    }
    return false;
}

test "rank estimates partition routed storage while retaining replicated and rank-zero shared operators" {
    const e: Entry = .{ .dtype = .u8, .rank = 2, .shape = .{ 640, 1280, 1, 1, 1 }, .begin = 0, .end = 640 * 1280 };
    const name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
    try std.testing.expectEqual(@as(u64, cuda.experts.Nvfp4Experts.bytes(1, 320, 2560, 1)), try tensorBytes(name, e, 2, 0));
    try std.testing.expectEqual(try tensorBytes(name, e, 2, 0), try tensorBytes(name, e, 2, 1));
    var shared = e;
    shared.dtype = .f8_e4m3;
    shared.shape = .{ 640, 2560, 1, 1, 1 };
    shared.end = 640 * 2560;
    const shared_name = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight";
    try std.testing.expectEqual(@as(u64, 640 * 2560 * 33 / 32), try tensorBytes(shared_name, shared, 2, 0));
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes(shared_name, shared, 2, 1));
    const norm: Entry = .{ .dtype = .bf16, .rank = 1, .shape = .{ 256, 1, 1, 1, 1 }, .begin = 0, .end = 512 };
    try std.testing.expectEqual(@as(u64, 1024), try tensorBytes("model.language_model.layers.3.self_attn.q_norm.weight", norm, 2, 1));
    try std.testing.expectEqual(@as(u64, 512), try tensorBytes("model.language_model.layers.0.linear_attn.norm.weight", norm, 2, 1));
}
