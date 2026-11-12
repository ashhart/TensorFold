//! Conservative rank-local checkpoint bytes: replicated dense operators, partitioned routed weights, one shared branch.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const dense_split = @import("cuda_dense_split.zig");
const Entry = core.safetensors.Entry;

pub fn estimate(a: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize) !u64 {
    if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
    var store = try wt.Store.open(a, io, dir);
    defer store.deinit();
    const split = try dense_split.applies(&store, world);
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
            total = try std.math.add(u64, total, try tensorBytes(tensor.key_ptr.*, tensor.value_ptr.*, world, rank, split));
        }
    }
    return total;
}

fn tensorBytes(name: []const u8, e: Entry, world: usize, rank: usize, split: bool) !u64 {
    const bytes: u64 = e.end - e.begin;
    if (wt.hostOnly(name)) return 0;
    if (split) if (try dense_split.headBytes(name, e, world)) |cut| return cut;
    if (split and e.dtype == .bf16 and e.rank == 2) if (dense_split.axisOf(name)) |axis| return dense_split.residentBytes(e.dim(0), e.dim(1), axis, world);
    if (rank != 0 and std.mem.indexOf(u8, name, ".mlp.shared_expert.") != null) return 0;
    if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) {
        if (std.mem.endsWith(u8, name, ".weight_scale_2")) return bytes;
        if (!std.mem.endsWith(u8, name, ".weight")) return 0;
        if (e.dtype != .u8 or e.rank != 2 or bytes % (8 * world) != 0) return error.UnexpectedTensor;
        // NVFP4 device blocks hold 128 code words and 16 scale words; a rank's whole experts average bytes / world.
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
    try std.testing.expectEqual(@as(u64, cuda.experts.Nvfp4Experts.bytes(1, 320, 2560, 1)), try tensorBytes(name, e, 2, 0, false));
    try std.testing.expectEqual(try tensorBytes(name, e, 2, 0, false), try tensorBytes(name, e, 2, 1, false));
    var shared = e;
    shared.dtype = .f8_e4m3;
    shared.shape = .{ 640, 2560, 1, 1, 1 };
    shared.end = 640 * 2560;
    const shared_name = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight";
    try std.testing.expectEqual(@as(u64, 640 * 2560 * 33 / 32), try tensorBytes(shared_name, shared, 2, 0, false));
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes(shared_name, shared, 2, 1, false));
    const norm: Entry = .{ .dtype = .bf16, .rank = 1, .shape = .{ 256, 1, 1, 1, 1 }, .begin = 0, .end = 512 };
    try std.testing.expectEqual(@as(u64, 1024), try tensorBytes("model.language_model.layers.3.self_attn.q_norm.weight", norm, 2, 1, false));
    try std.testing.expectEqual(@as(u64, 512), try tensorBytes("model.language_model.layers.0.linear_attn.norm.weight", norm, 2, 1, false));
}

test "rank estimates under the dense split count each rank's out-projection output rows, padded as BF16 packing pads" {
    const out: Entry = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2560, 6144, 1, 1, 1 }, .begin = 0, .end = 2560 * 6144 * 2 };
    const names = [_][]const u8{ "model.language_model.layers.0.linear_attn.out_proj.weight", "model.language_model.layers.3.self_attn.o_proj.weight" };
    for (names) |name| {
        for (0..2) |rank| try std.testing.expectEqual(@as(u64, 2560 * 3072 * 2), try tensorBytes(name, out, 2, rank, true));
        try std.testing.expectEqual(@as(u64, 2560 * 6144 * 2), try tensorBytes(name, out, 2, 0, false));
        try std.testing.expectEqual(@as(u64, 2560 * 6144 * 2), try tensorBytes(name, out, 1, 0, false));
    }
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes("mtp.layers.0.self_attn.o_proj.weight", out, 2, 0, true));
}

test "rank estimates under the head split count each rank's in-projection rows, conv channels and gate values" {
    const k = 2560;
    const Part = struct { name: []const u8, n: usize, kept: usize };
    const delta = [_]Part{ .{ .name = "in_proj_qkv", .n = 10240, .kept = 5120 }, .{ .name = "in_proj_z", .n = 6144, .kept = 3072 }, .{ .name = "in_proj_b", .n = 48, .kept = 24 }, .{ .name = "in_proj_a", .n = 48, .kept = 24 } };
    const attn = [_]Part{ .{ .name = "q_proj", .n = 12288, .kept = 6144 }, .{ .name = "k_proj", .n = 512, .kept = 256 }, .{ .name = "v_proj", .n = 512, .kept = 256 }, .{ .name = "indexer.index_qk_proj", .n = 640, .kept = 640 } };
    for ([_][]const Part{ &delta, &attn }, [_][]const u8{ "0.linear_attn.", "3.self_attn." }, [_]usize{ 8320, 7296 }) |parts, class, packed_rows| {
        var split: [2]u64 = .{ 0, 0 };
        var buf: [120]u8 = undefined;
        for (parts) |p| {
            const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{s}{s}.weight", .{ class, p.name });
            const e: Entry = .{ .dtype = .bf16, .rank = 2, .shape = .{ p.n, k, 1, 1, 1 }, .begin = 0, .end = p.n * k * 2 };
            for (&split, 0..) |*sum, rank| sum.* += try tensorBytes(name, e, 2, rank, true);
            // unsplit, each part is the whole tensor on every rank, padded on its own as today
            try std.testing.expectEqual(@as(u64, cuda.nvfp4.padded(p.n) * k * 2), try tensorBytes(name, e, 2, 1, false));
            try std.testing.expect(try tensorBytes(name, e, 2, 0, true) >= p.kept * k * 2);
        }
        // the parts' shares sum to the stitched cut as one BF16 pack pads it: 8,240 rows to 8,320, 7,296 to itself
        for (split) |sum| try std.testing.expectEqual(@as(u64, packed_rows * k * 2), sum);
    }
    const conv: Entry = .{ .dtype = .bf16, .rank = 3, .shape = .{ 10240, 1, 4, 1, 1 }, .begin = 0, .end = 10240 * 4 * 2 };
    const conv_name = "model.language_model.layers.0.linear_attn.conv1d.weight";
    for (0..2) |rank| try std.testing.expectEqual(@as(u64, 5120 * 4 * 2), try tensorBytes(conv_name, conv, 2, rank, true));
    try std.testing.expectEqual(@as(u64, 10240 * 4 * 2), try tensorBytes(conv_name, conv, 2, 1, false));
    const gates: Entry = .{ .dtype = .f32, .rank = 1, .shape = .{ 48, 1, 1, 1, 1 }, .begin = 0, .end = 48 * 4 };
    for ([_][]const u8{ "model.language_model.layers.0.linear_attn.A_log", "model.language_model.layers.0.linear_attn.dt_bias" }) |name| {
        for (0..2) |rank| try std.testing.expectEqual(@as(u64, 24 * 4), try tensorBytes(name, gates, 2, rank, true));
        try std.testing.expectEqual(@as(u64, 48 * 4), try tensorBytes(name, gates, 2, 0, false));
    }
    // the norms stay whole on both ranks, and the MTP head's layer is the draft's to price
    const norm: Entry = .{ .dtype = .bf16, .rank = 1, .shape = .{ 128, 1, 1, 1, 1 }, .begin = 0, .end = 256 };
    try std.testing.expectEqual(@as(u64, 256), try tensorBytes("model.language_model.layers.0.linear_attn.norm.weight", norm, 2, 1, true));
    const q: Entry = .{ .dtype = .bf16, .rank = 2, .shape = .{ 12288, k, 1, 1, 1 }, .begin = 0, .end = 12288 * k * 2 };
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes("mtp.layers.0.self_attn.q_proj.weight", q, 2, 0, true));
}

test "shared output-row cuts include padded gate/up rows and full-input down rows on both ranks" {
    const up: Entry = .{ .dtype = .bf16, .rank = 2, .shape = .{ 640, 2560, 1, 1, 1 }, .begin = 0, .end = 640 * 2560 * 2 };
    const down: Entry = .{ .dtype = .bf16, .rank = 2, .shape = .{ 2560, 640, 1, 1, 1 }, .begin = 0, .end = 2560 * 640 * 2 };
    for (0..2) |rank| {
        try std.testing.expectEqual(@as(u64, 384 * 2560 * 2), try tensorBytes("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", up, 2, rank, true));
        try std.testing.expectEqual(@as(u64, 1280 * 640 * 2), try tensorBytes("model.language_model.layers.0.mlp.shared_expert.down_proj.weight", down, 2, rank, true));
    }
}
