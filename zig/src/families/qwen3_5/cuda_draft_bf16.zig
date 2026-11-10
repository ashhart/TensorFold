//! Opt-in bf16 DFlash2 projection packs in one arena, dispatched through the shared tensor-core lane operator.
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;
const Entry = struct { original: u64, weight: cuda.qmmf.Weight };
const Align = @import("cuda_engine.zig").aligned;

pub fn setting(raw: ?[]const u8) !u8 {
    const value = raw orelse return 0;
    const mask = try std.fmt.parseInt(u8, value, 10);
    if (mask > 31) return error.BadDraftBf16;
    return mask;
}

fn category(name: []const u8) u8 {
    if (std.mem.eql(u8, name, "fc.weight")) return 1;
    if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return 2;
    if (std.mem.indexOf(u8, name, "conv.kernel_projection.weight") != null) return 4;
    if (std.mem.indexOf(u8, name, ".self_attn.") != null) return 8;
    if (std.mem.indexOf(u8, name, ".mlp.") != null) return 16;
    return 0;
}

fn packedBytes(n: usize, k: usize) usize {
    return Align(((n + 127) / 128 * 128) * k * 2);
}

fn total(s: anytype, mask: u8) usize {
    var bytes: usize = 0;
    if (mask & 1 != 0) bytes += packedBytes(s.hidden, s.taps * s.hidden);
    if (mask & 2 != 0) bytes += packedBytes(s.rank, s.hidden);
    if (mask & 4 != 0) bytes += s.layers * 2 * packedBytes(s.dynamic, s.hidden);
    if (mask & 8 != 0) bytes += s.layers * (packedBytes(s.q_width, s.hidden) +
        2 * packedBytes(s.kv_width, s.hidden) + packedBytes(s.hidden, s.q_width));
    if (mask & 16 != 0) bytes += s.layers * (2 * packedBytes(s.intermediate, s.hidden) +
        packedBytes(s.hidden, s.intermediate));
    return bytes;
}

pub const Pack = struct {
    arena: cuda.DeviceBuffer,
    lane: cuda.qmmf.Lane,
    mask: u8,
    entries: [64]Entry = undefined,
    count: usize = 0,
    at: usize = 0,

    pub fn init(e: *Engine, s: anytype) !?Pack {
        const mask = try setting(try @import("sys").environment.given("TF_DRAFT_BF16"));
        if (mask == 0) return null;
        if (try e.d.currentCapability() != 120 or !@import("cuda_geometry.zig").dense27(e.g))
            return error.UnsupportedDraftBf16;
        var arena = try cuda.DeviceBuffer.alloc(e.d, total(s, mask));
        errdefer arena.free();
        try arena.fill8(0, null);
        var lane = try cuda.qmmf.Lane.load(e.d, 12);
        errdefer lane.unload();
        std.log.info("DFlash2 bf16 projection mask {d}: {d} bytes", .{ mask, arena.len });
        return .{ .arena = arena, .lane = lane, .mask = mask };
    }

    pub fn deinit(p: *Pack) void {
        p.lane.unload();
        p.arena.free();
    }

    pub fn add(p: *Pack, gpa: std.mem.Allocator, name: []const u8, source: []const u8, original: u64, n: usize, k: usize) !void {
        if (p.mask & category(name) == 0) return;
        if (n == 0 or k == 0 or k % 64 != 0 or source.len != n * k * 2 or
            @intFromPtr(source.ptr) % 2 != 0 or p.count == p.entries.len) return error.BadDraftBf16;
        const npad = (n + 127) / 128 * 128;
        const words = try gpa.alloc(u16, npad * k);
        defer gpa.free(words);
        cuda.qmmf.packBf16(words, @alignCast(std.mem.bytesAsSlice(u16, source)), n, k);
        const at = Align(p.at);
        if (at + words.len * 2 > p.arena.len) return error.BadDraftBf16;
        try p.arena.upload(at, std.mem.sliceAsBytes(words));
        p.entries[p.count] = .{ .original = original, .weight = .{
            .mode = .bf16,
            .codes = p.arena.ptr + at,
            .scales = 0,
            .n = @intCast(n),
            .k = @intCast(k),
            .npad = @intCast(npad),
        } };
        p.count += 1;
        p.at = at + words.len * 2;
    }

    pub fn containsAny(p: *const Pack, weights: []const cuda.qlinear.Affine4) bool {
        for (weights) |w| for (p.entries[0..p.count]) |entry| if (entry.original == w.w) return true;
        return false;
    }

    pub fn project(p: *const Pack, stream: cuda.Stream, original: u64, x: u64, out: u64, rows: usize) !bool {
        for (p.entries[0..p.count]) |entry| if (entry.original == original) {
            if (p.lane.partBytes(rows, entry.weight) != 0) return error.UnsupportedDraftBf16;
            try p.lane.matmul(stream, x, entry.weight.k, rows, entry.weight, out, 0);
            return true;
        };
        return false;
    }
};

test "DFlash bf16 categories keep selectors, taps, dynamic convolutions and backbone matrices separate" {
    try std.testing.expectEqual(@as(u8, 1), category("fc.weight"));
    try std.testing.expectEqual(@as(u8, 2), category("candidate_selector.hidden_projection.weight"));
    try std.testing.expectEqual(@as(u8, 4), category("layers.0.attention_conv.kernel_projection.weight"));
    try std.testing.expectEqual(@as(u8, 4), category("layers.0.mlp_conv.kernel_projection.weight"));
    try std.testing.expectEqual(@as(u8, 8), category("layers.0.self_attn.q_proj.weight"));
    try std.testing.expectEqual(@as(u8, 16), category("layers.0.mlp.down_proj.weight"));
    try std.testing.expectEqual(@as(u8, 0), category("norm.weight"));
    try std.testing.expectEqual(@as(u8, 0), try setting(null));
    try std.testing.expectEqual(@as(u8, 31), try setting("31"));
    try std.testing.expectError(error.BadDraftBf16, setting("32"));
}

test "DFlash bf16 adapter: all 16 row counts use cluster reduction without partial scratch" {
    const Fake = struct {
        var calls: usize = 0;
        var rows: i32 = 0;
        var partial: u64 = 0;
        var clusters: usize = 0;
        fn launch(config: *const cuda.abi.LaunchConfig, _: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            calls += 1;
            rows = @as(*const i32, @ptrCast(@alignCast(args.?[6].?))).*;
            partial = @as(*const u64, @ptrCast(@alignCast(args.?[5].?))).*;
            clusters = 0;
            for (config.attrs.?[0..config.num_attrs]) |attr| if (attr.id == .cluster_dimension) {
                clusters = attr.value.cluster_dim.z;
            };
            return 0;
        }
    };
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuLaunchKernelEx = Fake.launch;
    const f: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(0x1000) };
    var p: Pack = .{ .arena = .{ .d = &d, .ptr = 0x2000, .len = 262144000 }, .lane = undefined, .mask = 1, .count = 1 };
    p.lane.clusters = true;
    p.lane.half_below = 0;
    p.lane.fns = @splat(@splat(f));
    p.entries[0] = .{ .original = 0x3000, .weight = .{ .mode = .bf16, .codes = 0x2000, .scales = 0, .n = 5120, .k = 25600, .npad = 5120 } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.calls = 0;
    try std.testing.expectEqual(@as(u32, 4), cuda.qmmf.slices(p.entries[0].weight));
    for (1..17) |rows| {
        try std.testing.expectEqual(@as(usize, 0), p.lane.partBytes(rows, p.entries[0].weight));
        try std.testing.expect(try p.project(stream, 0x3000, 0x4000, 0x5000, rows));
        try std.testing.expectEqual(@as(i32, @intCast(rows)), Fake.rows);
        try std.testing.expectEqual(@as(u64, 0), Fake.partial);
        try std.testing.expectEqual(@as(usize, 4), Fake.clusters);
    }
    try std.testing.expectEqual(@as(usize, 16), Fake.calls);
    p.lane.clusters = false;
    try std.testing.expectEqual(@as(usize, 1310720), p.lane.partBytes(16, p.entries[0].weight));
    try std.testing.expectError(error.UnsupportedDraftBf16, p.project(stream, 0x3000, 0x4000, 0x5000, 16));
    try std.testing.expectEqual(@as(usize, 16), Fake.calls);
}
