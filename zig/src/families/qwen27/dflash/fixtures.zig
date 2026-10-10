//! Tiny owned BF16 fixtures exercise native metadata and callback schedules without a model or GPU.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
pub const toy = Config{ .hidden = 64, .intermediate = 128, .vocab = 64, .heads = 8, .kv_heads = 4, .head_dim = 8, .mask = 63, .window = 16, .rank = 16, .topk = 2 };
const Dimensions = struct { values: [4]usize, rank: usize };

pub const Store = struct {
    arena: std.heap.ArenaAllocator,
    keys: [][]const u8 = undefined,
    tensors: []ck.Tensor = undefined,
    embed: ck.Tensor = undefined,
    pub fn init(gpa: std.mem.Allocator) !Store {
        var s = Store{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.deinit();
        const a = s.arena.allocator();
        s.keys = try weights.names(a, toy);
        s.tensors = try a.alloc(ck.Tensor, s.keys.len);
        for (s.keys, s.tensors) |name, *t| {
            const shape = dims(name);
            var count: usize = 2;
            for (shape.values[0..shape.rank]) |n| count = try std.math.mul(usize, count, n);
            const data = try a.alloc(u8, count);
            @memset(data, 0);
            t.* = .{ .dtype = .bf16, .rank = @intCast(shape.rank), .shape = shape.values, .bytes = data };
        }
        const data = try a.alloc(u8, toy.vocab * toy.hidden * 2);
        @memset(data, 0);
        s.embed = .{ .dtype = .bf16, .rank = 2, .shape = .{ toy.vocab, toy.hidden, 1, 1 }, .bytes = data };
        return s;
    }
    pub fn deinit(s: *Store) void {
        s.arena.deinit();
    }
    pub fn graph(s: *Store, gpa: std.mem.Allocator) !weights.Graph {
        return weights.load(gpa, .{ .source = .{ .ptr = s, .getFn = get }, .names = s.keys }, toy, .{ .dense = s.embed }, .{ .dense = s.embed });
    }
    fn get(ptr: *anyopaque, name: []const u8) !ck.Tensor {
        const s: *Store = @ptrCast(@alignCast(ptr));
        for (s.keys, s.tensors) |key, t| if (std.mem.eql(u8, key, name)) return t;
        return error.MissingTensor;
    }
    fn dims(name: []const u8) Dimensions {
        if (std.mem.eql(u8, name, "fc.weight")) return .{ .values = .{ toy.hidden, toy.tapWidth(), 1, 1 }, .rank = 2 };
        if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return .{ .values = .{ toy.rank, toy.hidden, 1, 1 }, .rank = 2 };
        if (std.mem.startsWith(u8, name, "candidate_selector.")) return .{ .values = .{ toy.vocab, toy.rank, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "base_kernel")) return .{ .values = .{ 2, 2, toy.hidden, 1 }, .rank = 3 };
        if (std.mem.endsWith(u8, name, "kernel_projection.weight")) return .{ .values = .{ toy.dynamicWidth(), toy.hidden, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "q_proj.weight")) return .{ .values = .{ toy.qWidth(), toy.hidden, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "k_proj.weight") or std.mem.endsWith(u8, name, "v_proj.weight")) return .{ .values = .{ toy.kvWidth(), toy.hidden, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "o_proj.weight")) return .{ .values = .{ toy.hidden, toy.qWidth(), 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "gate_proj.weight") or std.mem.endsWith(u8, name, "up_proj.weight")) return .{ .values = .{ toy.intermediate, toy.hidden, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "down_proj.weight")) return .{ .values = .{ toy.hidden, toy.intermediate, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "q_norm.weight") or std.mem.endsWith(u8, name, "k_norm.weight")) return .{ .values = .{ toy.head_dim, 1, 1, 1 }, .rank = 1 };
        return .{ .values = .{ toy.hidden, 1, 1, 1 }, .rank = 1 };
    }
};

test "complete tiny graph preserves every native tensor and target binding pointer" {
    const a = std.testing.allocator;
    var s = try Store.init(a);
    defer s.deinit();
    var g = try s.graph(a);
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 81), s.keys.len);
    try std.testing.expectEqual(s.tensors[0].bytes.ptr, g.fusion.bytes.ptr);
    try std.testing.expectEqual(s.embed.bytes.ptr, g.embed.dense.bytes.ptr);
    try std.testing.expectEqual(s.embed.bytes.ptr, g.head.dense.bytes.ptr);
    try std.testing.expect(try g.nativeBytes() > 0);
}
test "wrong BF16 metadata or shape cannot be fixed through a cast" {
    const a = std.testing.allocator;
    var s = try Store.init(a);
    defer s.deinit();
    s.tensors[0].dtype = .f16;
    try std.testing.expectError(error.DraftNativeDType, s.graph(a));
    s.tensors[0].dtype = .bf16;
    s.tensors[0].shape[1] -= 1;
    try std.testing.expectError(error.ProjectionShape, s.graph(a));
}
