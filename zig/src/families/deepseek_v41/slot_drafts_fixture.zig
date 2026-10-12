//! Slot drafts tests' fixture: the K2 widths, and call-list comparisons (roles, Triton classes, args).

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const config = @import("config.zig");
const Config = config.Config;

pub fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    // the ingest's main_proj (not in the backbone fixture): a width of the DSpark blocks' own
    try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
    return w;
}

pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    cfg: Config = .{},
    w: block.Widths = undefined,
    o: block.Options = .{ .taps = true },

    pub fn init(f: *Fixture) !void {
        f.arena = std.heap.ArenaAllocator.init(testing.allocator);
        f.w = try widths(f.arena.allocator());
    }
};

pub fn roleOf(t: calls.Tensor) []const u8 {
    return switch (t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

/// An argument's Triton specialization class: a tensor's role and address mod 16, an int's (== 1, % 16), else its tag.
pub fn class(x: calls.Arg) [3]i64 {
    return switch (x) {
        .t => |t| .{ 1, @mod(t.offset, 16), @backingInt(t.dt) },
        .i => |v| .{ 2, @intFromBool(v == 1), @intFromBool(@mod(v, 16) == 0) },
        .f => .{ 3, 0, 0 },
        .b => |v| .{ 4, @intFromBool(v), 0 },
        .none => .{ 5, 0, 0 },
        else => .{ 6, 0, 0 },
    };
}

pub fn sameCall(a: calls.Call, b: calls.Call) !void {
    try testing.expectEqualStrings(a.name, b.name);
    try testing.expectEqual(a.args.len, b.args.len);
    for (a.args, b.args) |x, y| try testing.expectEqual(std.meta.activeTag(x.arg), std.meta.activeTag(y.arg));
}

pub fn equalArgs(x: calls.Arg, y: calls.Arg) bool {
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            const u = switch (y) {
                .t, .opaque_table => |u| u,
                else => break :blk false,
            };
            break :blk std.mem.eql(u8, roleOf(t), roleOf(u)) and std.meta.activeTag(t.role) == std.meta.activeTag(u.role) and t.dt == u.dt and t.offset == u.offset and std.mem.eql(i64, t.shape, u.shape) and std.mem.eql(i64, t.stride, u.stride);
        },
        .list => |l| y == .list and l.len == y.list.len and for (l, y.list) |p, q| {
            if (!equalArgs(p, q)) break false;
        } else true,
        else => std.meta.eql(x, y),
    };
}

pub fn identical(a: []const calls.Call, b: []const calls.Call) !void {
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| {
        try sameCall(x, y);
        try testing.expectEqual(x.triton, y.triton);
        try testing.expectEqual(x.glue, y.glue);
        try testing.expectEqual(x.begin, y.begin);
        try testing.expectEqual(x.grid, y.grid);
        for (x.args, y.args) |p, q| {
            try testing.expectEqualStrings(p.name, q.name);
            try testing.expect(equalArgs(p.arg, q.arg));
        }
    }
}
