//! The buffer plan: every role a window's calls name, sized by its largest view, over every row bucket.

const std = @import("std");
const calls = @import("block_calls.zig");

pub const Scope = enum { persistent, window, layer }; // "s." own allocation; "w."/"L." one arena

pub fn scopeOf(role: []const u8) Scope {
    if (std.mem.startsWith(u8, role, "s.")) return .persistent;
    if (std.mem.startsWith(u8, role, "w.")) return .window;
    return .layer;
}

pub fn dtSize(dt: calls.Dt) u64 {
    return switch (dt) {
        .bool, .i8, .u8 => 1,
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
    };
}

/// Bytes from the role's first byte that a view reaches (0: an empty tensor).
pub fn extent(t: calls.Tensor) u64 {
    var last: i64 = 0;
    for (t.shape, t.stride) |n, s| {
        if (n == 0) return 0;
        last += (n - 1) * s;
    }
    return @as(u64, @intCast(t.offset)) + (@as(u64, @intCast(last)) + 1) * dtSize(t.dt);
}

pub const Plan = struct {
    a: std.mem.Allocator,
    sizes: std.StringArrayHashMapUnmanaged(u64) = .empty,
    /// the weights the programs read by name (forms_inplace.zig leaves those in place)
    weights: std.StringArrayHashMapUnmanaged(void) = .empty,

    fn note(p: *Plan, x: calls.Arg) !void {
        switch (x) {
            .list => |items| for (items) |y| try p.note(y),
            .t, .opaque_table => |t| switch (t.role) {
                .buf => |r| {
                    const n = extent(t);
                    if (n == 0) return;
                    const g = try p.sizes.getOrPut(p.a, r);
                    if (!g.found_existing) {
                        g.key_ptr.* = try p.a.dupe(u8, r);
                        g.value_ptr.* = 0;
                    }
                    g.value_ptr.* = @max(g.value_ptr.*, n);
                },
                .weight => |w| {
                    const g = try p.weights.getOrPut(p.a, w);
                    if (!g.found_existing) g.key_ptr.* = try p.a.dupe(u8, w);
                },
                else => {},
            },
            else => {},
        }
    }

    /// Takes every role of `cs` into the plan (call once a row bucket).
    pub fn add(p: *Plan, cs: []const calls.Call) !void {
        for (cs) |c| for (c.args) |x| try p.note(x.arg);
    }

    pub const Totals = struct { roles: [3]usize = @splat(0), bytes: [3]u64 = @splat(0) };

    /// Bytes by scope, each role rounded up to 256 (its place in the arena or its own allocation).
    pub fn totals(p: *const Plan) Totals {
        var t: Totals = .{};
        for (p.sizes.keys(), p.sizes.values()) |k, v| {
            const s = @backingInt(scopeOf(k));
            t.roles[s] += 1;
            t.bytes[s] += std.mem.alignForward(u64, v, 256);
        }
        return t;
    }
};

const testing = std.testing;

test "extent: a view's reach from the role's first byte" {
    const t: calls.Tensor = .{ .role = .{ .buf = "w.x" }, .dt = .bf16, .shape = &.{ 3, 4096 }, .stride = &.{ 16384, 1 }, .offset = 8192 };
    try testing.expectEqual(@as(u64, 8192 + (2 * 16384 + 4095 + 1) * 2), extent(t));
    const e: calls.Tensor = .{ .role = .{ .buf = "w.x" }, .dt = .f32, .shape = &.{0}, .stride = &.{1} };
    try testing.expectEqual(@as(u64, 0), extent(e));
}
