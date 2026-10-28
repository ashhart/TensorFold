//! Host-only learner ABI, reservation and native cache-epoch regressions.
const std = @import("std");
const to = @import("train_ops.zig");
const config = @import("config.zig");

test "qwen27 learning: Metal argument layouts match the generated wrappers" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(to.Shape));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(to.Shape, "f"));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(to.GdnShape));
    try std.testing.expectEqual(@as(usize, 28), @sizeOf(to.Heads));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(to.Heads, "scale"));
}

test "qwen27 learning: the reserve covers both native frame widths and every GDN checkpoint" {
    var c: config.Config = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 10000000 };
    for (0..64) |l| c.kinds[l] = if (l % 4 == 3) .attention else .linear;
    const small = try @import("train_memory.zig").bytes(c, 128, true);
    const wide = try @import("train_memory.zig").bytes(c, 2048, true);
    try std.testing.expect(wide > small);
    const checkpoints = (1024 / 4 + 8) * c.v_heads * c.dv * c.dk * 4;
    try std.testing.expect(small > checkpoints * 39);
    try std.testing.expect(wide < 128 << 30);
}

test "qwen27 learning: moving weights invalidates standing, rewind metadata, draft trees and the pending round" {
    const Fake = struct {
        gpa: std.mem.Allocator,
        standing: struct {
            tokens: usize = 17,
            rewind_at: u32 = 11,
            pub fn deinit(s: *@This(), _: std.mem.Allocator) void {
                s.* = .{ .tokens = 0, .rewind_at = 0 };
            }
        } = .{},
        trees: usize = 3,
        round: bool = true,
        pub fn dropTree(f: *@This()) void {
            f.trees = 0;
        }
        pub fn dropRound(f: *@This()) void {
            f.round = false;
        }
    };
    var fake: Fake = .{ .gpa = std.testing.allocator };
    @import("train_epoch.zig").changed(&fake);
    try std.testing.expectEqual(@as(usize, 0), fake.standing.tokens);
    try std.testing.expectEqual(@as(u32, 0), fake.standing.rewind_at);
    try std.testing.expectEqual(@as(usize, 0), fake.trees);
    try std.testing.expect(!fake.round);
}

test "qwen27 learning: detached disk rows retain their server owner until close" {
    const Row = struct { stopped: bool = false };
    const Store = struct {
        disk: ?*Row,
        pub fn clear(s: *@This()) void {
            if (s.disk) |r| r.stopped = true;
            s.disk = null;
        }
    };
    var rows: Row = .{};
    var host: struct { disk_rows: ?*Row } = .{ .disk_rows = &rows };
    var store: Store = .{ .disk = &rows };
    store.clear();
    try std.testing.expect(rows.stopped and store.disk == null);
    const owned = @import("train_epoch.zig").takeDisk(&host);
    try std.testing.expect(owned.? == &rows);
    try std.testing.expect(@import("train_epoch.zig").takeDisk(&host) == null);
}
