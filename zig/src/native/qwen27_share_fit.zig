//! The 27B's late Neural Engine budget, with optional guards that can only make admission stricter.
const std = @import("std");
const fit = @import("cache_fit.zig");

pub const Guards = struct {
    ram_cap: ?u64 = null,
    footprint_floor: ?u64 = null,

    pub fn read() error{BadShareGuard}!Guards {
        return .{
            .ram_cap = try parseBytes(env("TF_QWEN27_FIT_RAM_BYTES")),
            .footprint_floor = try parseBytes(env("TF_QWEN27_FIT_FOOTPRINT_BYTES")),
        };
    }
};

fn env(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

fn parseBytes(text: ?[]const u8) error{BadShareGuard}!?u64 {
    const s = text orelse return null;
    const n = std.fmt.parseInt(u64, s, 10) catch return error.BadShareGuard;
    if (n == 0) return error.BadShareGuard;
    return n;
}

/// Ready bytes already include the drafter and residency; `later` contains only buffers not allocated yet.
pub fn room(total: ?u64, ready: ?u64, later: u64, guards: Guards) u64 {
    const ram = total orelse return 0;
    const used = ready orelse return 0;
    if (ram == 0 or used == 0) return 0;
    const bounded = if (guards.ram_cap) |cap| @min(ram, cap) else ram;
    const held = if (guards.footprint_floor) |floor| @max(used, floor) else used;
    return bounded / 100 * fit.SHARE_PERCENT -| held -| later -| fit.MARGIN;
}

const GiB: u64 = 1 << 30;
const MiB: u64 = 1 << 20;

test "the share fits before the drafter and wiring, but is refused by the finished 32 GiB server" {
    const share: u64 = 5760 * MiB;
    const early: u64 = 14 * GiB;
    const ready: u64 = 19 * GiB + GiB / 2;
    try std.testing.expect(share <= room(32 * GiB, early, 0, .{}));
    try std.testing.expect(share > room(32 * GiB, ready, 0, .{}));
    try std.testing.expect(share <= room(48 * GiB, ready, 0, .{}));
}

test "later slots and image buffers reduce room, while the exact edge remains usable" {
    const total: u64 = 32 * GiB;
    const later: u64 = 2 * GiB;
    const edge = total / 100 * fit.SHARE_PERCENT - fit.MARGIN;
    const ready = edge - later - 512 * MiB;
    try std.testing.expectEqual(@as(u64, 512 * MiB), room(total, ready, later, .{}));
    try std.testing.expectEqual(@as(u64, 512 * MiB - 1), room(total, ready + 1, later, .{}));
    try std.testing.expectEqual(@as(u64, 256 * MiB), room(total, ready, later + 256 * MiB, .{}));
    try std.testing.expectEqual(@as(u64, 0), room(total, ready, later + GiB, .{}));
}

test "unknown or zero OS readings never admit a share, even with guards" {
    const g: Guards = .{ .ram_cap = 32 * GiB, .footprint_floor = 19 * GiB };
    for ([_]?u64{ null, 0 }) |unread| {
        try std.testing.expectEqual(@as(u64, 0), room(unread, 14 * GiB, 0, g));
        try std.testing.expectEqual(@as(u64, 0), room(32 * GiB, unread, 0, g));
    }
}

test "guard variables only reduce the real budget" {
    const real = room(256 * GiB, 14 * GiB, 0, .{});
    const limited = room(256 * GiB, 14 * GiB, 0, .{ .ram_cap = 32 * GiB, .footprint_floor = 20 * GiB });
    try std.testing.expect(limited < 5760 * MiB and limited < real);
    try std.testing.expectEqual(real, room(256 * GiB, 14 * GiB, 0, .{
        .ram_cap = 512 * GiB,
        .footprint_floor = 1 * GiB,
    }));
}

test "oversized readings and later allocations saturate to no room" {
    const most = std.math.maxInt(u64);
    try std.testing.expectEqual(@as(u64, 0), room(32 * GiB, most, most, .{}));
    try std.testing.expectEqual(@as(u64, 0), room(most, most - 1, most, .{}));
}

test "guard byte counts reject malformed, zero, negative and overflowing values" {
    try std.testing.expectEqual(@as(?u64, null), try parseBytes(null));
    try std.testing.expectEqual(@as(?u64, 32 * GiB), try parseBytes("34359738368"));
    for ([_][]const u8{ "", "0", "-1", "x", "18446744073709551616" }) |s| {
        try std.testing.expectError(error.BadShareGuard, parseBytes(s));
    }
}
