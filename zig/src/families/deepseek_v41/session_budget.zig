//! `--prompt-cache-gib` as the kept-state store's resident byte limit (sessions_gpu.zig `Options.resident_bytes`).

const std = @import("std");

/// Why the flag set nothing: the variable is set, or several ranks (the followers never see the flag).
pub const Ignored = enum {
    env,
    ranks,

    pub fn text(i: Ignored) []const u8 {
        return switch (i) {
            .env => "TF_DSV41_SESSION_RESIDENT_GIB is set and wins",
            .ranks => "with TP > 1 only rank 0 sees it; set TF_DSV41_SESSION_RESIDENT_GIB on every rank",
        };
    }
};

/// The resident byte limit (null: unlimited; 0: no idle state kept) and, when the flag lost, why.
pub const Limit = struct { bytes: ?u64, ignored: ?Ignored = null };

/// `env`: TF_DSV41_SESSION_RESIDENT_GIB's bytes; `gib`: the flag (the server's checked 0 to 2^34); `world`: ranks.
pub fn resident(env: ?u64, gib: ?f64, world: u32) Limit {
    const g = gib orelse return .{ .bytes = env };
    if (env != null) return .{ .bytes = env, .ignored = .env };
    if (world > 1) return .{ .bytes = null, .ignored = .ranks };
    return .{ .bytes = @intFromFloat(g * (1 << 30)) };
}

test "prompt cache: --prompt-cache-gib 4 sets a resident limit of 4 << 30 bytes" {
    try std.testing.expectEqual(Limit{ .bytes = 4 << 30 }, resident(null, 4, 1));
    try std.testing.expectEqual(Limit{ .bytes = 3 << 29 }, resident(null, 1.5, 1));
}

test "prompt cache: --prompt-cache-gib 0 keeps no idle state" {
    try std.testing.expectEqual(Limit{ .bytes = 0 }, resident(null, 0, 1));
}

test "prompt cache: a set TF_DSV41_SESSION_RESIDENT_GIB wins over the flag and says so" {
    try std.testing.expectEqual(Limit{ .bytes = 1 << 30, .ignored = .env }, resident(1 << 30, 4, 1));
    try std.testing.expectEqual(Limit{ .bytes = 2 << 30, .ignored = .env }, resident(2 << 30, 0, 1));
    try std.testing.expectEqual(Limit{ .bytes = 1 << 30 }, resident(1 << 30, null, 1));
}

test "prompt cache: neither the flag nor the variable keeps today's unlimited default" {
    try std.testing.expectEqual(Limit{ .bytes = null }, resident(null, null, 1));
    try std.testing.expectEqual(Limit{ .bytes = null }, resident(null, null, 4));
}

test "prompt cache: with TP > 1 the flag leaves the limit to the variable on every rank" {
    try std.testing.expectEqual(Limit{ .bytes = null, .ignored = .ranks }, resident(null, 4, 2));
    try std.testing.expectEqual(Limit{ .bytes = 1 << 30, .ignored = .env }, resident(1 << 30, 4, 2));
}
