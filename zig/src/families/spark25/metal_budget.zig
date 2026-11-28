//! What a Spark2.5 load on Metal may hold: weights, scratch and every stream's caches, below the panic line.
const std = @import("std");
const st = @import("metal_state.zig");

/// The default share of RAM: past about 80% wired, macOS stalls until its watchdog restarts the Mac.
pub const default_share = 0.75;
/// Kept free inside the limit for the command buffers, the keepalive and the host.
pub const margin: u64 = 2 << 30;

/// Bytes for caches: `ram` times the share (or SPARK_LOAD_LIMIT_GB's GB, at most the GPU's working set), less the rest.
pub fn caches(ram: u64, working_set: u64, text: ?[]const u8, kept: u64) error{BadLoadLimit}!u64 {
    var limit: u64 = @intFromFloat(@as(f64, @floatFromInt(ram)) * default_share);
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len > 0) {
        const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
        if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
        limit = @min(@as(u64, @intFromFloat(gb * 1e9)), @min(ram, working_set));
    }
    return if (limit > kept + margin) limit - kept - margin else 0;
}

/// What a loaded `model` keeps resident besides caches: its shards, its stacked matrices and the windows' scratch.
pub fn resident(model: anytype) u64 {
    return model.checkpoint.residentBytes() + model.weights.stacked_bytes + scratch;
}

/// The decode window's and the prompt chunk's buffers.
pub const scratch: u64 = st.scratchBytes(st.batch_rows) + st.scratchBytes(st.prompt_rows);

/// The longest context one stream's caches fit in `bytes`, at most `most`.
pub fn longest(bytes: u64, most: usize) usize {
    if (st.cacheBytes(1) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (st.cacheBytes(mid) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

test "spark metal budget: a share of RAM less the weights, a set limit capped at the working set, bad text refused" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 192 * gib - 6 * gib - margin), try caches(256 * gib, 240 * gib, null, 6 * gib));
    try std.testing.expectEqual(@as(u64, 0), try caches(256 * gib, 240 * gib, null, 191 * gib));
    try std.testing.expectEqual(@as(u64, 64 * gib - 6 * gib - margin), try caches(256 * gib, 240 * gib, "68.719476736", 6 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "lots", 6 * gib));
    try std.testing.expectEqual(@as(usize, 4096), longest(st.cacheBytes(4096), 1 << 20));
}

test "spark metal budget: the stacked matrices count as resident, so a load limit they push over leaves no caches" {
    const gib: u64 = 1 << 30;
    const Fake = struct {
        checkpoint: struct {
            bytes: usize,
            pub fn residentBytes(c: @This()) usize {
                return c.bytes;
            }
        },
        weights: struct { stacked_bytes: u64 },
    };
    const m: Fake = .{ .checkpoint = .{ .bytes = 1 * gib }, .weights = .{ .stacked_bytes = gib / 2 } };
    try std.testing.expectEqual(@as(u64, 1 * gib + gib / 2 + scratch), resident(m));
    const old = m.checkpoint.residentBytes() + scratch; // the accounting before: the shards and scratch only
    try std.testing.expectEqual(try caches(32 * gib, 24 * gib, null, old) - gib / 2, try caches(32 * gib, 24 * gib, null, resident(m)));
    // a limit with room for the shards, scratch and one token's caches, but not for the stacks as well
    const room = old + margin + st.cacheBytes(1) + gib / 8;
    const text = try std.fmt.allocPrint(std.testing.allocator, "{d}", .{@as(f64, @floatFromInt(room)) / 1e9});
    defer std.testing.allocator.free(text);
    try std.testing.expect(longest(try caches(32 * gib, 24 * gib, text, old), 1 << 20) > 0);
    try std.testing.expectEqual(@as(usize, 0), longest(try caches(32 * gib, 24 * gib, text, resident(m)), 1 << 20));
}
