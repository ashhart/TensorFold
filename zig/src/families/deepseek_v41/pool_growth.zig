//! A pre-launch pool transaction: every rank maps its rows or every rank restores the previous page ownership.

const sessions = @import("kv").sessions;
const lockstep = @import("lockstep.zig");

pub const Need = struct { slot: *sessions.Slot, end: u64 };

/// No device table or model state is changed here; failed agreement retains ownership for fatal teardown.
pub fn reserve(agree: ?lockstep.Agree, needs: []const Need) !void {
    if (needs.len > 16) return error.Slots;
    var before: [16]u64 = undefined;
    var over: [16]u32 = undefined;
    var quota: [16]?u32 = undefined;
    var origin: [16]@TypeOf(@as(sessions.Slot, undefined).origin) = undefined;
    for (needs, 0..) |n, i| {
        if (n.end > @as(u64, n.slot.max_pages) * n.slot.pool.page) return error.PastCapacity;
        before[i] = n.slot.tokens();
        over[i] = n.slot.over;
        quota[i] = n.slot.quota;
        origin[i] = n.slot.origin;
        for (needs[0..i]) |prior| if (prior.slot == n.slot) return error.Slots;
    }
    var ok = true;
    for (needs) |n| {
        // Unadmitted callers may grow only into pages that no live request reserved.
        if (n.slot.quota == null) n.slot.reserve(n.slot.len);
        n.slot.ensure(n.end) catch |err| switch (err) {
            error.PoolExhausted => {
                ok = false;
                break;
            },
            else => return err,
        };
    }
    if (agree) |a| ok = try a.all(a.ctx, ok);
    if (!ok) {
        for (needs, 0..) |n, i| {
            _ = try n.slot.truncate(before[i]);
            n.slot.reserve(quota[i]);
            n.slot.over = over[i];
            n.slot.origin = origin[i];
        }
        return error.PoolExhausted;
    }
    for (needs, 0..) |n, i| if (quota[i] == null) {
        n.slot.reserve(null);
        n.slot.over = over[i];
    };
}
