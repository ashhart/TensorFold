//! Whether a model can start now: room under --max-loaded and the machine's memory, or the idle child to stop first.
const std = @import("std");

/// A slot's life: a child counts as loaded from the start of its load to the end of its stop.
pub const State = enum { unloaded, loading, ready, stopping };

/// What the decision reads of one slot.
pub const View = struct { state: State, in_flight: u32 = 0, waiters: u32 = 0, used_ns: i96 = 0, floor: ?u64 = null };

/// Start now; stop that idle child first; wait for a busy one; or never, as the model needs more than the machine.
pub const Room = union(enum) { fits, stop: usize, wait, never };

/// The decision for ``want``: at most ``max_loaded`` loaded, their floors within ``capacity`` (0: no memory check).
pub fn decide(views: []const View, want: usize, max_loaded: u32, capacity: u64) Room {
    const need = views[want].floor;
    if (capacity > 0) if (need) |n| if (n > capacity) return .never;
    var loaded: u32 = 0;
    var held: u64 = 0;
    var idle: u32 = 0;
    var freeable: u64 = 0;
    var victim: ?usize = null;
    for (views, 0..) |v, i| {
        if (i == want or v.state == .unloaded) continue;
        loaded += 1;
        held += v.floor orelse 0;
        if (!isIdle(v)) continue;
        idle += 1;
        freeable += v.floor orelse 0;
        if (victim == null or v.used_ns < views[victim.?].used_ns) victim = i;
    }
    if (loaded < max_loaded and fits(held, need, capacity)) return .fits;
    // stop an idle child only when stopping every idle one would make room: a busy one must finish first
    if (victim) |v| if (loaded - idle < max_loaded and fits(held - freeable, need, capacity)) return .{ .stop = v };
    return .wait;
}

/// A ready child with no request in flight and none waiting for it.
pub fn isIdle(v: View) bool {
    return v.state == .ready and v.in_flight == 0 and v.waiters == 0;
}

fn fits(held: u64, need: ?u64, capacity: u64) bool {
    const n = need orelse return true;
    return capacity == 0 or held + n <= capacity;
}

test "serve --router: under --max-loaded a model starts, and past it the least recently used idle child stops first" {
    const t = std.testing;
    var v = [_]View{ .{ .state = .unloaded }, .{ .state = .unloaded }, .{ .state = .unloaded } };
    try t.expectEqual(Room.fits, decide(&v, 0, 2, 0));
    v[0] = .{ .state = .ready, .used_ns = 30 };
    v[1] = .{ .state = .ready, .used_ns = 10 };
    try t.expectEqual(Room{ .stop = 1 }, decide(&v, 2, 2, 0));
    v[1].used_ns = 40;
    try t.expectEqual(Room{ .stop = 0 }, decide(&v, 2, 2, 0));
    try t.expectEqual(Room.fits, decide(&v, 2, 3, 0));
    v[0].in_flight = 1;
    try t.expectEqual(Room{ .stop = 1 }, decide(&v, 2, 2, 0));
    v[1].waiters = 1;
    try t.expectEqual(Room.wait, decide(&v, 2, 2, 0));
    v[1] = .{ .state = .stopping };
    try t.expectEqual(Room.wait, decide(&v, 2, 2, 0));
    v[1] = .{ .state = .loading };
    try t.expectEqual(Room.wait, decide(&v, 2, 2, 0));
}

test "serve --router: memory floors bound what loads, a model larger than the machine never starts, an unknown floor passes" {
    const t = std.testing;
    const gib: u64 = 1 << 30;
    var v = [_]View{
        .{ .state = .ready, .used_ns = 1, .floor = 30 * gib },
        .{ .state = .ready, .used_ns = 2, .floor = 10 * gib },
        .{ .state = .unloaded, .floor = 40 * gib },
    };
    try t.expectEqual(Room{ .stop = 0 }, decide(&v, 2, 8, 64 * gib));
    v[0].in_flight = 1;
    try t.expectEqual(Room.wait, decide(&v, 2, 8, 64 * gib));
    v[2].floor = 20 * gib;
    try t.expectEqual(Room.fits, decide(&v, 2, 8, 64 * gib));
    v[2].floor = 65 * gib;
    try t.expectEqual(Room.never, decide(&v, 2, 8, 64 * gib));
    v[2].floor = null;
    try t.expectEqual(Room.fits, decide(&v, 2, 8, 64 * gib));
    try t.expectEqual(Room.fits, decide(&v, 2, 8, 0));
}
