//! Dory's kept states for the disk store (native/cuda_kept.zig): their layout, a blank one, what they are read under.
const std = @import("std");
const cuda = @import("cuda");
const be = @import("cuda_backend.zig");
const snap = @import("cuda_snap.zig");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

pub const Target = be.Dory;
pub const State = snap.Saved;
pub const probe_seed: u64 = 0x646f;

/// Every cache, K then V, as snap.each lays them: a full one's rows from position 0, a ring's tail the state's own.
pub fn parts(t: *const Target, at: u32, out: anytype, a: Allocator) !void {
    const m = t.m;
    const row = m.c.kvDim() * 2;
    for (m.sched.cache_kind[0..m.sched.caches]) |kind| for (0..2) |_| {
        if (kind == .full) {
            try out.append(a, .{ .bytes = @as(u64, at) * row, .row = @intCast(row) });
        } else try out.append(a, .{ .bytes = snap.tail(m.c.window, at) * row });
    };
}

pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
    return .{ st.buf, null };
}

pub fn blank(t: *Target, at: u32) !*State {
    return snap.blank(t.gpa, t.m, at);
}

pub fn drafts(_: *const Target) bool {
    return false; // Dory's drafts come from the prompt's text: no head's state
}

pub fn context(t: *const Target) usize {
    return t.m.context;
}

pub fn save(t: *Target, s: *lanes.Stream, at: u32) !*State {
    return t.saveState(s, at);
}

pub fn drop(t: *Target, st: *State) void {
    t.dropState(st);
}

/// The KV form past what the probe shows: loops and the state key, a ring's keys, a row, pages or whole caches.
pub fn facts(t: *const Target) [5]u64 {
    const m = t.m;
    const key = snap.keyOf(m.c);
    return .{ key.loops, @backingInt(key.mode), m.c.window, m.c.kvDim(), @intFromBool(m.paged) };
}

const testing = std.testing;
const fwd = @import("cuda_model.zig");
const config = @import("config.zig");

fn pinned(loops: usize) !config.Config {
    const text = try config.pinnedText(testing.allocator);
    defer testing.allocator.free(text);
    var why: config.Why = .{};
    return (try config.parse(testing.allocator, text, &why)).withLoops(loops);
}

const Piece = struct { bytes: u64, row: u32 = 0 };

const Offsets = struct {
    ends: *std.ArrayList(u64),
    fn add(c: Offsets, _: u64, off: usize, len: usize) anyerror!void {
        try c.ends.append(testing.allocator, off + len);
    }
};

test "Dory disk: the parts are snap.each's caches in its order, and a part's bytes are its copies'" {
    var m: fwd.Model = undefined;
    m.c = try pinned(2);
    m.sched = @import("schedule.zig").build(m.c);
    m.context = 8192;
    m.paging = null;
    m.ring = fwd.ringSize(fwd.prompt_rows, m.c.window);
    var ks: [64]u64 = @splat(1 << 32);
    m.k = ks[0..m.sched.caches];
    m.v = ks[0..m.sched.caches];
    var t: Target = undefined;
    t.m = &m;
    for ([_]u32{ 1, 1023, 1024, 4000, 8000 }) |at| {
        var list: std.ArrayList(Piece) = .empty;
        defer list.deinit(testing.allocator);
        try parts(&t, at, &list, testing.allocator);
        var ends: std.ArrayList(u64) = .empty;
        defer ends.deinit(testing.allocator);
        try snap.each(&m, 0, at, 0, Offsets{ .ends = &ends }, Offsets.add);
        var off: u64 = 0;
        for (list.items) |p| {
            off += p.bytes;
            try testing.expect(std.mem.indexOfScalar(u64, ends.items, off) != null); // a part ends where a copy does
        }
        try testing.expectEqual(snap.bytes(&m, at), off);
    }
}

test "Dory disk: another loop count, ring or paging is another key" {
    var m: fwd.Model = undefined;
    m.c = try pinned(2);
    m.paged = false;
    var t: Target = undefined;
    t.m = &m;
    const base = facts(&t);
    m.c = try pinned(4);
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
    m.c = try pinned(2);
    m.c.window -= 1;
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
    m.c = try pinned(2);
    m.paged = true;
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
}
