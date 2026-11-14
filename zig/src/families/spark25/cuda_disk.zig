//! Spark2.5's kept states for the disk store (native/cuda_kept.zig): layout, a blank one, what they are read under.
const std = @import("std");
const cuda = @import("cuda");
const be = @import("backend.zig");
const cfg = @import("config.zig");
const snap = @import("snap.zig");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

pub const Target = be.Spark;
pub const State = snap.Saved;
pub const probe_seed: u64 = 0x7370;

/// Every layer, K then V, as snap.each lays them: a full one's rows from position 0, a ring's tail the state's own.
pub fn parts(t: *const Target, at: u32, out: anytype, a: Allocator) !void {
    const m = t.m;
    const row = snap.rowBytes(m);
    for (0..m.layers) |i| for (0..2) |_| {
        if (cfg.full(i)) {
            try out.append(a, .{ .bytes = @as(u64, at) * row, .row = @intCast(row) });
        } else try out.append(a, .{ .bytes = snap.tail(at) * row });
    };
}

pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
    return .{ st.buf, null };
}

pub fn blank(t: *Target, at: u32) !*State {
    return snap.blank(t.gpa, t.m, at);
}

/// A state in the store's pages is not this image: TF_KV_SHARE refuses --learn-conversations, so it never comes.
pub fn whole(_: *const Target, st: *const State) bool {
    return st.from == 0 and st.blob.len == 0 and st.buf != null;
}

pub fn drafts(_: *const Target) bool {
    return false; // Spark2.5's drafts come from the prompt's text: no head's state
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

/// The KV form past what the probe shows: the layers, a row, a ring's keys, pages or whole caches.
pub fn facts(t: *const Target) [4]u64 {
    return .{ t.m.layers, snap.rowBytes(t.m), cfg.window, @intFromBool(t.m.paged) };
}

const testing = std.testing;
const fwd = @import("forward.zig");

const Piece = struct { bytes: u64, row: u32 = 0 };

const Offsets = struct {
    ends: *std.ArrayList(u64),
    fn add(c: Offsets, _: u64, off: usize, len: usize) anyerror!void {
        try c.ends.append(testing.allocator, off + len);
    }
};

test "Spark2.5 disk: the parts are snap.each's layers in its order, and a part's bytes are its copies'" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30 };
    var m: fwd.Model = undefined;
    m.geo = cfg.spark4b;
    m.layers = 4; // layer 3 is full
    m.context = 4096;
    m.paged = false;
    m.kc = &k;
    m.vc = &v;
    var t: Target = undefined;
    t.m = &m;
    for ([_]u32{ 1, 511, 512, 1500, 4000 }) |at| {
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
        try testing.expectEqual(snap.bytes(&m, at, 0), off);
    }
}

test "Spark2.5 disk: a state in pages stays off disk; paging or another geometry is another key" {
    var m: fwd.Model = undefined;
    m.geo = cfg.spark4b;
    m.layers = 4;
    m.paged = false;
    var t: Target = undefined;
    t.m = &m;
    var ids = [_]u32{0};
    var st: State = .{ .at = 8, .blob = &ids };
    try testing.expect(!whole(&t, &st));
    st.blob = &.{};
    try testing.expect(!whole(&t, &st)); // no buffer either
    const base = facts(&t);
    m.paged = true;
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
    m.paged = false;
    m.layers = 3;
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
}
