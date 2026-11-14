//! Gemma 4's kept states for the disk store (native/cuda_kept.zig): layout, a blank one, what they are read under.
const std = @import("std");
const cuda = @import("cuda");
const be = @import("cuda_backend.zig");
const cfg = @import("cuda_config.zig");
const fwd = @import("cuda_forward.zig");
const snap = @import("cuda_snap.zig");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

pub const Target = be.Gemma;
pub const State = snap.Saved;
pub const probe_seed: u64 = 0x6734;

/// Owned layers, K then V, as snap.each lays them; a ring's tail is the state's own, as is a drafter's context.
pub fn parts(t: *const Target, at: u32, out: anytype, a: Allocator) !void {
    const m = t.m;
    for (0..fwd.pages.ownLayers(m)) |i| for (0..2) |_| {
        const row = snap.rowBytes(m, i);
        if (cfg.global(i)) {
            try out.append(a, .{ .bytes = @as(u64, at) * row, .row = @intCast(row) });
        } else try out.append(a, .{ .bytes = snap.tail(at) * row });
    };
    if (t.keeper) |kp| try out.append(a, .{ .bytes = kp.bytes(kp.ctx, at) });
}

pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
    return .{ st.buf, st.draft };
}

pub fn blank(t: *Target, at: u32) !*State {
    return snap.blank(t.gpa, t.m, at, if (t.keeper) |kp| kp.bytes(kp.ctx, at) else 0);
}

/// The target's rows are all here: not in the store's pages, no image RAM alone can tell.
pub fn probed(_: *const Target, st: *const State) bool {
    return st.from == 0 and st.blob.len == 0 and !st.images;
}

/// A state with images, or without the context a drafter keeps with it, stays off disk.
pub fn whole(t: *const Target, st: *const State) bool {
    return probed(t, st) and (t.keeper == null or st.draft != null);
}

pub fn drafts(_: *const Target) bool {
    return false; // the probe hashes the target's rows; a drafter's checkpoint is in the key (promptHash)
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

/// The KV form past what the probe shows: the layers' rows, a ring's keys, pages or whole caches, a drafter's context.
pub fn facts(t: *const Target) [5]u64 {
    const m = t.m;
    var h = std.hash.Wyhash.init(0x6734);
    const layers = fwd.pages.ownLayers(m);
    for (0..layers) |i| h.update(std.mem.asBytes(&[_]u64{ snap.rowBytes(m, i), @intFromBool(cfg.global(i)) }));
    return .{ layers, h.final(), cfg.window, @intFromBool(m.paged), @intFromBool(t.keeper != null) };
}

const testing = std.testing;

const Piece = struct { bytes: u64, row: u32 = 0 };

const Offsets = struct {
    ends: *std.ArrayList(u64),
    fn add(c: Offsets, _: u64, off: usize, len: usize) anyerror!void {
        try c.ends.append(testing.allocator, off + len);
    }
};

fn model(m: *fwd.Model, k: []u64, v: []u64) void {
    m.geo = cfg.unified12b;
    m.layers = 6; // layer 5 is global
    m.context = 4096;
    m.paged = false;
    m.kc = k;
    m.vc = v;
}

test "Gemma disk: the parts are snap.each's layers in its order, and a part's bytes are its copies'" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30, 12 << 30 };
    var m: fwd.Model = undefined;
    model(&m, &k, &v);
    var t: Target = undefined;
    t.m = &m;
    t.keeper = null;
    for ([_]u32{ 1, 1023, 1024, 2000, 4000 }) |at| {
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

const Draft = struct {
    fn bytes(_: *anyopaque, at: usize) u64 {
        return at * 64;
    }
    fn keep(_: *anyopaque, _: *const lanes.Stream, _: *snap.Saved) anyerror!void {}
};

test "Gemma disk: images and a missing drafter context stay off disk; a drafter or paging is another key" {
    var k: [6]u64 = @splat(0);
    var m: fwd.Model = undefined;
    model(&m, &k, &k);
    var t: Target = undefined;
    t.m = &m;
    t.keeper = null;
    var st: State = .{ .at = 8, .buf = undefined };
    try testing.expect(whole(&t, &st));
    st.images = true;
    try testing.expect(!whole(&t, &st));
    st.images = false;
    const base = facts(&t);
    var ctx: u8 = 0;
    t.keeper = .{ .ctx = &ctx, .bytes = Draft.bytes, .keep = Draft.keep };
    try testing.expect(!whole(&t, &st) and probed(&t, &st)); // a kept state needs its drafter's context, a probe not
    var list: std.ArrayList(Piece) = .empty;
    defer list.deinit(testing.allocator);
    try parts(&t, 8, &list, testing.allocator);
    try testing.expectEqual(@as(u64, 8 * 64), list.items[list.items.len - 1].bytes);
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
    t.keeper = null;
    m.paged = true;
    try testing.expect(!std.mem.eql(u64, &base, &facts(&t)));
}
