//! A stream's state after its first `at` prompt tokens, copied to host memory and back: the prompt cache's entries.
const std = @import("std");
const builtin = @import("builtin");
const hip = @import("hip");
const cache = @import("cache.zig");

/// The device and the pinned memory a state's copies pass through (empty: none, a big copy pins the pages it reads).
pub const Copy = struct { d: *const hip.Runtime, bounce: []u8 = &.{} };

/// `dst.len` bytes from device address `src`, through the bounce buffer when there is one.
fn down(x: Copy, dst: []u8, src: u64) !void {
    if (x.bounce.len == 0) return hip.runtime.check(x.d.api.hipMemcpyDtoH(dst.ptr, @ptrFromInt(src), dst.len));
    var k: usize = 0;
    while (k < dst.len) : (k += x.bounce.len) {
        const n = @min(x.bounce.len, dst.len - k);
        try hip.runtime.check(x.d.api.hipMemcpyDtoH(x.bounce.ptr, @ptrFromInt(src + k), n));
        @memcpy(dst[k..][0..n], x.bounce[0..n]);
    }
}

/// `src` to device address `dst`, through the bounce buffer when there is one.
fn up(x: Copy, dst: u64, src: []const u8) !void {
    if (x.bounce.len == 0) return hip.runtime.check(x.d.api.hipMemcpyHtoD(@ptrFromInt(dst), @constCast(src.ptr), src.len));
    var k: usize = 0;
    while (k < src.len) : (k += x.bounce.len) {
        const n = @min(x.bounce.len, src.len - k);
        @memcpy(x.bounce[0..n], src[k..][0..n]);
        try hip.runtime.check(x.d.api.hipMemcpyHtoD(@ptrFromInt(dst + k), x.bounce.ptr, n));
    }
}

/// A kept state's bytes on small pages of their own (Linux): a huge-page fault per state compacted under the kernels.
fn take(n: usize, gpa: std.mem.Allocator) ![]u8 {
    if (comptime builtin.os.tag != .linux) return gpa.alloc(u8, n);
    if (n == 0) return &.{};
    const m = try std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    _ = std.os.linux.madvise(m.ptr, m.len, std.os.linux.MADV.NOHUGEPAGE);
    return m;
}

fn give(b: []u8, gpa: std.mem.Allocator) void {
    if (comptime builtin.os.tag != .linux) return gpa.free(b);
    if (b.len > 0) std.posix.munmap(@alignCast(b));
}

/// One span of a cache's device memory a state at `at` holds.
const Region = struct { offset: usize, len: usize };

/// The spans of `c` holding the state at `at`: the sparse layers' first `at` rows a kv head at a time, the rest whole.
fn regions(c: *const cache.Caches, at: usize, ratio: usize, out: *std.ArrayList(Region), gpa: std.mem.Allocator) !void {
    const l = c.layout;
    const kv_heads = l.kv_heads;
    const hd_bytes = l.head_dim * l.act;
    const di_bytes = l.index_dim * l.act;
    var base: usize = 0;
    for (0..l.sparse) |_| {
        for (0..2) |kv| for (0..kv_heads) |h| {
            try out.append(gpa, .{ .offset = base + kv * l.kv_bytes + h * l.cap * hd_bytes, .len = at * hd_bytes });
        };
        try out.append(gpa, .{ .offset = base + 2 * l.kv_bytes, .len = at * di_bytes });
        try out.append(gpa, .{ .offset = base + 2 * l.kv_bytes + l.raw_bytes, .len = at / ratio * di_bytes });
        base += 2 * l.kv_bytes + l.raw_bytes + l.pooled_bytes;
    }
    try out.append(gpa, .{ .offset = base, .len = l.bytes() - base });
}

pub const Saved = struct {
    at: u32,
    pooled: usize,
    history: [2]i64,
    main: []u8,
    head: []u8,
    /// The slot the other ranks keep their own copy under.
    slot: u32 = 0,

    pub fn bytes(s: *const Saved) u64 {
        return s.main.len + s.head.len;
    }
};

/// `side`: the part past the sparse layers (the last region) from a side buffer instead (forward.Mark).
fn copyOut(x: Copy, c: *const cache.Caches, at: usize, ratio: usize, side: ?u64, gpa: std.mem.Allocator) ![]u8 {
    var list: std.ArrayList(Region) = .empty;
    defer list.deinit(gpa);
    try regions(c, at, ratio, &list, gpa);
    var n: usize = 0;
    for (list.items) |r| n += r.len;
    const out = try take(n, gpa);
    errdefer give(out, gpa);
    var off: usize = 0;
    for (list.items, 0..) |r, i| {
        const src = if (side != null and i + 1 == list.items.len) side.? else c.mem.base() + r.offset;
        if (r.len > 0) try down(x, out[off..][0..r.len], src);
        off += r.len;
    }
    return out;
}

fn copyIn(
    x: Copy,
    c: *const cache.Caches,
    at: usize,
    ratio: usize,
    bytes: []const u8,
    gpa: std.mem.Allocator,
) !void {
    var list: std.ArrayList(Region) = .empty;
    defer list.deinit(gpa);
    try regions(c, at, ratio, &list, gpa);
    var off: usize = 0;
    for (list.items) |r| {
        if (off + r.len > bytes.len) return error.SnapshotMismatch;
        if (r.len > 0) try up(x, c.mem.base() + r.offset, bytes[off..][0..r.len]);
        off += r.len;
    }
    if (off != bytes.len) return error.SnapshotMismatch;
}

/// The state of `c` (and the head's `head`) after `at` prompt tokens, copied to host memory; the stream is idle.
pub fn save(
    gpa: std.mem.Allocator,
    x: Copy,
    s: hip.Stream,
    c: *const cache.Caches,
    head: ?*const cache.Caches,
    at: usize,
    ratio: usize,
) !*Saved {
    try s.synchronize();
    if (c.len != at) return error.SnapshotNotAtMark;
    return saveFrom(gpa, x, c, head, at, ratio, null, c.pooled, c.history);
}

/// The state at `at` inside the last prompt chunk (forward.Mark): rows below it from the caches, the rest from `side`.
pub fn saveMarked(
    gpa: std.mem.Allocator,
    x: Copy,
    s: hip.Stream,
    c: *const cache.Caches,
    head: ?*const cache.Caches,
    at: usize,
    ratio: usize,
    side: u64,
    history: [2]i64,
) !*Saved {
    try s.synchronize();
    if (c.len <= at) return error.SnapshotNotAtMark;
    return saveFrom(gpa, x, c, head, at, ratio, side, at / ratio, history);
}

fn saveFrom(
    gpa: std.mem.Allocator,
    x: Copy,
    c: *const cache.Caches,
    head: ?*const cache.Caches,
    at: usize,
    ratio: usize,
    side: ?u64,
    pooled: usize,
    history: [2]i64,
) !*Saved {
    const out = try gpa.create(Saved);
    errdefer gpa.destroy(out);
    const main = try copyOut(x, c, at, ratio, side, gpa);
    errdefer give(main, gpa);
    const h: []u8 = if (head) |hc| try copyOut(x, hc, at, ratio, null, gpa) else &.{};
    out.* = .{ .at = @intCast(at), .pooled = pooled, .history = history, .main = main, .head = h };
    return out;
}

/// `saved` made the live state of `c` and `head`, or a failure part way through that leaves either changed.
fn restore(
    gpa: std.mem.Allocator,
    x: Copy,
    s: hip.Stream,
    saved: *const Saved,
    c: *cache.Caches,
    head: ?*cache.Caches,
    ratio: usize,
) !void {
    try s.synchronize();
    if (saved.at + 1 > c.layout.cap) return error.SnapshotTooLong;
    try copyIn(x, c, saved.at, ratio, saved.main, gpa);
    if (head) |h| try copyIn(x, h, saved.at, ratio, saved.head, gpa);
    c.len = saved.at;
    c.pooled = saved.pooled;
    c.history = saved.history;
    if (head) |h| {
        h.len = saved.at;
        h.pooled = saved.at / ratio;
    }
}

/// `saved` (null: no copy here) restored on every rank or none: all resume from it, or all start over (RestoreRefused).
pub fn restoreAgreed(
    gpa: std.mem.Allocator,
    x: Copy,
    s: hip.Stream,
    saved: ?*const Saved,
    c: *cache.Caches,
    head: ?*cache.Caches,
    ratio: usize,
    eos: i64,
    link: ?hip.collective.Link,
) !void {
    const mine = if (saved) |v| restore(gpa, x, s, v, c, head, ratio) else error.UnknownSnapshot;
    if (hip.collective.everyRank(link, mine)) return;
    // a part-restored rank, and every rank that restored beside one that failed, starts the prompt over from zero
    if (!hip.collective.everyRank(link, fresh(s, c, head, eos))) return error.GroupStopped;
    return error.RestoreRefused;
}

/// `c` and `head` as a new stream's: zeros, no positions, eos before any row.
fn fresh(s: hip.Stream, c: *cache.Caches, head: ?*cache.Caches, eos: i64) !void {
    try s.synchronize();
    try c.reset(eos);
    if (head) |h| try h.reset(eos);
}

pub fn drop(gpa: std.mem.Allocator, saved: *Saved) void {
    give(saved.main, gpa);
    give(saved.head, gpa);
    gpa.destroy(saved);
}
