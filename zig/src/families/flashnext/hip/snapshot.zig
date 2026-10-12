//! A stream's state after its first `at` prompt tokens, copied to host memory and back: the prompt cache's entries.
const std = @import("std");
const hip = @import("hip");
const cache = @import("cache.zig");

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

fn copyOut(d: *const hip.Runtime, c: *const cache.Caches, at: usize, ratio: usize, gpa: std.mem.Allocator) ![]u8 {
    var list: std.ArrayList(Region) = .empty;
    defer list.deinit(gpa);
    try regions(c, at, ratio, &list, gpa);
    var n: usize = 0;
    for (list.items) |r| n += r.len;
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    var off: usize = 0;
    for (list.items) |r| {
        if (r.len > 0) try hip.runtime.check(d.api.hipMemcpyDtoH(out[off..].ptr, @ptrFromInt(c.mem.base() + r.offset), r.len));
        off += r.len;
    }
    return out;
}

fn copyIn(d: *const hip.Runtime, c: *const cache.Caches, at: usize, ratio: usize, bytes: []const u8, gpa: std.mem.Allocator) !void {
    var list: std.ArrayList(Region) = .empty;
    defer list.deinit(gpa);
    try regions(c, at, ratio, &list, gpa);
    var off: usize = 0;
    for (list.items) |r| {
        if (off + r.len > bytes.len) return error.SnapshotMismatch;
        if (r.len > 0) try hip.runtime.check(d.api.hipMemcpyHtoD(@ptrFromInt(c.mem.base() + r.offset), @constCast(bytes[off..].ptr), r.len));
        off += r.len;
    }
    if (off != bytes.len) return error.SnapshotMismatch;
}

/// The state of `c` (and the head's `head`) after `at` prompt tokens, copied to host memory; the stream is idle.
pub fn save(gpa: std.mem.Allocator, d: *const hip.Runtime, s: hip.Stream, c: *const cache.Caches, head: ?*const cache.Caches, at: usize, ratio: usize) !*Saved {
    try s.synchronize();
    if (c.len != at) return error.SnapshotNotAtMark;
    const out = try gpa.create(Saved);
    errdefer gpa.destroy(out);
    const main = try copyOut(d, c, at, ratio, gpa);
    errdefer gpa.free(main);
    const h: []u8 = if (head) |x| try copyOut(d, x, at, ratio, gpa) else &.{};
    out.* = .{ .at = @intCast(at), .pooled = c.pooled, .history = c.history, .main = main, .head = h };
    return out;
}

/// `saved` made the live state of `c` and `head`: the next prompt chunk starts at its position.
pub fn restore(gpa: std.mem.Allocator, d: *const hip.Runtime, s: hip.Stream, saved: *const Saved, c: *cache.Caches, head: ?*cache.Caches, ratio: usize) !void {
    try s.synchronize();
    if (saved.at + 1 > c.layout.cap) return error.SnapshotTooLong;
    try copyIn(d, c, saved.at, ratio, saved.main, gpa);
    if (head) |h| try copyIn(d, h, saved.at, ratio, saved.head, gpa);
    c.len = saved.at;
    c.pooled = saved.pooled;
    c.history = saved.history;
    if (head) |h| {
        h.len = saved.at;
        h.pooled = saved.at / ratio;
    }
}

pub fn drop(gpa: std.mem.Allocator, saved: *Saved) void {
    gpa.free(saved.main);
    if (saved.head.len > 0) gpa.free(saved.head);
    gpa.destroy(saved);
}
