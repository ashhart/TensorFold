//! Decode-time lookup scoring, name collection and prompt text helpers.
const std = @import("std");
const json = @import("json");
const index = @import("notes/index.zig");
const lookup = @import("notes/lookup.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// The index generation the first lookup read; chat write (a later slice) does not bump this mid-reply.
pub fn readFrom(u: *const @import("notes_route.zig").Used) u64 {
    return u.found.ver;
}

/// A failed exact fit retains the higher-scoring half, bounding repeated rendering work.
pub fn nextFitPrefix(n: usize) usize {
    return n / 2;
}

test "failed live fits visit less than twice the initial candidate volume" {
    for ([_]usize{ 1, 2, 3, 500, 50_000 }) |initial| {
        var n = initial;
        var volume: usize = 0;
        var attempts: usize = 0;
        while (n > 0) : (n = nextFitPrefix(n)) {
            volume += n;
            attempts += 1;
        }
        try std.testing.expect(volume < 2 * initial);
        try std.testing.expect(attempts <= @bitSizeOf(usize) - @clz(initial));
    }
}

/// The entities the query names: first the `boosted` ones lookup() matched (names of 2 to 8 keys), then one-key names.
const Named = struct { all: []const u32, boosted: usize };

pub fn named(a: Allocator, idx: *const index.Index, sc: *const lookup.Scratch) Allocator.Error!Named {
    var out: std.ArrayList(u32) = .empty;
    for (sc.found.items) |e| if (std.mem.indexOfScalar(u32, out.items, e) == null) try out.append(a, e);
    const boosted = out.items.len;
    for (sc.qkeys.items) |k| if (idx.nameOf(k)) |e| if (std.mem.indexOfScalar(u32, out.items, e) == null) try out.append(a, e);
    return .{ .all = out.items, .boosted = boosted };
}

/// A note's score in this lookup (0: it has none).
pub fn scoreOf(sc: *const lookup.Scratch, p: u32) f64 {
    return if (sc.slot[p].stamp == sc.gen) sc.slot[p].score else 0;
}

/// Each note in `notes` with its lookup score plus `extra` (the subject boost lookup() gives only longer names).
pub fn addNotes(a: Allocator, score: *std.AutoHashMapUnmanaged(u32, f64), sc: *const lookup.Scratch, notes: index.Iter, extra: f64) Allocator.Error!void {
    var it = notes;
    while (it.next()) |p| {
        const r = try score.getOrPut(a, p);
        r.value_ptr.* = (if (r.found_existing) r.value_ptr.* else scoreOf(sc, p)) + extra;
    }
}

pub fn message(a: Allocator, role: []const u8, content: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = role });
    try o.put(a, "content", .{ .string = content });
    return .{ .object = o };
}

/// The reply's text without the think markers, which are no words of it.
pub fn plain(a: Allocator, reply: []const u8, markers: [2][]const u8) Allocator.Error![]const u8 {
    var out: []const u8 = reply;
    for ([_][]const u8{ markers[0], markers[1], "<think>", "</think>" }) |m| if (m.len > 0 and std.mem.indexOf(u8, out, m) != null) {
        out = try std.mem.replaceOwned(u8, a, out, m, " ");
    };
    return out;
}

pub fn placesOf(a: Allocator, idx: *const index.Index, ids: []const u32) Allocator.Error![]const u32 {
    const out = try a.alloc(u32, ids.len);
    var n: usize = 0;
    for (ids) |id| if (idx.placeOfId(id)) |p| {
        out[n] = p;
        n += 1;
    };
    return out[0..n];
}
