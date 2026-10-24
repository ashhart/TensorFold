//! Durable retry work: identities and reversible token spans, retired after engine-reachable cleanup.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const Intent = struct {
    id: u32,
    source_num: u64,
    removed: []const []const u8,
    spans: []const []const u32,
    short: bool,

    pub fn deinit(i: Intent, g: Allocator) void {
        for (i.removed) |s| g.free(s);
        g.free(i.removed);
        for (i.spans) |s| g.free(s);
        g.free(i.spans);
    }

    pub fn dupe(i: Intent, g: Allocator) !Intent {
        const removed = try g.alloc([]const u8, i.removed.len);
        var nr: usize = 0;
        errdefer {
            for (removed[0..nr]) |s| g.free(s);
            g.free(removed);
        }
        for (i.removed, removed) |s, *d| {
            d.* = try g.dupe(u8, s);
            nr += 1;
        }
        const spans = try g.alloc([]const u32, i.spans.len);
        var ns: usize = 0;
        errdefer {
            for (spans[0..ns]) |s| g.free(s);
            g.free(spans);
        }
        for (i.spans, spans) |s, *d| {
            d.* = try g.dupe(u32, s);
            ns += 1;
        }
        return .{ .id = i.id, .source_num = i.source_num, .removed = removed, .spans = spans, .short = i.short };
    }
};

pub fn encode(a: Allocator, items: []const Intent) ![]u8 {
    const rows = try a.alloc(json.Value, items.len);
    for (items, rows) |i, *row| {
        const o = try json.newObject(a);
        try o.put(a, "id", try json.intValue(a, i.id));
        try o.put(a, "source_num", try json.intValue(a, i.source_num));
        try o.put(a, "short", .{ .bool = i.short });
        const removed = try a.alloc(json.Value, i.removed.len);
        for (removed, i.removed) |*v, s| v.* = .{ .string = s };
        try o.put(a, "removed", .{ .array = removed });
        const spans = try a.alloc(json.Value, i.spans.len);
        for (spans, i.spans) |*v, span| {
            const ids = try a.alloc(json.Value, span.len);
            for (ids, span) |*n, id| n.* = try json.intValue(a, id);
            v.* = .{ .array = ids };
        }
        try o.put(a, "spans", .{ .array = spans });
        row.* = .{ .object = o };
    }
    return json.stringify(a, .{ .array = rows }, .{});
}

pub fn decode(a: Allocator, v: json.Value) !Intent {
    const id = std.math.cast(u32, (v.get("id") orelse return error.BadJournal).int64() orelse return error.BadJournal) orelse return error.BadJournal;
    const source = std.math.cast(u64, (v.get("source_num") orelse return error.BadJournal).int64() orelse return error.BadJournal) orelse return error.BadJournal;
    const removed_v = (v.get("removed") orelse return error.BadJournal).items() orelse return error.BadJournal;
    const spans_v = (v.get("spans") orelse return error.BadJournal).items() orelse return error.BadJournal;
    const short_v = v.get("short") orelse return error.BadJournal;
    if (id == 0 or short_v != .bool) return error.BadJournal;
    const removed = try a.alloc([]const u8, removed_v.len);
    for (removed, removed_v) |*s, value| s.* = value.str() orelse return error.BadJournal;
    const spans = try a.alloc([]const u32, spans_v.len);
    for (spans, spans_v) |*s, value| {
        const ids_v = value.items() orelse return error.BadJournal;
        if (ids_v.len == 0) return error.BadJournal;
        const ids = try a.alloc(u32, ids_v.len);
        for (ids, ids_v) |*n, x| n.* = std.math.cast(u32, x.int64() orelse return error.BadJournal) orelse return error.BadJournal;
        s.* = ids;
    }
    return .{ .id = id, .source_num = source, .removed = removed, .spans = spans, .short = short_v.bool };
}
