//! An EXLR8 manifest: the width each routed expert-half (gate/up per 128 channels, down per expert) loads at.

const std = @import("std");
const Io = std.Io;

pub const experts = 256;
pub const halves = 16;
pub const layers = 79;

/// One MoE layer's widths, as the manifest's `layers[L]` gives them.
pub const LayerWidths = struct {
    gate: [experts][halves]u8,
    up: [experts][halves]u8,
    down: [experts]u8,
    shared_gate: [halves]u8,
    shared_up: [halves]u8,
    shared_down: u8,
};

pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    name: []const u8,
    version: []const u8,
    set_id: []const u8,
    layers: [layers]?*LayerWidths,

    pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Manifest {
        const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(text);
        return parse(gpa, text);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Manifest {
        var m: Manifest = .{ .arena = .init(gpa), .name = "", .version = "", .set_id = "", .layers = @splat(null) };
        errdefer m.arena.deinit();
        const a = m.arena.allocator();
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const o = if (root == .object) root.object else return error.BadExlr8Manifest;
        if (!std.mem.eql(u8, str(o.get("format")) orelse "", "exlr8-v2-manifest")) return error.BadExlr8Manifest;
        m.name = str(o.get("name")) orelse return error.BadExlr8Manifest;
        m.version = str(o.get("version")) orelse "";
        m.set_id = str(o.get("set_id")) orelse "";
        const ls = o.get("layers") orelse return error.BadExlr8Manifest;
        if (ls != .object) return error.BadExlr8Manifest;
        var it = ls.object.iterator();
        while (it.next()) |kv| {
            const n = std.fmt.parseInt(usize, kv.key_ptr.*, 10) catch return error.BadExlr8Manifest;
            if (n >= layers or m.layers[n] != null or kv.value_ptr.* != .object) return error.BadExlr8Manifest;
            const w = try a.create(LayerWidths);
            const l = kv.value_ptr.object;
            try grid(l.get("gate"), &w.gate);
            try grid(l.get("up"), &w.up);
            try row(l.get("down"), &w.down);
            const s = l.get("shared") orelse return error.BadExlr8Manifest;
            if (s != .object) return error.BadExlr8Manifest;
            try row(s.object.get("gate"), &w.shared_gate);
            try row(s.object.get("up"), &w.shared_up);
            w.shared_down = try widthOf(s.object.get("down") orelse return error.BadExlr8Manifest);
            m.layers[n] = w;
        }
        return m;
    }

    pub fn deinit(m: *Manifest) void {
        m.arena.deinit();
        m.* = undefined;
    }

    /// The expert-half's width (expert 256 is the shared expert; down ignores h), or null for a layer without experts.
    pub fn width(m: *const Manifest, layer: usize, expert: usize, proj: enum { gate, up, down }, h: usize) ?u8 {
        const w = m.layers[layer] orelse return null;
        if (expert == experts) return switch (proj) {
            .gate => w.shared_gate[h],
            .up => w.shared_up[h],
            .down => w.shared_down,
        };
        return switch (proj) {
            .gate => w.gate[expert][h],
            .up => w.up[expert][h],
            .down => w.down[expert],
        };
    }
};

fn str(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn widthOf(v: std.json.Value) !u8 {
    const k = if (v == .integer) v.integer else return error.BadExlr8Manifest;
    if (k < 2 or k > 4) return error.BadExlr8Manifest;
    return @intCast(k);
}

fn row(v: ?std.json.Value, out: []u8) !void {
    const x = v orelse return error.BadExlr8Manifest;
    if (x != .array or x.array.items.len != out.len) return error.BadExlr8Manifest;
    for (x.array.items, out) |item, *o| o.* = try widthOf(item);
}

fn grid(v: ?std.json.Value, out: *[experts][halves]u8) !void {
    const x = v orelse return error.BadExlr8Manifest;
    if (x != .array or x.array.items.len != experts) return error.BadExlr8Manifest;
    for (x.array.items, out) |item, *o| try row(item, o);
}

test "a manifest's widths read back per expert-half, and a width the format cannot store is refused" {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(std.testing.allocator);
    const gpa = std.testing.allocator;
    try text.appendSlice(gpa, "{\"format\":\"exlr8-v2-manifest\",\"name\":\"t\",\"set_id\":\"x\",\"layers\":{\"40\":{\"gate\":[");
    for (0..experts) |e| try text.print(gpa, "{s}[{d},3,3,3,3,3,3,3,3,3,3,3,3,3,3,4]", .{ if (e == 0) "" else ",", 2 + e % 3 });
    try text.appendSlice(gpa, "],\"up\":[");
    for (0..experts) |e| try text.appendSlice(gpa, if (e == 0) "[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]" else ",[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]");
    try text.appendSlice(gpa, "],\"down\":[");
    for (0..experts) |e| try text.print(gpa, "{s}{d}", .{ if (e == 0) "" else ",", 4 - e % 3 });
    try text.appendSlice(gpa, "],\"shared\":{\"gate\":[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],\"up\":[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],\"down\":3}}}}");
    var m = try Manifest.parse(gpa, text.items);
    defer m.deinit();
    try std.testing.expectEqual(@as(?u8, 3), m.width(40, 1, .gate, 0));
    try std.testing.expectEqual(@as(?u8, 4), m.width(40, 1, .gate, 15));
    try std.testing.expectEqual(@as(?u8, 3), m.width(40, 1, .down, 0));
    try std.testing.expectEqual(@as(?u8, 3), m.width(40, experts, .down, 0));
    try std.testing.expectEqual(@as(?u8, null), m.width(3, 0, .gate, 0));
    const bad = try std.mem.replaceOwned(u8, gpa, text.items, "\"down\":3}", "\"down\":5}");
    defer gpa.free(bad);
    try std.testing.expectError(error.BadExlr8Manifest, Manifest.parse(gpa, bad));
}
