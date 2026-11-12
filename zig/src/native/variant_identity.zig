//! Loaded shader variants and arithmetic policy extend a persisted serving namespace.
const std = @import("std");

pub fn hash(h: *std.hash.Wyhash, a: std.mem.Allocator, variants: anytype) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var names: std.ArrayList([]const u8) = .empty;
    var it = variants.keyIterator();
    while (it.next()) |name| try names.append(arena.allocator(), name.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    text(h, "loaded-variants-v1");
    for (names.items) |name| {
        const v = variants.get(name).?;
        text(h, name);
        text(h, v.text);
        inline for (.{ "inputs", "outputs", "meta" }) |field| {
            text(h, field);
            var length: [8]u8 = undefined;
            std.mem.writeInt(u64, &length, @field(v, field).len, .little);
            h.update(&length);
            for (@field(v, field)) |word| text(h, word);
        }
    }
}

pub fn policy(h: *std.hash.Wyhash, capacity: usize, factor: f64, fp32: bool) void {
    text(h, "rotary-precision-v1");
    const words = [_]u64{ @intCast(capacity), @bitCast(factor), @intFromBool(fp32) };
    h.update(std.mem.sliceAsBytes(&words));
}

fn text(h: *std.hash.Wyhash, bytes: []const u8) void {
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, bytes.len, .little);
    h.update(&length);
    h.update(bytes);
}

fn of(variants: anytype) !u64 {
    var h = std.hash.Wyhash.init(17);
    try hash(&h, std.testing.allocator, variants);
    return h.final();
}

test "persistence identity: loaded shader bytes and bindings move the key independently of a probe" {
    const V = struct { text: []const u8, inputs: []const []const u8, outputs: []const []const u8, meta: []const []const u8 };
    var v: V = .{ .text = "kernel one", .inputs = &.{"x"}, .outputs = &.{"y"}, .meta = &.{"rows"} };
    var map: std.StringHashMapUnmanaged(*V) = .empty;
    defer map.deinit(std.testing.allocator);
    try map.put(std.testing.allocator, "k", &v);
    const first = try of(map);
    try std.testing.expectEqual(first, try of(map));
    v.text = "kernel two";
    try std.testing.expect(first != try of(map));
    v.text = "kernel one";
    inline for (.{ "inputs", "outputs", "meta" }) |field| {
        const old = @field(v, field);
        @field(v, field) = &.{"other"};
        try std.testing.expect(first != try of(map));
        @field(v, field) = old;
    }
    var other = v;
    try map.put(std.testing.allocator, "another", &other);
    var reversed: @TypeOf(map) = .empty;
    defer reversed.deinit(std.testing.allocator);
    try reversed.put(std.testing.allocator, "another", &other);
    try reversed.put(std.testing.allocator, "k", &v);
    try std.testing.expectEqual(try of(map), try of(reversed));
}

test "persistence identity: rotary policy and attention precision move the serving namespace" {
    var base = std.hash.Wyhash.init(17);
    policy(&base, 131072, 0, false);
    var extended = std.hash.Wyhash.init(17);
    policy(&extended, 524288, 4, false);
    var precise = std.hash.Wyhash.init(17);
    policy(&precise, 131072, 0, true);
    try std.testing.expect(base.final() != extended.final());
    try std.testing.expect(base.final() != precise.final());
}
