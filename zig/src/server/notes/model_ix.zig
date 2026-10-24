//! The model's lookups by name, pair, source and a fact's normal form: a write finds them at any store size.
const std = @import("std");
const intake = @import("intake.zig");
const Allocator = std.mem.Allocator;

/// A fact's key in `facts`: its entity and the hash of its text's normal form (candidates are checked by text).
pub fn factKey(entity: u32, norm_text: []const u8) u64 {
    return std.hash.Wyhash.hash(entity, norm_text);
}

/// A key's places in `facts`: one, or (the high bit) a list in `lists`.
const many: u32 = 0x8000_0000;

/// Each map gives what the model's lists give, in their order: the first place of a name or pair.
pub const Ix = struct {
    entity: std.StringHashMapUnmanaged(u32) = .empty, // entity name -> its first place
    attribute: std.StringHashMapUnmanaged(u32) = .empty,
    pair: std.AutoHashMapUnmanaged(u64, u32) = .empty, // entity << 32 | attribute -> the first such pair
    source: std.StringHashMapUnmanaged(u32) = .empty, // a written note's source -> its first place
    facts: std.AutoHashMapUnmanaged(u64, u32) = .empty, // factKey -> a current fact's place, or many | a list's index
    lists: std.ArrayList(std.ArrayList(u32)) = .empty, // the places of keys several current facts share, ascending
    spare: std.ArrayList(u32) = .empty, // a list made ready by reserveFact for the next key that needs one

    pub fn deinit(x: *Ix, g: Allocator) void {
        x.entity.deinit(g);
        x.attribute.deinit(g);
        x.pair.deinit(g);
        x.source.deinit(g);
        x.facts.deinit(g);
        for (x.lists.items) |*l| l.deinit(g);
        x.lists.deinit(g);
        x.spare.deinit(g);
    }

    /// A name at place k, unless an earlier place has it (the lists' first match wins).
    pub fn putName(map: *std.StringHashMapUnmanaged(u32), g: Allocator, name: []const u8, k: u32) Allocator.Error!void {
        const r = try map.getOrPut(g, name);
        if (!r.found_existing) r.value_ptr.* = k;
    }

    pub fn putNameAssumeCapacity(map: *std.StringHashMapUnmanaged(u32), name: []const u8, k: u32) void {
        const r = map.getOrPutAssumeCapacity(name);
        if (!r.found_existing) r.value_ptr.* = k;
    }

    pub fn pairKey(entity: u32, attribute: u32) u64 {
        return @as(u64, entity) << 32 | attribute;
    }

    /// The current facts' places under `key`, ascending.
    pub fn factsOf(x: *const Ix, key: u64, one: *[1]u32) []const u32 {
        const v = x.facts.get(key) orelse return &.{};
        if (v & many != 0) return x.lists.items[v & ~many].items;
        one[0] = v;
        return one;
    }

    /// A current fact's place added under its key (places come in ascending order).
    pub fn addFact(x: *Ix, g: Allocator, key: u64, place: u32) Allocator.Error!void {
        try x.reserveFact(g, key);
        x.addFactAssumeCapacity(key, place);
    }

    /// Room for one more fact under `key`, so that addFactAssumeCapacity never allocates.
    pub fn reserveFact(x: *Ix, g: Allocator, key: u64) Allocator.Error!void {
        try x.facts.ensureUnusedCapacity(g, 1);
        const v = x.facts.get(key) orelse return;
        if (v & many != 0) return x.lists.items[v & ~many].ensureUnusedCapacity(g, 1);
        try x.lists.ensureUnusedCapacity(g, 1);
        try x.spare.ensureTotalCapacity(g, 2);
    }

    pub fn addFactAssumeCapacity(x: *Ix, key: u64, place: u32) void {
        const r = x.facts.getOrPutAssumeCapacity(key);
        if (!r.found_existing) {
            r.value_ptr.* = place;
        } else if (r.value_ptr.* & many != 0) {
            x.lists.items[r.value_ptr.* & ~many].appendAssumeCapacity(place);
        } else {
            var l = x.spare; // reserved: room for two
            x.spare = .empty;
            l.appendAssumeCapacity(r.value_ptr.*);
            l.appendAssumeCapacity(place);
            r.value_ptr.* = many | @as(u32, @intCast(x.lists.items.len));
            x.lists.appendAssumeCapacity(l);
        }
    }

    /// A fact no longer current leaves its key (never allocates: a list that empties stays, as its index does).
    pub fn dropFact(x: *Ix, key: u64, place: u32) void {
        const v = x.facts.get(key) orelse return;
        if (v & many == 0) {
            if (v == place) _ = x.facts.remove(key);
            return;
        }
        const l = &x.lists.items[v & ~many];
        const at = std.mem.indexOfScalar(u32, l.items, place) orelse return;
        _ = l.orderedRemove(at);
    }
};

/// The normal form's key of a fact's text.
pub fn factKeyOf(a: Allocator, entity: u32, text: []const u8) Allocator.Error!u64 {
    var buf: [4096]u8 = undefined; // a note's text fits: the caller's allocator is not asked
    var bf = std.heap.BufferFirstAllocator.init(&buf, a);
    const s = bf.allocator();
    const n = try intake.norm(s, text);
    defer s.free(n);
    return factKey(entity, n);
}

test "a name keeps its first place; a key's facts keep their order, one place or a list" {
    const g = std.testing.allocator;
    var x: Ix = .{};
    defer x.deinit(g);
    try Ix.putName(&x.entity, g, "my sister", 3);
    try Ix.putName(&x.entity, g, "my sister", 7);
    try std.testing.expectEqual(@as(u32, 3), x.entity.get("my sister").?);
    const k = try factKeyOf(g, 2, "My sister is Anna.");
    try std.testing.expectEqual(k, try factKeyOf(g, 2, "my sister  is anna"));
    try std.testing.expect(k != try factKeyOf(g, 3, "My sister is Anna."));
    var one: [1]u32 = undefined;
    try x.addFact(g, k, 4);
    try std.testing.expectEqualSlices(u32, &.{4}, x.factsOf(k, &one));
    for ([_]u32{ 9, 12 }) |p| try x.addFact(g, k, p);
    x.dropFact(k, 9);
    try std.testing.expectEqualSlices(u32, &.{ 4, 12 }, x.factsOf(k, &one));
    try x.addFact(g, 77, 5);
    x.dropFact(77, 5);
    try std.testing.expectEqual(@as(usize, 0), x.factsOf(77, &one).len);
    try x.reserveFact(g, k); // room reserved: the add allocates nothing
    x.addFactAssumeCapacity(k, 20);
    try std.testing.expectEqualSlices(u32, &.{ 4, 12, 20 }, x.factsOf(k, &one));
}
