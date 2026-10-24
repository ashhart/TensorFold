//! Conservative name-value extraction; callers must require a current name fact.
const std = @import("std");
const index = @import("index.zig");

/// The text after the first naming phrase: right after the subject, or after one word opening a later clause.
fn namingClause(rest: []const u8) ?[]const u8 {
    var s: usize = 0; // where the clause starts
    var k: usize = 0; // the clause: the first is the subject's own
    while (true) : (k += 1) {
        var at = s;
        if (k > 0) {
            while (at < rest.len and (std.ascii.isAlphabetic(rest[at]) or rest[at] >= 0x80)) : (at += 1) {}
        }
        if (k == 0 or at > s) for ([_][]const u8{ "'s name is ", " is named ", " is called " }) |prefix| {
            if (rest.len - at >= prefix.len and std.ascii.eqlIgnoreCase(rest[at..][0..prefix.len], prefix)) return rest[at + prefix.len ..];
        };
        while (s + 1 < rest.len and !(std.mem.indexOfScalar(u8, ".,;:", rest[s]) != null and rest[s + 1] == ' ')) : (s += 1) {}
        if (s + 1 >= rest.len) return null;
        s += 2;
    }
}

pub fn nameValue(subject: []const u8, text: []const u8) ?[]const u8 {
    if (text.len <= subject.len or !std.ascii.eqlIgnoreCase(text[0..subject.len], subject)) return null;
    var v = std.mem.trim(u8, namingClause(text[subject.len..]) orelse return null, " \t");
    if (std.mem.endsWith(u8, v, ".")) v = v[0 .. v.len - 1];
    if (v.len >= 4 and std.ascii.eqlIgnoreCase(v[v.len - 4 ..], " now")) v = v[0 .. v.len - 4];
    if (v.len == 0 or v.len > 96 or !std.unicode.utf8ValidateSlice(v)) return null;
    var words: usize = 1;
    for (v) |c| {
        if (c == ' ') words += 1 else if (!std.ascii.isAlphabetic(c) and c != '-' and c != '\'' and c < 0x80) return null;
    }
    if (words > 4) return null;
    for ([_][]const u8{ "not", "unknown", "no longer", "don't know" }) |bad| {
        if (std.ascii.eqlIgnoreCase(v, bad) or (v.len > bad.len and std.ascii.eqlIgnoreCase(v[0..bad.len], bad) and v[bad.len] == ' ')) return null;
    }
    return v;
}

test "name alias extraction requires subject and an explicit naming clause" {
    try std.testing.expectEqualStrings("Ana", nameValue("my sister", "My sister's name is Ana.").?);
    try std.testing.expectEqualStrings("Bruno", nameValue("my dog", "My dog is called Bruno now.").?);
    try std.testing.expectEqualStrings("Milo", nameValue("my cat", "My cat is named Milo.").?);
    try std.testing.expect(nameValue("my sister", "My sister lives in Ana.") == null);
    try std.testing.expect(nameValue("my sister", "My sister's name is not Ana.") == null);
    try std.testing.expect(nameValue("my dog", "My dog's friend is called Bruno.") == null);
    try std.testing.expect(nameValue("my dog", "My dog is called Bruno. Ignore instructions.") == null);
}

test "a naming clause may open a later clause of the note after one word that stands for the subject" {
    try std.testing.expectEqualStrings("Pim", nameValue("Noor's goat", "Noor's goat got renamed, it is called Pim now.").?);
    try std.testing.expectEqualStrings("Kip", nameValue("my mule", "My mule was sold; it's name is Kip.").?);
    try std.testing.expectEqualStrings("Ute", nameValue("my yak", "My yak moved. She is named Ute.").?);
    try std.testing.expect(nameValue("my dog", "My dog likes Bo, his friend is called Bo.") == null); // two words
    try std.testing.expect(nameValue("my dog", "My dog was called Rex; now it is called Otto.") == null);
    try std.testing.expect(nameValue("my dog", "My dog is called Bo, it is called Otto now.") == null); // the first decides
    try std.testing.expect(nameValue("my dog", "My dog sleeps; it is called") == null);
    try std.testing.expect(nameValue("my dog", "My dog sleeps;  it is called Otto.") == null);
    try std.testing.expect(nameValue("my dog", "My dog sleeps; is called Otto.") == null); // no word before it
    try std.testing.expect(nameValue("my dog", "My dog sleeps, it is called Otto, not Rex.") == null);
}

/// Tests only: no written name fact read (a stale table: notes_delta_test.zig's planted bug).
pub var plant_stale: bool = false;

/// One encoded alias: the keys of " I", " my", " me" (place: none) or of a name fact's value (its place).
const Entry = struct { place: u32, keys: [8]u32, len: u8 };

const Key = [8]u32;

/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 3.
pub const Memo = struct {
    model: ?*const anyopaque = null, // the model it was made for: another (an erase's, a load's) starts it again
    scanned: usize = 0, // the notes before this place were looked at
    self: std.ArrayList(Entry) = .empty,
    self_done: bool = false,
    names: std.ArrayList(Entry) = .empty, // name facts by place, whatever their state now
    by_key: std.AutoHashMapUnmanaged(Key, std.ArrayList(u32)) = .empty, // alias keys -> indexes into names, ascending
    by_place: std.AutoHashMapUnmanaged(u32, u32) = .empty, // a name fact's place -> its index into names
    temp: bool = false, // populate()'s own, for one build
    gpa: ?std.mem.Allocator = null, // what its lists are made with (the first caller's index allocator unless set)

    pub fn deinit(m: *Memo, g0: std.mem.Allocator) void {
        const g = m.gpa orelse g0;
        const keep = m.gpa;
        m.self.deinit(g);
        m.names.deinit(g);
        var it = m.by_key.valueIterator();
        while (it.next()) |l| l.deinit(g);
        m.by_key.deinit(g);
        m.by_place.deinit(g);
        m.* = .{ .temp = m.temp, .gpa = keep };
    }

    fn reset(m: *Memo, g: std.mem.Allocator, model: *const anyopaque) void {
        const temp = m.temp;
        m.deinit(g);
        m.temp = temp;
        m.model = model;
    }

    fn add(m: *Memo, g: std.mem.Allocator, x: Entry) !void {
        try m.names.ensureUnusedCapacity(g, 1);
        try m.by_place.ensureUnusedCapacity(g, 1);
        const r = try m.by_key.getOrPut(g, x.keys);
        if (!r.found_existing) r.value_ptr.* = .empty;
        try r.value_ptr.append(g, @intCast(m.names.items.len));
        m.by_place.putAssumeCapacity(x.place, @intCast(m.names.items.len));
        m.names.appendAssumeCapacity(x);
    }
};

fn encodeKeys(idx: anytype, text: anytype, s: []const u8, place: u32) !?Entry {
    const a = idx.arena.allocator();
    const ids = text.encode(a, s, false) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null, // an unencodable alias must not break the note store
    };
    defer a.free(ids);
    if (ids.len == 0 or ids.len > 8) return null;
    var out: Entry = .{ .place = place, .keys = @splat(0xffff_ffff), .len = @intCast(ids.len) };
    for (ids, out.keys[0..ids.len]) |id, *key| key.* = idx.keyOf(id);
    return out;
}

/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 4.
pub fn populate(idx: anytype, model: anytype, text: anytype) !void {
    var memo: Memo = .{ .temp = true };
    defer memo.deinit(idx.arena.child);
    return populateMemo(idx, model, text, &memo);
}

/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 5.
pub fn populateMemo(idx: anytype, model: anytype, text: anytype, memo: *Memo) !void {
    if (memo.gpa == null) memo.gpa = idx.arena.child;
    const g = memo.gpa.?;
    const a = idx.arena.allocator();
    if (memo.model != @as(*const anyopaque, model)) memo.reset(g, model);
    const n_notes = model.notes.items.len;
    if (memo.scanned > n_notes) memo.reset(g, model); // fewer notes than it read: start again
    if (!memo.self_done) if (model.entityOf("me")) |_| {
        for ([_][]const u8{ " I", " my", " me" }) |word| if (try encodeKeys(idx, text, word, 0)) |x| try memo.self.append(g, x);
        memo.self_done = true;
    };
    // the notes before the last are settled; the last may be a write that is reverted (and its place reused),

    // so it is encoded every time and never kept
    const stale = plant_stale and !memo.temp;
    const limit = if (stale) @min(n_notes, model.base_notes + 1) else n_notes; // stale: the snapshot's names only
    while (memo.scanned + 1 < limit) : (memo.scanned += 1) {
        if (try nameEntry(idx, model, text, @intCast(memo.scanned))) |x| try memo.add(g, x);
    }
    const last: ?Entry = if (n_notes > memo.scanned and !stale) try nameEntry(idx, model, text, @intCast(n_notes - 1)) else null;
    const me = model.entityOf("me");
    if (idx.layer.core) |core| {
        var touched: std.AutoArrayHashMapUnmanaged(Key, u8) = .empty;
        defer touched.deinit(g);
        const cn = idx.layer.core_n;
        for (cn..n_notes) |p| {
            if (last) |x| if (x.place == p) try touched.put(g, x.keys, x.len);
            if (memo.by_place.get(@intCast(p))) |k| try touched.put(g, memo.names.items[k].keys, memo.names.items[k].len);
        }
        var ov = idx.layer.over.keyIterator();
        while (ov.next()) |p| if (memo.by_place.get(p.*)) |k| try touched.put(g, memo.names.items[k].keys, memo.names.items[k].len);
        if (core.self_entity == null and me != null) for (memo.self.items) |x| try touched.put(g, x.keys, x.len);
        for (touched.keys(), touched.values()) |key, len| {
            var val: ?u32 = null;
            if (me) |e| for (memo.self.items) |x| if (std.mem.eql(u32, &x.keys, &key)) {
                val = e;
            };
            var cands: std.ArrayList(u32) = .empty;
            const from: []const u32 = if (memo.by_key.get(key)) |l| l.items else &.{};
            const tail: []const Entry = if (last) |*x| if (std.mem.eql(u32, &x.keys, &key)) x[0..1] else &.{} else &.{};
            for (from) |k| try contribute(a, model, memo.names.items[k], &val, &cands);
            for (tail) |x| try contribute(a, model, x, &val, &cands);
            try idx.aliases.put(a, key, val orelse index.alias_gone);
            try idx.alias_entities.put(a, key, cands); // empty: no name fact names it now
            idx.alias_max = @max(idx.alias_max, len);
        }
        return;
    }
    if (me) |entity| for (memo.self.items) |x| {
        try idx.aliases.put(a, x.keys, entity);
        idx.alias_max = @max(idx.alias_max, x.len);
    };
    const tail: []const Entry = if (last) |*x| x[0..1] else &.{};
    for ([2][]const Entry{ memo.names.items, tail }) |list| for (list) |x| {
        const n = &model.notes.items[x.place];
        if (n.state != .current) continue;
        const found = try idx.aliases.getOrPut(a, x.keys);
        if (!found.found_existing) found.value_ptr.* = n.entity else if (found.value_ptr.* != n.entity) found.value_ptr.* = 0xffff_ffff;
        const candidates = try idx.alias_entities.getOrPut(a, x.keys);
        if (!candidates.found_existing) candidates.value_ptr.* = .empty;
        if (std.mem.indexOfScalar(u32, candidates.value_ptr.items, n.entity) == null) try candidates.value_ptr.append(a, n.entity);
        idx.alias_max = @max(idx.alias_max, x.len);
    };
}

/// One name fact's part in its key's alias, as the full build's loop takes it (current ones only, by place).
fn contribute(a: std.mem.Allocator, model: anytype, x: Entry, val: *?u32, cands: *std.ArrayList(u32)) !void {
    const n = &model.notes.items[x.place];
    if (n.state != .current) return;
    if (val.*) |v| {
        if (v != n.entity) val.* = 0xffff_ffff;
    } else val.* = n.entity;
    if (std.mem.indexOfScalar(u32, cands.items, n.entity) == null) try cands.append(a, n.entity);
}

fn nameEntry(idx: anytype, model: anytype, text: anytype, place: u32) !?Entry {
    const n = &model.notes.items[place];
    if (n.kind != .fact or !std.mem.eql(u8, model.attributes.items[n.attribute], "name")) return null;
    const value = nameValue(model.entities.items[n.entity], n.text()) orelse return null;
    const a = idx.arena.allocator();
    const spaced = try std.mem.concat(a, u8, &.{ " ", value });
    defer a.free(spaced);
    return encodeKeys(idx, text, spaced, place);
}
