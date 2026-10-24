//! Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 9.
const std = @import("std");
const builtin = @import("builtin");
const index = @import("index.zig");
const sa = @import("sa.zig");
const exact = @import("exact_alloc.zig");
const Index = index.Index;
const Groups = index.Groups;
const Allocator = std.mem.Allocator;

/// Test plant: skip copying own_phrases in fromVersion; not threadlocal (merge thread); test-only.
pub var plant_skip_core_phrases: if (builtin.is_test) bool else void = if (builtin.is_test) false else {};

/// A full suffix-array index of `v`'s notes; `v` is a layered version (its core has a base).
pub fn fromVersion(gpa: Allocator, v: *const Index) index.LoadError!*Index {
    return fromVersionCancellable(gpa, v, null);
}

pub fn fromVersionCancellable(gpa: Allocator, v: *const Index, cancel: ?*const std.atomic.Value(bool)) index.LoadError!*Index {
    try cancelled(cancel);
    const c = v.layer.core orelse return error.Unsupported;
    const cn = v.layer.core_n;
    const old = c.base orelse return error.Unsupported;
    const n: u32 = v.n_notes;
    const self = try gpa.create(Index);
    errdefer gpa.destroy(self);
    self.* = c.*; // the snapshot's settings, header, vocabulary and titles; every per-note part is set below
    self.arena = exact.Exact.init(gpa);
    errdefer self.arena.deinit();
    const a = self.arena.allocator();
    self.owned = null;
    self.ver = 0;
    self.refs = .init(1);
    self.layer = .{};
    self.base = null;
    errdefer if (self.base) |b| b.release();
    self.n_notes = n;
    self.n_phrases = v.n_phrases;
    self.max_told = v.max_told;
    self.rank_by_note = true;
    self.own_trank = &.{};
    self.n_cur = v.n_cur;
    self.n_hist = v.n_hist;
    self.self_entity = v.self_entity;
    // the notes by place
    const id = try a.alloc(u32, n);
    const state = try a.alloc(index.State, n);
    const told = try a.alloc(i64, n);
    const entity = try a.alloc(u32, n);
    const pair = try a.alloc(u32, n);
    const value = try a.alloc(i32, n);
    const line = try a.alloc([]const u8, n);
    const cost = try a.alloc(u32, n);
    var least: i64 = std.math.maxInt(i64);
    for (0..n) |k| {
        if (k % 256 == 0) try cancelled(cancel);
        const p: u32 = @intCast(k);
        id[k] = v.idAt(p);
        state[k] = v.stateAt(p);
        told[k] = v.toldAt(p);
        entity[k] = v.entityAt(p);
        pair[k] = v.pairAt(p);
        value[k] = v.valueEntityAt(p); // the full build's value entity: v resolves its core's lazily (index.zig)
        line[k] = v.lineAt(p);
        cost[k] = v.costAt(p);
        least = @min(least, @as(i64, cost[k]) + self.fx_nl);
    }
    const id_place = try a.alloc(u32, if (n > 0) @as(usize, id[n - 1]) + 1 else 0); // ids ascend with places
    @memset(id_place, index.no_note);
    for (id, 0..) |x, k| id_place[x] = @intCast(k);
    self.own_id = id;
    self.state = state;
    self.own_told = told;
    self.own_entity = entity;
    self.own_pair = pair;
    self.own_value = value;
    self.own_line = line;
    self.own_cost = cost;
    self.min_line = least;
    self.own_id_place = id_place;
    // the groups, from v's own (each in the full build's order: places ascending, pairs ascending)
    const ne = v.entityCount();
    const np: usize = v.names_from[2] + v.pair_attributes.len;
    self.own_ent_cur = try groups(a, gpa, ne, v, .cur);
    self.own_ent_keys = try groups(a, gpa, ne, v, .keys);
    self.own_pair_notes = try groups(a, gpa, np, v, .pairs);
    // names: copied, so the core outlives the version it was made from
    const en = try a.alloc([]const u8, ne);
    for (en, 0..) |*x, e| x.* = try a.dupe(u8, v.entityName(@intCast(e)));
    const an = try a.alloc([]const u8, v.attributeCount());
    for (an, 0..) |*x, k| x.* = try a.dupe(u8, v.attributeName(@intCast(k)));
    const pa = try a.alloc(u32, np);
    for (pa, 0..) |*x, k| x.* = v.pairAttribute(@intCast(k));
    self.entity_names = en;
    self.attribute_names = an;
    self.pair_attributes = pa;
    self.names_from = .{ 0, 0, 0 };
    // phrases and one-key names: the core's, then the side's (a later phrase wins, as in the model's list)
    self.own_phrases = .empty;
    self.own_names = .empty;
    try self.own_phrases.ensureTotalCapacity(a, @intCast(c.own_phrases.count() + v.own_phrases.count()));
    for ([2]*const Index{ c, v }) |x| {
        const skip = if (comptime builtin.is_test) plant_skip_core_phrases and x == c else false;
        if (!skip) {
            var it = x.own_phrases.iterator();
            while (it.next()) |kv| self.own_phrases.putAssumeCapacity(kv.key_ptr.*, kv.value_ptr.*);
        }
        var nt = x.own_names.iterator();
        while (nt.next()) |kv| try self.own_names.put(a, kv.key_ptr.*, kv.value_ptr.*);
    }
    self.own_phrase_max = @max(c.own_phrase_max, v.own_phrase_max);
    // aliases: v's resolved tables (alias.zig), copied
    self.aliases = .empty;
    self.alias_entities = .empty;
    try copyAliases(self, a, v);
    // the written notes whose value entity a later name may give: the core's and the side's
    self.lazy_value = .empty;
    var nl: u32 = c.lazy_value.count();
    for (v.layer.side_unvalued) |u| nl += @intFromBool(u);
    try self.lazy_value.ensureTotalCapacity(a, nl);
    var lit = c.lazy_value.iterator();
    while (lit.next()) |kv| self.lazy_value.putAssumeCapacity(kv.key_ptr.*, kv.value_ptr.*);
    for (v.layer.side_unvalued, v.layer.side_tokens, cn..) |u, t, p| if (u) self.lazy_value.putAssumeCapacity(@intCast(p), t);
    // the suffix arrays: the old core's merged with the side's (sa.merge), the old notes' keys from the old base
    const dec = try sa.noteKeys(gpa, old);
    defer gpa.free(dec.keys);
    defer gpa.free(dec.off);
    const src = try gpa.alloc(sa.Source, n);
    defer gpa.free(src);
    const side_keys = try gpa.alloc(u32, blk: {
        var t: usize = 0;
        for (v.layer.side_tokens) |x| t += x.len;
        break :blk t;
    });
    defer gpa.free(side_keys);
    var at: usize = 0;
    for (src, 0..) |*s, k| {
        const cur = state[k] == .current;
        if (k < old.notes) {
            s.* = .{ .keys = dec.keys[dec.off[k]..dec.off[k + 1]], .ids = &.{}, .cur = cur };
        } else {
            const t = v.layer.side_tokens[k - cn];
            for (t, side_keys[at..][0..t.len]) |x, *y| y.* = v.keyOf(x);
            s.* = .{ .keys = side_keys[at..][0..t.len], .ids = t, .cur = cur };
            at += t.len;
        }
    }
    if (old.notes != cn) return error.Unsupported;
    self.retired = try index.retiredBits(a, n);
    try cancelled(cancel);
    self.base = try sa.mergeCancellable(gpa, old, src, self.vwordy, cancel);
    self.side_keys = .{};
    self.grams = .empty;
    self.post = &.{};
    return self;
}

fn cancelled(flag: ?*const std.atomic.Value(bool)) index.LoadError!void {
    if (flag) |f| if (f.load(.acquire)) return error.Unsupported;
}

fn groups(a: Allocator, gpa: Allocator, count: usize, v: *const Index, which: enum { cur, keys, pairs }) Allocator.Error!Groups {
    const off = try a.alloc(u32, count + 1);
    var items: std.ArrayList(u32) = .empty;
    defer items.deinit(gpa);
    for (0..count) |k| {
        off[k] = @intCast(items.items.len);
        var it = switch (which) {
            .cur => v.entCur(@intCast(k)),
            .keys => v.entKeys(@intCast(k)),
            .pairs => v.pairNotes(@intCast(k)),
        };
        while (it.next()) |p| try items.append(gpa, p);
    }
    off[count] = @intCast(items.items.len);
    return .{ .off = off, .items = try a.dupe(u32, items.items) };
}

/// The full tables of v's aliases: its core's, with v's overrides (alias.zig) put over them.
fn copyAliases(self: *Index, a: Allocator, v: *const Index) Allocator.Error!void {
    const c = v.layer.core.?;
    try self.aliases.ensureTotalCapacity(a, c.aliases.count() + v.aliases.count());
    for ([2]*const Index{ c, v }) |x| {
        var it = x.aliases.iterator();
        while (it.next()) |kv| self.aliases.putAssumeCapacity(kv.key_ptr.*, kv.value_ptr.*);
    }
    var gone = v.aliases.iterator();
    while (gone.next()) |kv| if (kv.value_ptr.* == index.alias_gone) {
        _ = self.aliases.remove(kv.key_ptr.*);
    };
    for ([2]*const Index{ c, v }) |x| {
        var ie = x.alias_entities.iterator();
        while (ie.next()) |kv| {
            if (kv.value_ptr.items.len == 0) {
                _ = self.alias_entities.remove(kv.key_ptr.*);
                continue;
            }
            var l: std.ArrayList(u32) = .empty;
            try l.appendSlice(a, kv.value_ptr.items);
            try self.alias_entities.put(a, kv.key_ptr.*, l);
        }
    }
    self.alias_max = v.aliasMax();
}
