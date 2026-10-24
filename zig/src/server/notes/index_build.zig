//! The index's build: the notes by place, their groups and phrases, the postings and the suffix-array base.
const std = @import("std");
const index = @import("index.zig");
const model_mod = @import("model.zig");
const sa = @import("sa.zig");
const Index = index.Index;
const Raw = index.Raw;
const State = index.State;
const Gram = index.Gram;
const GramCtx = index.GramCtx;
const Post = index.Post;
const Model = model_mod.Model;
const LoadError = index.LoadError;
const Allocator = std.mem.Allocator;
const max_phrase = index.max_phrase;
const packPhrase = index.packPhrase;
const no_note = index.no_note;
const none_key = index.none_key;
const key_id = index.key_id;

/// The side buffer's notes' keys, flat.
pub fn sideKeys(self: *Index, m: *const Model, side: []const Raw) LoadError!void {
    const a = self.arena.allocator();
    const off = try a.alloc(u32, side.len + 1);
    var total: u32 = 0;
    for (side, 0..) |x, k| {
        off[k] = total;
        total += @intCast(x.tokens.len);
    }
    off[side.len] = total;
    const items = try a.alloc(u32, total);
    var at: usize = 0;
    for (side) |x| for (x.tokens) |t| {
        items[at] = m.keyOf(t);
        at += 1;
    };
    self.side_keys = .{ .off = off, .items = items };
}

/// A suffix-array base over every note: their folded keys, each view's suffixes sorted (the build's memory freed).
pub fn baseOf(gpa: Allocator, m: *const Model, raw: []const Raw, prev: ?*const sa.Base) LoadError!*sa.Base {
    try saVocab(gpa, m);
    var total: usize = 0;
    for (raw) |x| total += x.tokens.len;
    const keys = try gpa.alloc(u32, total);
    defer gpa.free(keys);
    const src = try gpa.alloc(sa.Source, raw.len);
    defer gpa.free(src);
    var at: usize = 0;
    for (raw, src) |x, *o| {
        for (x.tokens, keys[at..][0..x.tokens.len]) |t, *k| k.* = m.keyOf(t);
        o.* = .{ .keys = keys[at..][0..x.tokens.len], .ids = x.tokens, .cur = x.state == .current };
        at += x.tokens.len;
    }
    return if (prev) |p| sa.merge(gpa, p, src, m.vwordy) else sa.build(gpa, src, m.vwordy);
}

/// Key sequences match as packed n-grams do only with no none_key key and one wordiness per key: else Unsupported.
pub fn saVocab(gpa: Allocator, m: *const Model) LoadError!void {
    var max: u32 = 0;
    for (m.vkey) |k| {
        if (k == none_key) return error.Unsupported;
        max = @max(max, k & ~key_id);
    }
    const seen = try gpa.alloc(u8, 2 * (@as(usize, max) + 1)); // by key: 0 unseen, 1 a word, 2 not
    defer gpa.free(seen);
    @memset(seen, 0);
    for (m.vkey, 0..) |k, t| {
        const at = 2 * @as(usize, k & ~key_id) + @intFromBool(k & key_id != 0);
        const w: u8 = if (t < m.vwordy.len and m.vwordy[t]) 1 else 2;
        if (seen[at] == 0) seen[at] = w else if (seen[at] != w) return error.Unsupported;
    }
}

/// The notes by place, with Python's tie order (-told, id) and the least a line can add.
pub fn placeNotes(self: *Index, gpa: Allocator, raw: []const Raw) LoadError!void {
    const a = self.arena.allocator();
    const n = raw.len;
    const note_id = try a.alloc(u32, n);
    const state = try a.alloc(State, n);
    const told = try a.alloc(i64, n);
    const entity = try a.alloc(u32, n);
    const pair = try a.alloc(u32, n);
    const value_ent = try a.alloc(i32, n);
    const line = try a.alloc([]const u8, n);
    const cost = try a.alloc(u32, n);
    var max_id: u32 = 0;
    for (raw) |x| max_id = @max(max_id, x.id);
    const id_place = try a.alloc(u32, @as(usize, max_id) + 1);
    @memset(id_place, no_note);
    var n_cur: u32 = 0;
    var n_hist: u32 = 0;
    for (raw, 0..) |x, k| {
        if (id_place[x.id] != no_note) return error.BadSnapshot;
        id_place[x.id] = @intCast(k);
        note_id[k] = x.id;
        state[k] = x.state;
        if (x.state == .current) n_cur += 1 else n_hist += 1;
        told[k] = x.told;
        entity[k] = x.entity;
        pair[k] = x.pair;
        value_ent[k] = x.value_ent;
        line[k] = x.line; // the model's own: a write frees a replaced line only once no version can read it
        cost[k] = x.cost;
    }
    var fits = index.rank_by_note_allowed;
    for (raw, 0..) |x, k| fits = fits and index.rankFits(k, x.told) and (k == 0 or x.id > raw[k - 1].id);
    self.rank_by_note = fits;
    if (fits) { // rankAt() reads each note's own told and id: no global order to make
        self.own_trank = &.{};
    } else try trankOf(self, gpa, a, told, note_id);
    self.own_id = note_id;
    self.state = state;
    self.own_told = told;
    self.own_entity = entity;
    self.own_pair = pair;
    self.own_value = value_ent;
    self.own_line = line;
    self.own_cost = cost;
    var least: i64 = std.math.maxInt(i64);
    for (cost) |c| least = @min(least, @as(i64, c) + self.fx_nl);
    self.min_line = least;
    self.own_id_place = id_place;
    self.n_cur = n_cur;
    self.n_hist = n_hist;
}

/// Full-build tie ranks in (-told, id) order, for values outside rankAt()'s range.
fn trankOf(self: *Index, gpa: Allocator, a: Allocator, told: []const i64, note_id: []const u32) Allocator.Error!void {
    const n = told.len;
    const by = try gpa.alloc(u32, n);
    defer gpa.free(by);
    for (by, 0..) |*x, k| x.* = @intCast(k);
    const TieOrder = struct {
        told: []const i64,
        id: []const u32,
        fn less(c: @This(), x: u32, y: u32) bool {
            if (c.told[x] != c.told[y]) return c.told[x] > c.told[y];
            return c.id[x] < c.id[y];
        }
    };
    std.sort.pdq(u32, by, TieOrder{ .told = told, .id = note_id }, TieOrder.less);
    const trank = try a.alloc(u32, n);
    for (by, 0..) |x, r| trank[x] = @intCast(r);
    self.own_trank = trank;
}

/// O.ent_cur, O.ent_keys and O.by_key as note places, flat: one pass to count, one to fill (place, pair order).
pub fn groupsOf(self: *Index, m: *const Model) LoadError!void {
    const a = self.arena.allocator();
    const ne = m.entities.items.len;
    const np = m.pairs.items.len;
    for (self.own_pair) |p| if (p >= np) return error.BadSnapshot;
    for (self.own_entity) |e| if (e >= ne) return error.BadSnapshot;
    for (m.pairs.items) |p| if (p.entity >= ne) return error.BadSnapshot;
    const cur_off = try a.alloc(u32, ne + 1);
    const key_off = try a.alloc(u32, ne + 1);
    @memset(cur_off, 0);
    @memset(key_off, 0);
    for (self.state, self.own_entity) |st, e| if (st == .current) {
        cur_off[e + 1] += 1;
    };
    for (m.pairs.items) |p| key_off[p.entity + 1] += 1;
    for (1..ne + 1) |e| {
        cur_off[e] += cur_off[e - 1];
        key_off[e] += key_off[e - 1];
    }
    const cur_items = try a.alloc(u32, cur_off[ne]);
    const key_items = try a.alloc(u32, key_off[ne]);
    // each group fills from its start: off[e] is moved along it, then back
    for (self.state, self.own_entity, 0..) |st, e, k| if (st == .current) {
        cur_items[cur_off[e]] = @intCast(k);
        cur_off[e] += 1;
    };
    for (m.pairs.items, 0..) |p, k| {
        key_items[key_off[p.entity]] = @intCast(k);
        key_off[p.entity] += 1;
    }
    var e = ne;
    while (e > 0) : (e -= 1) {
        cur_off[e] = cur_off[e - 1];
        key_off[e] = key_off[e - 1];
    }
    cur_off[0] = 0;
    key_off[0] = 0;
    var total: usize = 0;
    for (m.pairs.items) |p| total += p.ids.items.len;
    const pair_off = try a.alloc(u32, np + 1);
    const pair_items = try a.alloc(u32, total);
    var at: u32 = 0;
    for (m.pairs.items, 0..) |p, k| {
        pair_off[k] = at;
        for (p.ids.items) |id| {
            pair_items[at] = self.placeOfId(id) orelse return error.BadSnapshot;
            at += 1;
        }
    }
    pair_off[np] = at;
    self.own_ent_cur = .{ .off = cur_off, .items = cur_items };
    self.own_ent_keys = .{ .off = key_off, .items = key_items };
    self.own_pair_notes = .{ .off = pair_off, .items = pair_items };
}

/// The entity, attribute and pair-attribute names from these numbers on, copied into the version (names_from).
pub fn namesOf(self: *Index, m: *const Model, e_from: usize, a_from: usize, p_from: usize) Allocator.Error!void {
    const a = self.arena.allocator();
    const entities = try a.alloc([]const u8, m.entities.items.len - e_from);
    for (m.entities.items[e_from..], entities) |name, *out| out.* = try a.dupe(u8, name);
    const attributes = try a.alloc([]const u8, m.attributes.items.len - a_from);
    for (m.attributes.items[a_from..], attributes) |name, *out| out.* = try a.dupe(u8, name);
    const pair_attributes = try a.alloc(u32, m.pairs.items.len - p_from);
    for (m.pairs.items[p_from..], pair_attributes) |p, *out| out.* = p.attribute;
    self.entity_names = entities;
    self.attribute_names = attributes;
    self.pair_attributes = pair_attributes;
    self.names_from = .{ @intCast(e_from), @intCast(a_from), @intCast(p_from) };
}

/// Entity phrases: only 2 to 8 keys can ever match (subjects() looks at those lengths); a later one wins.
pub fn phrasesOf(self: *Index, phrases: []const model_mod.Phrase, n_entities: usize) LoadError!void {
    const a = self.arena.allocator();
    var count: u32 = 0;
    for (phrases) |p| count += @intFromBool(p.keys.len >= 2 and p.keys.len <= max_phrase);
    try self.own_phrases.ensureTotalCapacity(a, count); // once: no outgrown table left behind
    for (phrases) |p| {
        if (p.entity >= n_entities) return error.BadSnapshot;
        if (p.keys.len == 1) try self.own_names.put(a, p.keys[0], p.entity);
        if (p.keys.len < 2 or p.keys.len > max_phrase) continue;
        self.own_phrases.putAssumeCapacity(packPhrase(p.keys), p.entity);
        self.own_phrase_max = @max(self.own_phrase_max, p.keys.len);
    }
}

/// The postings of both views, from each note's token ids by the same grams() as a question's.
pub fn buildPostings(self: *Index, gpa: Allocator, raw: []const Raw, first: usize) LoadError!void {
    // raw: the notes from place `first` on (sa: the side buffer's)
    const a = self.arena.allocator();
    const Hit = struct { entry: u32, note: u32 };
    var hits: std.ArrayList(Hit) = .empty;
    defer hits.deinit(gpa);
    var keys: std.ArrayList(u32) = .empty;
    defer keys.deinit(gpa);
    var gl: std.ArrayList(Gram) = .empty;
    defer gl.deinit(gpa);
    var entries: std.ArrayList(Post) = .empty; // counts first, then offsets
    defer entries.deinit(gpa);
    var map: std.HashMapUnmanaged(Gram, u32, GramCtx, 80) = .empty;
    defer map.deinit(gpa);
    for (raw, 0..) |x, k| {
        keys.clearRetainingCapacity();
        for (x.tokens) |t| try keys.append(gpa, self.keyOf(t));
        gl.clearRetainingCapacity();
        try self.gramsOf(gpa, x.tokens, keys.items, &gl);
        for (gl.items) |g| {
            const r = try map.getOrPut(gpa, g);
            if (!r.found_existing) {
                r.value_ptr.* = @intCast(entries.items.len);
                try entries.append(gpa, .{ .cur_off = 0, .cur_len = 0, .hist_off = 0, .hist_len = 0 });
            }
            const e = &entries.items[r.value_ptr.*];
            if (self.stateAt(@intCast(first + k)) == .current) e.cur_len += 1 else e.hist_len += 1;
            try hits.append(gpa, .{ .entry = r.value_ptr.*, .note = @intCast(first + k) });
        }
    }
    var off: u32 = 0;
    for (entries.items) |*e| {
        e.cur_off = off;
        off += e.cur_len;
        e.hist_off = off;
        off += e.hist_len;
    }
    const post = try a.alloc(u32, off);
    const fill = try gpa.alloc(u32, entries.items.len * 2);
    defer gpa.free(fill);
    @memset(fill, 0);
    for (hits.items) |h| {
        const e = entries.items[h.entry];
        if (self.stateAt(h.note) == .current) {
            post[e.cur_off + fill[2 * h.entry]] = h.note;
            fill[2 * h.entry] += 1;
        } else {
            post[e.hist_off + fill[2 * h.entry + 1]] = h.note;
            fill[2 * h.entry + 1] += 1;
        }
    }
    self.post = post;
    self.grams = .empty;
    try self.grams.ensureTotalCapacity(a, @intCast(entries.items.len)); // counted first: one table, no regrowth
    var it = map.iterator();
    while (it.next()) |kv| self.grams.putAssumeCapacity(kv.key_ptr.*, entries.items[kv.value_ptr.*]);
}
