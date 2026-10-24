//! A version layered on a core: the core's columns and suffix array shared, the notes since it kept apart.
const std = @import("std");
const index = @import("index.zig");
const model_mod = @import("model.zig");
const value = @import("value.zig");
const sa = @import("sa.zig");
const exact = @import("exact_alloc.zig");
const ib = @import("index_build.zig");
const Index = index.Index;
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// A core note retired since the core was built: its version-local state, line and cost.
pub const Over = struct { state: index.State, line: []const u8, cost: u32 };

/// What a layered version keeps beside its core, in its own arena; a core's or a full index's is empty.
pub const Layer = struct {
    core: ?*Index = null, // retained: a full sa index, shared by the versions layered on it
    core_n: u32 = 0, // places 0..core_n-1 are the core's; the version's own columns hold the rest
    first_id: u32 = 0, // the first side note's id: own_id_place is indexed by id - first_id
    over: std.AutoHashMapUnmanaged(u32, Over) = .empty,
    lazy_values: bool = false, // names were added since the core: its written notes' value entities resolve at read
    side_cur: std.AutoHashMapUnmanaged(u32, []const u32) = .empty, // entity -> its side notes current now, by place
    side_keys_of: std.AutoHashMapUnmanaged(u32, []const u32) = .empty, // entity -> the pairs made since the core
    side_pairs: std.AutoHashMapUnmanaged(u32, []const u32) = .empty, // pair -> its side notes, by place
    more_entities: u32 = 0, // entities made since the core
    side_tokens: []const []const u32 = &.{}, // each side note's token ids (the model's bodies): a merge reads them
    side_unvalued: []const bool = &.{}, // a side note written with no value entity given (value.ofWrites names it)

    pub fn sidePlaceOfId(l: *const Layer, idx: *const Index, id: u32) ?u32 {
        if (id < l.first_id) return null;
        const k = id - l.first_id;
        if (k >= idx.own_id_place.len or idx.own_id_place[k] == index.no_note) return null;
        return idx.own_id_place[k];
    }
};

/// Tests only: the side's current notes left out of their entities' groups (notes_delta_test.zig's planted bug).
pub var plant_skip_side_cur: bool = false;

/// `core` with the notes past it as a side buffer; `retired`: places retired since; null: a full build is needed.
pub fn build(gpa: Allocator, m: *const Model, core: *Index, retired: []const u32) index.LoadError!?*Index {
    const n = m.notes.items.len;
    const cn: usize = core.n_notes;
    if (core.layer.core != null or core.kind != .sa or core.base == null or n < cn or !core.rank_by_note) return null;
    var last = core.max_told;
    var prev_id: u32 = if (cn > 0) core.idAt(@intCast(cn - 1)) else 0;
    for (m.notes.items[cn..], cn..) |*x, p| { // the tie rank is each note's own (index.zig rankAt): no order to keep
        if (!index.rankFits(p, x.told) or (p > 0 and x.id <= prev_id)) return null;
        prev_id = x.id;
        last = @max(last, x.told);
    }
    const self = try gpa.create(Index);
    errdefer gpa.destroy(self);
    self.* = .{ .arena = exact.Exact.init(gpa), .log = core.log, .settings = core.settings, .head = core.head, .cur_title = core.cur_title, .hist_title = core.hist_title, .fx_head = core.fx_head, .fx_cur = core.fx_cur, .fx_hist = core.fx_hist, .fx_nl = core.fx_nl, .vkey = core.vkey, .vwordy = core.vwordy, .tokenizer_sha256 = core.tokenizer_sha256, .own_id = &.{}, .state = &.{}, .own_told = &.{}, .own_trank = &.{}, .own_entity = &.{}, .own_pair = &.{}, .own_value = &.{}, .own_line = &.{}, .own_cost = &.{}, .min_line = core.min_line, .own_id_place = &.{}, .own_ent_cur = .{}, .own_ent_keys = .{}, .own_pair_notes = .{}, .kind = .sa, .base = core.base, .grams = .empty, .post = &.{}, .own_phrases = .empty, .own_phrase_max = 0, .own_names = .empty, .n_cur = 0, .n_hist = 0, .entity_names = &.{}, .attribute_names = &.{}, .pair_attributes = &.{}, .aliases = .empty, .alias_entities = .empty, .alias_max = 0, .self_entity = m.entityOf("me"), .rank_by_note = true };
    errdefer self.arena.deinit();
    const a = self.arena.allocator();
    self.n_notes = @intCast(n);
    self.n_phrases = @intCast(m.phrases.items.len);
    self.max_told = last;
    self.layer.core = core; // retained once the build cannot fail; read by the accessors below
    self.layer.core_n = @intCast(cn);
    self.layer.first_id = if (n > cn) m.notes.items[cn].id else 0;
    self.layer.more_entities = @intCast(m.entities.items.len - core.entity_names.len);
    try ib.namesOf(self, m, core.entity_names.len, core.attribute_names.len, core.pair_attributes.len);
    const state = try a.alloc(index.State, n - cn);
    try self.layer.over.ensureTotalCapacity(a, @intCast(retired.len));
    var gone: u32 = 0;
    for (retired) |p| if (p < cn) {
        const x = &m.notes.items[p];
        if (!self.layer.over.contains(p) and core.stateAt(p) == .current and x.state != .current) gone += 1;
        self.layer.over.putAssumeCapacity(p, .{ .state = x.state, .line = x.line(), .cost = x.cost });
        self.min_line = @min(self.min_line, @as(i64, x.cost) + self.fx_nl);
        core.markRetired(@intCast(p));
    };
    const side = m.notes.items[cn..];
    const raw = try gpa.alloc(index.Raw, side.len); // the build's own, freed when it returns
    defer gpa.free(raw);
    for (side, raw, cn..) |*x, *r, p| {
        r.* = .{ .id = x.id, .state = x.state, .entity = x.entity, .pair = x.pair, .value_ent = x.value_ent, .told = x.told, .line = x.line(), .cost = x.cost, .tokens = x.tokens() };
        state[p - cn] = x.state;
    }
    self.state = state;
    const toks = try a.alloc([]const u32, side.len);
    const unvalued = try a.alloc(bool, side.len);
    for (side, toks, unvalued) |*x, *t, *u| {
        t.* = x.tokens();
        u.* = x.source() != null and x.value_ent < 0;
    }
    self.layer.side_tokens = toks;
    self.layer.side_unvalued = unvalued;
    try sideColumns(self, a, raw);
    var cur: u32 = 0;
    for (raw) |r| cur += @intFromBool(r.state == .current);
    self.n_cur = core.n_cur - gone + cur;
    self.n_hist = self.n_notes - self.n_cur;
    try sideGroups(self, gpa, a, m, raw, core);
    try ib.phrasesOf(self, m.phrases.items[core.n_phrases..], m.entities.items.len);
    try sideValues(self, m);
    self.layer.lazy_values = m.phrases.items.len > core.n_phrases;
    try ib.sideKeys(self, m, raw);
    try ib.buildPostings(self, gpa, raw, cn);
    _ = core.retain(); // last: an error before it leaves the core's count as it was
    return self;
}

/// The side notes' own columns (places core_n..), by place; their ids give own_id_place.
fn sideColumns(self: *Index, a: Allocator, raw: []const index.Raw) Allocator.Error!void {
    const k = raw.len;
    const id = try a.alloc(u32, k);
    const told = try a.alloc(i64, k);
    const entity = try a.alloc(u32, k);
    const pair = try a.alloc(u32, k);
    const val = try a.alloc(i32, k);
    const line = try a.alloc([]const u8, k);
    const cost = try a.alloc(u32, k);
    for (raw, 0..) |r, j| {
        id[j] = r.id;
        told[j] = r.told;
        entity[j] = r.entity;
        pair[j] = r.pair;
        val[j] = r.value_ent;
        line[j] = r.line;
        cost[j] = r.cost;
        self.min_line = @min(self.min_line, @as(i64, r.cost) + self.fx_nl);
    }
    const span: usize = if (k > 0) raw[k - 1].id - self.layer.first_id + 1 else 0;
    const id_place = try a.alloc(u32, span);
    @memset(id_place, index.no_note);
    for (raw, 0..) |r, j| id_place[r.id - self.layer.first_id] = self.layer.core_n + @as(u32, @intCast(j));
    self.own_id = id;
    self.own_told = told;
    self.own_entity = entity;
    self.own_pair = pair;
    self.own_value = val;
    self.own_line = line;
    self.own_cost = cost;
    self.own_id_place = id_place;
}

/// The side's part of each group: entity -> its current side notes, pair -> its side notes, entity -> new pairs.
fn sideGroups(self: *Index, gpa: Allocator, a: Allocator, m: *const Model, raw: []const index.Raw, core: *const Index) Allocator.Error!void {
    const cn = self.layer.core_n;
    var cur: std.AutoArrayHashMapUnmanaged(u32, std.ArrayList(u32)) = .empty;
    var prs: std.AutoArrayHashMapUnmanaged(u32, std.ArrayList(u32)) = .empty;
    var keys: std.AutoArrayHashMapUnmanaged(u32, std.ArrayList(u32)) = .empty;
    defer for ([_]*std.AutoArrayHashMapUnmanaged(u32, std.ArrayList(u32)){ &cur, &prs, &keys }) |h| {
        for (h.values()) |*l| l.deinit(gpa);
        h.deinit(gpa);
    };
    for (raw, 0..) |r, j| {
        const p: u32 = cn + @as(u32, @intCast(j));
        if (r.state == .current and !plant_skip_side_cur) try appendTo(gpa, &cur, r.entity, p);
        try appendTo(gpa, &prs, r.pair, p);
    }
    const core_np = core.own_pair_notes.len();
    for (m.pairs.items[core_np..], core_np..) |pr, k| try appendTo(gpa, &keys, pr.entity, @intCast(k));
    try settle(a, &cur, &self.layer.side_cur);
    try settle(a, &prs, &self.layer.side_pairs);
    try settle(a, &keys, &self.layer.side_keys_of);
}

fn appendTo(gpa: Allocator, h: *std.AutoArrayHashMapUnmanaged(u32, std.ArrayList(u32)), k: u32, v: u32) Allocator.Error!void {
    const r = try h.getOrPut(gpa, k);
    if (!r.found_existing) r.value_ptr.* = .empty;
    try r.value_ptr.append(gpa, v);
}

/// The build's lists copied into the version's arena at their sizes, one table sized once.
fn settle(a: Allocator, from: *const std.AutoArrayHashMapUnmanaged(u32, std.ArrayList(u32)), to: *std.AutoHashMapUnmanaged(u32, []const u32)) Allocator.Error!void {
    try to.ensureTotalCapacity(a, @intCast(from.count()));
    for (from.keys(), from.values()) |k, l| to.putAssumeCapacity(k, try a.dupe(u32, l.items));
}

/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 13.
fn sideValues(self: *Index, m: *const Model) Allocator.Error!void {
    const cn = self.layer.core_n;
    const own = @constCast(self.own_value);
    for (m.notes.items[cn..], 0..) |*x, j| {
        if (x.source() == null or x.value_ent >= 0) continue;
        if (value.named(self, x.tokens(), x.entity)) |e| own[j] = @intCast(e);
    }
}
