//! Vendored from LLWZ notes-zig a81d1a9 (zig/notes/src/notes.zig; MIT, (c) 2026 ThinkOff): the notes' index.
const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const model_mod = @import("model.zig");
const exact = @import("exact_alloc.zig");
const sa = @import("sa.zig");
const ib = @import("index_build.zig");
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// A log function, called through a pointer.
pub const LogFn = *const fn (f64) callconv(.c) f64;

fn zigLog(x: f64) callconv(.c) f64 {
    return @log(x);
}

/// The C library's log (CPython's math.log calls it), found by name: Zig's own differs in the last bit for some inputs.
pub fn libmLog() LogFn {
    if (comptime builtin.os.tag == .windows) {
        const crt = struct {
            pub extern "c" fn log(x: f64) f64;
        };
        return &crt.log;
    }
    // RTLD_DEFAULT: the handle that searches every loaded image (-2 on Apple systems, 0 elsewhere)
    const default: ?*anyopaque = if (builtin.os.tag.isDarwin()) @ptrFromInt(@as(usize, @bitCast(@as(isize, -2)))) else null;
    if (std.c.dlsym(default, "log")) |p| return @ptrCast(@alignCast(p));
    return &zigLog;
}

pub fn isZigLog(f: LogFn) bool {
    return f == &zigLog;
}

/// A key that is the token id itself has this bit; a fold string's key is its place in the interned list.
pub const key_id: u32 = 0x8000_0000;
/// No key: an unused place in a packed n-gram, and the key of a token id the vocabulary lacks (it matches nothing).
pub const none_key = types.none_key;
/// N-grams are packed into one u128, four keys of 32 bits, so n_max is at most 4.
pub const max_n = types.max_n;
/// subjects() looks for entity phrases of 2 to 8 keys.
pub const max_phrase = 8;
pub const no_note: u32 = 0xFFFF_FFFF;
/// A layered version's alias override for a key that names nobody now.
pub const alias_gone: u32 = 0xFFFF_FFFE;
/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 10.
pub const id_bits = 24;
pub const told_span: i64 = (1 << 40) - 1;

/// A core's retired bits, all clear.
pub fn retiredBits(a: Allocator, n: usize) Allocator.Error![]std.atomic.Value(u64) {
    const bits = try a.alloc(std.atomic.Value(u64), (n + 63) / 64);
    for (bits) |*w| w.* = .init(0);
    return bits;
}

/// Tests only: false makes every full build rank by the global sort (the old path, the differential test's oracle).
pub threadlocal var rank_by_note_allowed: bool = true;

/// Whether a note's place and told make its rankAt() key.
pub fn rankFits(place: usize, told: i64) bool {
    return place < (1 << id_bits) and told >= 0 and told <= told_span;
}

pub const State = types.State;
/// The n-gram index: LLWZ's posting lists, or a suffix array over the notes' folded keys.
pub const Kind = enum { postings, sa };

/// What --notes-index and the notes benches build unless told: the suffix array (postings stays selectable).
pub const default_kind: Kind = .sa;

/// Matches over n_max keys, sa only, off by default: all n-grams up to `max`, or each position's longest (lookup.zig).
pub const Long = struct { mode: enum { off, all, longest } = .off, max: u8 = 8 };
pub const Hop = types.Hop;
pub const Settings = types.Settings;
pub const LoadError = types.LoadError;

pub const Gram = u128;

/// Keys packed into one u128; the unused places hold none_key, so (a, b) and (a, b, c) never meet.
pub fn packGram(keys: []const u32) Gram {
    var g: Gram = 0;
    var j: usize = 0;
    while (j < max_n) : (j += 1) {
        const k: u32 = if (j < keys.len) keys[j] else none_key;
        g |= @as(Gram, k) << @intCast(32 * j);
    }
    return g;
}

pub const GramCtx = struct {
    pub fn hash(_: GramCtx, g: Gram) u64 {
        const lo: u64 = @truncate(g);
        const hi: u64 = @truncate(g >> 64);
        const m = @as(u128, lo ^ 0xa076_1d64_78bd_642f) *% @as(u128, hi ^ 0xe703_7ed1_a0b4_28db);
        return @as(u64, @truncate(m)) ^ @as(u64, @truncate(m >> 64));
    }
    pub fn eql(_: GramCtx, a: Gram, b: Gram) bool {
        return a == b;
    }
};

pub const Phrase = [max_phrase]u32;
pub const PhraseCtx = struct {
    pub fn hash(_: PhraseCtx, p: Phrase) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&p));
    }
    pub fn eql(_: PhraseCtx, a: Phrase, b: Phrase) bool {
        return std.mem.eql(u32, &a, &b);
    }
};

pub fn packPhrase(keys: []const u32) Phrase {
    var p: Phrase = @splat(none_key);
    @memcpy(p[0..keys.len], keys);
    return p;
}

/// Postings of one n-gram: note places in the current view, then in the history view.
pub const Post = struct { cur_off: u32, cur_len: u32, hist_off: u32, hist_len: u32 };

/// Groups of note places (or pairs) by a number, flat: group k is items[off[k]..off[k+1]].
pub const Groups = struct {
    off: []const u32 = &.{},
    items: []const u32 = &.{},

    pub inline fn of(g: Groups, k: usize) []const u32 {
        return g.items[g.off[k]..g.off[k + 1]];
    }

    pub fn len(g: Groups) usize {
        return g.off.len -| 1;
    }
};

/// One note's places in a group's slices: a core's (live: only those current now), then a layered version's side.
pub const Iter = struct {
    a: []const u32,
    b: []const u32 = &.{},
    i: usize = 0,
    idx: *const Index,
    live: bool = false,

    pub inline fn next(it: *Iter) ?u32 {
        while (it.i < it.a.len) {
            const p = it.a[it.i];
            it.i += 1;
            if (!it.live or it.idx.stateAt(p) == .current) return p;
        }
        const k = it.i - it.a.len;
        if (k >= it.b.len) return null;
        it.i += 1;
        return it.b[k];
    }
};

/// One note as the snapshot gives it.
pub const Raw = struct { id: u32, state: State, entity: u32, pair: u32, value_ent: i32, told: i64, line: []const u8, cost: u32, tokens: []const u32 };

pub const Index = struct {
    arena: exact.Exact, // every array at its exact size: no arena node grown 1.5 times past the postings
    log: LogFn,
    settings: Settings,
    head: []const u8,
    cur_title: []const u8,
    hist_title: []const u8,
    fx_head: u32,
    fx_cur: u32,
    fx_hist: u32,
    fx_nl: u32,
    vkey: []const u32,
    vwordy: []const bool,
    /// The tokenizer.json SHA-256 the snapshot was exported with ("" when it names none).
    tokenizer_sha256: []const u8,
    // the notes by place (ids: Python's); layered (layer.zig): own_ columns from core_n on, read by the accessors
    own_id: []const u32,
    state: []const State, // every place's, in every version
    own_told: []const i64,
    own_trank: []const u32, // place in the order (-told, id): Python's tie-break after the score
    own_entity: []const u32,
    own_pair: []const u32,
    own_value: []const i32,
    own_line: []const []const u8,
    own_cost: []const u32,
    min_line: i64, // the least any line can add in pack(): its cost and a line break (a layered version's: at most it)
    own_id_place: []const u32, // a note id (layered: less first_id) -> its place, or no_note
    own_ent_cur: Groups, // entity -> places of its current notes (O.ent_cur)
    own_ent_keys: Groups, // entity -> its pairs (O.ent_keys)
    own_pair_notes: Groups, // pair -> places of its notes, every state, oldest first (O.by_key)
    // names by number, this version's own from names_from on (a layered version shares its core's)
    entity_names: []const []const u8,
    attribute_names: []const []const u8,
    pair_attributes: []const u32,
    kind: Kind = .postings,
    /// sa: the suffix arrays of places 0..base.notes-1, shared by the versions built on it (null: postings only)
    base: ?*sa.Base = null,
    /// sa: the side buffer's notes' keys (place base.notes + k is group k), for matches longer than n_max
    side_keys: Groups = .{},
    /// postings: every note's; sa: the side buffer's, the notes written since the base (places base.notes..)
    grams: std.HashMapUnmanaged(Gram, Post, GramCtx, 80),
    post: []const u32,
    own_phrases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80), // layered: the phrases the side buffer added
    own_phrase_max: usize,
    own_names: std.AutoHashMapUnmanaged(u32, u32), // one-key entity phrases: key -> entity (the decode check's)
    // aliases (alias.zig): a full build's whole tables; a layered version's hold only the keys a write touched since

    // its core (alias_gone: no alias now), read before the core's by aliasOf() and aliasCandidates()
    aliases: std.HashMapUnmanaged(Phrase, u32, PhraseCtx, 80),
    alias_entities: std.HashMapUnmanaged(Phrase, std.ArrayList(u32), PhraseCtx, 80),
    alias_max: usize,
    self_entity: ?u32,
    n_cur: u32,
    n_hist: u32,
    owned: ?*Model = null, // the model a loaded snapshot made, freed with the index
    ver: u64 = 0, // the store's version number for this build (0: a loaded snapshot)
    refs: std.atomic.Value(u32) = .init(1), // a core is shared by the versions layered on it
    n_notes: u32 = 0,
    max_told: i64 = std.math.minInt(i64), // the latest told of its notes
    n_phrases: u32 = 0, // the model's phrases it was built from (a layered version's side adds the rest)
    layer: @import("layer.zig").Layer = .{}, // a version layered on a core: the core and what changed since
    rank_by_note: bool = false, // rankAt() from each note's told and place (every one fits), else own_trank
    retired: []std.atomic.Value(u64) = &.{}, // sa core: a bit per place some version on it overrides (overOf)
    lazy_value: std.AutoHashMapUnmanaged(u32, []const u32) = .empty, // sa: written notes no model value names: tokens
    names_from: [3]u32 = .{ 0, 0, 0 },
    reaper: ?*@import("bounded.zig").Reaper = null, // frees this layered version's core off the caller's thread
    alias_candidates_off: bool = false, // a lookup view that reads no alias candidates (notes_decode.zig's) // entity_names, attribute_names, pair_attributes start at these numbers

    pub fn notes(self: *const Index) usize {
        return self.n_notes;
    }

    // Version-local metadata: a layered version (layer.zig) reads places below core_n from its core, unless a

    // later write retired them (layer.over, layer.values), and its own columns from core_n on.
    pub inline fn stateAt(self: *const Index, place: u32) State {
        if (place < self.layer.core_n) return if (self.overOf(place)) |o| o.state else self.layer.core.?.state[place];
        return self.state[place - self.layer.core_n];
    }

    /// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 11.
    inline fn overOf(self: *const Index, p: u32) ?@import("layer.zig").Over {
        if (self.layer.over.count() == 0) return null;
        const c = self.layer.core.?;
        if (c.retired.len > 0 and c.retired[p >> 6].load(.monotonic) & (@as(u64, 1) << @intCast(p & 63)) == 0) return null;
        return self.layer.over.get(p);
    }

    /// The writer marks a core place retired (layer.zig), before any version that overrides it is published.
    pub fn markRetired(self: *const Index, p: u32) void {
        if (self.retired.len > 0) _ = self.retired[p >> 6].fetchOr(@as(u64, 1) << @intCast(p & 63), .monotonic);
    }

    pub fn lineAt(self: *const Index, p: u32) []const u8 {
        if (p < self.layer.core_n) return if (self.overOf(p)) |o| o.line else self.layer.core.?.own_line[p];
        return self.own_line[p - self.layer.core_n];
    }

    pub fn costAt(self: *const Index, p: u32) u32 {
        if (p < self.layer.core_n) return if (self.overOf(p)) |o| o.cost else self.layer.core.?.own_cost[p];
        return self.own_cost[p - self.layer.core_n];
    }

    pub inline fn idAt(self: *const Index, p: u32) u32 {
        return if (p < self.layer.core_n) self.layer.core.?.own_id[p] else self.own_id[p - self.layer.core_n];
    }

    pub inline fn toldAt(self: *const Index, p: u32) i64 {
        return if (p < self.layer.core_n) self.layer.core.?.own_told[p] else self.own_told[p - self.layer.core_n];
    }

    pub inline fn entityAt(self: *const Index, p: u32) u32 {
        return if (p < self.layer.core_n) self.layer.core.?.own_entity[p] else self.own_entity[p - self.layer.core_n];
    }

    pub inline fn pairAt(self: *const Index, p: u32) u32 {
        return if (p < self.layer.core_n) self.layer.core.?.own_pair[p] else self.own_pair[p - self.layer.core_n];
    }

    pub fn valueEntityAt(self: *const Index, p: u32) i32 {
        if (p < self.layer.core_n) {
            const c = self.layer.core.?;
            if (self.layer.lazy_values) if (c.lazy_value.get(p)) |tokens| { // value.ofWrites, against this version
                return if (@import("value.zig").named(self, tokens, c.own_entity[p])) |e| @intCast(e) else -1;
            };
            return c.own_value[p];
        }
        return self.own_value[p - self.layer.core_n];
    }

    /// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 12.
    pub inline fn rankAt(self: *const Index, p: u32) u64 {
        if (self.rank_by_note) return (@as(u64, @intCast(told_span - self.toldAt(p))) << id_bits) | p;
        return (@as(u64, self.own_trank[p]) << 32) | p;
    }

    /// The place a rankAt() key was made from.
    pub inline fn placeOfRank(self: *const Index, r: u64) u32 {
        if (self.rank_by_note) return @intCast(r & ((1 << id_bits) - 1));
        return @truncate(r);
    }

    /// The entity an alias names (no_note: several do), or null.
    pub fn aliasOf(self: *const Index, p: Phrase) ?u32 {
        if (self.aliases.get(p)) |e| return if (e == alias_gone) null else e;
        return if (self.layer.core) |c| c.aliases.get(p) else null;
    }

    /// Every entity an alias names, in the order their name facts were written, or null.
    pub fn aliasCandidates(self: *const Index, p: Phrase) ?[]const u32 {
        if (self.alias_candidates_off) return null;
        if (self.alias_entities.get(p)) |l| return if (l.items.len == 0) null else l.items;
        return if (self.layer.core) |c| if (c.alias_entities.get(p)) |l| l.items else null else null;
    }

    /// The longest alias, in keys (at least the longest: only a bound for the lookups' loops).
    pub fn aliasMax(self: *const Index) usize {
        return @max(self.alias_max, if (self.layer.core) |c| c.alias_max else 0);
    }

    /// The entities this version knows (entity numbers below it).
    pub fn entityCount(self: *const Index) usize {
        if (self.layer.core) |c| return c.own_ent_cur.len() + self.layer.more_entities;
        return self.own_ent_cur.len();
    }

    /// The attributes this version knows.
    pub fn attributeCount(self: *const Index) usize {
        return self.names_from[1] + self.attribute_names.len;
    }

    pub fn entityName(self: *const Index, e: u32) []const u8 {
        if (e < self.names_from[0]) return self.layer.core.?.entityName(e);
        return self.entity_names[e - self.names_from[0]];
    }

    pub fn attributeName(self: *const Index, x: u32) []const u8 {
        if (x < self.names_from[1]) return self.layer.core.?.attributeName(x);
        return self.attribute_names[x - self.names_from[1]];
    }

    pub fn pairAttribute(self: *const Index, pr: u32) u32 {
        if (pr < self.names_from[2]) return self.layer.core.?.pairAttribute(pr);
        return self.pair_attributes[pr - self.names_from[2]];
    }

    /// A note id's place in the index, if the snapshot has that note.
    pub fn placeOfId(self: *const Index, id: u32) ?u32 {
        if (self.layer.core) |c| {
            if (c.placeOfId(id)) |q| return q;
            return self.layer.sidePlaceOfId(self, id);
        }
        if (id >= self.own_id_place.len or self.own_id_place[id] == no_note) return null;
        return self.own_id_place[id];
    }

    /// One more reader of this index (a version layered on it); deinit() drops one.
    pub fn retain(self: *Index) *Index {
        _ = self.refs.fetchAdd(1, .monotonic);
        return self;
    }

    /// Drops one reference; the last frees the index (a layered version's core goes with its last version).
    pub fn deinit(self: *Index) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        self.destroy();
    }

    /// Frees it now (its last reference gone); a layered version's core goes to `reaper` when it has one.
    pub fn destroy(self: *Index) void {
        const gpa = self.arena.child;
        if (self.layer.core) |c| {
            if (self.reaper) |r| r.drop(c) else c.deinit();
        } else if (self.base) |b| b.release();
        self.arena.deinit();
        if (self.owned) |m| m.deinit();
        gpa.destroy(self);
    }

    /// The places of an entity's current notes, by place (O.ent_cur).
    pub fn entCur(self: *const Index, e: u32) Iter {
        if (self.layer.core) |c| return .{ .a = c.groupOf(.cur, e), .b = self.layer.side_cur.get(e) orelse &.{}, .idx = self, .live = true };
        return .{ .a = self.own_ent_cur.of(e), .idx = self };
    }

    /// An entity's pairs (O.ent_keys).
    pub fn entKeys(self: *const Index, e: u32) Iter {
        if (self.layer.core) |c| return .{ .a = c.groupOf(.keys, e), .b = self.layer.side_keys_of.get(e) orelse &.{}, .idx = self };
        return .{ .a = self.own_ent_keys.of(e), .idx = self };
    }

    /// A pair's notes, every state, oldest first (O.by_key).
    pub fn pairNotes(self: *const Index, pr: u32) Iter {
        if (self.layer.core) |c| return .{ .a = c.groupOf(.pairs, pr), .b = self.layer.side_pairs.get(pr) orelse &.{}, .idx = self };
        return .{ .a = self.own_pair_notes.of(pr), .idx = self };
    }

    /// A core's group k of `which`, or none when k came after it was built.
    fn groupOf(self: *const Index, which: enum { cur, keys, pairs }, k: u32) []const u32 {
        const g = switch (which) {
            .cur => self.own_ent_cur,
            .keys => self.own_ent_keys,
            .pairs => self.own_pair_notes,
        };
        return if (k < g.len()) g.of(k) else &.{};
    }

    /// The entity a phrase of 2 to 8 keys names (a later phrase wins, as the model's list order makes it).
    pub fn phraseOf(self: *const Index, p: Phrase) ?u32 {
        if (self.own_phrases.get(p)) |e| return e;
        return if (self.layer.core) |c| c.own_phrases.get(p) else null;
    }

    /// The entity a one-key name names.
    pub fn nameOf(self: *const Index, k: u32) ?u32 {
        if (self.own_names.get(k)) |e| return e;
        return if (self.layer.core) |c| c.own_names.get(k) else null;
    }

    pub fn phraseMax(self: *const Index) usize {
        return @max(self.own_phrase_max, if (self.layer.core) |c| c.own_phrase_max else 0);
    }

    pub fn phraseCount(self: *const Index) usize {
        return self.own_phrases.count() + if (self.layer.core) |c| c.own_phrases.count() else 0;
    }

    /// Reads a snapshot (notes_export.py's JSON) and builds the index, checking notes and distinct n-grams per view.
    pub fn load(gpa: Allocator, text: []const u8) LoadError!*Index {
        const m = try model_mod.parse(gpa, text);
        errdefer m.deinit();
        const self = try build(gpa, m);
        self.owned = m;
        errdefer {
            self.owned = null;
            self.deinit();
        }
        var gc: u32 = 0;
        var gh: u32 = 0;
        var it = self.grams.valueIterator();
        while (it.next()) |p| {
            if (p.cur_len > 0) gc += 1;
            if (p.hist_len > 0) gh += 1;
        }
        if (gc != m.grams_current or gh != m.grams_history) return error.BadSnapshot;
        return self;
    }

    /// build() with the n-gram index of `kind` (sa: a new base over every note, nothing in the side buffer).
    pub fn buildKind(gpa: Allocator, m: *const Model, kind: Kind) LoadError!*Index {
        return buildWith(gpa, m, kind);
    }

    /// The index of a model's notes, as a snapshot of them would load; the model must outlive it (it reads its vocab).
    pub fn build(gpa: Allocator, m: *const Model) LoadError!*Index {
        return buildWith(gpa, m, .postings);
    }

    /// The index of `kind` over every note (sa: a new base, nothing in the side buffer); layer.zig builds on one.
    pub fn buildWith(gpa: Allocator, m: *const Model, kind: Kind) LoadError!*Index {
        return buildMerging(gpa, m, kind, null);
    }

    /// buildWith(); sa: the base a linear merge of `prev` (a base over the model's first notes) and the notes after it.
    pub fn buildMerging(gpa: Allocator, m: *const Model, kind: Kind, prev: ?*const sa.Base) LoadError!*Index {
        const self = try gpa.create(Index);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.arena = exact.Exact.init(gpa);
        errdefer self.arena.deinit();
        self.owned = null;
        self.ver = 0;
        self.refs = .init(1);
        self.layer = .{};
        self.alias_candidates_off = false;
        self.n_notes = @intCast(m.notes.items.len);
        self.n_phrases = @intCast(m.phrases.items.len);
        self.max_told = std.math.minInt(i64);
        for (m.notes.items) |*n| self.max_told = @max(self.max_told, n.told);
        self.kind = kind;
        self.base = null;
        errdefer if (self.base) |b| b.release();
        self.log = libmLog();
        self.settings = m.settings;
        self.head = m.head;
        self.cur_title = m.cur_title;
        self.hist_title = m.hist_title;
        self.fx_head = m.fixed.head;
        self.fx_cur = m.fixed.current;
        self.fx_hist = m.fixed.history;
        self.fx_nl = m.fixed.nl;
        self.tokenizer_sha256 = m.tokenizer_sha256;
        self.vkey = m.vkey;
        self.vwordy = m.vwordy;
        const raw = try gpa.alloc(Raw, m.notes.items.len); // the build's own: freed when it returns
        defer gpa.free(raw);
        for (m.notes.items, raw) |n, *r| r.* = .{ .id = n.id, .state = n.state, .entity = n.entity, .pair = n.pair, .value_ent = n.value_ent, .told = n.told, .line = n.line(), .cost = n.cost, .tokens = n.tokens() };
        try ib.placeNotes(self, gpa, raw);
        try ib.groupsOf(self, m);
        self.own_phrases = .empty;
        self.own_phrase_max = 0;
        self.own_names = .empty;
        try ib.phrasesOf(self, m.phrases.items, m.entities.items.len);
        try ib.namesOf(self, m, 0, 0, 0);
        self.aliases = .empty;
        self.alias_entities = .empty;
        self.alias_max = 0;
        self.self_entity = m.entityOf("me");
        const ve = try @import("value.zig").ofWrites(self.arena.allocator(), self, m);
        if (ve.ptr != self.own_value.ptr) self.arena.allocator().free(self.own_value); // one array, as placed
        self.own_value = ve;
        self.lazy_value = .empty;
        if (kind == .sa) { // what a layered version reads to resolve these notes' value entities by its own names
            var k: u32 = 0;
            for (m.notes.items) |*n| if (n.source() != null and n.value_ent < 0) {
                k += 1;
            };
            try self.lazy_value.ensureTotalCapacity(self.arena.allocator(), k);
            for (m.notes.items, 0..) |*n, p| if (n.source() != null and n.value_ent < 0) self.lazy_value.putAssumeCapacity(@intCast(p), n.tokens());
        }
        self.side_keys = .{};
        self.retired = &.{};
        if (kind == .sa) self.retired = try retiredBits(self.arena.allocator(), raw.len);
        if (kind == .sa) self.base = try ib.baseOf(gpa, m, raw, prev);
        const first: usize = if (kind == .sa) raw.len else 0; // sa: the side buffer is empty
        try ib.buildPostings(self, gpa, raw[first..], first);
        return self;
    }

    pub inline fn keyOf(self: *const Index, t: u32) u32 {
        return if (t < self.vkey.len) self.vkey[t] else none_key;
    }

    inline fn wordyOf(self: *const Index, t: u32) bool {
        return t < self.vwordy.len and self.vwordy[t];
    }

    /// NoteStore.grams: n from n_min to n_max at every start, distinct, in the order first met (Python keeps a set).
    pub fn gramsOf(self: *const Index, gpa: Allocator, ids: []const u32, keys: []const u32, out: *std.ArrayList(Gram)) Allocator.Error!void {
        const st = self.settings;
        var n: usize = st.n_min;
        while (n <= st.n_max) : (n += 1) {
            if (keys.len < n) break;
            var j: usize = 0;
            while (j + n <= keys.len) : (j += 1) {
                if (n == 1 and !st.punct and !self.wordyOf(ids[j])) continue; // punctuation alone is no key
                const g = packGram(keys[j..][0..n]);
                if (std.mem.indexOfScalar(Gram, out.items, g) == null) try out.append(gpa, g);
            }
        }
    }
};

/// Reads a snapshot file and loads it.
pub fn loadFile(gpa: Allocator, io: std.Io, path: []const u8) !*Index {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(text);
    return Index.load(gpa, text);
}

test "packGram keeps lengths apart" {
    const a = packGram(&.{ 1, 2 });
    const b = packGram(&.{ 1, 2, none_key });
    const c = packGram(&.{ 1, 2, 3 });
    try std.testing.expect(a == b); // none_key is never a real key, so this pair cannot occur in a snapshot
    try std.testing.expect(a != c);
    try std.testing.expect(packGram(&.{1}) != packGram(&.{ 1, 0 }));
}
