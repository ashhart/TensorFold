//! One owner's notes as LLWZ's NoteStore keeps them: a snapshot's, then the writes since. The index is built from it.
const std = @import("std");
const json = @import("json");
const index = @import("types.zig");
const line_mod = @import("line.zig");
const model_ix = @import("model_ix.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Kind = enum(u8) { fact = 0, tombstone = 1 };

/// One note as NoteStore keeps it: ids, text and source in one block (`body`); the line (replaced on retire) its own.
pub const Note = struct {
    told: i64,
    body: [*]align(4) u8 = undefined, // the token ids (u32), then the text, then the source
    line_ptr: [*]u8 = undefined,
    id: u32,
    entity: u32,
    attribute: u32,
    pair: u32,
    value_ent: i32,
    cost: u32,
    version: u32,
    by: u32 = 0, // the note that superseded or withdrew this one (0: none yet)
    n_tokens: u32 = 0,
    text_len: u32 = 0,
    line_len: u32 = 0,
    source_len: u16 = 0, // the write that made it; 0 for a snapshot's note (a write's source is never empty)
    state: index.State,
    kind: Kind,
    multi: bool = false, // its attribute holds several values (a take-back's label says so under "explicit")

    pub fn tokens(n: *const Note) []const u32 {
        return @as([*]const u32, @ptrCast(n.body))[0..n.n_tokens];
    }

    pub fn text(n: *const Note) []const u8 {
        return n.body[4 * @as(usize, n.n_tokens) ..][0..n.text_len];
    }

    pub fn source(n: *const Note) ?[]const u8 {
        if (n.source_len == 0) return null;
        return n.body[4 * @as(usize, n.n_tokens) + n.text_len ..][0..n.source_len];
    }

    pub fn line(n: *const Note) []u8 {
        return n.line_ptr[0..n.line_len];
    }

    pub fn setLine(n: *Note, l: []u8) void {
        n.line_ptr = l.ptr;
        n.line_len = @intCast(l.len);
    }

    fn bodyBlock(n: *const Note) []align(4) u8 {
        return n.body[0 .. 4 * @as(usize, n.n_tokens) + n.text_len + n.source_len];
    }

    /// The body made: room for `n_tokens` ids (filled by the caller through tokensMut), the text and the source.
    pub fn makeBody(n: *Note, g: Allocator, n_tokens: usize, txt: []const u8, src: ?[]const u8) Allocator.Error!void {
        const s = src orelse "";
        if (s.len > std.math.maxInt(u16) or txt.len > std.math.maxInt(u32) or n_tokens > std.math.maxInt(u32)) return error.OutOfMemory;
        const block = try g.alignedAlloc(u8, .@"4", 4 * n_tokens + txt.len + s.len);
        @memcpy(block[4 * n_tokens ..][0..txt.len], txt);
        @memcpy(block[4 * n_tokens + txt.len ..][0..s.len], s);
        n.body = block.ptr;
        n.n_tokens = @intCast(n_tokens);
        n.text_len = @intCast(txt.len);
        n.source_len = @intCast(s.len);
    }

    pub fn tokensMut(n: *Note) []u32 {
        return @as([*]u32, @ptrCast(n.body))[0..n.n_tokens];
    }
};

/// An (entity, attribute) pair and its notes' ids, every state, oldest first (NoteStore's by_key).
pub const Pair = struct { entity: u32, attribute: u32, ids: std.ArrayList(u32) = .empty };

/// An entity's words as index keys (O.phrases), in the order they were added; the last one added for a key wins.
pub const Phrase = struct { keys: []u32, entity: u32 };

pub const Fixed = struct { head: u32, current: u32, history: u32, nl: u32 };

pub const LoadError = index.LoadError;

pub const Model = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator, // what never changes: the vocabulary, the head and titles
    settings: index.Settings,
    takeback: line_mod.Takeback,
    unknown_rule: bool,
    head: []const u8,
    cur_title: []const u8,
    hist_title: []const u8,
    fixed: Fixed,
    tokenizer_sha256: []const u8,
    vkey: []const u32,
    vwordy: []const bool,
    entities: std.ArrayList([]u8) = .empty,
    attributes: std.ArrayList([]u8) = .empty,
    pairs: std.ArrayList(Pair) = .empty,
    phrases: std.ArrayList(Phrase) = .empty,
    notes: std.ArrayList(Note) = .empty, // by id, ascending
    id_floor: u32 = 1, // persisted high-water mark: an erased id is never assigned again
    base_notes: usize = 0, // the snapshot's notes come first
    told_last: i64 = 0,
    grams_current: u32 = 0, // the snapshot's own n-gram counts, which Index.load checks
    grams_history: u32 = 0,
    ix: model_ix.Ix = .{}, // names, pairs, sources and current facts' normal forms: a write's lookups (model_ix.zig)

    pub fn deinit(m: *Model) void {
        const g = m.gpa;
        deinitLists(m);
        g.destroy(m);
    }

    pub fn freeNote(g: Allocator, n: *Note) void {
        // A write installs cleanup before allocating the body and line. Do

        // not evaluate their undefined pointers until ownership exists.
        if (n.n_tokens != 0 or n.text_len != 0 or n.source_len != 0) g.free(n.bodyBlock());
        if (n.line_len != 0) g.free(n.line());
    }

    /// A note id's place, or null.
    pub fn placeOf(m: *const Model, id: u32) ?u32 {
        const items = m.notes.items;
        var lo: usize = 0;
        var hi: usize = items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (items[mid].id == id) return @intCast(mid);
            if (items[mid].id < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    pub fn nextId(m: *const Model) u32 {
        return @max(m.id_floor, if (m.notes.items.len == 0) 1 else m.notes.items[m.notes.items.len - 1].id +| 1);
    }

    /// The first entity with this name.
    pub fn entityOf(m: *const Model, name: []const u8) ?u32 {
        return m.ix.entity.get(name);
    }

    pub fn attributeOf(m: *const Model, name: []const u8) ?u32 {
        return m.ix.attribute.get(name);
    }

    pub fn pairOf(m: *const Model, entity: u32, attribute: u32) ?u32 {
        return m.ix.pair.get(model_ix.Ix.pairKey(entity, attribute));
    }

    /// The place of the note a write made, by its source.
    pub fn placeOfSource(m: *const Model, source: []const u8) ?u32 {
        return m.ix.source.get(source);
    }

    /// The lookups made from the lists, as a parse ends (a write keeps them up to date: write.zig).
    fn indexAll(m: *Model) LoadError!void {
        const g = m.gpa;
        const x = &m.ix;
        for (m.entities.items, 0..) |e, k| try model_ix.Ix.putName(&x.entity, g, e, @intCast(k));
        for (m.attributes.items, 0..) |e, k| try model_ix.Ix.putName(&x.attribute, g, e, @intCast(k));
        for (m.pairs.items, 0..) |p, k| {
            const r = try x.pair.getOrPut(g, model_ix.Ix.pairKey(p.entity, p.attribute));
            if (!r.found_existing) r.value_ptr.* = @intCast(k);
        }
        for (m.notes.items, 0..) |*n, k| {
            if (n.source()) |s| try model_ix.Ix.putName(&x.source, g, s, @intCast(k));
            if (n.state == .current and n.kind == .fact) try x.addFact(g, try model_ix.factKeyOf(g, n.entity, n.text()), @intCast(k));
        }
    }

    pub fn keyOf(m: *const Model, t: u32) u32 {
        return if (t < m.vkey.len) m.vkey[t] else index.none_key;
    }

    /// Every list copied to an exact block (a shrink in place would leave the allocator's block its old size).
    pub fn trim(m: *Model) Allocator.Error!void {
        const g = m.gpa;
        try trimList(model_note, g, &m.notes);
        try trimList([]u8, g, &m.entities);
        try trimList([]u8, g, &m.attributes);
        for (m.pairs.items) |*p| try trimList(u32, g, &p.ids);
        try trimList(Pair, g, &m.pairs);
        try trimList(Phrase, g, &m.phrases);
    }

    /// Notes counted by view: current, and the rest.
    pub fn counts(m: *const Model) [2]u32 {
        var c: [2]u32 = .{ 0, 0 };
        for (m.notes.items) |n| c[if (n.state == .current) 0 else 1] += 1;
        return c;
    }
};

const model_note = Note;

fn trimList(comptime T: type, g: Allocator, list: *std.ArrayList(T)) Allocator.Error!void {
    if (list.capacity == list.items.len) return;
    const exact = try g.alloc(T, list.items.len);
    @memcpy(exact, list.items);
    g.free(list.allocatedSlice());
    list.* = .fromOwnedSlice(exact);
}

/// Reads a snapshot (zengine/notes_export.py's JSON): its settings, vocabulary, notes, pairs and phrases.
pub fn parse(gpa: Allocator, text: []const u8) LoadError!*Model {
    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const s = switch (try json.parse(parse_arena.allocator(), text)) {
        .ok => |v| v,
        .err => return error.BadSnapshot,
    };
    if (!std.mem.eql(u8, try str(s, "format"), "lw-notes-snapshot/1")) return error.BadSnapshot;
    const vocab = try need(s, "vocab");
    if (vocab.get("dense")) |d| if (d.truthy()) return error.Unsupported; // a stand-in tokenizer's own numbering
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa), .settings = undefined, .takeback = .plain, .unknown_rule = false, .head = "", .cur_title = "", .hist_title = "", .fixed = undefined, .tokenizer_sha256 = "", .vkey = &.{}, .vwordy = &.{} };
    errdefer deinitLists(m);
    const a = m.arena.allocator();
    try readSettings(m, try need(s, "settings"));
    m.head = try a.dupe(u8, try str(s, "head"));
    // Python already joined UNKNOWN_RULE into head; a true flag without that sentence is a broken snap
    if (m.unknown_rule and std.mem.indexOf(u8, m.head, line_mod.unknown_rule) == null) return error.BadSnapshot;
    m.cur_title = try a.dupe(u8, try str(s, "cur_title"));
    m.hist_title = try a.dupe(u8, try str(s, "hist_title"));
    const fixed = try need(s, "fixed");
    m.fixed = .{ .head = try uint(fixed, "head"), .current = try uint(fixed, "current"), .history = try uint(fixed, "history"), .nl = try uint(fixed, "nl") };
    const tok = s.get("tokenizer");
    m.tokenizer_sha256 = try a.dupe(u8, if (tok) |t| t.strField("sha256") orelse "" else "");
    const size = try uint(vocab, "size");
    const vk = try arr(vocab, "key");
    const vw_text = try str(vocab, "wordy");
    if (vk.len != size or vw_text.len != size) return error.BadSnapshot;
    const vkey = try a.alloc(u32, size);
    for (vk, vkey) |k, *slot| slot.* = try asU32(k);
    m.vkey = vkey;
    const vw = try a.alloc(bool, size);
    for (vw_text, vw) |c, *slot| slot.* = c == '1';
    m.vwordy = vw;
    for (try arr(s, "entities")) |e| try appendName(gpa, &m.entities, try str(e, "name"));
    for (try arr(s, "attributes")) |x| try appendName(gpa, &m.attributes, x.str() orelse return error.BadSnapshot);
    try readNotes(m, try arr(s, "notes"));
    try readPairs(m, try arr(s, "pairs"));
    try readPhrases(m, try arr(s, "phrases"));
    const c = m.counts();
    const n = try need(s, "n");
    if (c[0] != try uint(n, "current") or c[1] != try uint(n, "history")) return error.BadSnapshot;
    const grams = try need(s, "grams");
    m.grams_current = try uint(grams, "current");
    m.grams_history = try uint(grams, "history");
    m.base_notes = m.notes.items.len;
    try m.trim(); // the entity and attribute names grew one at a time
    try m.indexAll(); // after trim: the maps' keys are the names themselves, which trim never moves
    return m;
}

/// The model's growable lists freed, for a parse that failed half way.
fn deinitLists(m: *Model) void {
    const g = m.gpa;
    for (m.notes.items) |*x| Model.freeNote(g, x);
    m.notes.deinit(g);
    for (m.entities.items) |e| g.free(e);
    m.entities.deinit(g);
    for (m.attributes.items) |x| g.free(x);
    m.attributes.deinit(g);
    for (m.pairs.items) |*p| p.ids.deinit(g);
    m.pairs.deinit(g);
    for (m.phrases.items) |p| g.free(p.keys);
    m.phrases.deinit(g);
    m.ix.deinit(g);
    m.arena.deinit();
}

fn appendName(g: Allocator, list: *std.ArrayList([]u8), name: []const u8) LoadError!void {
    try list.ensureUnusedCapacity(g, 1);
    list.appendAssumeCapacity(try g.dupe(u8, name));
}

fn readSettings(m: *Model, js: Value) LoadError!void {
    const n_min = try uint(js, "n_min");
    const n_max = try uint(js, "n_max");
    if (n_min < 1 or n_min > n_max) return error.BadSnapshot;
    if (n_max > index.max_n) return error.Unsupported;
    const hop = try str(js, "hop");
    m.settings = .{
        .cap = try uint(js, "cap"),
        .n_min = n_min,
        .n_max = n_max,
        .fold = try boolean(js, "fold"),
        .hop = if (std.mem.eql(u8, hop, "subject")) .subject else if (std.mem.eql(u8, hop, "subject+value"))
            .subject_value
        else if (std.mem.eql(u8, hop, "none")) .none else return error.BadSnapshot,
        .hop_weight = try num(js, "hop_weight"),
        .hop_from = try uint(js, "hop_from"),
        .hop_history = try boolean(js, "hop_history"),
        .min_rel = try num(js, "min_rel"),
        .subject = try num(js, "subject"),
        .punct = try boolean(js, "punct"),
    };
    // a snapshot from before the take-back setting reads as "plain", the wording it always had
    const tb = js.strField("takeback") orelse "plain";
    m.takeback = if (std.mem.eql(u8, tb, "plain")) .plain else if (std.mem.eql(u8, tb, "explicit")) .explicit else return error.BadSnapshot;
    // a snapshot from before the unknown-rule setting reads as false; Python already joined UNKNOWN_RULE into head
    m.unknown_rule = false;
    if (js.get("unknown_rule")) |u| {
        if (u == .null) {} else if (u == .bool) m.unknown_rule = u.bool else return error.BadSnapshot;
    }
}

/// The notes, by ascending id; each line must be one line() gives for its state and kind.
fn readNotes(m: *Model, list: []const Value) LoadError!void {
    const g = m.gpa;
    try m.notes.ensureTotalCapacityPrecise(g, list.len);
    for (list) |x| {
        const id = try uint(x, "id");
        if (m.notes.items.len > 0 and id <= m.notes.items[m.notes.items.len - 1].id) return error.BadSnapshot;
        const state = std.enums.fromInt(index.State, try uint(x, "state")) orelse return error.BadSnapshot;
        const kind = std.enums.fromInt(Kind, try uint(x, "kind")) orelse return error.BadSnapshot;
        const entity = try uint(x, "entity");
        const attribute = try uint(x, "attribute");
        const pair = try uint(x, "pair");
        const value_ent = try int(x, "value_ent");
        const n_ent: i64 = @intCast(m.entities.items.len);
        if (entity >= m.entities.items.len or attribute >= m.attributes.items.len or value_ent < -1 or value_ent >= n_ent) return error.BadSnapshot;
        const line = try str(x, "line");
        const read = line_mod.parse(g, id, line, m.takeback) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.BadLine => error.BadSnapshot,
        };
        defer g.free(read.text);
        var note: Note = .{ .id = id, .state = state, .kind = kind, .entity = entity, .attribute = attribute, .pair = pair, .value_ent = @intCast(value_ent), .told = try int(x, "told"), .cost = 0, .version = 0, .by = read.by, .multi = read.multi };
        const agrees = switch (read.kind) {
            .none => state == .current and kind == .fact,
            .takeback => state == .current and kind == .tombstone,
            .superseded => state == .superseded,
            .withdrawn => state == .tombstoned,
        };
        if (!agrees) return error.BadSnapshot;
        note.cost = try uint(x, "cost");
        const toks = try arr(x, "tokens");
        try note.makeBody(g, toks.len, read.text, null);
        errdefer g.free(note.bodyBlock());
        for (toks, note.tokensMut()) |v, *slot| slot.* = try asU32(v);
        note.setLine(try g.dupe(u8, line));
        m.told_last = @max(m.told_last, note.told);
        m.notes.appendAssumeCapacity(note);
    }
}

/// by_key: each pair's notes oldest first; a note's version is its place in its pair, from 1.
fn readPairs(m: *Model, list: []const Value) LoadError!void {
    const g = m.gpa;
    try m.pairs.ensureTotalCapacityPrecise(g, list.len);
    const seen = try g.alloc(bool, m.notes.items.len);
    defer g.free(seen);
    @memset(seen, false);
    for (list, 0..) |p, k| {
        const entity = try uint(p, "entity");
        const attribute = try uint(p, "attribute");
        if (entity >= m.entities.items.len or attribute >= m.attributes.items.len) return error.BadSnapshot;
        m.pairs.appendAssumeCapacity(.{ .entity = entity, .attribute = attribute });
        const pr = &m.pairs.items[m.pairs.items.len - 1];
        const ids = try arr(p, "ids");
        try pr.ids.ensureTotalCapacityPrecise(g, ids.len);
        for (ids, 1..) |v, version| {
            const place = m.placeOf(try asU32(v)) orelse return error.BadSnapshot;
            const n = &m.notes.items[place];
            if (seen[place] or n.pair != k or n.entity != entity or n.attribute != attribute) return error.BadSnapshot;
            seen[place] = true;
            n.version = @intCast(version);
            pr.ids.appendAssumeCapacity(n.id);
        }
    }
    for (seen) |s| if (!s) return error.BadSnapshot;
}

fn readPhrases(m: *Model, list: []const Value) LoadError!void {
    const g = m.gpa;
    try m.phrases.ensureTotalCapacityPrecise(g, list.len);
    for (list) |p| {
        const entity = try uint(p, "entity");
        if (entity >= m.entities.items.len) return error.BadSnapshot;
        const ks = try arr(p, "keys");
        const keys = try g.alloc(u32, ks.len);
        errdefer g.free(keys);
        for (ks, keys) |k, *slot| slot.* = try asU32(k);
        m.phrases.appendAssumeCapacity(.{ .keys = keys, .entity = entity });
    }
}

fn need(v: Value, key: []const u8) LoadError!Value {
    return v.get(key) orelse error.BadSnapshot;
}

fn int(v: Value, key: []const u8) LoadError!i64 {
    return (try need(v, key)).int64() orelse error.BadSnapshot;
}

fn asU32(v: Value) LoadError!u32 {
    const x = v.int64() orelse return error.BadSnapshot;
    return std.math.cast(u32, x) orelse error.BadSnapshot;
}

fn uint(v: Value, key: []const u8) LoadError!u32 {
    return asU32(try need(v, key));
}

fn num(v: Value, key: []const u8) LoadError!f64 {
    const x = try need(v, key);
    if (x == .bool) return error.BadSnapshot;
    return x.float64() orelse error.BadSnapshot;
}

fn boolean(v: Value, key: []const u8) LoadError!bool {
    const x = try need(v, key);
    return if (x == .bool) x.bool else error.BadSnapshot;
}

fn str(v: Value, key: []const u8) LoadError![]const u8 {
    return (try need(v, key)).str() orelse error.BadSnapshot;
}

fn arr(v: Value, key: []const u8) LoadError![]Value {
    return (try need(v, key)).items() orelse error.BadSnapshot;
}

test "partial note cleanup before body and before line ownership" {
    const g = std.testing.allocator;
    var n: Note = .{ .id = 1, .entity = 0, .attribute = 0, .pair = 0, .value_ent = -1, .cost = 0, .version = 1, .told = 0, .state = .current, .kind = .fact };
    var failing = std.testing.FailingAllocator.init(g, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, n.makeBody(failing.allocator(), 1, "body", "source"));
    Model.freeNote(g, &n); // neither pointer has been initialized
    try n.makeBody(g, 1, "body", "source");
    Model.freeNote(g, &n); // body owned, line not allocated yet
}
