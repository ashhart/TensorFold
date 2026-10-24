//! One write to the notes as LLWZ's NoteStore._apply makes it, with the deterministic part of the intake check.
const std = @import("std");
const model_mod = @import("model.zig");
const line_mod = @import("line.zig");
const intake = @import("intake.zig");
const model_ix = @import("model_ix.zig");
const model_text = @import("../model_text.zig");
const Model = model_mod.Model;
const Note = model_mod.Note;
const Verdict = intake.Verdict;
const Allocator = std.mem.Allocator;

pub const Op = enum { add, change, take_back };

/// The note a change or take-back names: a snapshot's note by its id, or a written one by its write's source.
pub const Target = union(enum) { none, note: u32, source: []const u8 };

/// One write as the journal keeps it: what replay needs to make it again, nothing derived.
pub const Record = struct {
    op: Op,
    id: u32 = 0, // persisted identity; zero only for legacy records and direct model callers
    source: []const u8, // the write's own name ("w1"), which the note keeps as its source
    told: i64,
    subject: []const u8, // the entity; a change or take-back keeps its target's
    attribute: []const u8, // "note:SOURCE" for a new note's own pair; a change or take-back keeps its target's
    target: Target = .none,
    text: []const u8,
};

/// What a write did, as the reply reports it: ids are note ids.
pub const Outcome = struct {
    verdict: Verdict,
    note: u32 = 0, // the note written (0: none)
    retired: []const u32 = &.{}, // the notes it superseded or withdrew
    same_as: []const u32 = &.{}, // ALREADY_KNOWN: the current notes that say the same
};

pub const Error = error{ OutOfMemory, Tokenizer };

/// What apply changed, so revert can put it back and commit can free what it replaced.
pub const Undo = struct {
    told_last: i64 = 0,
    notes: usize = 0,
    entities: usize = 0,
    attributes: usize = 0,
    pairs: usize = 0,
    phrases: usize = 0,
    pair: ?u32 = null, // an existing pair the note joined
    fact: ?u64 = null, // the new note's model_ix fact key, when it is a fact (room reserved)
    gone: std.ArrayList(Saved) = .empty,

    const Saved = struct { place: u32, state: @import("types.zig").State, by: u32, line: []u8, cost: u32, fact: ?u64 };

    pub fn deinit(u: *Undo, g: Allocator) void {
        u.gone.deinit(g);
    }
};

/// The take-back text when the client sends none: a note keeps a sentence, and this one names nothing taken back.
pub const default_takeback = "I take that back.";

/// Whether the pair currently holds more than one fact: the set wording of an explicit take-back.
fn pairIsMulti(m: *const Model, pair: ?u32) bool {
    const p = pair orelse return false;
    var n_cur: u32 = 0;
    for (m.pairs.items[p].ids.items) |id| {
        const n = m.notes.items[m.placeOf(id) orelse continue];
        if (n.state == .current and n.kind == .fact) n_cur += 1;
    }
    return n_cur > 1;
}

fn sameText(a: Allocator, x: []const u8, y_norm: []const u8) Allocator.Error!bool {
    const nx = try intake.norm(a, x);
    defer a.free(nx);
    return std.mem.eql(u8, nx, y_norm);
}

/// The current facts of `entity` (or, with a pair, of that pair) whose normal form is the new text's, by place.
fn duplicates(m: *const Model, a: Allocator, entity: ?u32, pair: ?u32, text: []const u8) Allocator.Error![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    const e = entity orelse return out.toOwnedSlice(a);
    const want = try intake.norm(a, text);
    defer a.free(want);
    var one: [1]u32 = undefined;
    for (m.ix.factsOf(model_ix.factKey(e, want), &one)) |at| { // the candidates of that normal form's hash, checked by text
        const n = m.notes.items[at];
        if (n.state != .current or n.kind != .fact or n.entity != e) continue;
        if (pair) |p| if (n.pair != p) continue;
        if (try sameText(a, n.text(), want)) try out.append(a, n.id);
    }
    return out.toOwnedSlice(a);
}

fn targetPlace(m: *const Model, t: Target) ?u32 {
    return switch (t) {
        .none => null,
        .note => |id| if (m.placeOf(id)) |p| (if (p < m.base_notes) p else null) else null,
        .source => |s| m.placeOfSource(s),
    };
}

/// The model's notes a write would retire, and its verdict (no note yet): the intake's part that needs no rule book.
fn sort(m: *const Model, a: Allocator, r: Record, pair: ?u32, gone: *std.ArrayList(u32)) Allocator.Error!Verdict {
    const t = targetPlace(m, r.target);
    switch (r.op) {
        .add => return .NEW,
        .change => {
            if (t) |p| if (m.notes.items[p].state == .current) {
                try gone.append(a, p);
                return if (m.notes.items[p].kind == .fact) .REPLACES else .NEW;
            };
            // the note it changed is gone (erased since): it changes what its pair holds now
            var fact = false;
            if (pair) |pr| {
                const from = gone.items.len;
                for (m.pairs.items[pr].ids.items) |id| { // the pair's notes, not every note
                    const k = m.placeOf(id).?;
                    const n = m.notes.items[k];
                    if (n.state != .current) continue;
                    try gone.append(a, k);
                    fact = fact or n.kind == .fact;
                }
                std.sort.pdq(u32, gone.items[from..], {}, std.sort.asc(u32)); // by place, as a scan of the notes
            }
            return if (fact) .REPLACES else .NEW;
        },
        .take_back => {
            const p = t orelse return .IMPLIED; // what it took back was never taught, so nothing changes
            const n = m.notes.items[p];
            if (n.state != .current or n.kind != .fact) return .IMPLIED;
            try gone.append(a, p);
            return .REPLACES;
        },
    }
}

/// NoteStore._apply for one record. On an error the model is as it was; on success revert() or commit() follows.
pub fn apply(m: *Model, text: model_text.Text, a: Allocator, r: Record, out: *Outcome, undo: *Undo) Error!void {
    const g = m.gpa;
    undo.* = .{ .told_last = m.told_last, .notes = m.notes.items.len, .entities = m.entities.items.len, .attributes = m.attributes.items.len, .pairs = m.pairs.items.len, .phrases = m.phrases.items.len };
    const entity = m.entityOf(r.subject);
    const attribute = m.attributeOf(r.attribute);
    const pair: ?u32 = if (entity != null and attribute != null) m.pairOf(entity.?, attribute.?) else null;
    const same: []const u32 = if (r.op == .take_back) &.{} else try duplicates(m, a, entity, if (r.op == .change) pair orelse std.math.maxInt(u32) else null, r.text);
    if (same.len > 0) {
        m.told_last = @max(m.told_last, r.told);
        out.* = .{ .verdict = .ALREADY_KNOWN, .same_as = same };
        return;
    }
    var gone: std.ArrayList(u32) = .empty;
    const verdict = try sort(m, a, r, pair, &gone);
    if (!verdict.learn()) {
        m.told_last = @max(m.told_last, r.told);
        out.* = .{ .verdict = verdict };
        return;
    }
    const id = if (r.id == 0) m.nextId() else r.id;
    if (id < m.nextId() or id == std.math.maxInt(u32)) return error.Tokenizer;
    const kind: model_mod.Kind = if (r.op == .take_back) .tombstone else .fact;
    const ids = try encode(g, text, r.text);
    defer g.free(ids);
    var note: Note = .{ .id = id, .state = .current, .kind = kind, .entity = entity orelse @intCast(m.entities.items.len), .attribute = attribute orelse @intCast(m.attributes.items.len), .pair = pair orelse @intCast(m.pairs.items.len), .value_ent = -1, .told = r.told, .cost = 0, .version = 1 + if (pair) |p| @as(u32, @intCast(m.pairs.items[p].ids.items.len)) else 0 };
    errdefer Model.freeNote(g, &note);
    try note.makeBody(g, ids.len, r.text, r.source);
    @memcpy(note.tokensMut(), ids);
    // several current facts on this pair: the set wording under takeback "explicit" (notes.py 13b, schema.multi)
    note.multi = kind == .tombstone and pairIsMulti(m, pair);
    note.setLine(try line_mod.render(g, id, r.text, if (kind == .tombstone) .{ .takeback = note.multi } else .none, m.takeback));
    note.cost = try cost(g, text, note.line());
    // the retired notes' new lines, made before anything changes
    try undo.gone.ensureTotalCapacity(g, gone.items.len);
    const lines = try a.alloc([]u8, gone.items.len);
    const costs = try a.alloc(u32, gone.items.len);
    var made: usize = 0;
    errdefer for (lines[0..made]) |l| g.free(l);
    for (gone.items, 0..) |p, k| {
        const old = m.notes.items[p];
        const ret: line_mod.Retired = .{ .by = id, .told = r.told };
        lines[k] = try line_mod.render(g, old.id, old.text(), if (kind == .tombstone) .{ .withdrawn = ret } else .{ .superseded = ret }, m.takeback);
        made += 1;
        costs[k] = try cost(g, text, lines[k]);
    }
    const new_entity = entity == null;
    var keys: []u32 = &.{};
    errdefer g.free(keys);
    if (new_entity and !std.mem.eql(u8, r.subject, "me")) keys = try phraseKeys(m, g, text, r.subject);
    var ename: []u8 = &.{};
    errdefer g.free(ename);
    if (new_entity) ename = try g.dupe(u8, r.subject);
    var aname: []u8 = &.{};
    errdefer g.free(aname);
    if (attribute == null) aname = try g.dupe(u8, r.attribute);
    var new_pair: model_mod.Pair = .{ .entity = note.entity, .attribute = note.attribute };
    errdefer new_pair.ids.deinit(g);
    if (pair) |p| try m.pairs.items[p].ids.ensureUnusedCapacity(g, 1) else try new_pair.ids.ensureTotalCapacity(g, 1);
    try m.notes.ensureUnusedCapacity(g, 1);
    try m.entities.ensureUnusedCapacity(g, 1);
    try m.attributes.ensureUnusedCapacity(g, 1);
    try m.pairs.ensureUnusedCapacity(g, 1);
    try m.phrases.ensureUnusedCapacity(g, 1);
    const retired = try a.alloc(u32, gone.items.len);
    // the model's lookups get room now, and their entries at commit (model_ix.zig)
    const facts = try a.alloc(?u64, gone.items.len);
    for (gone.items, facts) |p, *f| {
        const n = &m.notes.items[p];
        f.* = if (n.kind == .fact) try model_ix.factKeyOf(a, n.entity, n.text()) else null;
    }
    try m.ix.entity.ensureUnusedCapacity(g, 1);
    try m.ix.attribute.ensureUnusedCapacity(g, 1);
    try m.ix.pair.ensureUnusedCapacity(g, 1);
    try m.ix.source.ensureUnusedCapacity(g, 1);
    undo.fact = if (kind == .fact) try model_ix.factKeyOf(a, note.entity, r.text) else null;
    if (undo.fact) |k| try m.ix.reserveFact(g, k);
    // from here nothing allocates: the write happens whole
    if (new_entity) m.entities.appendAssumeCapacity(ename);
    if (new_entity and !std.mem.eql(u8, r.subject, "me")) m.phrases.appendAssumeCapacity(.{ .keys = keys, .entity = note.entity });
    if (attribute == null) m.attributes.appendAssumeCapacity(aname);
    if (pair) |p| {
        m.pairs.items[p].ids.appendAssumeCapacity(id);
        undo.pair = p;
    } else {
        new_pair.ids.appendAssumeCapacity(id);
        m.pairs.appendAssumeCapacity(new_pair);
    }
    for (gone.items, lines, costs, retired, facts) |p, l, c, *rid, f| {
        const n = &m.notes.items[p];
        undo.gone.appendAssumeCapacity(.{ .place = p, .state = n.state, .by = n.by, .line = n.line(), .cost = n.cost, .fact = f });
        n.state = if (kind == .tombstone) .tombstoned else .superseded;
        n.by = id;
        n.setLine(l);
        n.cost = c;
        rid.* = n.id;
    }
    m.notes.appendAssumeCapacity(note);
    m.told_last = @max(m.told_last, r.told);
    out.* = .{ .verdict = verdict, .note = id, .retired = retired };
}

/// Puts the model back as it was before apply().
pub fn revert(m: *Model, undo: *Undo) void {
    const g = m.gpa;
    var k = undo.gone.items.len;
    while (k > 0) {
        k -= 1;
        const s = undo.gone.items[k];
        const n = &m.notes.items[s.place];
        g.free(n.line());
        n.setLine(s.line);
        n.state = s.state;
        n.by = s.by;
        n.cost = s.cost;
    }
    undo.gone.clearRetainingCapacity();
    while (m.notes.items.len > undo.notes) {
        var n = m.notes.pop().?;
        Model.freeNote(g, &n);
    }
    if (undo.pair) |p| _ = m.pairs.items[p].ids.pop();
    while (m.pairs.items.len > undo.pairs) {
        var p = m.pairs.pop().?;
        p.ids.deinit(g);
    }
    while (m.phrases.items.len > undo.phrases) g.free(m.phrases.pop().?.keys);
    while (m.attributes.items.len > undo.attributes) g.free(m.attributes.pop().?);
    while (m.entities.items.len > undo.entities) g.free(m.entities.pop().?);
    m.told_last = undo.told_last;
    undo.pair = null;
    undo.fact = null;
}

/// The model's lookups told what apply() added and retired (their room reserved there: nothing allocates).
fn indexWrite(m: *Model, undo: *const Undo) void {
    const x = &m.ix;
    for (m.entities.items[undo.entities..], undo.entities..) |e, k| model_ix.Ix.putNameAssumeCapacity(&x.entity, e, @intCast(k));
    for (m.attributes.items[undo.attributes..], undo.attributes..) |e, k| model_ix.Ix.putNameAssumeCapacity(&x.attribute, e, @intCast(k));
    for (m.pairs.items[undo.pairs..], undo.pairs..) |p, k| {
        const r = x.pair.getOrPutAssumeCapacity(model_ix.Ix.pairKey(p.entity, p.attribute));
        if (!r.found_existing) r.value_ptr.* = @intCast(k);
    }
    for (undo.gone.items) |s| if (s.fact) |k| x.dropFact(k, s.place);
    if (m.notes.items.len > undo.notes) {
        const place: u32 = @intCast(undo.notes);
        const n = &m.notes.items[place];
        if (n.source()) |s| model_ix.Ix.putNameAssumeCapacity(&x.source, s, place);
        if (undo.fact) |k| x.addFactAssumeCapacity(k, place);
    }
}

/// Frees what apply() replaced: the retired notes' old lines (for a model no index reads yet).
pub fn commit(m: *Model, undo: *Undo) void {
    indexWrite(m, undo);
    for (undo.gone.items) |s| m.gpa.free(s.line);
    undo.gone.clearRetainingCapacity();
    undo.pair = null;
    undo.fact = null;
}

fn encode(g: Allocator, text: model_text.Text, s: []const u8) Error![]u32 {
    return text.encode(g, s, false) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => error.Tokenizer,
    };
}

/// NoteStore.cost: the tokens of a rendered line.
fn cost(g: Allocator, text: model_text.Text, line: []const u8) Error!u32 {
    const ids = try encode(g, text, line);
    defer g.free(ids);
    return @intCast(ids.len);
}

/// NoteStore.add's phrase: the index keys of " " + the entity's name.
fn phraseKeys(m: *const Model, g: Allocator, text: model_text.Text, name: []const u8) Error![]u32 {
    const spaced = try std.mem.concat(g, u8, &.{ " ", name });
    defer g.free(spaced);
    const ids = try encode(g, text, spaced);
    for (ids) |*t| t.* = m.keyOf(t.*);
    return ids;
}
