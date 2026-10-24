//! Storage erase prepares a replay without the target closure, then rewrites disk before swapping ownership.
const std = @import("std");
const Storage = @import("storage.zig").Storage;
const model = @import("model.zig");
const places = @import("../notes_erase_places.zig");
const intent_mod = @import("erase_intent.zig");
const write = @import("write.zig");
const records = @import("../notes_records.zig");
const index = @import("index.zig");
const alias = @import("alias.zig");
const Allocator = std.mem.Allocator;

/// Caller serializes the entire preparation/commit lifetime, including response serialization.
pub const Prepared = struct {
    storage: *Storage,
    fresh: *model.Model,
    log: records.Log,
    kept: []const write.Record,
    removed: []const []const u8,
    moved: []const [2]u32,
    texts: []const []const u8,
    source_num: u64,
    intent: intent_mod.Intent,
    retry: bool = false,
    idx: ?*index.Index = null,
    committed: bool = false,
    unsynced: bool = false,
    retire_failed: bool = false,

    pub fn deinit(p: *Prepared) void {
        if (!p.committed and !p.retry) {
            if (p.idx) |idx| idx.deinit();
            p.log.deinit();
            p.fresh.deinit();
        }
    }

    /// A renamed-but-unsynced rewrite has changed disk, so memory must change too.
    pub fn commit(p: *Prepared, a: Allocator) !void {
        if (p.committed) return error.AlreadyCommitted;
        const s = p.storage;
        if (p.retry) {
            p.committed = true;
            return;
        }
        const j = &s.journal.?;
        const owned = try p.intent.dupe(j.gpa);
        errdefer owned.deinit(j.gpa);
        try j.erases.append(j.gpa, owned);
        errdefer _ = j.erases.pop();
        s.journal.?.rewrite(a, p.kept, p.log.next_source) catch |e| switch (e) {
            error.RenamedUnsynced => p.unsynced = true,
            else => return e,
        };
        const old = s.model;
        var old_log = s.log;
        const old_idx = s.idx;
        s.model = p.fresh;
        s.log = p.log;
        s.idx = p.idx;
        p.committed = true;
        old_log.deinit();
        old.deinit();
        if (old_idx) |x| x.deinit();
    }
    /// Retire the token spans after engine-reachable cleanup; external-copy warnings do not retain the note.
    pub fn finish(p: *Prepared, a: Allocator) !void {
        const j = &p.storage.journal.?;
        if (j.poisoned) return error.JournalPoisoned;
        for (j.erases.items, 0..) |i, k| if (i.id == p.intent.id) {
            const removed = j.erases.orderedRemove(k);
            j.rewrite(a, p.kept, p.log.next_source) catch |e| {
                j.erases.insertAssumeCapacity(k, removed);
                return e;
            };
            removed.deinit(j.gpa);
            return;
        };
    }
};

pub fn prepare(s: *Storage, a: Allocator, id: u32) !Prepared {
    const j = if (s.journal) |*v| v else return error.ReadOnly;
    if (j.poisoned) return error.JournalPoisoned;
    if (!s.started) return error.MissingRun;
    const old = s.model;
    for (j.erases.items) |i| if (i.id == id) {
        var kept: std.ArrayList(write.Record) = .empty;
        for (s.log.items.items) |e| try kept.append(a, e.r);
        return .{ .storage = s, .fresh = old, .log = s.log, .kept = kept.items, .removed = i.removed, .moved = &.{}, .texts = &.{}, .source_num = i.source_num, .intent = i, .retry = true };
    };
    const place = old.placeOf(id) orelse return error.NoNote;
    if (place < old.base_notes) return error.SnapshotNote;
    const source = old.notes.items[place].source().?;
    const source_num = records.sourceNumber(source) orelse 0;
    var gone: std.ArrayList(usize) = .empty;
    try records.erased(&s.log, .{ .source = source }, &gone, a);
    const g = s.zeroing.allocator();
    const fresh = try model.parse(g, try snapshotBytes(s, a));
    errdefer fresh.deinit();
    var log: records.Log = .{ .gpa = g, .next_source = s.log.next_source };
    errdefer log.deinit();
    var kept: std.ArrayList(write.Record) = .empty;
    var removed: std.ArrayList([]const u8) = .empty;
    for (s.log.items.items, 0..) |entry, k| {
        if (std.mem.indexOfScalar(usize, gone.items, k) != null) {
            try removed.append(a, try a.dupe(u8, entry.r.source));
            continue;
        }
        // These records borrow the old log until commit has finished rewriting the journal.
        var outcome: write.Outcome = undefined;
        var undo: write.Undo = .{};
        defer undo.deinit(g);
        try write.apply(fresh, s.text, a, entry.r, &outcome, &undo);
        write.commit(fresh, &undo);
        // Replaying without a take-back may make a later write redundant. Name that removal explicitly.
        if (outcome.note == 0 and old.placeOfSource(entry.r.source) != null) {
            try removed.append(a, try a.dupe(u8, entry.r.source));
            continue;
        }
        try log.add(entry.r, outcome, fresh);
        try kept.append(a, log.items.items[log.items.items.len - 1].r);
        fresh.id_floor = @max(fresh.nextId(), entry.r.id + 1);
    }
    fresh.id_floor = @max(fresh.nextId(), old.nextId());
    try fresh.trim();
    const idx = try index.Index.buildKind(g, fresh, s.kind);
    errdefer idx.deinit();
    try alias.populate(idx, fresh, s.text);
    var moved: std.ArrayList([2]u32) = .empty;
    for (old.notes.items[old.base_notes..]) |*note| if (fresh.placeOfSource(note.source().?)) |q| {
        if (fresh.notes.items[q].id != note.id) try moved.append(a, .{ note.id, fresh.notes.items[q].id });
    };
    var texts: std.ArrayList([]const u8) = .empty;
    for (old.notes.items[old.base_notes..]) |*note| for (removed.items) |src| {
        if (std.mem.eql(u8, note.source().?, src)) {
            try texts.append(a, try a.dupe(u8, note.text()));
            break;
        }
    };
    // An already-known write can have different raw words without creating a new note.
    for (gone.items) |k| {
        const raw = s.log.items.items[k].r.text;
        for (texts.items) |seen| {
            if (std.mem.eql(u8, seen, raw)) break;
        } else try texts.append(a, try a.dupe(u8, raw));
    }
    const spans = try places.discover(a, s.text, texts.items);
    return .{
        .intent = .{ .id = id, .source_num = source_num, .removed = removed.items, .spans = spans.items, .short = spans.short },
        .storage = s,
        .fresh = fresh,
        .log = log,
        .kept = kept.items,
        .removed = removed.items,
        .moved = moved.items,
        .texts = texts.items,
        .source_num = source_num,
        .idx = idx,
    };
}

/// File-backed erase refuses changed snapshots; fixture-backed stores retain their original bytes.
fn snapshotBytes(s: *Storage, a: Allocator) ![]const u8 {
    const path = s.snapshot_path orelse return s.snapshot;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(s.io.?, path, a, .unlimited);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(sha, .lower), &s.snapshot_sha)) return error.SnapshotChanged;
    return bytes;
}
