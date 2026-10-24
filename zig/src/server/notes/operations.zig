//! Storage-only note operations; callers serialize access and use a request arena for returned views.
const std = @import("std");
const Store = @import("storage.zig").Storage;
const model_mod = @import("model.zig");
const write = @import("write.zig");
const intake = @import("intake.zig");
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// Why a write was refused: the HTTP status and the message.
pub const Refusal = struct { status: u16 = 400, message: []const u8 = "" };

pub const Error = error{ Refused, OutOfMemory };

fn refuse(r: *Refusal, status: u16, message: []const u8) Error {
    r.* = .{ .status = status, .message = message };
    return error.Refused;
}

/// A note as a listing or a write's reply shows it.
pub const View = struct {
    id: u32,
    version: u32,
    state: @import("types.zig").State,
    kind: model_mod.Kind,
    subject: []const u8,
    text: []const u8,
    line: []const u8,
    source: ?[]const u8,
    told: i64,
    by: u32,
};

pub fn view(m: *const Model, a: Allocator, place: u32) Allocator.Error!View {
    const n = m.notes.items[place];
    return .{ .id = n.id, .version = n.version, .state = n.state, .kind = n.kind, .subject = try a.dupe(u8, m.entities.items[n.entity]), .text = try a.dupe(u8, n.text()), .line = try a.dupe(u8, n.line()), .source = if (n.source()) |s| try a.dupe(u8, s) else null, .told = n.told, .by = n.by };
}

/// What a write did: its verdict, its source, the note it wrote (or is the same as), and the notes it retired.
pub const Written = struct {
    verdict: intake.Verdict,
    source: []const u8,
    told: i64,
    note: ?View,
    retired: []const u32,
    same_as: []const u32,
    ver: u64 = 0, // the index version it swapped in (0: none, as for a write that learns nothing)
};

/// What a write asks: a new note, a change of note `target`, or a take-back of it.
pub const Ask = struct {
    op: write.Op,
    text: []const u8,
    subject: ?[]const u8 = null,
    target: u32 = 0,
};

/// Build the complete reply before committing the journal; no fallible allocation follows commit.
pub fn perform(s: *Store, a: Allocator, ask: Ask, told: i64, why: *Refusal) Error!Written {
    var p = try prepare(s, a, ask, told, why);
    defer p.deinit();
    try p.commit(a, why);
    return p.reply;
}

/// Keep this prepared operation alive and locked while serializing the HTTP body, then commit it.
pub fn prepare(s: *Store, a: Allocator, ask: Ask, told: i64, why: *Refusal) Error!Prepared {
    const m = s.model;
    const source = try std.fmt.allocPrint(a, "w{d}", .{s.log.next_source});
    var r: write.Record = .{ .op = ask.op, .source = source, .told = told, .subject = ask.subject orelse "me", .attribute = "", .text = ask.text };
    if (ask.op == .add) {
        r.attribute = try std.fmt.allocPrint(a, "note:{s}", .{source});
    } else {
        const p = m.placeOf(ask.target) orelse return refuse(why, 404, "no note has this id");
        const t = m.notes.items[p];
        if (t.state != .current) return refuse(why, 409, if (t.state == .superseded) "this note is no longer current (a later note replaced it): change or take back the current one" else "this note was taken back already");
        if (ask.op == .take_back and t.kind == .tombstone) return refuse(why, 409, "this note is a take-back; a take-back cannot be taken back (change it to say something new)");
        if (ask.op == .take_back and p < m.base_notes and m.takeback == .explicit)
            return refuse(why, 409, "under the snapshot's \"explicit\" take-back wording a take-back's label depends on whether the attribute holds several values, which the snapshot does not record");
        if (ask.subject) |sub| if (!std.mem.eql(u8, sub, m.entities.items[t.entity])) return refuse(why, 400, "a change keeps the subject of the note it changes");
        r.subject = m.entities.items[t.entity];
        r.attribute = m.attributes.items[t.attribute];
        r.target = if (p < m.base_notes) .{ .note = t.id } else .{ .source = t.source().? };
    }

    var pending = s.prepare(a, r) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Tokenizer => refuse(why, 400, "the tokenizer cannot encode this note"),
        error.ReadOnly => refuse(why, 503, "the store has no journal"),
        error.MissingRun => refuse(why, 503, "the process writer flags have not been recorded"),
        error.JournalPoisoned => refuse(why, 503, "the notes journal needs recovery; restart the server"),
        error.Unsupported => refuse(why, 500, "the notes index cannot be built from this vocabulary"),
        error.BadSnapshot => refuse(why, 500, "the notes index cannot be built from this store"),
    };
    errdefer pending.deinit();
    const out = pending.out;
    const shown: ?u32 = if (out.note != 0) out.note else if (out.same_as.len > 0) out.same_as[0] else null;
    const reply: Written = .{
        .verdict = out.verdict,
        .source = source,
        .told = told,
        .note = if (shown) |id| try view(m, a, m.placeOf(id).?) else null,
        .retired = out.retired,
        .same_as = out.same_as,
    };
    return .{ .pending = pending, .reply = reply };
}

pub const Prepared = struct {
    pending: @import("storage.zig").Pending,
    reply: Written,

    pub fn deinit(p: *Prepared) void {
        p.pending.deinit();
    }

    pub fn commit(p: *Prepared, a: Allocator, why: *Refusal) Error!void {
        const s = p.pending.storage;
        p.pending.commit(a) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.JournalIo => refuse(why, 500, "the write could not be stored on disk; nothing changed"),
            error.JournalPoisoned => refuse(why, 500, if (s.journal.?.doubt)
                "nothing applied in memory; disk state unknown, restart may apply the write; writes stopped"
            else
                "nothing applied in memory; incomplete journal write drops on restart; writes stopped"),
            error.AlreadyCommitted => unreachable,
        };
    }
};

/// Caller holds the store's writer lock; returned views are copied into the request arena.
pub fn list(s: *Store, a: Allocator, history: bool) Allocator.Error![]View {
    var out: std.ArrayList(View) = .empty;
    for (s.model.notes.items, 0..) |n, k| if (history or n.state == .current)
        try out.append(a, try view(s.model, a, @intCast(k)));
    return out.items;
}
