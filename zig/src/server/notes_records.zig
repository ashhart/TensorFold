//! The writes since the snapshot, as the journal holds them, with what each did (NoteStore's O.log).
const std = @import("std");
const model_mod = @import("notes/model.zig");
const write = @import("notes/write.zig");
const intake = @import("notes/intake.zig");
const Model = model_mod.Model;
const Allocator = std.mem.Allocator;

/// One write and its outcome; `same_as` names notes as targets, which an erase's renumbering does not move.
pub const Entry = struct {
    r: write.Record,
    verdict: intake.Verdict,
    same_as: []write.Target,
};

pub const Log = struct {
    gpa: Allocator,
    items: std.ArrayList(Entry) = .empty,
    next_source: u64 = 1, // "w1", "w2", ...: a source is never used twice, an erased one included

    pub fn deinit(l: *Log) void {
        for (l.items.items) |*e| freeEntry(l.gpa, e);
        l.items.deinit(l.gpa);
    }

    /// An entry with its own copies of the record's strings, ready for push() (which cannot fail).
    pub fn prepare(l: *Log, r: write.Record, out: write.Outcome, m: *const Model) Allocator.Error!Entry {
        const g = l.gpa;
        var e: Entry = .{ .r = r, .verdict = out.verdict, .same_as = &.{} };
        e.r.source = try g.dupe(u8, r.source);
        errdefer g.free(e.r.source);
        e.r.subject = try g.dupe(u8, r.subject);
        errdefer g.free(e.r.subject);
        e.r.attribute = try g.dupe(u8, r.attribute);
        errdefer g.free(e.r.attribute);
        e.r.text = try g.dupe(u8, r.text);
        errdefer g.free(e.r.text);
        e.r.target = switch (r.target) {
            .none, .note => r.target,
            .source => |s| .{ .source = try g.dupe(u8, s) },
        };
        errdefer freeTarget(g, e.r.target);
        const same = try g.alloc(write.Target, out.same_as.len);
        var made: usize = 0;
        errdefer {
            for (same[0..made]) |t| freeTarget(g, t);
            g.free(same);
        }
        for (out.same_as) |id| {
            same[made] = try targetOf(g, m, id);
            made += 1;
        }
        e.same_as = same;
        try l.items.ensureUnusedCapacity(g, 1);
        return e;
    }

    pub fn push(l: *Log, e: Entry) void {
        l.items.appendAssumeCapacity(e);
        if (sourceNumber(e.r.source)) |n| l.next_source = @max(l.next_source, n + 1);
    }

    /// prepare() and push() at once (a replay).
    pub fn add(l: *Log, r: write.Record, out: write.Outcome, m: *const Model) Allocator.Error!void {
        l.push(try l.prepare(r, out, m));
    }
};

/// A note as a target: a snapshot's by its id, a written one by its write's source.
pub fn targetOf(g: Allocator, m: *const Model, id: u32) Allocator.Error!write.Target {
    const p = m.placeOf(id) orelse return .{ .note = id };
    if (p < m.base_notes) return .{ .note = id };
    return .{ .source = try g.dupe(u8, m.notes.items[p].source().?) };
}

pub fn sameTarget(x: write.Target, y: write.Target) bool {
    return switch (x) {
        .none => y == .none,
        .note => |n| y == .note and y.note == n,
        .source => |s| y == .source and std.mem.eql(u8, y.source, s),
    };
}

pub fn freeTarget(g: Allocator, t: write.Target) void {
    if (t == .source) g.free(t.source);
}

pub fn freeEntry(g: Allocator, e: *Entry) void {
    g.free(e.r.source);
    g.free(e.r.subject);
    g.free(e.r.attribute);
    g.free(e.r.text);
    freeTarget(g, e.r.target);
    for (e.same_as) |t| freeTarget(g, t);
    g.free(e.same_as);
}

/// The n of "wN", or null.
pub fn sourceNumber(s: []const u8) ?u64 {
    if (s.len < 2 or s[0] != 'w') return null;
    return std.fmt.parseInt(u64, s[1..], 10) catch null;
}

/// The writes erasing note `x` takes out: its own, those kept out as the same as it, and its take-backs (now void).
pub fn erased(l: *const Log, x: write.Target, out: *std.ArrayList(usize), a: Allocator) Allocator.Error!void {
    for (l.items.items, 0..) |e, k| {
        const own = x == .source and std.mem.eql(u8, e.r.source, x.source);
        var same = false;
        if (e.verdict == .ALREADY_KNOWN) for (e.same_as) |t| {
            same = same or sameTarget(t, x);
        };
        const takes = e.r.op == .take_back and sameTarget(e.r.target, x);
        if (own or same or takes) try out.append(a, k);
    }
}
