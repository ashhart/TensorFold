//! Where an erased note may still be outside the store and its journal, as the erase reply names them.
const std = @import("std");
const api = @import("engine_api");
const Server = @import("server.zig").Server;
const model_text = @import("model_text.zig");
const line = @import("notes/line.zig");
const json = @import("json");

/// A place outside the store and the journal that can hold a note's text or tokens.
pub const Place = enum {
    journal,
    prompt_cache,
    learn_files,
    responses,
    cortheon,
    compaction_memory,
    request_log,
    promotion,
    requests,

    /// What may still hold the note there, said to the client.
    pub fn why(p: Place) []const u8 {
        return switch (p) {
            .journal => "the rewritten journal's folder could not be synced; a power cut may restore the old journal containing the note",
            .prompt_cache => "the engine could not show this note's prompt states gone: an unsupported hook, fewer than four stable span tokens, or requests restoring states may prevent clearance",
            .learn_files => "this or an earlier run may have written learned prompt states; not every possible learned folder has been shown free of this note's tokens",
            .responses => "stored Responses (/v1/responses with store) keep their conversations in memory, and one may quote this note",
            .cortheon => "Cortheon keeps the hidden rounds of its replies in memory, and one may quote this note",
            .compaction_memory => "this or an earlier run may have written compaction memory files quoting this note; those files have not been searched",
            .request_log => "this or an earlier run may have logged a chat body quoting this note to an operator-selected path; those files have not been searched",
            .promotion => "this or an earlier Sliding Weights run may retain this fact in its graph or saved model weights; disabling --slide does not remove it",
            .requests => "the engine is running or queueing requests now, and one may carry this note in its prompt or its reply; they are not cancelled",
        };
    }
};

/// Clearance distinguishes disabled writers, empty stores, removed copies and unresolved places.
pub const Status = enum { may_hold, off, empty, forgotten };

/// Which start made the erased note: this process, or an earlier one whose flags the journal may still name.
pub const Origin = struct {
    this_process: bool = false,
    source_num: u64 = 0, // "wN"; 0: unknown (no place report tied to a live write)
};

/// What the engine's forget of the erased notes' tokens did (Engine.forgetPrompts, once per note).
pub const Forget = struct {
    holders: bool = false,
    asked: bool = false, // the engine was asked (the erase removed at least one note)
    ok: bool = false, // every call answered: none Unsupported, Closed or Busy
    short: bool = false, // a note too short to find by its tokens alone: a match cannot be shown
    out: api.prompt_cache.Forgotten = .{ .disk = true }, // summed; disk: every call searched the learned index

    /// Whether this forget shows the engine's kept prompt states hold none of the notes.
    pub fn cacheClear(f: Forget) bool {
        return f.asked and f.ok and !f.short and f.out.held == 0;
    }

    /// Whether it shows the learned states on disk hold none of them.
    pub fn diskClear(f: Forget) bool {
        return f.cacheClear() and f.out.disk and f.out.failed == 0;
    }
};

/// Find an interior token span stable across note labels and surrounding prompt text.
pub fn spanOf(a: std.mem.Allocator, text: model_text.Text, note: []const u8) !?[]const u32 {
    var q: std.ArrayList(u8) = .empty;
    try line.quote(a, &q, note);
    const p = q.items;
    var lo = std.mem.indexOfScalar(u8, p, ' ') orelse return null;
    var hi = std.mem.lastIndexOfScalar(u8, p, ' ') orelse return null;
    while (lo < hi) {
        const span = text.encode(a, p[lo..hi], false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return null,
        };
        if (span.len >= 4 and try insideEveryWrapper(a, text, note, span)) return span;
        lo = std.mem.indexOfScalarPos(u8, p, lo + 1, ' ') orelse return null;
        hi = std.mem.lastIndexOfScalar(u8, p[0..hi], ' ') orelse return null;
    }
    return null;
}

/// Interior raw text: teaching turns do not contain the JSON escapes used by rendered notes.
pub fn rawSpanOf(a: std.mem.Allocator, text: model_text.Text, note: []const u8) !?[]const u32 {
    const lo = std.mem.indexOfScalar(u8, note, ' ') orelse return null;
    const hi = std.mem.lastIndexOfScalar(u8, note, ' ') orelse return null;
    if (lo >= hi) return null;
    const span = try text.encode(a, note[lo..hi], false);
    if (span.len < 4) return null;
    const quoted = try json.quote(a, note, .{});
    const histories = [_][]const u8{
        "",
        "{\"role\":\"system\",\"content\":\"Be concise.\"},",
        "{\"role\":\"user\",\"content\":\"Hello\"},{\"role\":\"assistant\",\"content\":\"Hello!\"},",
    };
    for (histories) |history| {
        const messages = try std.fmt.allocPrint(a, "[{s}{{\"role\":\"user\",\"content\":{s}}}]", .{ history, quoted });
        var problem: []const u8 = "";
        const ids = try text.renderIds(a, (try json.parseText(a, messages)).ok, .{}, &problem);
        if (std.mem.indexOf(u32, ids, span) == null) return null;
    }
    return span;
}

/// Whether `span` is inside the tokens of `note`'s line in each way a block can render and surround it.
fn insideEveryWrapper(a: std.mem.Allocator, text: model_text.Text, note: []const u8, span: []const u32) !bool {
    const told = line.ordinalOf(2026, 10, 16) * line.tpd;
    const labels = [_]line.Label{ .none, .{ .takeback = false }, .{ .takeback = true }, .{ .superseded = .{ .by = 98765, .told = told } }, .{ .withdrawn = .{ .by = 7, .told = told } } };
    const ids = [_]u32{ 1, 12, 123456 };
    const before = [_][]const u8{ "", "\n", "Current notes:\n" };
    const after = [_][]const u8{ "", "\n", "\n[3] \"x y z\"", "\n\nWhat is it?" };
    for (labels) |label| for ([_]line.Takeback{ .plain, .explicit }) |tb| for (ids) |id| {
        const l = try line.render(a, id, note, label, tb);
        for (before) |b| for (after) |e| {
            const s = try std.mem.concat(a, u8, &.{ b, l, e });
            const ids_ = text.encode(a, s, false) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return false,
            };
            if (std.mem.indexOf(u32, ids_, span) == null) return false;
        };
    };
    return true;
}

/// Ask the engine about quoted-note and raw teaching-turn spans, retaining conservative failures.
pub fn forget(srv: *Server, a: std.mem.Allocator, notes: []const []const u8) Forget {
    var f: Forget = .{ .holders = srv.prompt_holders.load(.acquire) > 0 or srv.preparing.load(.acquire) > 0 };
    for (notes) |note| {
        const found = spanOf(a, srv.text, note) catch null;
        const span = found orelse blk: {
            f.short = true; // still forgotten where its whole quoted text is found; only not shown gone
            var q: std.ArrayList(u8) = .empty;
            line.quote(a, &q, note) catch break :blk &[_]u32{};
            break :blk srv.text.encode(a, q.items, false) catch &[_]u32{};
        };
        if (span.len == 0) {
            f.short = true;
            continue;
        }
        ask(srv, &f, span);
        const raw = (rawSpanOf(a, srv.text, note) catch null) orelse {
            f.short = true;
            continue;
        };
        if (!std.mem.eql(u32, raw, span)) ask(srv, &f, raw);
    }
    return f;
}

fn ask(srv: *Server, f: *Forget, span: []const u32) void {
    if (!f.asked) f.ok = true;
    f.asked = true;
    const got = srv.engine.forgetPrompts(span) catch {
        f.ok = false;
        return;
    };
    f.out.states += got.states;
    f.out.scrubbed += got.scrubbed;
    f.out.learned += got.learned;
    f.out.held += got.held;
    f.out.failed += got.failed;
    f.out.in_flight += got.in_flight;
    f.out.folders.merge(got.folders);
    f.out.conversations += got.conversations;
    f.out.conversation_disk = f.out.conversation_disk or got.conversation_disk;
    f.out.disk = f.out.disk and got.disk;
}

/// Each place's status, by the server's own flags and stores now and what the engine's forget showed.
pub fn check(srv: *Server, forgot: Forget, origin: Origin) std.EnumArray(Place, Status) {
    var out: std.EnumArray(Place, Status) = .initFill(.off);
    const held = forgot.holders or srv.prompt_holders.load(.acquire) > 0 or srv.preparing.load(.acquire) > 0;
    out.set(.prompt_cache, if (!srv.info.prompt_cache) .off else if (!held and forgot.cacheClear()) .forgotten else .may_hold);
    out.set(.learn_files, if (srv.config.learn) .may_hold else diskOff(srv, origin, .learn)); // on: current-root clearance cannot rule out earlier operator-selected roots
    out.set(.responses, if (srv.store.count(srv.io) > 0) .may_hold else .empty); // always on: empty is what shows it holds nothing
    out.set(.cortheon, if (srv.cortheon) |c| (if (c.memory.count() > 0) .may_hold else .empty) else .off);
    out.set(.compaction_memory, if (srv.config.compact_memory != null) .may_hold else diskOff(srv, origin, .compact));
    out.set(.request_log, if (srv.config.request_log != null) .may_hold else diskOff(srv, origin, .log));
    out.set(.promotion, if (pocketMade(srv)) .may_hold else diskOff(srv, origin, .slide));
    var engine: @import("engine_api").Status = .{};
    srv.engine.status(&engine, &.{});
    out.set(.requests, if (held or engine.running + engine.waiting > 0 or forgot.out.in_flight > 0) .may_hold else .empty);
    return out;
}

const Disk = enum { learn, compact, log, slide };

fn pocketMade(srv: *Server) bool {
    if (srv.slide.model) |m| if (m.pocket > 0) return true;
    return srv.slide.facts.items.len > 0;
}

/// off when this process wrote the note and the writer is off, or every recorded start covering the note had it off.
fn diskOff(srv: *Server, origin: Origin, which: Disk) Status {
    if (origin.this_process) return .off;
    if (origin.source_num == 0) return .may_hold;
    const j = if (srv.notes) |s| s.journal else null;
    const runs = if (j) |jr| jr.runs.items else &.{};
    var born = false;
    var any_on = false;
    for (runs) |run| {
        if (run.next_source <= origin.source_num) born = true;
        if (!born) continue;
        any_on = any_on or switch (which) {
            .learn => run.learn,
            .compact => run.compact_memory.len != 0,
            .log => run.request_log.len != 0,
            .slide => run.slide or run.pocket,
        };
    }
    if (!born) return .may_hold;
    return if (any_on) .may_hold else .off;
}
