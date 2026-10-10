//! Prompts in pieces on the lane core: a stream begun, its prompt rows run in pieces, then its first token.
const std = @import("std");
const be = @import("backend.zig");
const sm = @import("stream.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const open = @import("open.zig");
const Engine = @import("engine.zig").Engine;
const Stream = sm.Stream;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// A stream whose prompt runs in pieces between rounds: `piece`s, then `finishStream`; a failed begin holds nothing.
pub fn beginStream(e: *Engine, s: *Stream) !be.Backend.Begun {
    try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    const begin = e.backend.vtable.begin orelse return error.NoPieces;
    const b = try begin(e.backend.ptr, s);
    s.cached = @intCast(b.cached);
    return b;
}

/// Prompt rows [start, end) of a begun stream (`save`: the prompt's snapshot right after them).
pub fn piece(e: *Engine, s: *Stream, start: u64, end: u64, save: bool) !void {
    const run = e.backend.vtable.piece orelse return error.NoPieces;
    return run(e.backend.ptr, s, start, end, save);
}

/// A round's pieces of several streams in one backend call (Config.piece_runs, Backend.pieces); else one at a time.
pub fn pieces(e: *Engine, list: []const be.Backend.Piece) !void {
    if (e.cfg.piece_runs) if (e.backend.vtable.pieces) |run| return run(e.backend.ptr, list);
    for (list) |p| try piece(e, p.stream, p.start, p.end, p.save);
}

/// A round's finished prompts before their `finishStream` calls (Config.replay_runs, Backend.finals).
pub fn prepareFinals(e: *Engine, streams: []const *Stream) !void {
    if (!e.cfg.replay_runs or streams.len < 2) return;
    const run = e.backend.vtable.finals orelse return;
    return run(e.backend.ptr, streams);
}

/// A begun stream's prompt is in: first token and drafts; a failure is the host's to discard (Cancelled: released).
pub fn finishStream(e: *Engine, s: *Stream) !void {
    _ = e.arena.reset(.retain_capacity);
    // Config.join_tail: the last token waits for the next round (a forced or budget-cut first token keeps its row)
    if (e.cfg.join_tail and e.cfg.family_mtp and s.drafts and s.prompt_len > 0 and s.force.items.len == 0 and !s.cutsNext()) {
        if (e.backend.vtable.join) |join| {
            if (join(e.backend.ptr, s) catch |x| return released(e, s, x)) return joined(e, s);
            return open.openedWith(e, s, .head);
        }
    }
    const fin = e.backend.vtable.finish orelse return error.NoPieces;
    fin(e.backend.ptr, s) catch |x| return released(e, s, x);
    return open.openedWith(e, s, .head);
}

/// A round's finished prompts as `finishStream`, joined ones' first drafts in one call; returns how many joined.
pub fn finishStreams(e: *Engine, streams: []const *Stream, errs: []?anyerror) !usize {
    _ = e.arena.reset(.retain_capacity);
    const a = e.arena.allocator();
    var reqs: std.ArrayList(be.DraftRequest) = .empty;
    var joins: std.ArrayList(usize) = .empty;
    for (streams, errs, 0..) |s, *err, i| {
        err.* = null;
        const can = e.cfg.join_tail and e.cfg.family_mtp and s.drafts and s.prompt_len > 0 and s.force.items.len == 0 and !s.cutsNext();
        const join = if (can) e.backend.vtable.join else null;
        if (join) |j| {
            const ok = j(e.backend.ptr, s) catch |x| {
                err.* = released(e, s, x);
                continue;
            };
            if (ok) {
                try reqs.append(a, joinRequest(e, s) catch |x| {
                    err.* = x;
                    continue;
                });
                try joins.append(a, i);
                continue;
            }
        } else {
            const fin = e.backend.vtable.finish orelse return error.NoPieces;
            fin(e.backend.ptr, s) catch |x| {
                err.* = released(e, s, x);
                continue;
            };
        }
        open.openedWith(e, s, .head) catch |x| {
            err.* = x;
        };
    }
    if (reqs.items.len == 0) return 0;
    e.backend.draft(reqs.items) catch |x| {
        for (joins.items) |i| errs[i] = x;
        return 0;
    };
    for (reqs.items) |r| joinedDrafted(e, r) catch |x| {
        for (joins.items) |i| if (streams[i] == r.stream) {
            errs[i] = x;
        };
    };
    return reqs.items.len;
}

/// A cancelled stream's lane is released here: the host finishes it without the core.
fn released(e: *Engine, s: *Stream, err: anyerror) anyerror {
    if (err == error.Cancelled) e.backend.release(s);
    return err;
}

/// A joined prompt (Backend.join): its last token pending, drafts asked now, verified in its first shared round.
fn joined(e: *Engine, s: *Stream) !void {
    s.context.shrinkRetainingCapacity(s.prompt_len);
    const last = s.context.items[s.prompt_len - 1];
    s.cache_len = s.prompt_len - 1;
    s.pending = last;
    const position: u64 = s.prompt_len;
    const d = try open.firstDepth(e, s, .head);
    try e.backend.draft(&.{.{ .stream = s, .follow = &.{}, .first = .{ .value = last }, .rows = null, .start = s.cache_len, .position = position, .depth = d }});
    s.dropHeld(e.gpa);
    s.next = .{ .count = d };
    try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position)), f("follow", .{ .u32s = &.{last} }), f("rows", .null) });
    try trail.resolve(e);
    try e.live.append(e.gpa, s);
}

/// After its first draft: the joined stream holds it and takes part from the next round.
fn joinedDrafted(e: *Engine, r: be.DraftRequest) !void {
    const s = r.stream;
    s.dropHeld(e.gpa);
    s.next = .{ .count = r.depth };
    try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(r.depth)), f("position", int(r.position)), f("follow", .{ .u32s = &.{s.pending.?} }), f("rows", .null) });
    try trail.resolve(e);
    try e.live.append(e.gpa, s);
}

/// A joined prompt's state and its first draft request (the last token pending at the prompt's last position).
fn joinRequest(e: *Engine, s: *Stream) !be.DraftRequest {
    s.context.shrinkRetainingCapacity(s.prompt_len);
    const last = s.context.items[s.prompt_len - 1];
    s.cache_len = s.prompt_len - 1;
    s.pending = last;
    const d = try open.firstDepth(e, s, .head);
    return .{ .stream = s, .follow = &.{}, .first = .{ .value = last }, .rows = null, .start = s.cache_len, .position = s.prompt_len, .depth = d };
}
