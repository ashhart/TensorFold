//! Admitting streams: prompt passes (shared, as multi.py's _fill_batch), first tokens, then first drafts.
const std = @import("std");
const be = @import("backend.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const LogRow = @import("logprob.zig").Row;
const Feed = be.Feed;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// Prompt passes (shared when the backend has them), first tokens, every stream's first drafts; failures in `errs`.
pub fn streams(e: *Engine, list: []const *Stream, errs: []?anyerror) !void {
    _ = e.arena.reset(.retain_capacity);
    const a = e.arena.allocator();
    @memset(errs, null);
    for (list) |s| try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    if (list.len > 1 and e.backend.vtable.prefill_many != null) {
        try e.backend.vtable.prefill_many.?(e.backend.ptr, list, errs);
    } else for (list, errs) |s, *err| e.backend.prefill(s) catch |x| {
        err.* = x;
    };
    const feeds = try a.alloc(Feed, list.len);
    const drawn = try a.alloc(u64, list.len);
    const asked = try a.alloc(?u32, list.len);
    var requests: std.ArrayList(be.DraftRequest) = .empty;
    for (list, errs, feeds, drawn, asked) |s, *err, *feed, *handle, *ask| {
        ask.* = null;
        if (err.*) |x| {
            if (x == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
            continue;
        }
        if (s.isCancelled()) { // cancelled in its last chunk: no first token
            e.backend.release(s);
            err.* = error.Cancelled;
            continue;
        }
        s.context.shrinkRetainingCapacity(s.prompt_len);
        s.rows.clearRetainingCapacity();
        s.pending = null;
        s.cache_len = s.prompt_len;
        const position: u64 = s.prompt_len;
        handle.* = e.backend.first(s, position) catch |x| {
            err.* = x; // the host discards it, as after a failed prompt pass
            continue;
        };
        feed.* = .{ .handle = handle.* };
        if (try e.forcedNext(s)) |t| feed.* = .{ .value = t };
        if (e.cfg.family_mtp and s.drafts) {
            // the head reads the prompt's last row and the first token, and drafts the one after it
            const d: u32 = @intCast(try e.rule.depth(win.who(s)));
            ask.* = d;
            try requests.append(a, .{ .stream = s, .follow = &.{}, .first = feed.*, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = d });
        } else if (e.cfg.pipelined and s.logprobs == null) {
            try e.queueNext(s, feed.*);
        }
    }
    if (requests.items.len > 0) try e.backend.draft(requests.items);
    for (list, errs, feeds, drawn, asked) |s, err, feed, handle, ask| {
        if (err != null) continue;
        const position: u64 = s.prompt_len;
        if (ask) |d| {
            s.dropHeld(e.gpa);
            s.next = .{ .count = d };
            if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
                s.next = held; // a tree head's first drafts as host tokens, as every later round's
            };
        }
        const value = try e.readFeed(feed);
        if (ask) |d| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
        if (e.log != null) {
            const first = if (feed == .handle) value else try e.backend.read(handle);
            try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(first)), f("token", int(value)) });
        }
        var first_row: [1]LogRow = undefined;
        if (s.logprobs != null) first_row[0] = (try e.backend.firstRow(s)).forToken(value);
        _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &first_row else &.{});
        s.pending = value;
        try trail.resolve(e);
        if (s.finished) {
            try trail.finish(e, s);
            e.release(s);
            continue;
        }
        try e.live.append(e.gpa, s);
    }
}
