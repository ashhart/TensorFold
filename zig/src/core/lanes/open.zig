//! A stream's way into the rounds: its prompt pass whole or a chunk a round, then its first token and drafts.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const be = @import("backend.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const Stream = sm.Stream;
const Feed = be.Feed;
const LogRow = @import("logprob.zig").Row;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// Prefill a stream, draw its first token and ask for its first drafts; it takes part from the next round.
pub fn add(e: *Engine, s: *Stream) !void {
    return addCommitted(e, s, null);
}

pub fn addCommitted(e: *Engine, s: *Stream, hook: ?be.Committed) !void {
    _ = e.arena.reset(.retain_capacity);
    try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    e.backend.prefill(s) catch |err| {
        if (err == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
        return err;
    };
    try openedCommitted(e, s, .rule, hook);
}

/// How a stream's first drafts are sized: the depth rule, or the head's whole reach under node probabilities.
pub const Reach = enum { rule, head };

/// Streams a backend prefills in one pass at most.
pub const max_together = 16;

/// `add` for several streams, prompts in one pass when the backend takes them; `errs[i]`: stream i's error, or null.
pub fn addStreams(e: *Engine, ss: []const *Stream, errs: []?anyerror) void {
    for (errs) |*x| x.* = null;
    const many = e.backend.vtable.prefill_many orelse return addEach(e, ss, errs);
    if (ss.len < 2 or ss.len > max_together) return addEach(e, ss, errs);
    _ = e.arena.reset(.retain_capacity);
    for (ss) |s| trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) }) catch {};
    const together = many(e.backend.ptr, ss) catch |err| blk: { // one stream's refusal is its own: each goes alone
        std.log.warn("{d} prompts together: {s}; each goes alone", .{ ss.len, @errorName(err) });
        break :blk false;
    };
    if (!together) return addEach(e, ss, errs);
    for (ss, errs) |s, *x| opened(e, s) catch |err| { // a stream cancelled in the pass gets no first token
        x.* = err;
    };
}

/// Whether `s` may join a batched prompt pass beside chunked fills: short, text only, nothing kept to resume from.
pub fn batchable(e: *const Engine, s: *const Stream) bool {
    const n = e.backend.vtable.many_rows;
    return n > 0 and s.prompt_len <= n and s.images.len == 0 and s.reuse.saved == null;
}

fn addEach(e: *Engine, ss: []const *Stream, errs: []?anyerror) void {
    for (ss, errs) |s, *x| add(e, s) catch |err| {
        x.* = err;
    };
}

/// After the prompt pass: the first token drawn and committed, the first drafts asked; rounds from the next.
pub fn opened(e: *Engine, s: *Stream) !void {
    return openedWith(e, s, .rule);
}

/// `opened` with its first drafts sized by `reach` (a family planner's prompts in pieces: `.head`).
pub fn openedWith(e: *Engine, s: *Stream, reach: Reach) !void {
    return openedCommitted(e, s, reach, null);
}

fn openedCommitted(e: *Engine, s: *Stream, reach: Reach, hook: ?be.Committed) !void {
    if (s.embedding != null) {
        if (s.isCancelled()) {
            e.backend.release(s);
            return error.Cancelled;
        }
        s.cache_len = s.prompt_len;
        s.pending = null;
        return;
    }
    if (e.backend.first_before_draft and e.cfg.family_mtp and s.drafts) return @import("open_first.zig").opened(e, s, reach, hook);
    if (s.isCancelled()) { // cancelled in its last chunk: no first token
        e.backend.release(s);
        return error.Cancelled;
    }
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
    const drawn = try e.drawFirst(s, position);
    var feed: Feed = .{ .handle = drawn };
    if (try e.forcedNext(s)) |t| feed = .{ .value = t };
    var asked: ?u32 = null;
    if (e.cfg.family_mtp and s.drafts) {
        // the head reads the prompt's last row and the first token, and drafts the one after it
        const wanted = try firstDepth(e, s, reach);
        const requests = try @import("root_copy.zig").prepare(e, &.{.{ .stream = s, .follow = &.{}, .first = feed, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = wanted }});
        var d = requests[0].depth;
        asked = d;
        try e.backend.draft(requests);
        if (e.backend.proposals()) d = requests[0].heldDepth(e.backend);
        s.dropHeld(e.gpa);
        s.next = .{ .count = d };
        if (d > 0) if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
            s.next = held; // a tree head's first drafts as host tokens, as every later round's
        };
    } else if (e.cfg.pipelined and s.logprobs == null and s.grammar == null) {
        try e.queueNext(s, feed);
    }
    const value = try e.readFeed(feed);
    if (asked) |d| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
    if (e.log != null) {
        const first = if (feed == .handle) value else try e.backend.read(drawn);
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
        return;
    }
    try e.live.append(e.gpa, s);
}

/// The first drafts' depth for `reach`.
pub fn firstDepth(e: *Engine, s: *Stream, reach: Reach) !u32 {
    var who = win.who(s);
    if (reach == .head and e.cfg.node_probabilities) return @intCast(try e.rule.headDepth(&who, null, win.probe(e)));
    return @intCast(try e.rule.depth(who));
}

/// Whether the backend can run a prompt pass a chunk at a time between rounds (`fillStream`).
pub fn fills(e: *const Engine) bool {
    return e.backend.vtable.prefill_step != null;
}

/// One chunk of a stream's prompt pass; true once the stream is opened and joins the next round.
pub fn fillStream(e: *Engine, s: *Stream, first: bool) !bool {
    return fillStreamCommitted(e, s, first, null);
}

pub fn fillStreamCommitted(e: *Engine, s: *Stream, first: bool, hook: ?be.Committed) !bool {
    _ = e.arena.reset(.retain_capacity);
    if (first) try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    const chunk = e.backend.vtable.prefill_step orelse return error.NoPrefillSteps;
    const done = chunk(e.backend.ptr, s) catch |err| {
        if (err == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
        return err;
    };
    if (!done) return false;
    try openedCommitted(e, s, .rule, hook);
    return true;
}
