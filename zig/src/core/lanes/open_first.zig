//! Opt-in drafted admission commits and hands off its first token before preparing initial drafts.
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const open = @import("open.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const LogRow = @import("logprob.zig").Row;
const f = ev.f;
const str = trail.str;
const int = trail.int;

pub fn opened(e: *Engine, s: *Stream, reach: open.Reach, hook: ?be.Committed) !void {
    if (s.isCancelled()) {
        e.backend.release(s);
        return error.Cancelled;
    }
    try e.live.ensureUnusedCapacity(e.gpa, 1);
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
    const drawn = try e.drawFirst(s, position);
    var feed: be.Feed = .{ .handle = drawn };
    if (try e.forcedNext(s)) |t| feed = .{ .value = t };
    const depth = try open.firstDepth(e, s, reach);
    const value = try e.readFeed(feed);
    if (e.log != null) {
        const actual = if (feed == .handle) value else try e.backend.read(drawn);
        try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(actual)), f("token", int(value)) });
    }
    var row: [1]LogRow = undefined;
    if (s.logprobs != null) row[0] = (try e.backend.firstRow(s)).forToken(value);
    _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &row else &.{});
    s.pending = value;
    if (s.finished) {
        {
            // The host will not discard a finished stream if tracing fails.
            defer e.release(s);
            try trail.resolve(e);
            try trail.finish(e, s);
        }
        if (hook) |h| h.call(h.ptr);
        return;
    }
    try trail.resolve(e);
    if (hook) |h| h.call(h.ptr);
    if (s.isCancelled()) {
        e.release(s);
        return error.Cancelled;
    }
    e.backend.draft(&.{.{ .stream = s, .follow = &.{}, .first = .{ .value = value }, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = depth }}) catch |err| {
        if (err == error.Cancelled) e.release(s);
        return err;
    };
    s.dropHeld(e.gpa);
    s.next = .{ .count = depth };
    if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
        s.next = held;
    };
    try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(depth)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
    e.live.appendAssumeCapacity(s);
}
