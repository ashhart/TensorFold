//! A lone stream's pipelined steps: the next forward queued before the last one's token is read.
const be = @import("backend.zig");
const sm = @import("stream.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Result = @import("engine.zig").Result;
const Stream = sm.Stream;
const Feed = be.Feed;
const f = ev.f;
const str = trail.str;

/// One token, the next forward queued first; a copied continuation ahead switches to verify windows.
pub fn round(e: *Engine, s: *Stream) !Result {
    const before = s.mode;
    // an adopted stream holds neither a mode nor a queued step: it starts where a drain step leaves one
    const mode: sm.Mode = s.mode orelse if (s.inflight == null) .verify else .pipe;
    var result: Result = .{ .rows = 1, .keep = 1, .got = &.{} };
    if (mode == .verify or mode == .exit) {
        const verifying = mode == .verify and e.cfg.family_width >= 2;
        const proposal: win.Copy = if (verifying) try win.copyProposal(e, s) else .{};
        if (proposal.tokens.len > 0) {
            const copied = try e.arena.allocator().dupe(u32, proposal.tokens);
            result = try e.familyRound(s, .{ .tokens = copied, .recalled = proposal.recalled });
            if (result.keep == 1 or s.force.items.len > 0) s.mode = .exit;
        } else {
            // nothing to copy: back to steps queued ahead (this one lands next round)
            s.mode = .pipe;
            try e.queueNext(s, .{ .value = s.pending.? });
        }
        try trail.pipe(e, s, before, result.got);
        return result;
    }
    var current: Feed = .{ .handle = s.inflight.? };
    s.inflight = null;
    if (try e.forcedNext(s)) |t| current = .{ .value = t };
    var token: u32 = undefined;
    if (mode == .drain) {
        token = try e.readFeed(current); // the last queued step: no new one
        s.mode = .verify;
    } else {
        try e.queueNext(s, current); // the GPU starts the next step first
        token = try e.readFeed(current);
    }
    s.rounds += 1;
    const landed = try s.commit(e.gpa, &.{token}, &.{});
    s.pending = token;
    if (s.recall) |r| r.noteHead(1); // a step lands one token: the bar a copy must beat
    if (e.cfg.root_copies) s.root_copy_miss = false;
    const piping = e.cfg.family_width >= 2 and (s.mode orelse .pipe) == .pipe and !s.finished and s.drafts;
    if (piping and (try win.copyProposal(e, s)).tokens.len > 0)
        s.mode = .drain; // a copy window is ahead: land the queued step
    result.got = try e.arena.allocator().dupe(u32, s.emitted()[s.emitted().len - landed ..]);
    try trail.pipe(e, s, before, result.got);
    return result;
}

/// Land the queued token without queuing another (a shared round is synchronous).
pub fn land(e: *Engine, s: *Stream) !void {
    var current: Feed = .{ .handle = s.inflight.? };
    s.inflight = null;
    if (try e.forcedNext(s)) |t| current = .{ .value = t };
    const token = try e.readFeed(current);
    s.rounds += 1;
    const landed = try s.commit(e.gpa, &.{token}, &.{});
    s.pending = token;
    s.mode = .verify;
    try trail.event(e, &.{ f("ev", str("land")), f("stream", str(s.id)), f("got", .{ .u32s = s.emitted()[s.emitted().len - landed ..] }) });
}
