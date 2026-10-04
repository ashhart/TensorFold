//! What the round loop writes to the decision log: the recorder's events, field for field (events.zig formats).
const std = @import("std");
const Allocator = std.mem.Allocator;
const ev = @import("events.zig");
const sm = @import("stream.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Engine = @import("engine.zig").Engine;
const Table = @import("table.zig").Table;
const Stream = sm.Stream;
const Value = ev.Value;
const f = ev.f;

pub fn str(s: []const u8) Value {
    return .{ .str = s };
}

pub fn int(x: anytype) Value {
    return .{ .int = @intCast(x) };
}

pub fn event(e: *Engine, fields: []const ev.Field) !void {
    if (e.log) |log| _ = try log.add(fields);
}

pub fn pipe(e: *Engine, s: *Stream, before: ?sm.Mode, got: []const u32) !void {
    try event(e, &.{ f("ev", str("pipe")), f("stream", str(s.id)), f("mode", modeName(before)), f("after", modeName(s.mode)), f("got", .{ .u32s = got }) });
}

pub fn modeName(m: ?sm.Mode) Value {
    return if (m) |x| str(@tagName(x)) else .null;
}

/// Fill in queued draws' tokens where the recorder reads them: the end of an admission or a step.
pub fn resolve(e: *Engine) !void {
    const log = e.log orelse return;
    for (e.unread.items) |u| try log.set(u.event, "token", int(try e.backend.read(u.handle)));
    e.unread.clearRetainingCapacity();
}

pub fn finish(e: *Engine, s: *Stream) !void {
    try event(e, &.{ f("ev", str("finish")), f("stream", str(s.id)), f("reason", str(s.reason.name())), f("emitted", .{ .u32s = s.emitted() }), f("rounds", int(s.rounds)), f("drafted", int(s.drafted)), f("accepted", int(s.accepted)), f("cache_len", int(s.cache_len)) });
}

pub fn table(a: Allocator, t: Table) !Value {
    var rows: std.ArrayList(Value) = .empty;
    for (t.slots, 0..) |slot, k| {
        const v = slot orelse continue;
        const pair = try a.alloc(Value, 2);
        pair[0] = int(k);
        pair[1] = ev.bits(v);
        try rows.append(a, .{ .list = pair });
    }
    return .{ .list = rows.items };
}

/// The loop's state after a step: the recorder's `state` event, field for field.
pub fn state(e: *Engine) !void {
    if (e.log == null) return;
    const a = e.arena.allocator();
    const overhead = try a.alloc(Value, e.rule.overhead.items.len);
    for (overhead, e.rule.overhead.items) |*o, x| {
        const pair = try a.alloc(Value, 2);
        pair[0] = int(x.streams);
        pair[1] = ev.bits(x.ms);
        o.* = .{ .list = pair };
    }
    const streams = try a.alloc(Value, e.live.items.len);
    for (streams, e.live.items) |*out, s| {
        const pending: []const u32 = if (s.pending) |p| try a.dupe(u32, &.{p}) else &.{};
        const d: Value = if (s.depth) |st| blk: {
            const p = try a.alloc(u64, st.p.len);
            for (p, st.p) |*b, x| b.* = @bitCast(x);
            break :blk .{ .obj = try a.dupe(ev.Field, &.{ f("p", .{ .u64s = p }), f("rounds", int(st.rounds)), f("plain", if (st.plain) |x| int(x) else .null), f("wait", if (st.wait) |x| int(x) else .null), f("ms", if (st.ms) |t| try table(a, t) else .null) }) };
        } else .null;
        const proposer: Value = if (s.proposer) |p| (if (p.vtable == &SuffixLookup.vtable) blk: {
            const x: *SuffixLookup = @ptrCast(@alignCast(p.ptr));
            break :blk .{ .obj = try a.dupe(ev.Field, &.{ f("silent_for", int(x.silent_for)), f("recent", .{ .i64s = x.recent.items }), f("proposals", int(x.proposals)), f("proposed_tokens", int(x.proposed_tokens)), f("judged_tokens", int(x.judged_tokens)), f("accepted_tokens", int(x.accepted_tokens)), f("silenced_rounds", int(x.silenced_rounds)), f("last_match", int(x.last_match)) }) };
        } else .null) else .null;
        out.* = .{ .obj = try a.dupe(ev.Field, &.{
            f("id", str(s.id)),                                       f("cache_len", int(s.cache_len)),
            f("emitted", int(s.emitted().len)),                       f("pending", .{ .u32s = pending }),
            f("rounds", int(s.rounds)),                               f("drafted", int(s.drafted)),
            f("accepted", int(s.accepted)),                           f("force", .{ .u32s = s.force.items }),
            f("depth", d),                                            f("mode", modeName(s.mode)),
            f("copy_width", if (s.copy_width) |x| int(x) else .null), f("served", if (s.served) |x| int(x) else .null),
            f("granted", if (s.granted) |x| int(x) else .null),       f("next", if (s.next) |h| int(h.count) else .null),
            f("inflight", .{ .bool = s.inflight != null }),           f("proposer", proposer),
        }) };
    }
    try event(e, &.{
        f("ev", str("state")),                        f("alone", .{ .bool = e.alone }),
        f("drafted", int(e.drafted)),                 f("accepted", int(e.accepted)),
        f("round_ms", try table(a, e.rule.round_ms)), f("overhead_ms", .{ .list = overhead }),
        f("shared_rounds", int(e.shared_rounds)),     f("streams", .{ .list = streams }),
    });
}
