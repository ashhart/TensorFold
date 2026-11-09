//! The draft head's absorb-and-chain step over a shared round's lanes: serial, or one batched head pass a step.
const std = @import("std");
const hip = @import("hip");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const head_mod = @import("head.zig");
const lanes_mod = @import("lanes.zig");

const be = @import("lanes").backend;
const Fx = lanes_mod.Fx;
const Lane = lanes_mod.Lane;

/// A stream drafting in a batched call: its lane, head caches, chain depth and its rows in the packed absorb.
const Slot = struct {
    lane: *Lane,
    hc: *cache.Caches,
    depth: usize,
    stop_under: f32,
    tok: [1]u32 = undefined,
    row_at: usize = 0,
    row_n: usize = 0,
    chained: usize = 0,
};

/// The head absorbs each stream's kept rows (residual streams and the token after each), then chains its drafts.
pub fn draft(f: *Fx, requests: []const be.DraftRequest) anyerror!void {
    if (requests.len > 1) return batched(f, requests);
    return serial(f, requests);
}

/// The serial path: a stream at a time, one head forward and one draw a step.
fn serial(f: *Fx, requests: []const be.DraftRequest) anyerror!void {
    const e = f.e;
    const m = &e.weights.model;
    const row_bytes = m.dims.wide() * m.act.size();
    for (requests) |r| {
        const lane: *Lane = f.lanes.get(r.stream) orelse return error.UnknownStream;
        if (lane.pending) |p| try f.mark(lane, if (r.rows) |kept| kept.len else p.rows);
        lane.held_n = 0;
        if (r.position != lane.caches.len + 1) return error.PositionMismatch;
        const hc = if (lane.head) |*h| h else continue;
        const token: u32 = if (r.rows != null)
            r.follow[r.follow.len - 1]
        else switch (r.first orelse return error.NoFirstToken) {
            .handle => |at| f.drawn[at % lanes_mod.ring],
            .value => |v| v,
        };
        var follow_buf: [cache.max_window]u32 = undefined;
        const follow: []const u32 = if (r.rows != null) r.follow else blk: {
            follow_buf[0] = token;
            break :blk follow_buf[0..1];
        };
        const start = lane.caches.len - follow.len;
        try e.stream.synchronize();
        e.host.used = 0;
        e.arena.reset();
        const o = e.ops(&e.arena);
        const seg = [_]forward.Seg{.{ .c = hc, .tokens = follow, .pos = start }};
        var out = try head_mod.run(o, false, m, &e.host, &seg, .{ .ptr = lane.streams.base(), .kind = m.act }, f.ones);
        var mixed = forward.at(out.mixed, (follow.len - 1) * m.dims.hidden);
        var streams = out.streams.ptr + (follow.len - 1) * row_bytes;
        const depth = @min(r.depth, lanes_mod.max_depth);
        // a chain past the first round ends after a draft the head gives under the confidence (--mtp-confidence)
        const stop_under: f32 = if (r.rows != null) @floatCast(e.o.policy.mtp.confidence) else 0.0;
        var d: usize = 0;
        var chained: usize = 0;
        while (d < depth) : (d += 1) {
            var tok: [1]u32 = undefined;
            var prob: [1]f32 = undefined;
            try head_mod.draws(o, m, &e.host, e.stream, mixed, 1, f.draft_ids, &tok, &prob);
            lane.held[d] = tok[0];
            chained = d + 1;
            if (d + 1 == depth or prob[0] < stop_under) break;
            const step = [_]forward.Seg{.{ .c = hc, .tokens = &tok, .pos = lane.caches.len + d }};
            out = try head_mod.run(o, false, m, &e.host, &step, .{ .ptr = streams, .kind = m.act }, f.ones);
            mixed = out.mixed;
            streams = out.streams.ptr;
        }
        lane.held_n = chained;
    }
}

/// The batched path: a packed absorb forward and one draw over every stream, then one head pass a step.
fn batched(f: *Fx, requests: []const be.DraftRequest) anyerror!void {
    const e = f.e;
    const m = &e.weights.model;
    const d = m.dims;
    const row_bytes = d.wide() * m.act.size();
    const hidden_bytes = d.hidden * m.act.size();
    if (requests.len > forward.max_segs) return error.TooManyStreams;
    // the staging is reused across streams: the last forward's uploads must have left it
    try e.stream.synchronize();
    e.host.used = 0;
    e.arena.reset();
    const o = e.ops(&e.arena);
    var slots: [forward.max_segs]Slot = undefined;
    var segs: [forward.max_segs]forward.Seg = undefined;
    var firsts: [forward.max_segs][1]u32 = undefined;
    var n: usize = 0;
    var rows: usize = 0;
    for (requests) |r| {
        const lane = f.lanes.get(r.stream) orelse return error.UnknownStream;
        if (lane.pending) |p| try f.mark(lane, if (r.rows) |kept| kept.len else p.rows);
        lane.held_n = 0;
        if (r.position != lane.caches.len + 1) return error.PositionMismatch;
        const hc = if (lane.head) |*h| h else continue;
        const token: u32 = if (r.rows != null)
            r.follow[r.follow.len - 1]
        else switch (r.first orelse return error.NoFirstToken) {
            .handle => |at| f.drawn[at % lanes_mod.ring],
            .value => |v| v,
        };
        const follow: []const u32 = if (r.rows != null) r.follow else blk: {
            firsts[n][0] = token;
            break :blk firsts[n][0..1];
        };
        slots[n] = .{
            .lane = lane,
            .hc = hc,
            .depth = @min(r.depth, lanes_mod.max_depth),
            // a chain past the first round ends after a draft the head gives under the confidence (--mtp-confidence)
            .stop_under = if (r.rows != null) @floatCast(e.o.policy.mtp.confidence) else 0.0,
            .row_at = rows,
            .row_n = follow.len,
        };
        segs[n] = .{ .c = hc, .tokens = follow, .pos = lane.caches.len - follow.len };
        rows += follow.len;
        n += 1;
    }
    if (n == 0) return;
    // the kept residual rows of every stream packed one contiguous batch for the head's single forward
    const packed_rows = try e.arena.take(rows * row_bytes);
    for (slots[0..n]) |*s| {
        try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(
            @ptrFromInt(packed_rows + s.row_at * row_bytes),
            @ptrFromInt(s.lane.streams.base()),
            s.row_n * row_bytes,
            e.stream.handle,
        ));
    }
    const out = try head_mod.run(o, false, m, &e.host, segs[0..n], .{ .ptr = packed_rows, .kind = m.act }, f.ones);
    var active: [forward.max_segs]usize = undefined;
    var active_n: usize = 0;
    for (slots[0..n], 0..) |*s, i| if (s.depth > 0) {
        active[active_n] = i;
        active_n += 1;
    };
    if (active_n == 0) return;
    // each stream chains from its last absorbed row: its mixed row gathered, its residual row kept in its lane
    const mixed_rows = try e.arena.take(n * hidden_bytes);
    for (slots[0..n], 0..) |*s, i| {
        try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(
            @ptrFromInt(mixed_rows + i * hidden_bytes),
            @ptrFromInt(out.mixed.ptr + (s.row_at + s.row_n - 1) * hidden_bytes),
            hidden_bytes,
            e.stream.handle,
        ));
        if (s.depth > 0) try hip.raw.copy(
            s.lane.streams,
            0,
            out.streams.ptr + (s.row_at + s.row_n - 1) * row_bytes,
            row_bytes,
            e.stream.handle,
        );
    }
    var toks: [forward.max_segs]u32 = undefined;
    var probs: [forward.max_segs]f32 = undefined;
    try head_mod.draws(
        o,
        m,
        &e.host,
        e.stream,
        .{ .ptr = mixed_rows, .kind = m.act },
        n,
        f.draft_ids,
        toks[0..n],
        probs[0..n],
    );
    var next: usize = 0;
    for (active[0..active_n]) |si| {
        const s = &slots[si];
        s.lane.held[0] = toks[si];
        s.chained = 1;
        if (1 < s.depth and !(probs[si] < s.stop_under)) {
            active[next] = si;
            next += 1;
        }
    }
    active_n = next;
    var step: usize = 1;
    while (active_n > 0) : (step += 1) {
        const step_rows = try e.arena.take(active_n * row_bytes);
        var step_segs: [forward.max_segs]forward.Seg = undefined;
        for (active[0..active_n], 0..) |si, a| {
            const s = &slots[si];
            s.tok[0] = s.lane.held[step - 1];
            step_segs[a] = .{ .c = s.hc, .tokens = &s.tok, .pos = s.lane.caches.len + step - 1 };
            try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(
                @ptrFromInt(step_rows + a * row_bytes),
                @ptrFromInt(s.lane.streams.base()),
                row_bytes,
                e.stream.handle,
            ));
        }
        const step_out = try head_mod.run(
            o,
            false,
            m,
            &e.host,
            step_segs[0..active_n],
            .{ .ptr = step_rows, .kind = m.act },
            f.ones,
        );
        try head_mod.draws(
            o,
            m,
            &e.host,
            e.stream,
            step_out.mixed,
            active_n,
            f.draft_ids,
            toks[0..active_n],
            probs[0..active_n],
        );
        for (active[0..active_n], 0..) |si, a| try hip.raw.copy(
            slots[si].lane.streams,
            0,
            step_out.streams.ptr + a * row_bytes,
            row_bytes,
            e.stream.handle,
        );
        next = 0;
        for (active[0..active_n], 0..) |si, a| {
            const s = &slots[si];
            s.lane.held[step] = toks[a];
            s.chained = step + 1;
            if (step + 1 < s.depth and !(probs[a] < s.stop_under)) {
                active[next] = si;
                next += 1;
            }
        }
        active_n = next;
    }
    for (slots[0..n]) |*s| s.lane.held_n = s.chained;
}
