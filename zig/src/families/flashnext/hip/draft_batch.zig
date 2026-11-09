//! The draft head's absorb-and-chain step over a shared round's lanes.
const std = @import("std");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const head_mod = @import("head.zig");
const lanes_mod = @import("lanes.zig");

const be = @import("lanes").backend;
const Fx = lanes_mod.Fx;
const Lane = lanes_mod.Lane;

/// The head absorbs each stream's kept rows (residual streams and the token after each), then chains its drafts.
pub fn draft(f: *Fx, requests: []const be.DraftRequest) anyerror!void {
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
        var out = try head_mod.run(
            o,
            false,
            m,
            &e.host,
            hc,
            .{ .ptr = lane.streams.base(), .kind = m.act },
            follow,
            start,
            f.ones,
        );
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
            out = try head_mod.run(
                o,
                false,
                m,
                &e.host,
                hc,
                .{ .ptr = streams, .kind = m.act },
                &tok,
                lane.caches.len + d,
                f.ones,
            );
            mixed = out.mixed;
            streams = out.streams.ptr;
        }
        lane.held_n = chained;
    }
}
