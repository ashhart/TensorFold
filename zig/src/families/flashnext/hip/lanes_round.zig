//! A lane round on Flash Next's HIP backend: every window verified in one forward, kept, then the head's drafts.
const hip = @import("hip");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const hip_lanes = @import("lanes.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");
const head_mod = @import("head.zig");
const worker = @import("worker.zig");

const Fx = hip_lanes.Fx;
const be = lanes.backend;
const draw = serve.draw;
const ring = hip_lanes.ring;
const max_depth = hip_lanes.max_depth;
const of = Fx.of;
const sampling = Fx.sampling;

/// Every window in one forward, each from its stream's kept length; every row drawn at its position.
pub fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const f = of(ptr);
    const e = f.e;
    const m = &e.weights.model;
    if (windows.len > forward.max_segs) return error.WindowTooWide;
    var tokens: [forward.max_segs][cache.max_window]u32 = undefined;
    var segs: [forward.max_segs]forward.Seg = undefined;
    var total: usize = 0;
    if (f.link != null) {
        f.msg.clearRetainingCapacity();
        try f.msg.appendSlice(f.gpa, &.{ @backingInt(worker.Op.verify), @intCast(windows.len), 0 });
    }
    for (windows, 0..) |w, i| {
        if (w.parents != null) return error.TreesNotBuilt;
        const lane = f.lanes.get(w.stream) orelse return error.UnknownStream;
        if (lane.pending) |p| try f.mark(lane, p.rows);
        const n = w.rows();
        if (n > cache.max_window) return error.WindowTooWide;
        if (w.held > lane.held_n or (w.held > 0 and w.tokens.len > 0)) return error.NoHeldDrafts;
        tokens[i][0] = w.pending;
        if (w.held > 0) @memcpy(tokens[i][1..n], lane.held[0..w.held]) else @memcpy(tokens[i][1..n], w.tokens);
        const len = lane.caches.len;
        for (w.positions, 0..) |p, r| if (p != len + 1 + r) return error.PositionMismatch;
        segs[i] = .{ .c = &lane.caches, .tokens = tokens[i][0..n], .pos = len };
        total += n;
        if (f.link != null) {
            try f.msg.appendSlice(f.gpa, &.{ lane.id, @intCast(n) });
            try f.msg.appendSlice(f.gpa, tokens[i][0..n]);
        }
    }
    const pick = try e.choose(segs[0..windows.len]);
    if (f.link != null) {
        f.msg.items[2] = @backingInt(pick);
        try f.send(f.msg.items);
    }
    const done = try e.round(segs[0..windows.len], pick);
    const fw = done.out;
    const row_bytes = m.dims.wide() * m.act.size();
    var off: usize = 0;
    for (windows) |w| {
        const lane = f.lanes.get(w.stream).?;
        try hip.raw.copy(lane.streams, 0, fw.streams.ptr + off * row_bytes, w.rows() * row_bytes, e.stream.handle);
        off += w.rows();
    }
    const o = e.ops(&e.arena);
    const logits = done.logits;
    var reqs: [forward.max_segs * cache.max_window]draw.Request = undefined;
    var drawn: [forward.max_segs * cache.max_window]u32 = undefined;
    off = 0;
    for (windows) |w| {
        const n = w.rows();
        const words = if (w.masks.len > 0) w.masks.len / n else 0;
        for (w.positions, 0..) |p, r| reqs[off + r] = .{
            .sampling = sampling(w.stream),
            .position = p,
            .mask = w.masks[r * words ..][0..words],
        };
        off += n;
    }
    if (total > e.drawer.rows) return error.WindowTooWide;
    try e.drawer.draw(o, e.stream, logits, reqs[0..total], drawn[0..total], false);
    off = 0;
    for (windows, out, 0..) |w, *v, i| {
        const n = w.rows();
        const lane = f.lanes.get(w.stream).?;
        @memcpy(v.sampled[0..n], drawn[off..][0..n]);
        @memcpy(v.drafts, if (w.held > 0) lane.held[0..w.held] else w.tokens);
        lane.held_n = 0;
        lane.pending = .{ .tokens = tokens[i], .rows = n };
        off += n;
    }
}

pub fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const f = of(ptr);
    for (windows, paths) |w, path| {
        const lane = f.lanes.get(w.stream) orelse return error.UnknownStream;
        for (path, 0..) |r, j| if (r != j) return error.TreesNotBuilt;
        if (lane.pending != null) try f.mark(lane, path.len);
    }
}

/// The head absorbs each stream's kept rows (residual streams and the token after each), then chains its drafts.
pub fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const f = of(ptr);
    const e = f.e;
    const m = &e.weights.model;
    const row_bytes = m.dims.wide() * m.act.size();
    for (requests) |r| {
        const lane = f.lanes.get(r.stream) orelse return error.UnknownStream;
        if (lane.pending) |p| try f.mark(lane, if (r.rows) |kept| kept.len else p.rows);
        lane.held_n = 0;
        if (r.position != lane.caches.len + 1) return error.PositionMismatch;
        const hc = if (lane.head) |*h| h else continue;
        const token: u32 = if (r.rows != null)
            r.follow[r.follow.len - 1]
        else switch (r.first orelse return error.NoFirstToken) {
            .handle => |at| f.drawn[at % ring],
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
        const depth = @min(r.depth, max_depth);
        // a chain past the first round ends after a draft the head gives under the confidence (--mtp-confidence)
        const stop_under: f32 = if (r.rows != null) @floatCast(e.o.policy.mtp.confidence) else 0.0;
        var d: usize = 0;
        var chained: usize = 0;
        while (d < depth) : (d += 1) {
            var tok: [1]u32 = undefined;
            var prob: [1]f32 = .{1.0};
            // a sampled stream drafts the target's keyed choice (head.keyed), its probability only for the gate
            const keyed = if (sampling(r.stream)) |sm|
                try head_mod.keyed(o, m, &e.host, e.stream, mixed, f.draft_ids, sm, r.position + d, if (stop_under > 0) &prob[0] else null)
            else
                null;
            if (keyed) |t| tok[0] = t else try head_mod.draws(o, m, &e.host, e.stream, mixed, 1, f.draft_ids, &tok, &prob);
            lane.held[d] = tok[0];
            chained = d + 1;
            if (d + 1 == depth or prob[0] < stop_under) break;
            out = try head_mod.run(
                o,
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
