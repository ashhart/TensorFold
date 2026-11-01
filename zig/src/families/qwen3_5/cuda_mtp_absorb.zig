//! Stateful MTP absorption keeps keys during a priced skip; proposal arithmetic runs only for useful drafts.
const mtp = @import("cuda_mtp.zig");
const Mtp = mtp.Mtp;
const Ctx = mtp.Ctx;
const TapRows = mtp.TapRows;
const engine = @import("cuda_engine.zig");
const max_rows = engine.most_verify;
const most_window = engine.most_window;
const ranks = @import("lanes").shape.ranks;
pub fn absorbRole(d: *Mtp, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize, rows_: ?[]const u32, follow: []const u32, prompt: bool, propose: bool) !void {
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = if (prompt) .prompt else .live;
    const e = d.e;
    const g = e.g;
    var start = start_;
    var n = n_;
    var fi: usize = 0;
    if (follow.len < n) return error.BadDraftRows;
    if (rows_) |rs| if (rs.len != n or n > most_window) return error.BadDraftRows;
    if (start < ctx.end) ctx.end = start;
    if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
    const held = engine.tap_rows - engine.max_rows; // the target's ring holds this many rows behind its newest chunk
    if (n > held) {
        fi = n - held;
        start += n - held;
        n = held;
        ctx.begin = start;
    }
    ctx.end = start;
    while (n > 0) {
        const rows = @min(n, max_rows);
        if (rows_) |rs| {
            var tr: TapRows = .{ .n = @intCast(rows), .rows = @splat(0) };
            for (rs, 0..) |r, i| tr.rows[i] = @intCast(r);
            try e.run(d.ring_fns.gather_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), tr, @as(c_int, @intCast(g.hidden)), d.taps_in });
        } else try e.run(d.ring_fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, engine.tap_rows), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(g.hidden)), d.taps_in });
        try d.norm(d.taps_in, 0, e.final_norm, d.hin, rows); // the target's final norm: the rows the head read
        try d.putIds(follow[fi..][0..rows]);
        try d.layerMode(ctx, rows, start, .ring, propose);
        start += rows;
        n -= rows;
        fi += rows;
        ctx.end = start;
        if (!propose) continue;
        // the newest row: its state kept for the drafts, its best tokens drawn
        try ctx.last.copyFrom(0, d.x + (rows - 1) * g.hidden * 2, g.hidden * 2, e.stream.handle);
        try d.draw(rows - 1, 1, true);
    }
    if (!propose) return e.stream.synchronize();
    var cands: [1][ranks]u32 = undefined;
    ctx.cands[0] = try d.fetch(1, &cands);
    if (d.trees) ctx.cands = cands[0];
}
