//! The lane allocator's prices for shared rounds, timed at load on a free session: shared forwards and a chained draft.
const std = @import("std");
const lanes = @import("lanes");
const mtl = @import("metal");
const fz = @import("replay.zig");
const forward = @import("batch_forward.zig");
const Session = @import("session.zig").Session;

pub const Costs = struct {
    shared: [fz.MAXR]lanes.config.Cost = undefined,
    step_ms: f64 = 0, // a chained head step
};

const PROMPT = 192; // the scratch prompt the windows follow
const TRIES = 4; // the fastest of these at each width
const WARM = 6; // widest windows first
/// Cache rows a session needs to be timed on: the scratch prompt, then windows of up to MAXR rows past it.
pub const ROWS = PROMPT + fz.MAXR + 1;

/// Times shared-round costs on the backend's last free session, or leaves them unpriced when it cannot fit.
pub fn timeCosts(b: anytype) !void {
    if (std.c.getenv("FZ_BATCH_UNPRICED") != null) {
        std.log.info("flash next: FZ_BATCH_UNPRICED=1: shared rounds unpriced, nothing timed", .{});
        return;
    }
    const s = &b.sessions[b.sessions.len - 1];
    if (!fits(s.m.cap)) {
        std.log.info("flash next: a session holds {d} rows, under the {d} that timing shared rounds takes: shared rounds unpriced", .{ s.m.cap, ROWS });
        return;
    }
    const c = try time(b, s);
    b.costs = c;
    std.log.info("flash next: shared forwards {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} ms at 1, 2, 4, 8, 16 rows; a chained draft {d:.2} ms", .{ c.shared[0].ms, c.shared[1].ms, c.shared[3].ms, c.shared[7].ms, c.shared[15].ms, c.step_ms });
}

/// Whether a session of `cap` rows can be timed on (the chained drafts stay inside the widest window).
pub fn fits(cap: usize) bool {
    return cap >= ROWS;
}

fn tokenAt(i: usize) u32 {
    return @intCast((37 * i + 11) % 50_000 + 1000);
}

/// The costs timed on `s` (a free session, empty again after), `b` a *Backend.
pub fn time(b: anytype, s: *Session) !Costs {
    if (!fits(s.m.cap)) return error.SessionTooSmall;
    const e = b.e;
    var prompt: [PROMPT]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = tokenAt(i);
    {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const old = try b.bind(s);
        defer old.restore(b.e);
        s.m.reset();
        const ps = [1]*fz.Prompt{e.pr};
        _ = try fz.Prompt.chunkN(&ps, &s.m, b.gpa, &prompt);
        try e.pr.mtpKeys(&s.m, 0, prompt[1..], e.pr.last);
        Session.copy(e.pr.last.at((PROMPT - 1) * fz.WIDE * 2), s.last, fz.WIDE * 2);
        s.normalize();
    }
    defer s.m.reset();
    var c: Costs = .{};
    for (0..WARM) |_| _ = try windowMs(b, s, fz.MAXR);
    for (&c.shared, 0..) |*x, i| {
        var best = std.math.inf(f64);
        for (0..TRIES) |_| best = @min(best, try windowMs(b, s, i + 1));
        x.* = .{ .width = @intCast(i + 1), .ms = best };
    }
    var one = std.math.inf(f64);
    var eight = std.math.inf(f64);
    for (0..TRIES) |_| {
        one = @min(one, try chainMs(b, s, 1));
        eight = @min(eight, try chainMs(b, s, 8));
    }
    c.step_ms = @max(0, (eight - one) / 7);
    return c;
}

/// One shared forward of `rows` rows on `s` from its position (nothing kept: the next window overwrites them).
fn windowMs(b: anytype, s: *Session, rows: usize) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t = &b.primary.t;
    var toks: [fz.MAXR]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = tokenAt(PROMPT + i);
    const t0 = mtl.clock.seconds();
    s.m.windowMeta(rows);
    s.m.pleIds(toks[0..rows]);
    @memcpy(t.ids8.b.slice(u32, fz.MAXR)[0..rows], toks[0..rows]);
    @memcpy(t.ple_ids.b.slice(u32, fz.MAXR * 16)[0 .. rows * 16], s.m.t.ple_ids.b.slice(u32, rows * 16));
    b.outputs(s, 0);
    t.rows.b.slice(i32, 1)[0] = @intCast(rows);
    t.mdims.b.slice(i32, 2)[0] = @intCast(rows);
    const pos8 = t.pos8.b.slice(u32, fz.MAXR);
    for (pos8[0..rows], 0..) |*p, j| p.* = @intCast(s.m.pos + j);
    const segs = [1]forward.Seg{.{ .s = s, .row0 = 0, .rows = rows }};
    const cb = b.e.r.queue.commandBuffer();
    b.e.r.begin(cb, .serial);
    errdefer if (b.e.r.fsr != null) b.e.r.abort();
    try forward.encode(b.e, &segs, rows, t.ids8, null);
    try b.primary.finish(cb);
    return (mtl.clock.seconds() - t0) * 1e3;
}

/// The head's absorb of one row and `depth` chained drafts in one command buffer, from the prompt's end each time.
fn chainMs(b: anytype, s: *Session, depth: usize) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const old = try b.bind(s);
    defer old.restore(b.e);
    s.m.mtp.pos = PROMPT - 1;
    s.m.mtp.drafted = 0;
    const follow = [1]u32{tokenAt(PROMPT)};
    const t0 = mtl.clock.seconds();
    const cb = b.e.r.queue.commandBuffer();
    b.e.r.begin(cb, if (b.e.r.serial) .serial else .concurrent);
    errdefer if (b.e.r.fsr != null) b.e.r.abort();
    s.encodeDrafts(&follow, s.last, depth) catch |err| {
        b.e.r.abort();
        return err;
    };
    try s.m.finish(cb);
    s.heldDrafts(depth);
    return (mtl.clock.seconds() - t0) * 1e3;
}

test "timing needs the scratch prompt and the widest window past it" {
    try std.testing.expectEqual(@as(usize, PROMPT + fz.MAXR + 1), ROWS);
    try std.testing.expect(!fits(ROWS - 1));
    try std.testing.expect(fits(ROWS));
    try std.testing.expect(!fits(160)); // --context 128's sessions
}

pub fn facts(b: anytype) lanes.Model {
    const c: ?*const Costs = if (b.costs) |*x| x else null;
    // Timed forwards price shared rounds; each extra draft costs one chained head step.
    const lone_priced = c != null and std.c.getenv("FZ_LONE_UNPRICED") == null; // not `lone`: that names the lone driver
    return .{ .exact_width = fz.MAXR, .first_copy_rows = 4, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .streams_exact = true, .hidden_rows = true, .batch_rows = fz.MAXR, .max_streams = @intCast(b.sessions.len), .draft_streams = !b.serial_drafts, .shared_costs = if (c) |x| &x.shared else &.{}, .draft_step_ms = if (c != null and !b.serial_drafts) c.?.step_ms else 0, .window_costs = if (lone_priced) &c.?.shared else &.{}, .mtp_step_ms = if (lone_priced) c.?.step_ms else 0 };
}
