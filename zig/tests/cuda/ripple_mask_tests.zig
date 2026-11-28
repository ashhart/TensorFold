//! Ripple's draws under grammar masks on CUDA (ripple-serve check 7): a masked draw is the plain draw over the cut row.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");
const serve = @import("ripple_serve_tests.zig");

const be = rc.backend;
const Stream = lanes.Stream;
const Window = lanes.backend.Window;
const Verified = lanes.backend.Verified;
const LogRow = lanes.LogprobRow;

/// A grammar row's kinds: every token allowed, about half of them, five of them, or all but one.
pub const MaskKind = enum { all, half, five, all_but };

/// A row of grammar bits over `v` tokens (lanes/grammar.zig: token t at bit t % 32 of word t / 32).
pub fn maskRow(out: []u32, rnd: std.Random, kind: MaskKind, v: usize, not: u32) void {
    switch (kind) {
        .all => @memset(out, 0xFFFF_FFFF),
        .half => for (out) |*w| {
            w.* = rnd.int(u32);
        },
        .five => {
            @memset(out, 0);
            for (0..5) |_| {
                const t = rnd.uintLessThan(usize, v);
                out[t / 32] |= @as(u32, 1) << @intCast(t % 32);
            }
        },
        .all_but => {
            @memset(out, 0xFFFF_FFFF);
            out[not / 32] &= ~(@as(u32, 1) << @intCast(not % 32));
        },
    }
}

/// `src`'s bf16 logits into `dst`, the tokens `mask` disallows at -inf (0xFF80).
fn underMask(dst: []u8, src: []const u8, mask: []const u32) void {
    for (0..src.len / 2) |t| {
        const allowed = lanes.grammar.allows(mask, t);
        dst[2 * t] = if (allowed) src[2 * t] else 0x80;
        dst[2 * t + 1] = if (allowed) src[2 * t + 1] else 0xFF;
    }
}

/// The first largest allowed logit's token: the greedy draw under a mask, on the host.
fn firstLargest(row: []const u8, mask: []const u32) u32 {
    var best: usize = 0;
    var top: f32 = 0;
    var found = false;
    for (0..row.len / 2) |t| {
        if (!lanes.grammar.allows(mask, t)) continue;
        const x: f32 = @bitCast((@as(u32, row[2 * t + 1]) << 24) | (@as(u32, row[2 * t]) << 16));
        if (found and !(x > top)) continue;
        best = t;
        top = x;
        found = true;
    }
    return @intCast(best);
}

/// Device scratch for check 7: host rows put on the device, and the plain draw's tokens over them.
const Redraw = struct {
    rows: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,

    /// The plain draw (no mask) of stream `s`'s rule over the host rows `bytes`, row 0 at `position0`.
    fn draw(d: Redraw, r: serve.Rig, s: *const Stream, bytes: []const u8, position0: u64, out: []u32) !void {
        try r.m.s.synchronize();
        try d.rows.upload(0, bytes);
        try r.x.drawRows(r.x.lanes.getPtr(s).?, d.rows.ptr, out.len, position0, d.picks.ptr, 0);
        try r.m.download(d.picks.ptr, std.mem.sliceAsBytes(out));
    }
};

/// Check 7: a stream's first token and a 4-row window under masks == the plain draw over the rows cut to the masks.
pub fn maskedDraws(r: serve.Rig, p: []const u32, plain: *const serve.Trace, sp: ?lanes.Sampling, label: []const u8) !usize {
    const v = r.m.c.vocab;
    const row_bytes = v * 2;
    const nw = be.Ripple.maskWords(v);
    const n = p.len;
    const win = 4;
    const k: u8 = lanes.logprob.max_top;
    var prng = std.Random.DefaultPrng.init(0x6d61_736b);
    const rnd = prng.random();
    // the model's rows must keep their bits, and the logprob rows must be the unmasked rows' for the token drawn
    var redraw: Redraw = .{ .rows = try cuda.DeviceBuffer.alloc(r.x.d, win * row_bytes), .picks = undefined };
    defer redraw.rows.free();
    redraw.picks = try cuda.DeviceBuffer.alloc(r.x.d, win * 4);
    defer redraw.picks.free();
    const under = try r.gpa.alloc(u8, win * row_bytes); // the model's rows of the window drawn under masks
    defer r.gpa.free(under);
    const bare = try r.gpa.alloc(u8, win * row_bytes); // the same window's rows with no mask
    defer r.gpa.free(bare);
    const cut = try r.gpa.alloc(u8, win * row_bytes); // rows with their disallowed tokens at -inf
    defer r.gpa.free(cut);
    const masks = try r.gpa.alloc(u32, win * nw);
    defer r.gpa.free(masks);
    var bad: usize = 0;
    var worst: f64 = 0;

    var sm = try Stream.init(r.gpa, .{ .id = "masked", .prompt = p, .max_new = serve.max_new, .sampling = sp, .drafts = false, .logprobs = k });
    defer sm.deinit(r.gpa);
    var tm = try serve.Trace.init(r.gpa, v);
    defer tm.deinit(r.gpa);
    _ = try r.open(&sm);
    defer r.b.release(&sm);

    // the first token: every token but the plain draw's allowed, so the draw must change
    const m0 = masks[0..nw];
    maskRow(m0, rnd, .all_but, v, plain.tokens[0]);
    const t0 = try r.firstUnder(&sm, &tm, m0);
    var want1: [1]u32 = undefined;
    underMask(cut[0..row_bytes], tm.logits[0], m0);
    try redraw.draw(r, &sm, cut[0..row_bytes], n, &want1);
    var ok = t0 == want1[0] and t0 != plain.tokens[0] and std.mem.eql(u8, tm.logits[0], plain.logits[0]);
    if (sp == null) ok = ok and t0 == firstLargest(tm.logits[0], m0);
    ok = serve.rowMatches(serve.Host.init(tm.logits[0]), try r.b.firstRow(&sm), t0, k, false, &worst) and ok;
    if (!ok) std.debug.print("  DIFF {s} first token under a mask: drew {d}, the plain draw over the cut row {d}, the plain token {d}\n", .{ label, t0, want1[0], plain.tokens[0] });
    bad += @intFromBool(!ok);

    // the window with no mask, in a second slot: the same pending token and drafts
    const drafts = plain.tokens[1..win];
    var pos: [win]u64 = undefined;
    for (&pos, 1..) |*q, i| q.* = n + i;
    var free_picks: [win]u32 = undefined;
    {
        var s0 = try Stream.init(r.gpa, .{ .id = "unmasked", .prompt = p, .max_new = serve.max_new, .sampling = sp, .drafts = false });
        defer s0.deinit(r.gpa);
        var t0u = try serve.Trace.init(r.gpa, v);
        defer t0u.deinit(r.gpa);
        _ = try r.open(&s0);
        defer r.b.release(&s0);
        _ = try r.first(&s0, &t0u);
        var dr: [win - 1]u32 = undefined;
        const w: Window = .{ .stream = &s0, .pending = t0, .held = 0, .tokens = drafts, .parents = null, .positions = &pos };
        var out = [_]Verified{.{ .sampled = &free_picks, .drafts = &dr }};
        try r.b.verify(&.{w}, &out);
        try r.m.download(r.m.b.rows, bare);
    }

    // the window under masks: a row of each kind, the last one without the unmasked draw's token
    const kinds = [win]MaskKind{ .all, .half, .five, .all_but };
    for (kinds, 0..) |kind, i| maskRow(masks[i * nw ..][0..nw], rnd, kind, v, free_picks[i]);
    var picks: [win]u32 = undefined;
    var dr: [win - 1]u32 = undefined;
    var rows: [win]LogRow = undefined;
    const w: Window = .{ .stream = &sm, .pending = t0, .held = 0, .tokens = drafts, .parents = null, .positions = &pos, .masks = masks };
    var out = [_]Verified{.{ .sampled = &picks, .drafts = &dr, .rows = &rows }};
    try r.b.verify(&.{w}, &out);
    try r.m.download(r.m.b.rows, under);
    for (0..win) |i| underMask(cut[i * row_bytes ..][0..row_bytes], under[i * row_bytes ..][0..row_bytes], masks[i * nw ..][0..nw]);
    var want: [win]u32 = undefined;
    try redraw.draw(r, &sm, cut, n + 1, &want);
    ok = std.mem.eql(u8, under, bare) and std.mem.eql(u32, &picks, &want) and picks[0] == free_picks[0] and picks[win - 1] != free_picks[win - 1];
    worst = 0;
    for (rows, picks, 0..) |lr, pick, i| {
        const row = under[i * row_bytes ..][0..row_bytes];
        const mask = masks[i * nw ..][0..nw];
        ok = lanes.grammar.allows(mask, pick) and ok;
        if (sp == null) ok = ok and pick == firstLargest(row, mask);
        ok = serve.rowMatches(serve.Host.init(row), lr, pick, k, false, &worst) and ok;
    }
    if (!ok) std.debug.print("  DIFF {s} window under masks: drew {any}, the plain draw over the cut rows {any}, with no mask {any}; the model's rows equal: {}\n", .{ label, picks, want, free_picks, std.mem.eql(u8, under, bare) });
    bad += @intFromBool(!ok);

    // a mask of the wrong length is refused by name, before any forward
    const short: Window = .{ .stream = &sm, .pending = t0, .held = 0, .tokens = drafts, .parents = null, .positions = &pos, .masks = masks[0 .. masks.len - 1] };
    const refused = if (r.b.verify(&.{short}, &out)) |_| false else |e| e == error.InvalidGrammarMasks;
    bad += @intFromBool(!refused);

    if (bad == 0) {
        check.pass("ripple-serve {s} draws under grammar masks: stream A's first token and a 4-row window (every token, about half, five tokens, all but the plain one) == the plain draw over the rows with the other tokens at -inf; the model's rows == those with no mask; logprob rows from the unmasked rows ({d} best, worst {e:.2}); a short mask refused", .{ label, k, worst });
        return 0;
    }
    std.debug.print("FAIL ripple-serve {s} draws under grammar masks: {d} of 3 parts differ\n", .{ label, bad });
    return 1;
}
