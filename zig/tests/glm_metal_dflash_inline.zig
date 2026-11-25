//! Same-buffer drafter hook over stand-in slots: the pass behind a window, served or left for fallback.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const syn = @import("glm_metal_dflash_synthetic.zig");
const check = @import("glm_metal_dflash_ahead.zig");
const config = syn.config;
const rounds_mod = glm.metal_dflash_rounds;
const Context = glm.metal_dflash_model.Context;
const accept = glm.metal_dflash_accept;

/// Rounds behind stand-in windows: served or (every fourth, the core keeping fewer) not, chains equal today's.
fn testRounds(a: std.mem.Allocator, h: *syn.Harness, n: usize) !void {
    const m = try syn.create(a, h, true);
    defer m.deinit();
    const opts = mtl.ResourceOptions.shared;
    var e: glm.engine.Engine = undefined;
    e.sc.ids = .{ .buf = try h.device.buffer(64 * 4, opts), .off = 0 };
    e.sc.picks = .{ .buf = try h.device.buffer(64 * 4, opts), .off = 0 };
    defer e.sc.ids.buf.deinit();
    defer e.sc.picks.buf.deinit();
    var slots: [3]glm.slots.Slot = undefined;
    for (&slots) |*s| {
        s.s.pos = 0;
        s.rows = 0;
        s.row0 = 0;
    }
    var sl: glm.slots.Slots = undefined;
    sl.e = &e;
    sl.slots = slots[0..n];
    sl.windows = 0;
    sl.log = false;
    var taps = try glm.metal_dflash_taps.Taps.init(h.device, m.pending, config.window, config.hidden, a, glm.state.max_rows);
    defer taps.deinit(a);
    const r = try rounds_mod.Rounds.init(a, &sl, m, &taps);
    defer r.deinit();
    const x = try glm.metal_dflash_inline.Inline.init(a, r, h.device);
    r.inline_pass = x; // freed with the rounds
    const hook = x.hook();
    var feeds: [3]mtl.Buffer = undefined;
    for (feeds[0..n], 0..) |*f, k| f.* = try check.seeded(h.device, 400, 700 + k);
    defer for (feeds[0..n]) |f| f.deinit();
    const lens = [3]u32{ 30, 61, 9 };
    const rows = [3]u32{ 8, 5, 8 };
    var ctl: [3]rounds_mod.StreamRounds = undefined; // the same streams drafting as today
    for (0..n) |k| {
        const sr = r.make(@intCast(k)).?;
        try sr.ctx.reset(config, 0);
        try syn.feedStream(m, sr.ctx, feeds[k], 0, lens[k]);
        ctl[k] = .{ .ctx = try check.primed(a, m, feeds[k], lens[k]) };
        try m.swap(sr.ctx);
        defer m.swap(sr.ctx) catch unreachable;
        try m.flush();
    }
    defer for (ctl[0..n]) |c| c.ctx.deinit(a);
    var random: std.Random.DefaultPrng = .init(77 + n);
    const rng = random.random();
    var served: u32 = 0;
    for (0..20) |round| {
        var wins: [3]glm.slots.Win = undefined;
        var jobs: [3]rounds_mod.Job = undefined;
        var ctl_jobs: [3]rounds_mod.Job = undefined;
        var row0: u32 = 0;
        const cut = round % 4 == 3;
        for (0..n) |k| {
            const sr = r.get(@intCast(k)).?;
            const first = sr.ctx.session.context.end;
            slots[k].s.pos = @intCast(first);
            slots[k].rows = rows[k];
            slots[k].row0 = row0;
            wins[k] = .{ .slot = @intCast(k), .pending = 0, .held = 0, .tokens = &.{} };
            const tokens = e.sc.ids.buf.slice(u32, 64)[row0..][0..rows[k]];
            const picks = e.sc.picks.buf.slice(u32, 64)[row0..][0..rows[k]];
            for (tokens, picks) |*t, *p| {
                t.* = rng.uintLessThan(u32, config.vocab);
                p.* = rng.uintLessThan(u32, config.vocab);
            }
            const keep = rng.uintLessThan(u32, rows[k]);
            for (0..keep) |j| picks[j] = tokens[j + 1];
            const got = accept.accept(tokens, picks);
            // a cut: the core keeps one draft fewer than the GPU (a stop token), its anchor that row's pick
            const kept = if (cut and got[0] > 0) got[0] - 1 else got[0];
            jobs[k] = .{ .sr = sr, .position = first + kept + 2, .anchor = picks[kept], .depth = 7 };
            ctl_jobs[k] = jobs[k];
            ctl_jobs[k].sr = &ctl[k];
            sr.ctx.committed_end = first;
            try syn.feedStream(m, sr.ctx, feeds[k], @intCast(first), rows[k]); // the window's taps capture
            try syn.feedStream(m, ctl[k].ctx, feeds[k], @intCast(first), kept + 1);
            row0 += rows[k];
        }
        const cb = h.queue.commandBuffer();
        const enc = cb.compute(.serial);
        hook.encode(hook.ctx, &sl, cb, enc, wins[0..n]);
        enc.end();
        cb.commit();
        cb.wait();
        hook.land(hook.ctx, &sl, true);
        sl.windows += 1;
        rounds_mod.sharePass(jobs[0..n]);
        rounds_mod.sharePass(ctl_jobs[0..n]);
        var out: [3]rounds_mod.Round = @splat(.{});
        var want: [3]rounds_mod.Round = @splat(.{});
        const took = try x.serve(&sl, jobs[0..n], out[0..n]);
        const cut_any = for (jobs[0..n], 0..) |j, k| {
            if (j.position != slots[k].s.pos + accept.accept(e.sc.ids.buf.slice(u32, 64)[slots[k].row0..][0..rows[k]], e.sc.picks.buf.slice(u32, 64)[slots[k].row0..][0..rows[k]])[0] + 2) break true;
        } else false;
        try std.testing.expectEqual(!cut_any, took);
        if (took) served += 1;
        const members = [3]usize{ 0, 1, 2 };
        if (!took) { // today's pass on the contexts the window absorbed ahead into
            if (n == 1) try r.round(jobs[0].sr, jobs[0].position, jobs[0].anchor, jobs[0].depth, &out[0]) else try r.roundBatch(jobs[0..n], members[0..n], out[0..n]);
        }
        if (n == 1) try r.round(&ctl[0], ctl_jobs[0].position, ctl_jobs[0].anchor, ctl_jobs[0].depth, &want[0]) else try r.roundBatch(ctl_jobs[0..n], members[0..n], want[0..n]);
        for (0..n) |k| {
            try check.sameCaches(r.get(@intCast(k)).?.ctx, ctl[k].ctx);
            try std.testing.expectEqual(want[k].held_n, out[k].held_n);
            try std.testing.expectEqualSlices(u32, want[k].tokens[0..want[k].held_n], out[k].tokens[0..out[k].held_n]);
        }
    }
    try std.testing.expect(served >= 10);
    try std.testing.expectEqual(@as(u64, served), x.served);
}

pub fn run(a: std.mem.Allocator, h: *syn.Harness) !void {
    try testRounds(a, h, 1);
    try testRounds(a, h, 3);
}
