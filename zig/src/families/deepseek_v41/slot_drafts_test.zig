//! DSpark over several slots on the host (`zig build test-dsv41-slot-drafts`): 1-3 slot passes equal the one-slot pass.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const Config = @import("config.zig").Config;
const graphs = @import("graphs.zig");
const rowmode = @import("rowmode.zig");
const run = @import("run.zig");
const dsd = @import("dspark_dev.zig");
const dsk = @import("draft/dspark.zig");
const fixture = @import("slot_drafts_fixture.zig");

test {
    _ = rows;
}

test {
    _ = @import("slot_drafts_chain_test.zig");
}

const widths = fixture.widths;
const Fixture = fixture.Fixture;
const roleOf = fixture.roleOf;
const class = fixture.class;
const sameCall = fixture.sameCall;
const equalArgs = fixture.equalArgs;
const identical = fixture.identical;

test "one slot: Layout.of(1) is the one-slot statics, and the default emission is the one-slot pass / ingest" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const n: i64 = f.cfg.dspark_block;
    const one: emit.Statics = .{ .n = n, .window = f.cfg.window };
    const l = emit.Layout.of(1, n, f.cfg.window);
    const lay: emit.Statics = .{ .n = n, .window = f.cfg.window, .lay = l };
    inline for (.{ "ids", "positions", "readRows", "writeRows", "len64", "tokens", "counts", "lo", "hi", "len32" }) |m|
        try testing.expectEqual(@field(emit.Statics, m)(one), @field(emit.Statics, m)(lay));
    try identical(try emit.emitPass(a, &f.cfg, &f.w, f.o, n), try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, 1, 1));
    try identical(try emit.emitIngest(a, &f.cfg, &f.w, f.o, 7, 3), try emit.emitIngestAt(a, &f.cfg, &f.w, f.o, 7, 3, .{}));
}

test "several slots: the same launches in the one-slot pass's Triton specializations, rows x slots, rings stacked" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const n: i64 = f.cfg.dspark_block;
    const S: i64 = 4;
    const ring = emit.ringRows(&f.cfg);
    const base = try emit.emitPass(a, &f.cfg, &f.w, f.o, n);
    for (1..rows.maxGroup(f.cfg.dspark_block) + 1) |k| {
        const cs = try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, @intCast(k), S);
        try testing.expect(@as(i64, @intCast(k)) * n < 16);
        try testing.expectEqual(base.len, cs.len);
        var rings: usize = 0;
        for (base, cs) |x, y| {
            try sameCall(x, y);
            if (x.triton) for (x.args, y.args) |p, q| {
                // a Triton launch's ints that are rows here are never 1 or a multiple of 16 at k x n rows
                try testing.expectEqual(class(p.arg), class(q.arg));
            };
            for (y.args) |q| if (q.arg == .t) {
                const r = roleOf(q.arg.t);
                if (std.mem.startsWith(u8, r, "s.L") and std.mem.indexOf(u8, r, ".ring.") != null) {
                    rings += 1;
                    try testing.expectEqual(S * ring, q.arg.t.shape[0]);
                }
            };
        }
        try testing.expect(rings > 0);
    }
    // the plan sizes the rings for every slot and the statics for the largest pass
    var p: buffers.Plan = .{ .a = a };
    const g = rows.maxGroup(f.cfg.dspark_block);
    try p.add(try emit.emitPassSlots(a, &f.cfg, &f.w, f.o, n, g, S));
    try testing.expectEqual(@as(u64, @intCast(S * ring * 576)), p.sizes.get("s.L40.ring.v").?);
    const l = emit.Layout.of(g, n, f.cfg.window);
    try testing.expect(p.sizes.get("w.ds.i64").? >= 8 * @as(u64, @intCast(l.len64)));
    try testing.expect(p.sizes.get("w.ds.i32").? >= 4 * @as(u64, @intCast(l.end32)));
}

test "an ingest writes its slot's ring (a view at slot x ring rows) from the chosen taps role" {
    var f: Fixture = .{ .arena = undefined };
    try f.init();
    defer f.arena.deinit();
    const a = f.arena.allocator();
    const ring = emit.ringRows(&f.cfg);
    const base = try emit.emitIngest(a, &f.cfg, &f.w, f.o, 6, 2);
    const cs = try emit.emitIngestAt(a, &f.cfg, &f.w, f.o, 6, 2, .{ .ring_slots = 4, .slot = 3, .taps_role = "s.rows.taps" });
    try testing.expectEqual(base.len, cs.len);
    var stores: usize = 0;
    var taps: usize = 0;
    for (base, cs) |x, y| {
        try sameCall(x, y);
        if (x.triton) for (x.args, y.args) |p, q| try testing.expectEqual(class(p.arg), class(q.arg));
        for (x.args, y.args) |p, q| if (p.arg == .t) {
            const r = roleOf(p.arg.t);
            if (std.mem.indexOf(u8, r, ".ring.v") != null) {
                stores += 1;
                try testing.expectEqual(3 * ring * 576, q.arg.t.offset);
                try testing.expectEqual(ring, q.arg.t.shape[0]);
            } else if (std.mem.indexOf(u8, r, ".ring.s") != null) {
                try testing.expectEqual(3 * ring * 8, q.arg.t.offset);
            } else if (std.mem.eql(u8, r, "w.taps")) {
                taps += 1;
                try testing.expectEqualStrings("s.rows.taps", roleOf(q.arg.t));
                try testing.expectEqual(p.arg.t.offset, q.arg.t.offset);
            }
        };
    }
    try testing.expectEqual(@as(usize, f.cfg.mtp_layers), stores);
    try testing.expect(taps > 0);
}

test "stage: Python's row-mode DraftInputs (ids, positions, ring slots, the context in the stacked ring)" {
    const n: u32 = 5;
    const W: u32 = 128;
    const ring: u32 = 256;
    const lay = emit.Layout.of(2, n, W);
    var h64: [256]i64 = undefined;
    var h32: [4096]i32 = undefined;
    const ms = [_]rows.Member{ .{ .slot = 3, .anchor = 77, .start = 300, .valid = 0 }, .{ .slot = 1, .anchor = 9, .start = 40, .valid = 35 } };
    rows.stage(lay, n, W, ring, 2, &ms, &h64, &h32);
    const at = struct {
        fn i64s(h: []const i64, o: i64, i: usize) i64 {
            return h[@as(usize, @intCast(o)) + i];
        }
        fn i32s(h: []const i32, o: i64, i: usize) i32 {
            return h[@as(usize, @intCast(o)) + i];
        }
    };
    try testing.expectEqual(77, at.i64s(&h64, lay.ids, 0));
    try testing.expectEqual(2, at.i64s(&h64, lay.ids, 1));
    try testing.expectEqual(9, at.i64s(&h64, lay.ids, 5));
    try testing.expectEqual(304, at.i64s(&h64, lay.positions, 4));
    try testing.expectEqual(40, at.i64s(&h64, lay.positions, 5));
    try testing.expectEqual(3, at.i64s(&h64, lay.read, 4));
    try testing.expectEqual(1, at.i64s(&h64, lay.write, 9));
    // slot 3 at 300: positions 172 .. 299 (the window), rows 3 x 256 + p % 256, every row of the slot alike
    try testing.expectEqual(W, @as(u32, @intCast(at.i32s(&h32, lay.counts, 0))));
    try testing.expectEqual(3 * 256 + 172, at.i32s(&h32, lay.tokens, 0));
    try testing.expectEqual(3 * 256 + 299 % 256, at.i32s(&h32, lay.tokens, 4 * W + W - 1));
    // slot 1 at 40 with valid 35: positions 35 .. 39 only
    try testing.expectEqual(5, at.i32s(&h32, lay.counts, 5));
    try testing.expectEqual(256 + 35, at.i32s(&h32, lay.tokens, 5 * W));
    try testing.expectEqual(-1, at.i32s(&h32, lay.tokens, 5 * W + 5));
    try testing.expectEqual(40, at.i32s(&h32, lay.lo, 9));
    try testing.expectEqual(44, at.i32s(&h32, lay.hi, 9));
    // no array overlaps the next one
    try testing.expect(lay.positions >= lay.ids + 10 and lay.read >= lay.positions + 10 and lay.write >= lay.read + 10);
    try testing.expect(lay.counts >= lay.tokens + 10 * W and lay.lo >= lay.counts + 10 and lay.hi >= lay.lo + 10);
}

test "groups, messages, taps placement, the gather's order" {
    var gs: [16][2]usize = undefined;
    try testing.expectEqual(@as(usize, 2), rows.groups(4, 3, &gs));
    try testing.expectEqualSlices(usize, &.{ 0, 2 }, &gs[0]);
    try testing.expectEqualSlices(usize, &.{ 2, 4 }, &gs[1]);
    try testing.expectEqual(@as(usize, 1), rows.groups(3, 3, &gs));
    try testing.expectEqual(@as(usize, 2), rows.groups(5, 3, &gs));
    try testing.expectEqualSlices(usize, &.{ 3, 5 }, &gs[1]);
    try testing.expectEqual(@as(usize, 4), rows.groups(4, 1, &gs));
    try testing.expectEqual(@as(u32, 3), rows.maxGroup(5));
    try testing.expectEqual(@as(u32, 1), rows.maxGroup(16));

    const ms = [_]rows.Member{ .{ .slot = 0, .anchor = 5, .start = 10, .valid = 0 }, .{ .slot = 2, .anchor = 6, .start = 70, .valid = 0 } };
    var buf: [50]i64 = undefined;
    const msg = try rows.proposeMsg(&ms, &buf);
    const valid = [_]u64{ 4, 0, 61, 0 };
    var out: [3]rows.Member = undefined;
    const got = try rows.proposeOf(msg, &valid, &out);
    try testing.expectEqual(@as(usize, 2), got.len);
    try testing.expectEqual(@as(u64, 4), got[0].valid);
    try testing.expectEqual(@as(u64, 61), got[1].valid);
    try testing.expectEqual(@as(u32, 6), got[1].anchor);
    try testing.expectError(error.BadPlan, rows.proposeOf(msg, &valid, out[0..1]));

    const taps: u64 = 0x10000;
    const stride: u64 = 2 * 3 * 4096;
    const stash = [2]u64{ 0x900000, 16 * 4 * stride };
    // in "w.taps": the slot's rows from row 4 of the window, rows 1 .. 3 of them
    try testing.expectEqual(rows.TapsAt{ .src = 0, .skip = 5 }, try rows.locate(taps + 4 * stride, 6, stride, 1, 3, 6, taps, stash));
    // stashed: slot 2's block of the stash
    try testing.expectEqual(rows.TapsAt{ .src = 1, .skip = 32 }, try rows.locate(stash[0] + 32 * stride, 4, stride, 0, 4, 6, taps, stash));
    // past the taps (a resumed prompt) or past the planned rows: not on the device
    try testing.expectEqual(@as(i64, -1), (try rows.locate(taps, 6, stride, 0, 7, 16, taps, stash)).src);
    try testing.expectEqual(@as(i64, -1), (try rows.locate(taps, 128, stride, 0, 100, 16, taps, stash)).src);
    try testing.expectError(error.Rows, rows.locate(taps + 3, 6, stride, 0, 1, 6, taps, stash));

    // [W][R][2k] -> rows of [W k]
    const all = [_]f32{ 1, 2, @bitCast(@as(i32, 10)), @bitCast(@as(i32, 11)), 3, 4, @bitCast(@as(i32, 12)), @bitCast(@as(i32, 13)), 5, 6, @bitCast(@as(i32, 20)), @bitCast(@as(i32, 21)), 7, 8, @bitCast(@as(i32, 22)), @bitCast(@as(i32, 23)) };
    var cand: [8]i32 = undefined;
    var cval: [8]f32 = undefined;
    rows.ungather(&all, 2, 2, 2, &cand, &cval);
    try testing.expectEqualSlices(i32, &.{ 10, 11, 20, 21, 12, 13, 22, 23 }, &cand);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 3, 4, 7, 8 }, &cval);
}

test "the speculative pass's device start: ds_stage's arithmetic over the placeholder staging == the host's statics" {
    // afterPick's device staging (glue.cu ds_stage, mirrored by stageMirror) must equal stageOne(bonus, next position)
    const dspark = @import("draft/dspark.zig");
    const cfg: Config = .{};
    const st: emit.Statics = .{ .n = cfg.dspark_block, .window = cfg.window };
    const ring: u64 = @intCast(emit.ringRows(&cfg));
    const shape: dspark.Shape = .{ .block = cfg.dspark_block, .window = cfg.window };
    var want64: [64]i64 = undefined;
    var want32: [1024]i32 = undefined;
    var got64: [64]i64 = undefined;
    var got32: [1024]i32 = undefined;
    const l64: usize = @intCast(st.len64());
    const l32: usize = @intCast(st.len32());
    for ([_]u64{ 1, 2, 5, 127, 128, 129, 200, 255, 256, 257, 4000, 1 << 20 }) |P| {
        for ([_]u64{ 0, 1, 64, 150, 3999 }) |valid| {
            if (valid > P) continue;
            for ([_]u32{ 1, 3, 6 }) |nw| {
                const window_start = P - @min(P, nw); // the window the pick came from (any start below P)
                var pick: [16]i64 = @splat(0);
                pick[nw] = 2; // accepted (unused by the statics)
                pick[nw + 1] = 77_123; // bonus
                pick[nw + 2] = @intCast(P);
                rows.stageOne(st, ring, cfg.dspark_noise_token, valid, 77_123, P, want64[0..l64], want32[0..l32]);
                rows.stageOne(st, ring, cfg.dspark_noise_token, valid, 0, window_start, got64[0..l64], got32[0..l32]);
                rows.stageMirror(st, ring, cfg.dspark_noise_token, valid, &pick, nw, got64[0..l64], got32[0..l32]);
                try testing.expectEqualSlices(i64, want64[0..l64], got64[0..l64]);
                try testing.expectEqualSlices(i32, want32[0..l32], got32[0..l32]);
                // stageOne's context == Context.of's (the old fillStatics' rule)
                const c = dspark.Context.of(shape, P, valid);
                try testing.expectEqual(@as(i32, @intCast(c.count)), want32[@intCast(st.counts())]);
                if (c.count > 0) try testing.expectEqual(@as(i32, @intCast(c.first % ring)), want32[@intCast(st.tokens())]);
            }
        }
    }
}

test "the slot passes' device start: ds_accept_rows / ds_stage_rows over the placeholder staging == the host's statics" {
    // dspark_slots.afterPick: each armed slot's statics rewritten from its picks must equal stage() at (bonus, next)
    const cfg: Config = .{};
    const n: u32 = cfg.dspark_block;
    const W: u32 = cfg.window;
    const ring: u32 = @intCast(emit.ringRows(&cfg));
    const noise: u32 = cfg.dspark_noise_token;
    // a 4-slot row window: slot 2 (rows 0-5), slot 0 (6-8), slot 3 (9-14), slot 1 (15)
    const segs = [_][3]i64{ .{ 0, 6, 300 }, .{ 6, 3, 129 }, .{ 9, 6, 5000 }, .{ 15, 1, 40 } };
    const seg_slot = [_]u32{ 2, 0, 3, 1 };
    var ids: [16]i64 = undefined;
    var picks: [16]i64 = undefined;
    for (&ids, &picks, 0..) |*t, *q, i| {
        t.* = @intCast(1000 + i);
        q.* = @intCast(5000 + i);
    }
    // accepted 3 for slot 2 (picks 0..2 equal the next rows' ids), 0 for slot 0, 5 (all) for slot 3, 0 for slot 1
    for (0..3) |i| picks[i] = ids[i + 1];
    for (9..14) |i| picks[i] = ids[i + 1];
    var acc: [4][3]i64 = undefined;
    rows.acceptMirror(&picks, &ids, &segs, &acc);
    try testing.expectEqual([3]i64{ 3, picks[3], 304 }, acc[0]);
    try testing.expectEqual([3]i64{ 0, picks[6], 130 }, acc[1]);
    try testing.expectEqual([3]i64{ 5, picks[14], 5006 }, acc[2]);
    try testing.expectEqual([3]i64{ 0, picks[15], 41 }, acc[3]);
    const valids = [4]u64{ 0, 100, 4990, 0 };
    // groups of 1-4 members in sorted-slot order, as afterPick forms them
    for ([_][2]usize{ .{ 0, 4 }, .{ 0, 2 }, .{ 2, 4 }, .{ 1, 2 } }) |g| {
        const gk = g[1] - g[0];
        const lay = emit.Layout.of(@intCast(gk), n, W);
        var want64: [256]i64 = undefined;
        var want32: [8192]i32 = undefined;
        var got64: [256]i64 = undefined;
        var got32: [8192]i32 = undefined;
        var ms_want: [4]rows.Member = undefined;
        var ms_ph: [4]rows.Member = undefined;
        var mem: [4][3]i64 = undefined;
        for (g[0]..g[1], 0..) |j, i| {
            const sl = seg_slot[j];
            ms_want[i] = .{ .slot = sl, .anchor = @intCast(acc[j][1]), .start = @intCast(acc[j][2]), .valid = valids[j] };
            ms_ph[i] = .{ .slot = sl, .anchor = 0, .start = @intCast(segs[j][2]), .valid = valids[j] };
            mem[i] = .{ @intCast(j), sl, @intCast(valids[j]) };
        }
        const l64: usize = @intCast(lay.len64);
        const l32: usize = @intCast(lay.len32);
        rows.stage(lay, n, W, ring, noise, ms_want[0..gk], want64[0..l64], want32[0..l32]);
        rows.stage(lay, n, W, ring, noise, ms_ph[0..gk], got64[0..l64], got32[0..l32]);
        rows.stageRowsMirror(lay, n, W, ring, noise, &acc, mem[0..gk], got64[0..l64], got32[0..l32]);
        try testing.expectEqualSlices(i64, want64[0..l64], got64[0..l64]);
        try testing.expectEqualSlices(i32, want32[0..l32], got32[0..l32]);
    }
}
