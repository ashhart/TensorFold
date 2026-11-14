//! A window's plan path against its per-row path at Flash Next's shape: the same rows, y, state and conv tail.
const std = @import("std");
const rig = @import("rig.zig");
const plan = @import("../launches/plan.zig");
const plan_ops = @import("../ops/plan.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Ops = @import("../ops/ops.zig").Ops;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const key_heads = 16;
const value_heads = 48;
const dk = 128;
const dv = 128;
const conv = 4;
const taps = conv - 1;
const channels = 2 * key_heads * dk + value_heads * dv;
const kw = key_heads * dk;
const vw = value_heads * dv;
const sections = kw + vw + 2 * value_heads;
const vh = value_heads;
const eps = 1e-6;

/// A plan's words: a slot's first row, its rows, its kept rows and its descriptor, then an empty slot.
const Words = struct {
    first: DeviceBuffer,
    count: DeviceBuffer,
    keep: DeviceBuffer,
    desc: DeviceBuffer,
    ptrs: DeviceBuffer,
    snaps: DeviceBuffer,
    slots: usize,

    /// `firsts` a slot's first row, `counts` its rows, `descs` its descriptor pair; the last slot must be empty.
    fn make(t: *Rig, firsts: []const u32, counts: []const u32, descs: []const [4]u64, snaps: [2]u64) !Words {
        var w: Words = .{
            .first = try t.upload(firsts),
            .count = undefined,
            .keep = undefined,
            .desc = undefined,
            .ptrs = undefined,
            .snaps = undefined,
            .slots = firsts.len,
        };
        errdefer w.first.free();
        w.count = try t.upload(counts);
        errdefer w.count.free();
        w.keep = try t.alloc(firsts.len * 4);
        errdefer w.keep.free();
        w.desc = try t.alloc(descs.len * 4 * 8);
        errdefer w.desc.free();
        w.ptrs = try t.alloc(firsts.len * 8);
        errdefer w.ptrs.free();
        w.snaps = try t.alloc(2 * 8);
        errdefer w.snaps.free();
        for (descs, 0..) |d, i| try w.desc.upload(i * 32, std.mem.sliceAsBytes(&d));
        const held = try gpa.alloc(u64, firsts.len);
        defer gpa.free(held);
        for (held, 0..) |*p, i| p.* = at(w.desc) + i * 32;
        try w.ptrs.upload(0, std.mem.sliceAsBytes(held));
        try w.snaps.upload(0, std.mem.sliceAsBytes(&snaps));
        const marks = try gpa.alloc(i32, firsts.len);
        defer gpa.free(marks);
        @memset(marks, -1);
        try w.keep.upload(0, std.mem.sliceAsBytes(marks));
        return w;
    }

    /// The empty slot after the window's: no rows, no state, no kept inputs.
    fn none() [4]u64 {
        return .{ 0, 0, 0, 0 };
    }

    fn ref(w: *const Words, rows: usize) plan.PlanRef {
        return .{
            .rows = rows,
            .slots = w.slots,
            .args = .{
                .first = at(w.first),
                .count = at(w.count),
                .desc = at(w.ptrs),
                .snaps = at(w.snaps),
            },
        };
    }

    /// The kept rows `n` of slot 0 (`-1` leaves it alone).
    fn mark(w: *const Words, o: Ops, n: i32) !void {
        try w.keep.upload(0, std.mem.sliceAsBytes(&[_]i32{ n, -1 }));
        _ = o;
    }

    fn free(w: *Words) void {
        for ([_]*DeviceBuffer{ &w.snaps, &w.ptrs, &w.desc, &w.keep, &w.count, &w.first }) |b| b.free();
    }
};

test "a window's split from its plan gives the per-row launch's q, k and v bits, at one to four rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    const act_kind: @import("../ops/types.zig").Kind = if (o.bf16) .bf16 else .f16;
    inline for (1..5) |rows| {
        var rng: rig.Rng = .{ .state = 0xC04 + rows };
        const hx = try gpa.alloc(u16, rows * channels);
        defer gpa.free(hx);
        for (hx) |*e| e.* = t.bits(rng.unit() * 2);
        const hw = try gpa.alloc(f32, channels * conv);
        defer gpa.free(hw);
        for (hw) |*e| e.* = rng.unit() / 4;
        const hs = try gpa.alloc(f32, taps * channels);
        defer gpa.free(hs);
        for (hs) |*e| e.* = rng.unit();
        var x = try t.upload(hx);
        defer x.free();
        var w = try t.upload(hw);
        defer w.free();
        var state = try t.upload(hs);
        defer state.free();
        var states = try t.alloc(rows * taps * channels * 4);
        defer states.free();
        var a: [3]DeviceBuffer = .{
            try t.alloc(rows * kw * 4),
            try t.alloc(rows * kw * 4),
            try t.alloc(rows * vw * 4),
        };
        defer for (&a) |*b| b.free();
        var b: [3]DeviceBuffer = .{
            try t.alloc(rows * kw * 4),
            try t.alloc(rows * kw * 4),
            try t.alloc(rows * vw * 4),
        };
        defer for (&b) |*x2| x2.free();
        var samples = try t.alloc(rows * channels * 4);
        defer samples.free();
        var tail = try t.upload(hs);
        defer tail.free();
        var words = try Words.make(t, &.{ 0, 0 }, &.{ @intCast(rows), 0 }, &.{ .{ 0, 0, at(tail), 0 }, Words.none() }, .{ at(samples), 0 });
        defer words.free();
        // the head norm's weights, and the gate's a, b, a_log and dt_bias, which it writes gate and beta from
        const hq_norm = try gpa.alloc(f32, kw);
        defer gpa.free(hq_norm);
        for (hq_norm) |*e| e.* = rng.unit() + 1;
        const hk_norm = try gpa.alloc(f32, kw);
        defer gpa.free(hk_norm);
        for (hk_norm) |*e| e.* = rng.unit() + 1;
        const ga = try gpa.alloc(u16, rows * vh);
        defer gpa.free(ga);
        for (ga) |*e| e.* = t.bits(rng.unit() * 2 - 1);
        const gb = try gpa.alloc(u16, rows * vh);
        defer gpa.free(gb);
        for (gb) |*e| e.* = t.bits(rng.unit() * 2 - 1);
        const a_log = try gpa.alloc(f32, vh);
        defer gpa.free(a_log);
        for (a_log) |*e| e.* = rng.unit() - 1;
        const dt_bias = try gpa.alloc(f32, vh);
        defer gpa.free(dt_bias);
        for (dt_bias) |*e| e.* = rng.unit();
        var qw = try t.upload(hq_norm);
        defer qw.free();
        var kw_w = try t.upload(hk_norm);
        defer kw_w.free();
        var ag = try t.upload(ga);
        defer ag.free();
        var bg = try t.upload(gb);
        defer bg.free();
        var alog = try t.upload(a_log);
        defer alog.free();
        var dt = try t.upload(dt_bias);
        defer dt.free();
        var gate_a = try t.alloc(rows * vh * 4);
        defer gate_a.free();
        var beta_a = try t.alloc(rows * vh * 4);
        defer beta_a.free();
        var gate_b = try t.alloc(rows * vh * 4);
        defer gate_b.free();
        var beta_b = try t.alloc(rows * vh * 4);
        defer beta_b.free();
        // today's window: one launch over the rows, each row's conv tail kept aside
        try o.l.tf_conv_split(.{
            .x = at(x),
            .kind = if (o.bf16) 2 else 1,
            .weight = at(w),
            .state = at(state),
            .states = at(states),
            .qn = at(a[0]),
            .kn = at(a[1]),
            .v = at(a[2]),
            .channels = channels,
            .kernel = conv,
            .rows = @intCast(rows),
            .kw = kw,
            .vw = vw,
            .qw = at(qw),
            .kw_w = at(kw_w),
            .eps = eps,
            .norm = 128,
            .ga = at(ag),
            .gb = at(bg),
            .a_log = at(alog),
            .dt_bias = at(dt),
            .gate = at(gate_a),
            .beta = at(beta_a),
            .gcount = @intCast(rows * vh),
            .heads = vh,
        }, o.stream);
        // the plan's: the same launch from the window's words, which leaves the state where it found it
        try plan_ops.convSplit(
            o,
            .{ .ptr = at(x), .kind = if (o.bf16) .bf16 else .f16 },
            at(w),
            at(b[0]),
            at(b[1]),
            at(b[2]),
            words.ref(rows),
            0,
            channels,
            conv,
            kw,
            vw,
            .{ .q = at(qw), .k = at(kw_w), .eps = eps },
            .{ .a = .{ .ptr = at(ag), .kind = act_kind }, .b = .{ .ptr = at(bg), .kind = act_kind }, .a_log = at(alog), .dt_bias = at(dt), .gate = at(gate_b), .beta = at(beta_b), .count = rows * vh, .heads = vh },
        );
        try t.stream.synchronize();
        for ([_][]const u8{ "q", "k", "v" }, 0..) |what, i| {
            const want = try rig.download(u32, a[i]);
            defer gpa.free(want);
            const got = try rig.download(u32, b[i]);
            defer gpa.free(got);
            if (!std.mem.eql(u32, want, got)) std.debug.print("{s} differs at {d} rows\n", .{ what, rows });
            try std.testing.expectEqualSlices(u32, want, got);
        }
        for ([_][]const u8{ "gate", "beta" }, [_]DeviceBuffer{ gate_a, beta_a }, [_]DeviceBuffer{ gate_b, beta_b }) |what, want_b, got_b| {
            const want = try rig.download(u32, want_b);
            defer gpa.free(want);
            const got = try rig.download(u32, got_b);
            defer gpa.free(got);
            if (!std.mem.eql(u32, want, got)) std.debug.print("{s} differs at {d} rows\n", .{ what, rows });
            try std.testing.expectEqualSlices(u32, want, got);
        }
        // the kept rows' tails come from the conv inputs: a keep takes row n's tail from the samples it kept
        for (1..rows + 1) |n| {
            try words.mark(o, @intCast(n));
            try plan_ops.keep(o, words.ref(rows), .{
                .keep = at(words.keep),
                .hidden = 0,
                .hidden_words = 0,
                .channels = channels,
                .taps = taps,
                .layers = 1,
            });
            try t.stream.synchronize();
            const want = try rig.download(u32, .{ .r = states.r, .ptr = @ptrFromInt(at(states) + (n - 1) * taps * channels * 4), .len = taps * channels * 4 });
            defer gpa.free(want);
            const got = try rig.download(u32, tail);
            defer gpa.free(got);
            if (!std.mem.eql(u32, want, got)) std.debug.print("conv tail differs after {d} of {d} rows\n", .{ n, rows });
            try std.testing.expectEqualSlices(u32, want, got);
            try tail.upload(0, std.mem.sliceAsBytes(hs));
        }
    }
}

test "a window's delta rule from its plan, the first n rows kept, leaves the state today's snapshot of row n holds" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    inline for (1..5) |rows| {
        var rng: rig.Rng = .{ .state = 0x6D3 + rows };
        const hq = try gpa.alloc(f32, rows * kw);
        defer gpa.free(hq);
        const hk = try gpa.alloc(f32, rows * kw);
        defer gpa.free(hk);
        const hv = try gpa.alloc(f32, rows * vw);
        defer gpa.free(hv);
        const hg = try gpa.alloc(f32, rows * value_heads);
        defer gpa.free(hg);
        const hb = try gpa.alloc(f32, rows * value_heads);
        defer gpa.free(hb);
        const hs = try gpa.alloc(f32, value_heads * dk * dv);
        defer gpa.free(hs);
        for ([_][]f32{ hq, hk, hv }) |set| {
            for (set) |*e| e.* = rng.unit();
        }
        for (hg) |*e| e.* = 0.5 + 0.25 * @abs(rng.unit());
        for (hb) |*e| e.* = 0.5 + 0.25 * @abs(rng.unit());
        for (hs) |*e| e.* = 0.1 * rng.unit();
        var q = try t.upload(hq);
        defer q.free();
        var k = try t.upload(hk);
        defer k.free();
        var v = try t.upload(hv);
        defer v.free();
        var gate = try t.upload(hg);
        defer gate.free();
        var beta = try t.upload(hb);
        defer beta.free();
        var serial_state = try t.upload(hs);
        defer serial_state.free();
        const snap = value_heads * dk * dv;
        var snaps_a = try t.alloc(rows * snap * 4);
        defer snaps_a.free();
        var y_a = try t.alloc(rows * vw * 4);
        defer y_a.free();
        var planned_state = try t.upload(hs);
        defer planned_state.free();
        // the window's start state, kept for comparison: a round leaves it there
        var start = try t.upload(hs);
        defer start.free();
        var y_b = try t.alloc(rows * vw * 4);
        defer y_b.free();
        // the rows the conv split keeps: k, v, gate and beta one section after another
        const hkp = try gpa.alloc(f32, rows * sections);
        defer gpa.free(hkp);
        @memcpy(hkp[0 .. rows * kw], hk);
        @memcpy(hkp[rows * kw .. rows * (kw + vw)], hv);
        @memcpy(hkp[rows * (kw + vw) .. rows * (kw + vw) + rows * value_heads], hg);
        @memcpy(hkp[rows * (kw + vw) + rows * value_heads ..], hb);
        var kept = try t.upload(hkp);
        defer kept.free();
        var words = try Words.make(t, &.{ 0, 0 }, &.{ @intCast(rows), 0 }, &.{ .{ 0, 0, 0, at(planned_state) }, Words.none() }, .{ 0, at(kept) });
        defer words.free();
        try o.gatedDelta(
            at(q),
            at(k),
            at(v),
            at(gate),
            at(beta),
            at(serial_state),
            at(y_a),
            rows,
            key_heads,
            value_heads,
            dk,
            dv,
            at(snaps_a),
        );
        try plan_ops.gatedDelta(o, at(q), at(k), at(v), at(gate), at(beta), at(y_b), words.ref(rows), 0, key_heads, value_heads, dk, dv);
        try t.stream.synchronize();
        const want_y = try rig.download(u32, y_a);
        defer gpa.free(want_y);
        const got_y = try rig.download(u32, y_b);
        defer gpa.free(got_y);
        try std.testing.expectEqualSlices(u32, want_y, got_y);
        // a window of one row writes its state back in place; a longer one leaves it as the round found it
        const round_state = try rig.download(u32, planned_state);
        defer gpa.free(round_state);
        const row1 = try rig.download(u32, .{ .r = snaps_a.r, .ptr = @ptrFromInt(at(snaps_a)), .len = snap * 4 });
        defer gpa.free(row1);
        if (rows == 1) {
            // the round is its own keep: the state it holds is the serial run's after its one row
            try std.testing.expectEqualSlices(u32, row1, round_state);
            // and a keep replays nothing (the kernel stops at a one-row window): the state stays that one
            try words.mark(o, 1);
            try plan_ops.gdnReplay(o, at(words.keep), words.ref(rows), key_heads, value_heads, dk, dv, 1);
            try t.stream.synchronize();
            const got = try rig.download(u32, planned_state);
            defer gpa.free(got);
            try std.testing.expectEqualSlices(u32, row1, got);
        } else {
            const as_found = try rig.download(u32, start);
            defer gpa.free(as_found);
            try std.testing.expectEqualSlices(u32, as_found, round_state);
            for (1..rows + 1) |n| {
                // a round leaves the state where it found it: a keep replays from the window's start
                try planned_state.upload(0, std.mem.sliceAsBytes(hs));
                try words.mark(o, @intCast(n));
                try plan_ops.gdnReplay(o, at(words.keep), words.ref(rows), key_heads, value_heads, dk, dv, 1);
                try t.stream.synchronize();
                const want = try rig.download(u32, .{ .r = snaps_a.r, .ptr = @ptrFromInt(at(snaps_a) + (n - 1) * snap * 4), .len = snap * 4 });
                defer gpa.free(want);
                const got = try rig.download(u32, planned_state);
                defer gpa.free(got);
                if (!std.mem.eql(u32, want, got)) std.debug.print("state differs after {d} of {d} rows\n", .{ n, rows });
                try std.testing.expectEqualSlices(u32, want, got);
            }
        }
    }
}

test "two streams' plans keep their own rows: each state as its serial run's snapshot of the kept row holds it" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    const counts = [2]usize{ 2, 3 };
    const snap = value_heads * dk * dv;
    var rng: rig.Rng = .{ .state = 0x77E };
    const hs = try gpa.alloc(f32, snap);
    defer gpa.free(hs);
    for (hs) |*e| e.* = 0.1 * rng.unit();
    // a stream's rows: q, k, v, gate and beta, and the ones the conv split keeps (k, v, gate, beta one section each)
    var q: [2]DeviceBuffer = undefined;
    var k: [2]DeviceBuffer = undefined;
    var v: [2]DeviceBuffer = undefined;
    var gate: [2]DeviceBuffer = undefined;
    var beta: [2]DeviceBuffer = undefined;
    var kept: [2]DeviceBuffer = undefined;
    var serial: [2]DeviceBuffer = undefined;
    var planned: [2]DeviceBuffer = undefined;
    var ys: [2]DeviceBuffer = undefined;
    var snaps: [2]DeviceBuffer = undefined;
    var words: [2]Words = undefined;
    var made: usize = 0;
    errdefer for ([_]*DeviceBuffer{
        &q[0],    &q[1],    &k[0],      &k[1],      &v[0],       &v[1],       &gate[0], &gate[1], &beta[0],  &beta[1],
        &kept[0], &kept[1], &serial[0], &serial[1], &planned[0], &planned[1], &ys[0],   &ys[1],   &snaps[0], &snaps[1],
    }) |b| b.free();
    defer for (words[0..made]) |*w| w.free();
    for (0..2) |i| {
        const n = counts[i];
        const hq = try gpa.alloc(f32, n * kw);
        defer gpa.free(hq);
        const hk = try gpa.alloc(f32, n * kw);
        defer gpa.free(hk);
        const hv = try gpa.alloc(f32, n * vw);
        defer gpa.free(hv);
        const hg = try gpa.alloc(f32, n * value_heads);
        defer gpa.free(hg);
        const hb = try gpa.alloc(f32, n * value_heads);
        defer gpa.free(hb);
        for ([_][]f32{ hq, hk, hv }) |set| {
            for (set) |*e| e.* = rng.unit();
        }
        for (hg) |*e| e.* = 0.5 + 0.25 * @abs(rng.unit());
        for (hb) |*e| e.* = 0.5 + 0.25 * @abs(rng.unit());
        const hk_ = try gpa.alloc(f32, n * sections);
        defer gpa.free(hk_);
        @memcpy(hk_[0 .. n * kw], hk);
        @memcpy(hk_[n * kw .. n * (kw + vw)], hv);
        @memcpy(hk_[n * (kw + vw) .. n * (kw + vw) + n * value_heads], hg);
        @memcpy(hk_[n * (kw + vw) + n * value_heads ..], hb);
        q[i] = try t.upload(hq);
        k[i] = try t.upload(hk);
        v[i] = try t.upload(hv);
        gate[i] = try t.upload(hg);
        beta[i] = try t.upload(hb);
        kept[i] = try t.upload(hk_);
        serial[i] = try t.upload(hs);
        planned[i] = try t.upload(hs);
        ys[i] = try t.alloc(n * vw * 4);
        snaps[i] = try t.alloc(n * snap * 4);
        words[i] = try Words.make(
            t,
            &.{ 0, 0 },
            &.{ @intCast(n), 0 },
            &.{ .{ 0, 0, 0, at(planned[i]) }, Words.none() },
            .{ 0, at(kept[i]) },
        );
        made += 1;
        // the stream's serial run, a snapshot a row, and its own plan's, in the same launch
        try o.gatedDelta(at(q[i]), at(k[i]), at(v[i]), at(gate[i]), at(beta[i]), at(serial[i]), at(ys[i]), n, key_heads, value_heads, dk, dv, at(snaps[i]));
        try plan_ops.gatedDelta(o, at(q[i]), at(k[i]), at(v[i]), at(gate[i]), at(beta[i]), at(ys[i]), words[i].ref(n), 0, key_heads, value_heads, dk, dv);
    }
    try t.stream.synchronize();
    for (0..2) |i| {
        const n = counts[i];
        for (1..n + 1) |j| {
            try words[i].keep.upload(0, std.mem.sliceAsBytes(&[_]i32{ @intCast(j), -1 }));
            try planned[i].upload(0, std.mem.sliceAsBytes(hs));
            try plan_ops.gdnReplay(o, at(words[i].keep), words[i].ref(n), key_heads, value_heads, dk, dv, 1);
            try t.stream.synchronize();
            const want = try rig.download(u32, .{ .r = snaps[i].r, .ptr = @ptrFromInt(at(snaps[i]) + (j - 1) * snap * 4), .len = snap * 4 });
            defer gpa.free(want);
            const got = try rig.download(u32, planned[i]);
            defer gpa.free(got);
            if (!std.mem.eql(u32, want, got)) std.debug.print("stream {d} kept {d} of {d}: state differs\n", .{ i, j, n });
            try std.testing.expectEqualSlices(u32, want, got);
        }
    }
}
