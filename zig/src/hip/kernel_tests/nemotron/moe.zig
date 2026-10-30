//! Nemotron-H's MoE on the GPU: the pick and the whole layer against float64, and a row's bytes however rows are cut.

const std = @import("std");
const rig = @import("../rig.zig");
const fx = @import("fixture.zig");
const nm = @import("../../moe_nemotron.zig");
const ops = @import("../../ops/ops.zig");

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// The real layer's shape (Nemotron 3.5 Lightning 30B-A3B), and smaller ones for many row counts.
const lightning: fx.Shape = .{ .experts = 128, .top_k = 6, .hidden = 2688, .width = 1856 };
const small: fx.Shape = .{ .experts = 64, .top_k = 6, .hidden = 512, .width = 256 };

/// A pick within this of the float64 order is a tie the fp32 logits may break either way.
const pick_tol = 1e-5;

/// Worst errors of one run against float64: pick order, weight over the scaling, output over its row's largest.
const Errors = struct {
    pick: f64 = 0,
    weight: f64 = 0,
    out: f64 = 0,
    /// Picks that break the rule outright: a repeat, an id out of range, a tie taken by the higher id.
    bad: usize = 0,

    fn merge(a: *Errors, b: Errors) void {
        a.pick = @max(a.pick, b.pick);
        a.weight = @max(a.weight, b.weight);
        a.out = @max(a.out, b.out);
        a.bad += b.bad;
    }
};

/// Bounds on Errors: the output's is a few roundings of the activation (relu^2's input and output, the sum's).
fn outBound(t: *const Rig) f64 {
    return if (t.bf16) 1.0 / 64.0 else 1.0 / 256.0;
}

fn ok(t: *const Rig, e: Errors) bool {
    return e.bad == 0 and e.pick <= pick_tol and e.weight <= 1e-5 and e.out <= outBound(t);
}

/// Runs the router and pick (and with `full` the whole layer) on rows [first, first + rows), checked against float64.
fn measure(
    t: *Rig,
    o: ops.Ops,
    l: *const fx.Layer,
    view: nm.Layer,
    x: fx.Input,
    first: usize,
    rows: usize,
    full: bool,
) !Errors {
    const s = l.shape;
    const d = s.hidden;
    const e = s.experts;
    const slots = view.top_k;
    o.arena.reset();
    const xt = x.tensor(t, first, d);
    const logits = try o.arena.of(f32, rows * e);
    const pick = try o.arena.of(i32, rows * slots);
    const wts = try o.arena.of(f32, rows * slots);
    try o.moeRouter(xt, view.router, logits, rows, d, e);
    try nm.select(o, view, logits, pick, wts, null, 0, rows);
    const out: u64 = if (full) (try nm.run(o, view, xt, rows)).ptr else 0;
    const hp = try fx.bytesAt(t, pick, rows * slots * 4);
    defer gpa.free(hp);
    const hw = try fx.bytesAt(t, wts, rows * slots * 4);
    defer gpa.free(hw);
    const ho = try fx.bytesAt(t, out, if (full) rows * d * 2 else 0);
    defer gpa.free(ho);
    const picks: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, hp));
    const weights: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, hw));
    const got: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, ho));
    const score = try gpa.alloc(f64, 2 * e);
    defer gpa.free(score);
    const sel = score[e..];
    const want = try gpa.alloc(f64, d + slots);
    defer gpa.free(want);
    var errs: Errors = .{};
    for (0..rows) |r| {
        const xr = x.values[(first + r) * d ..][0..d];
        l.scores(xr, score[0..e], sel);
        const pr = picks[r * slots ..][0..slots];
        const wr = weights[r * slots ..][0..slots];
        const taken = try gpa.alloc(bool, e);
        defer gpa.free(taken);
        @memset(taken, false);
        var total: f64 = 0;
        for (pr[0..s.top_k], 0..) |p, k| {
            if (p < 0 or p >= e or taken[@intCast(p)]) {
                errs.bad += 1;
                continue;
            }
            taken[@intCast(p)] = true;
            total += score[@intCast(p)];
            if (k > 0) errs.pick = @max(errs.pick, sel[@intCast(p)] - sel[@intCast(pr[k - 1])]);
        }
        if (errs.bad > 0) return errs;
        var least: f64 = std.math.inf(f64);
        for (pr[0..s.top_k]) |p| least = @min(least, sel[@intCast(p)]);
        for (0..e) |i| if (!taken[i]) {
            errs.pick = @max(errs.pick, sel[i] - least);
        };
        // ties: an expert's twin past E / 2 never goes ahead of it
        if (s.ties) for (pr[0..s.top_k], 0..) |p, k| if (p >= e / 2) {
            const twin = p - @as(i32, @intCast(e / 2));
            if (std.mem.indexOfScalar(i32, pr[0..k], twin) == null) errs.bad += 1;
        };
        for (pr[0..s.top_k], 0..) |p, k| {
            const sc = score[@intCast(p)];
            want[d + k] = (if (s.norm) sc / total else sc) * s.scaling;
            errs.weight = @max(errs.weight, @abs(wr[k] - want[d + k]) / s.scaling);
        }
        if (!full) continue;
        try l.output(xr, pr[0..s.top_k], want[d..][0..s.top_k], want[0..d]);
        var top: f64 = 0;
        for (want[0..d]) |v| top = @max(top, @abs(v));
        for (0..d) |c| errs.out = @max(errs.out, @abs(t.value(got[r * d + c]) - want[c]) / top);
    }
    return errs;
}

fn report(what: []const u8, t: *const Rig, e: Errors) void {
    std.debug.print(
        "  nemotron moe {s} ({s}): pick {e:.2}, weight {e:.2}, out {e:.2} of bound {e:.2}, bad {d}\n",
        .{ what, if (t.bf16) "bf16" else "fp16", e.pick, e.weight, e.out, outBound(t), e.bad },
    );
}

test "the Nemotron pick ranks sigmoid scores plus bias, the lower id on a tie, and weights the scores scaled" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x51ED };
    const shapes = [_]fx.Shape{
        small,
        .{ .experts = 128, .top_k = 6, .hidden = 2688, .width = 64 },
        .{ .experts = 32, .top_k = 8, .hidden = 256, .width = 64, .norm = false, .scaling = 1.0 },
        .{ .experts = 64, .top_k = 6, .hidden = 256, .width = 64, .ties = true },
        .{ .experts = 256, .top_k = 22, .hidden = 128, .width = 64 },
    };
    for (shapes) |s| {
        const l = try fx.Layer.init(t, &rng, s);
        defer l.deinit();
        var x = try fx.Input.init(t, &rng, 40, s.hidden);
        defer x.deinit();
        var all: Errors = .{};
        for ([_]bool{ false, true }) |window| for ([_]usize{ 1, 2, 5, 16, 40 }) |rows| {
            var o = t.ops(&t.on);
            o.window = window;
            all.merge(try measure(t, o, l, l.view, x, 0, rows, false));
        };
        report("pick", t, all);
        try rig.expect(ok(t, all), "nemotron pick E{d} top {d}: out of bounds", .{ s.experts, s.top_k });
    }
}

test "a one-row Nemotron pick writes its plan: an item a slot" {
    const t = try Rig.open(16 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xA11 };
    const l = try fx.Layer.init(t, &rng, small);
    defer l.deinit();
    var x = try fx.Input.init(t, &rng, 1, small.hidden);
    defer x.deinit();
    const o = t.ops(&t.on);
    const slots = l.view.top_k;
    const cap = 20;
    const logits = try o.arena.of(f32, small.experts);
    const pick = try o.arena.of(i32, slots);
    const wts = try o.arena.of(f32, slots);
    const items = try o.arena.of(i32, 3 * cap);
    const members = try o.arena.of(i32, slots);
    try o.moeRouter(x.tensor(t, 0, small.hidden), l.view.router, logits, 1, small.hidden, small.experts);
    try nm.select(o, l.view, logits, pick, wts, .{ .items = items, .members = members }, cap, 1);
    const hp = try fx.bytesAt(t, pick, slots * 4);
    defer gpa.free(hp);
    const hi = try fx.bytesAt(t, items, 3 * cap * 4);
    defer gpa.free(hi);
    const hm = try fx.bytesAt(t, members, slots * 4);
    defer gpa.free(hm);
    const p: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, hp));
    const it: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, hi));
    const m: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, hm));
    for (0..cap) |k| {
        const want = if (k < slots) [3]i32{ p[k], @intCast(k), 1 } else [3]i32{ 0, 0, 0 };
        try std.testing.expectEqualSlices(i32, &want, it[3 * k ..][0..3]);
    }
    for (m, 0..) |v, k| try std.testing.expectEqual(@as(i32, @intCast(k)), v);
}

test "the Nemotron MoE layer sits within a few activation roundings of float64, decode, rounds and prompts" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xBEEF };
    for ([_]fx.Shape{
        small,
        .{ .experts = 32, .top_k = 8, .hidden = 256, .width = 128, .norm = false, .scaling = 1.0 },
    }) |s| {
        const l = try fx.Layer.init(t, &rng, s);
        defer l.deinit();
        var x = try fx.Input.init(t, &rng, 80, s.hidden);
        defer x.deinit();
        var all: Errors = .{};
        for ([_]usize{ 1, 2, 7, 16 }) |rows| {
            var o = t.ops(&t.on);
            o.window = rows > 1;
            all.merge(try measure(t, o, l, l.view, x, 0, rows, true));
        }
        for ([_]usize{ 1, 16, 80 }) |rows| {
            var o = t.ops(&t.on);
            o.prefill = true;
            all.merge(try measure(t, o, l, l.view, x, 0, rows, true));
        }
        all.merge(try measure(t, t.ops(&t.off), l, l.view, x, 0, 3, true));
        report("layer", t, all);
        try rig.expect(ok(t, all), "nemotron layer E{d} D{d}: out of bounds", .{ s.experts, s.hidden });
    }
    // the served shape on a few rows: the float64 sums are long
    const l = try fx.Layer.init(t, &rng, lightning);
    defer l.deinit();
    var x = try fx.Input.init(t, &rng, 2, lightning.hidden);
    defer x.deinit();
    var all: Errors = .{};
    all.merge(try measure(t, t.ops(&t.on), l, l.view, x, 0, 1, true));
    var o = t.ops(&t.on);
    o.window = true;
    all.merge(try measure(t, o, l, l.view, x, 0, 2, true));
    report("layer lightning", t, all);
    try rig.expect(ok(t, all), "nemotron layer lightning: out of bounds", .{});
}

test "a layer whose pick drops the correction bias, or its normalization, fails the float64 check" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xBAD };
    const l = try fx.Layer.init(t, &rng, small);
    defer l.deinit();
    var x = try fx.Input.init(t, &rng, 16, small.hidden);
    defer x.deinit();
    var zeros = try t.alloc(small.experts * 4);
    defer zeros.free();
    try zeros.fill8(0);
    var no_bias = l.view;
    no_bias.bias = at(zeros);
    var no_norm = l.view;
    no_norm.norm = false;
    for ([_]nm.Layer{ no_bias, no_norm }, [_][]const u8{ "no bias", "no norm" }) |broken, name| {
        const e = try measure(t, t.ops(&t.on), l, broken, x, 0, 16, true);
        report(name, t, e);
        try rig.expect(!ok(t, e), "the broken pick ({s}) passed the float64 check", .{name});
    }
}

/// The layer's output bytes for rows [first, first + rows) under `o`.
fn layerBytes(t: *Rig, o: ops.Ops, view: nm.Layer, x: fx.Input, first: usize, rows: usize) ![]u8 {
    o.arena.reset();
    const d: usize = view.fc2.n;
    const y = try nm.run(o, view, x.tensor(t, first, d), rows);
    return fx.bytesAt(t, y.ptr, rows * d * 2);
}

/// Rows of `got` (from row `first` of `want`) whose bytes differ.
fn differ(got: []const u8, want: []const u8, first: usize, row_bytes: usize) usize {
    var n: usize = 0;
    for (0..got.len / row_bytes) |r| n += @intFromBool(!std.mem.eql(
        u8,
        got[r * row_bytes ..][0..row_bytes],
        want[(first + r) * row_bytes ..][0..row_bytes],
    ));
    return n;
}

test "a Nemotron MoE row writes the same bytes alone and in a round of 1 to 16, and in every prompt cut" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xC0FFEE };
    const l = try fx.Layer.init(t, &rng, lightning);
    defer l.deinit();
    const d = lightning.hidden;
    const row_bytes = d * 2;
    var x = try fx.Input.init(t, &rng, 80, d);
    defer x.deinit();
    const alone = try gpa.alloc(u8, 16 * row_bytes);
    defer gpa.free(alone);
    for (0..16) |r| {
        const b = try layerBytes(t, t.ops(&t.on), l.view, x, r, 1);
        defer gpa.free(b);
        @memcpy(alone[r * row_bytes ..][0..row_bytes], b);
    }
    var bad: usize = 0;
    for (1..17) |rows| {
        var o = t.ops(&t.on);
        o.window = true;
        const b = try layerBytes(t, o, l.view, x, 0, rows);
        defer gpa.free(b);
        bad += differ(b, alone, 0, row_bytes);
    }
    try rig.expect(bad == 0, "nemotron moe rounds: {d} rows differ from the row alone", .{bad});
    var o = t.ops(&t.on);
    o.prefill = true;
    const whole = try layerBytes(t, o, l.view, x, 0, 80);
    defer gpa.free(whole);
    const cuts = [_][]const usize{ &.{ 16, 64 }, &.{ 1, 15, 48, 16 }, &.{ 7, 9, 13, 51 }, &.{ 64, 1, 15 } };
    for (cuts) |cut| {
        var first: usize = 0;
        for (cut) |n| {
            const b = try layerBytes(t, o, l.view, x, first, n);
            defer gpa.free(b);
            bad += differ(b, whole, first, row_bytes);
            first += n;
        }
    }
    for (1..17) |rows| {
        const b = try layerBytes(t, o, l.view, x, 0, rows);
        defer gpa.free(b);
        bad += differ(b, whole, 0, row_bytes);
    }
    try rig.expect(bad == 0, "nemotron moe prompts: {d} rows differ from the same row in the whole prompt", .{bad});
    std.debug.print(
        "  nemotron moe: prompt rows equal to decode rows: {d} of 16\n",
        .{16 - differ(whole[0 .. 16 * row_bytes], alone, 0, row_bytes)},
    );
}

test "two ranks' fp32 shares of the Nemotron MoE, half the experts each and the shared one on rank 0, sum to float64" {
    const t = try Rig.open(64 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x7A9 };
    const l = try fx.Layer.init(t, &rng, small);
    defer l.deinit();
    const d = small.hidden;
    const half = small.experts / 2;
    var x = try fx.Input.init(t, &rng, 16, d);
    defer x.deinit();
    var worst: f64 = 0;
    for ([_]usize{ 1, 3, 16 }) |rows| for ([_]bool{ false, true }) |prefill| {
        var o = t.ops(&t.on);
        o.prefill = prefill;
        o.window = !prefill and rows > 1;
        var shares: [2][]u8 = undefined;
        for (&shares, 0..) |*sh, r| {
            o.arena.reset();
            const y = try nm.run(o, l.rank(r * half, half, r == 0, true), x.tensor(t, 0, d), rows);
            try rig.expect(y.kind == .f32, "a rank's share is not fp32", .{});
            sh.* = try fx.bytesAt(t, y.ptr, rows * d * 4);
        }
        defer for (shares) |sh| gpa.free(sh);
        o.arena.reset();
        const logits = try o.arena.of(f32, rows * small.experts);
        const pick = try o.arena.of(i32, rows * small.top_k);
        const wts = try o.arena.of(f32, rows * small.top_k);
        try o.moeRouter(x.tensor(t, 0, d), l.view.router, logits, rows, d, small.experts);
        try nm.select(o, l.view, logits, pick, wts, null, 0, rows);
        const hp = try fx.bytesAt(t, pick, rows * small.top_k * 4);
        defer gpa.free(hp);
        const picks: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, hp));
        const a: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, shares[0]));
        const b: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, shares[1]));
        var score: [64]f64 = undefined;
        var sel: [64]f64 = undefined;
        var want: [512]f64 = undefined;
        for (0..rows) |r| {
            const xr = x.values[r * d ..][0..d];
            l.scores(xr, &score, &sel);
            const pr = picks[r * small.top_k ..][0..small.top_k];
            var total: f64 = 0;
            for (pr) |p| total += score[@intCast(p)];
            var w: [6]f64 = undefined;
            for (pr, &w) |p, *wv| wv.* = score[@intCast(p)] / total * small.scaling;
            try l.output(xr, pr, &w, &want);
            var top: f64 = 0;
            for (want) |v| top = @max(top, @abs(v));
            for (0..d) |c| worst = @max(worst, @abs(@as(f64, a[r * d + c] + b[r * d + c]) - want[c]) / top);
        }
    };
    std.debug.print("  nemotron moe two ranks: worst {e:.2} of the row's largest\n", .{worst});
    try rig.expect(worst <= outBound(t), "nemotron moe two ranks: {e} from float64", .{worst});
}
