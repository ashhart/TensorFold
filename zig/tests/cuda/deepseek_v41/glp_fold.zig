//! Real fold/writer launches between canary bands, repeated five times and checked against FP32/FP64 models.
//! GLP format and paired site: Matt Suiche/weightless; additive convention: llama.cpp; Arditi et al. is background.

const std = @import("std");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");

pub const DirFault = enum { none, nan, inf };

pub const FoldCase = struct {
    name: []const u8,
    width: usize,
    rows: usize,
    xs: usize = 0, // row stride in elements; 0: dense (4 width)
    tap: bool = false,
    ts: usize = 0, // tap row stride; 0: width
    alpha: f32 = 0.75,
    fill: ref.Fill = .random,
    fault: DirFault = .none,
    witness: bool = false, // the last column's direction is 1 and its copies 64.0: that column must be edited
    seed: u64,
};

pub const Outcome = struct {
    t: ref.Tally = .{},
    nonfinite: usize = 0, // FP64 outputs that are NaN or infinite
    repeat_diff: usize = 0,
    witness_bad: usize = 0,
    guards: bool = true,
    inputs_kept: bool = true,

    pub fn ok(o: Outcome) bool {
        return o.t.clean() and o.repeat_diff == 0 and o.witness_bad == 0 and o.guards and o.inputs_kept;
    }
};

fn poison(i: usize) u16 {
    return ref.padWord(i ^ 0x55) ^ 0x8000;
}

pub fn foldRun(c: rig.Ctx, ops: rig.glp.Ops, k: FoldCase) !Outcome {
    const a = c.gpu.gpa;
    const w = k.width;
    const nout = 4 * w;
    const xs = if (k.xs == 0) nout else k.xs;
    const ts = if (k.ts == 0) w else k.ts;
    var rng = ref.Rng.init(k.seed);
    const d = try a.alloc(f32, w);
    defer a.free(d);
    ref.fillDir(k.fill, &rng, d);
    switch (k.fault) {
        .none => {},
        .nan => d[w / 3] = std.math.nan(f32),
        .inf => d[5] = std.math.inf(f32),
    }
    if (k.witness) d[w - 1] = 1;
    const x = try a.alloc(u16, k.rows * xs);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = ref.padWord(i);
    for (0..k.rows) |r| {
        const row = x[r * xs ..][0..nout];
        ref.fillRow(k.fill, &rng, r, row, d);
        if (k.witness) for (0..4) |cc| {
            row[cc * w + w - 1] = 0x4280; // 64.0
        };
    }
    const tap0 = try a.alloc(u16, k.rows * ts);
    defer a.free(tap0);
    for (tap0, 0..) |*v, i| v.* = poison(i);
    const ym = try a.alloc(u16, nout);
    defer a.free(ym);
    const tm = try a.alloc(u16, w);
    defer a.free(tm);
    const y64 = try a.alloc(f64, nout);
    defer a.free(y64);

    var xb = try rig.Buf.init(c.gpu, rig.bytes(x));
    defer xb.free();
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    var tb = try rig.Buf.init(c.gpu, rig.bytes(tap0));
    defer tb.free();
    var o: Outcome = .{};
    var first_x: ?[]align(16) u8 = null;
    var first_t: ?[]align(16) u8 = null;
    defer if (first_x) |f| a.free(f);
    defer if (first_t) |f| a.free(f);
    for (0..5) |rep| {
        try xb.write(rig.bytes(x));
        try tb.write(rig.bytes(tap0));
        try rig.fold(ops, xb.ptr(), xs, if (k.tap) tb.ptr() else 0, ts, db.ptr(), k.alpha, k.rows, w);
        try c.sync();
        const gx = try xb.read(a);
        const gt = try tb.read(a);
        if (rep > 0) {
            if (!std.mem.eql(u8, gx, first_x.?) or !std.mem.eql(u8, gt, first_t.?)) o.repeat_diff += 1;
            a.free(gx);
            a.free(gt);
            continue;
        }
        first_x = gx;
        first_t = gt;
        const gxw = rig.words(gx);
        const gtw = rig.words(gt);
        for (0..k.rows) |r| {
            const xin = x[r * xs ..][0..nout];
            ref.foldRow(xin, d, k.alpha, .{}, ym, if (k.tap) tm else null);
            ref.foldRef(xin, d, k.alpha, y64);
            const got = gxw[r * xs ..][0..xs];
            for (0..nout) |i| {
                if (!std.math.isFinite(y64[i])) o.nonfinite += 1;
                const bound = ref.allow(y64[i], ref.widen(xin[i]), d[i % w], k.alpha);
                o.t.add(r * xs + i, got[i], ym[i], y64[i], bound);
            }
            for (nout..xs) |i| if (got[i] != x[r * xs + i]) {
                o.t.bad_pad += 1;
            };
            if (k.witness) for (0..4) |cc| if (got[cc * w + w - 1] == xin[cc * w + w - 1]) {
                o.witness_bad += 1;
            };
            if (!k.tap) continue;
            for (0..w) |col| {
                var m: f64 = 0; // the mean of the rounded FP64 copies q_c = RNE16(y_c): the tap's FP64 target
                var err: f64 = 0; // E: each copy's enforced bound plus its distance from its rounded value
                var mag: f64 = 0; // A: the mean of |q_c|
                for (0..4) |cc| {
                    const y = y64[cc * w + col];
                    const q = ref.widen(ref.rne64(y));
                    m += 0.25 * q;
                    err += 0.25 * (ref.allow(y, ref.widen(xin[cc * w + col]), d[col], k.alpha) + @abs(y - q));
                    mag += 0.25 * @abs(q);
                }
                const g3 = 3 * 0x1p-24 / (1 - 3 * 0x1p-24); // the three FP32 additions of the mean (0.25 scaling is exact)
                const e = err + g3 * (mag + err);
                const bound = e + 0x1p-8 * (@abs(m) + e) + 0x1p-133; // then the tap's one BF16 rounding
                o.t.add(r * ts + col, gtw[r * ts + col], tm[col], m, bound);
            }
            for (w..ts) |i| if (gtw[r * ts + i] != tap0[r * ts + i]) {
                o.t.bad_pad += 1;
            };
        }
        if (!k.tap and !std.mem.eql(u8, gt, rig.bytes(tap0))) o.t.bad_pad += 1; // no tap buffer given: no write to it
    }
    o.guards = try xb.guards(a) and try db.guards(a) and try tb.guards(a);
    const dk = try db.read(a);
    defer a.free(dk);
    o.inputs_kept = std.mem.eql(u8, dk, rig.bytes(d));
    return o;
}

/// Reports one outcome as a line and fails the run if it is not clean.
pub fn report(what: []const u8, name: []const u8, o: Outcome, seed: u64, planted: bool, mixed: bool) !void {
    const t = o.t;
    var bad = !o.ok();
    if (planted and o.nonfinite == 0) bad = true; // the planted values must reach the FP64 outputs
    if (mixed and o.nonfinite == t.n) bad = true; // and leave finite ones beside them
    if (bad) {
        std.debug.print("  {d} elements: {d} differ from the fp32 order model, {d} class, {d} past the FP64 bound (first {?d}), " ++
            "{d} pad words changed, repeats differ {d}, witness {d}, guards {any}, inputs kept {any}, max {d} ulp, {d} nonfinite\n", .{
            t.n, t.model_diff, t.class, t.over, t.first, t.bad_pad, o.repeat_diff, o.witness_bad, o.guards, o.inputs_kept, t.max_ulp, o.nonfinite,
        });
        return rig.fail("{s} {s}: seed 0x{x}", .{ what, name, seed });
    }
    check.pass("glp {s} {s}: {d} elements, {d} off the fp32 order model, {d} past the FP64 bound, {d} FP64 rounding flips (max {d} ulp)," ++
        " {d} nonfinite, pads and guards intact, 5 repeats identical, seed 0x{x}", .{ what, name, t.n, t.model_diff + t.class, t.over, t.flips, t.max_ulp, o.nonfinite, seed });
}

pub fn foldCase(c: rig.Ctx, k: FoldCase) !void {
    const o = try foldRun(c, c.ops, k);
    const planted = k.fill == .nonfinite or k.fault != .none;
    try report("fold", k.name, o, k.seed, planted, planted);
}

// ---- the writer ------------------------------------------------------------------------------------------------

pub const WriterCase = struct {
    name: []const u8,
    width: usize,
    rows: usize, // launched
    total: usize = 0, // rows the buffer holds (0: rows); the others must stay untouched
    world: usize,
    gs: usize = 0, // row stride in elements; 0: width
    gr_extra: usize = 0, // elements between one rank's rows and the next rank's
    alpha: f32 = 0.75,
    fill: ref.Fill = .random,
    witness: bool = false, // the last column's direction is 1 and its only nonzero partial 64.0: it must be edited
    seed: u64,
};

pub const WOutcome = struct {
    t: ref.Tally = .{},
    nonfinite: usize = 0,
    repeat_diff: usize = 0,
    slot_bad: usize = 0, // an other rank's slot that is not -0.0, or a rank-order sum that is not the edited vector
    witness_bad: usize = 0,
    o_flips: usize = 0, // partial sums FP64 rounds differently from the fp32 sum
    guards: bool = true,
    inputs_kept: bool = true,

    pub fn ok(o: WOutcome) bool {
        return o.t.clean() and o.slot_bad == 0 and o.witness_bad == 0 and o.repeat_diff == 0 and o.guards and o.inputs_kept;
    }
};

pub fn writerRun(c: rig.Ctx, ops: rig.glp.Ops, k: WriterCase) !WOutcome {
    const a = c.gpu.gpa;
    const w = k.width;
    const gs = if (k.gs == 0) w else k.gs;
    const total = if (k.total == 0) k.rows else k.total;
    const gr = total * gs + k.gr_extra;
    var rng = ref.Rng.init(k.seed);
    const d = try a.alloc(f32, w);
    defer a.free(d);
    ref.fillDir(if (k.fill == .ties) .random else k.fill, &rng, d);
    if (k.witness) d[w - 1] = 1;
    const g = try a.alloc(u16, k.world * gr);
    defer a.free(g);
    for (g, 0..) |*v, i| v.* = ref.padWord(i);
    for (0..k.world) |rk| for (0..total) |r| {
        const part = g[rk * gr + r * gs ..][0..w];
        ref.fillPartial(if (r < k.rows) k.fill else .random, &rng, r, rk, k.world, part);
        if (k.witness and r < k.rows) part[w - 1] = if (rk == 0) 0x4280 else 0; // 64.0 + 0 + ...
    };
    const ym = try a.alloc(u16, w);
    defer a.free(ym);
    const o16 = try a.alloc(u16, w);
    defer a.free(o16);
    const y64 = try a.alloc(f64, w);
    defer a.free(y64);
    var gb = try rig.Buf.init(c.gpu, rig.bytes(g));
    defer gb.free();
    var db = try rig.Buf.init(c.gpu, rig.bytes(d));
    defer db.free();
    var o: WOutcome = .{};
    var first: ?[]align(16) u8 = null;
    defer if (first) |f| a.free(f);
    for (0..5) |rep| {
        try gb.write(rig.bytes(g));
        try rig.writer(ops, gb.ptr(), gr, gs, db.ptr(), k.alpha, k.rows, w, k.world);
        try c.sync();
        const got_b = try gb.read(a);
        if (rep > 0) {
            if (!std.mem.eql(u8, got_b, first.?)) o.repeat_diff += 1;
            a.free(got_b);
            continue;
        }
        first = got_b;
        const got = rig.words(got_b);
        for (0..k.rows) |r| {
            var ranks: [ref.max_world][]const u16 = undefined;
            for (0..k.world) |rk| ranks[rk] = g[rk * gr + r * gs ..][0..w];
            ref.writerRow(k.world, ranks[0..k.world], d, k.alpha, .{}, ym);
            ref.writerRef(k.world, ranks[0..k.world], d, k.alpha, o16, y64);
            var flip: [ref.max_width]bool = @splat(false);
            for (0..w) |col| {
                var s = ref.widen(ranks[0][col]);
                for (1..k.world) |rk| s += ref.widen(ranks[rk][col]);
                if (!ref.same(ref.rne32(s), o16[col])) {
                    flip[col] = true; // the fp32 sum rounds to another bf16 than the exact one: only the model is held
                    o.o_flips += 1;
                }
            }
            for (0..w) |col| {
                if (!std.math.isFinite(y64[col])) o.nonfinite += 1;
                const bound = if (flip[col]) std.math.inf(f64) else ref.allow(y64[col], ref.widen(o16[col]), d[col], k.alpha);
                o.t.add(r * gs + col, got[r * gs + col], ym[col], y64[col], bound);
                // the other ranks read -0.0, so the rank-order sum of the slots is the edited vector bit for bit
                var s = ref.widen(got[r * gs + col]);
                for (1..k.world) |rk| {
                    const slot = got[rk * gr + r * gs + col];
                    if (slot != 0x8000) o.slot_bad += 1;
                    s += ref.widen(slot);
                }
                if (!ref.same(ref.rne32(s), got[r * gs + col])) o.slot_bad += 1;
            }
            if (k.witness and got[r * gs + w - 1] == 0x4280) o.witness_bad += 1;
        }
        for (0..k.world) |rk| for (0..total) |r| for (0..gs) |col| {
            if (r < k.rows and col < w) continue;
            const at = rk * gr + r * gs + col;
            if (got[at] != g[at]) o.t.bad_pad += 1; // pad columns and the rows beyond the launch
        };
        for (total * gs..gr) |i| for (0..k.world) |rk| if (got[rk * gr + i] != g[rk * gr + i]) {
            o.t.bad_pad += 1;
        };
    }
    o.guards = try gb.guards(a) and try db.guards(a);
    const dk = try db.read(a);
    defer a.free(dk);
    o.inputs_kept = std.mem.eql(u8, dk, rig.bytes(d));
    return o;
}

pub fn writerCase(c: rig.Ctx, k: WriterCase) !void {
    const o = try writerRun(c, c.ops, k);
    var bad = !o.ok();
    if (k.fill == .nonfinite and (o.nonfinite == 0 or o.nonfinite == o.t.n)) bad = true;
    if (bad) {
        std.debug.print("  {d} elements: {d} off the model, {d} class, {d} past the FP64 bound (first {?d}), {d} pad words, " ++
            "{d} slot errors, witness {d}, repeats {d}, guards {any}, inputs kept {any}\n", .{ o.t.n, o.t.model_diff, o.t.class, o.t.over, o.t.first, o.t.bad_pad, o.slot_bad, o.witness_bad, o.repeat_diff, o.guards, o.inputs_kept });
        return rig.fail("writer {s}: seed 0x{x}", .{ k.name, k.seed });
    }
    check.pass("glp writer {s} world {d}: {d} elements, {d} off the fp32 order model, {d} past the FP64 bound, {d} FP64 rounding flips" ++
        " (max {d} ulp; {d} partial sums rounded differently), other slots -0.0 and the rank-order sum bit-equal, rows beyond and pads untouched," ++
        " 5 repeats identical, seed 0x{x}", .{ k.name, k.world, o.t.n, o.t.model_diff + o.t.class, o.t.over, o.t.flips, o.t.max_ulp, o.o_flips, k.seed });
}
