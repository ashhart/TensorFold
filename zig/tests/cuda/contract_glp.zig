//! CUDA GLPv3 additive FP64/fixed-order contracts and mandatory fault controls. The edit is add only, y = x + alpha d:
//! no sum of products, so gate 4 does not apply to the fold and the writer (the final keeps its fp32 sums). GLP format
//! and DeepSeek paired site: Matt Suiche/weightless (spec b181a07d3980fc48abe39d95ddb044459dd0b868); the additive
//! convention is llama.cpp's; Arditi et al. (NeurIPS 2024) is background only.
//!
//! Each launch's output is held to two tests at once: bit-equal to the independent fp32 order model of
//! deepseek_v41/glp_ref.zig, and inside its FP64 bound (NaN / infinity classes equal where FP64 has them).

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const images = @import("contract_deepseek_images.zig");
const ref = @import("deepseek_v41/glp_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;

// Mirror glp_ops.zig's by-value launch arguments (mode-less, 56 / 48 / 56 bytes) without importing the family module.
const StreamArgs = extern struct { x: u64, xs: i64, tap: u64 = 0, ts: i64 = 0, dir: u64, alpha: f32, R: c_int, width: c_int };
const WriterArgs = extern struct { g: u64, gr: i64, gs: i64, dir: u64, alpha: f32, R: c_int, width: c_int, world: c_int };
const FinalArgs = extern struct { x: u64, xs: i64, pre: u64, nw: u64, out: u64, eps: f32, nwbf16: c_int, R: c_int, width: c_int };

comptime {
    std.debug.assert(@sizeOf(StreamArgs) == 56 and @offsetOf(StreamArgs, "dir") == 32 and @offsetOf(StreamArgs, "alpha") == 40 and @offsetOf(StreamArgs, "R") == 44 and @offsetOf(StreamArgs, "width") == 48);
    std.debug.assert(@sizeOf(WriterArgs) == 48 and @offsetOf(WriterArgs, "dir") == 24 and @offsetOf(WriterArgs, "alpha") == 32 and @offsetOf(WriterArgs, "R") == 36 and @offsetOf(WriterArgs, "world") == 44);
    std.debug.assert(@sizeOf(FinalArgs) == 56 and @offsetOf(FinalArgs, "eps") == 40 and @offsetOf(FinalArgs, "R") == 48);
}

const eps: f32 = 1e-6;

const Kind = enum { stream, writer, final };

/// The planted faults and the gate each is registered red at: the nonfinite gate for the flushed values, the shapes
/// gate (fp32 order model bits and the FP64 bound) for the rest.
const Ctl = enum {
    sum, // the tap adds its copies where it averages them
    half, // the edit in fp16
    drop_tail, // the last column of the last group left unedited
    drop_nonfinite, // NaN / infinity flushed to zero by the edit
    rank_order, // the writer's ranks added last to first
    truncate, // truncation in place of round to nearest even

    fn gate(k: Ctl) c.Gate {
        return switch (k) {
            .drop_nonfinite => .nonfinite,
            .sum, .half, .drop_tail, .rank_order, .truncate => .shapes,
        };
    }
};

/// Rows: the steered rows; n and k: the row's width. The served widths are the production 5120 and two small ones.
const edit_served = [_]c.Shape{ .{ .rows = 4, .n = 5120, .k = 5120 }, .{ .rows = 17, .n = 1032, .k = 1032 }, .{ .rows = 3, .n = 128, .k = 128 } };
const final_served = [_]c.Shape{ .{ .rows = 3, .n = 5120, .k = 5120 }, .{ .rows = 17, .n = 128, .k = 128 } };

/// The edit's fp32 chain: a rank sum of up to eight partials, the product and the add (the bound is the model's own).
fn depthEdit(_: c.Shape) usize {
    return 10;
}

/// The final's square chains: 128 fmas, then the (at most 40) chains' adds.
fn depthFinal(_: c.Shape) usize {
    return 170;
}

/// Bound the fp32 product and sum separately from the BF16 output; final also includes RMS and collapse quantization.
const edit_ops = 3;
const final_ops = 5;

const final_why = "one-hot pre-mix, integer copies: fp32 exact, fp16 overflows; a pairwise bf16 sum of positive squares errs by under 3%";
const add_why = "the add edit has no sum of products; its fp32 sums (the tap's mean, the writer's ranks) are checked bit for bit against the model";

fn specOf(kind: Kind, name: []const u8) c.Spec {
    const fin = kind == .final;
    return .{
        .name = name,
        .accum = if (fin) .f32 else .none,
        .depth = if (fin) depthFinal else depthEdit,
        .ops = if (fin) final_ops else edit_ops,
        .mids = 0,
        .out = .bf16,
        .in_place = !fin,
        .served = if (fin) &final_served else &edit_served,
        .sums = if (fin) .{ .f16 = 100, .bf16 = 0, .why = final_why } else .{ .f16 = 0, .bf16 = 0, .why = add_why },
    };
}

/// One prepared call's host data and the three comparisons made on it, apart from any device buffer.
const Fixture = struct {
    kind: Kind,
    /// The fold's tap: [R, w] right after the copies in the one output
    tap: bool = false,
    /// The writer's ranks.
    world: usize = 1,
    /// The final's weights in bf16 (else fp32).
    nw16: bool = false,
    /// The writer's ranks cancel, so that the order of the sum shows in the bits (glp_ref.fillPartial).
    cancel: bool = false,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    alpha: f32 = 1.5,
    /// The prepared inputs are gate 4's (the final only).
    stress: bool = false,
    /// The prepared input words: the fold's copies, the writer's partials, the final's streams.
    words: []u16 = &.{},
    d: []f32 = &.{},
    pre: []f32 = &.{},
    nw: []u8 = &.{},
    /// The writer's FP64 partial sums rounded once to BF16.
    o: []u16 = &.{},

    fn free(fx: *Fixture, a: std.mem.Allocator) void {
        a.free(fx.words);
        a.free(fx.d);
        a.free(fx.pre);
        a.free(fx.nw);
        a.free(fx.o);
        fx.words = &.{};
        fx.d = &.{};
        fx.pre = &.{};
        fx.nw = &.{};
        fx.o = &.{};
    }

    fn admits(fx: *const Fixture, shape: c.Shape) bool {
        const w = shape.n;
        return shape.rows != 0 and w != 0 and w % 8 == 0 and w <= ref.max_width and (fx.kind != .final or w % ref.chain == 0);
    }

    fn q(fx: *const Fixture, r: usize) [4]f32 {
        return .{ fx.pre[4 * r], fx.pre[4 * r + 1], fx.pre[4 * r + 2], fx.pre[4 * r + 3] };
    }

    fn nwAt(fx: *const Fixture, k: usize) f64 {
        return if (fx.nw16) ref.widen(std.mem.readInt(u16, fx.nw[2 * k ..][0..2], .little)) else @as(f32, @bitCast(std.mem.readInt(u32, fx.nw[4 * k ..][0..4], .little)));
    }

    /// Output elements: the fold's copies (and tap), the writer's gathered partials, the final's rows.
    fn count(fx: *const Fixture) usize {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        return switch (fx.kind) {
            .stream => R * 4 * w + if (fx.tap) R * w else 0,
            .writer => fx.world * R * w,
            .final => R * w,
        };
    }

    /// Inputs for a shape filled as asked (seeded). Every fold and writer fixture carries the witness: the last
    /// column's direction is 1 and its input 64.0, so that column must be edited (a dropped tail edit shows at once).
    fn build(fx: *Fixture, a: std.mem.Allocator, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        const w = shape.n;
        const R = shape.rows;
        fx.free(a);
        errdefer fx.free(a);
        fx.shape = shape;
        fx.stress = fill == .stress;
        var rng = ref.Rng.init(seed);
        const kf: ref.Fill = switch (fill) {
            .random => if (fx.cancel and fx.kind == .writer) .cancel else .random,
            .nonfinite => .nonfinite,
            .stress => .random, // replaced below by the stress fixture
        };
        fx.alpha = if (fx.cancel) 1 else 1.5;
        fx.d = try a.alloc(f32, w);
        ref.fillDir(kf, &rng, fx.d);
        switch (fx.kind) {
            .stream, .final => {
                fx.words = try a.alloc(u16, R * 4 * w);
                for (0..R) |r| ref.fillRow(kf, &rng, r, fx.words[r * 4 * w ..][0 .. 4 * w], fx.d);
            },
            .writer => {
                fx.words = try a.alloc(u16, fx.world * R * w);
                for (0..R) |r| for (0..fx.world) |rk| ref.fillPartial(kf, &rng, r, rk, fx.world, fx.words[rk * R * w + r * w ..][0..w]);
                fx.o = try a.alloc(u16, R * w);
            },
        }
        if (fx.kind != .final and kf != .nonfinite) {
            fx.d[w - 1] = 1;
            for (0..R) |r| switch (fx.kind) {
                .stream => for (0..4) |cc| {
                    fx.words[r * 4 * w + cc * w + w - 1] = 0x4280; // 64.0
                },
                .writer => for (0..fx.world) |rk| {
                    fx.words[rk * R * w + r * w + w - 1] = if (rk == 0) 0x4280 else 0;
                },
                .final => unreachable,
            };
        }
        if (fx.kind == .final) {
            fx.pre = try a.alloc(f32, 4 * R);
            for (fx.pre) |*v| v.* = 0.25 * (1 + @as(f32, @floatCast(rng.unit())));
            fx.nw = try a.alloc(u8, w * if (fx.nw16) @as(usize, 2) else 4);
            for (0..w) |k| {
                const v = ref.dirEntry(&rng);
                if (fx.nw16) std.mem.writeInt(u16, fx.nw[2 * k ..][0..2], ref.rne32(v), .little) else std.mem.writeInt(u32, fx.nw[4 * k ..][0..4], @bitCast(v), .little);
            }
            if (fx.stress) {
                // exact FP32 fixtures: one-hot [16,57] copies for the RMS squares (see the recipe)
                for (fx.pre, 0..) |*v, i| v.* = if (i % 4 == 0) 1 else 0;
                for (fx.words) |*v| v.* = ref.rne32(@floatFromInt(16 + rng.below(42)));
            }
        }
    }

    /// FP64 outputs of the prepared inputs (the fold's and the writer's unrounded edit, the tap's mean of the rounded
    /// copies, the writer's other slots -0.0).
    fn reference(fx: *Fixture, want: []f64) void {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        for (0..R) |r| switch (fx.kind) {
            .stream => {
                const y = want[r * 4 * w ..][0 .. 4 * w];
                ref.foldRef(fx.words[r * 4 * w ..][0 .. 4 * w], fx.d, fx.alpha, y);
                if (fx.tap) for (0..w) |col| {
                    var m: f64 = 0; // the mean of the rounded FP64 copies q_c = RNE16(y_c): the tap's FP64 target
                    for (0..4) |cc| m += 0.25 * ref.widen(ref.rne64(y[cc * w + col]));
                    want[R * 4 * w + r * w + col] = m;
                };
            },
            .writer => {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..fx.world) |rk| ranks[rk] = fx.words[rk * R * w + r * w ..][0..w];
                ref.writerRef(fx.world, ranks[0..fx.world], fx.d, fx.alpha, fx.o[r * w ..][0..w], want[r * w ..][0..w]);
                for (1..fx.world) |rk| @memset(want[rk * R * w + r * w ..][0..w], -0.0);
            },
            .final => ref.finalRef(fx.words[r * 4 * w ..][0 .. 4 * w], fx.q(r), fx.nw, fx.nw16, eps, want[r * w ..][0..w]),
        };
    }

    /// The tap's bound: each copy's enforced bound plus its distance from its rounded value, the mean's three fp32
    /// additions (the 0.25 scaling is exact), then the tap's one BF16 rounding.
    fn tapBound(fx: *const Fixture, want: []const f64, j: usize) f64 {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        const r = j / w;
        const col = j % w;
        const m = want[R * 4 * w + j];
        var err: f64 = 0;
        var mag: f64 = 0;
        for (0..4) |cc| {
            const y = want[r * 4 * w + cc * w + col];
            const qv = ref.widen(ref.rne64(y));
            err += 0.25 * (ref.allow(y, ref.widen(fx.words[r * 4 * w + cc * w + col]), fx.d[col], fx.alpha) + @abs(y - qv));
            mag += 0.25 * @abs(qv);
        }
        const g3 = 3 * 0x1p-24 / (1 - 3 * 0x1p-24);
        const e = err + g3 * (mag + err);
        return e + 0x1p-8 * (@abs(m) + e) + 0x1p-133;
    }

    /// Ordinary bounds include the edit's fp32 product and sum, the writer's BF16 tie flips and the final's chains.
    fn bounds(fx: *const Fixture, want: []const f64, allow: []f64) void {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        const sp = specOf(fx.kind, "glp");
        for (allow, want, 0..) |*l, v, i| {
            var b: f64 = 1;
            switch (fx.kind) {
                .stream => b = if (i < R * 4 * w) ref.allow(v, ref.widen(fx.words[i]), fx.d[i % w], fx.alpha) else fx.tapBound(want, i - R * 4 * w),
                .writer => {
                    const rem = i % (R * w);
                    if (i / (R * w) == 0) {
                        const o = ref.widen(fx.o[rem]);
                        // the fp32 rank sum may round to another bf16 than the exact one: one ulp more of the vector
                        const flip: f64 = if (fx.world > 1) 0x1p-7 * @abs(o) else 0;
                        b = ref.allow(v, o, fx.d[rem % w], fx.alpha) + flip;
                    } else b = 0x1p-133;
                },
                .final => b = c.boundExact(sp, v) + (0x1p-7 + c.gamma(170, 0x1p-24)) * @abs(v),
            }
            l.* = if (std.math.isFinite(b)) b else 1;
        }
    }

    /// The independent fp32 model's bits for the whole output (glp_ref.zig; the writer's other slots read -0.0).
    fn model(fx: *const Fixture, exp: []u16) void {
        const w = fx.shape.n;
        const R = fx.shape.rows;
        for (0..R) |r| switch (fx.kind) {
            .stream => ref.foldRow(fx.words[r * 4 * w ..][0 .. 4 * w], fx.d, fx.alpha, .{}, exp[r * 4 * w ..][0 .. 4 * w], if (fx.tap) exp[R * 4 * w + r * w ..][0..w] else null),
            .writer => {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..fx.world) |rk| ranks[rk] = fx.words[rk * R * w + r * w ..][0..w];
                ref.writerRow(fx.world, ranks[0..fx.world], fx.d, fx.alpha, .{}, exp[r * w ..][0..w]);
                for (1..fx.world) |rk| @memset(exp[rk * R * w + r * w ..][0..w], 0x8000);
            },
            .final => ref.finalRow(fx.words[r * 4 * w ..][0 .. 4 * w], fx.q(r), fx.nw, fx.nw16, eps, exp[r * w ..][0..w]),
        };
    }

    /// Recompute the final with a chosen reduced accumulator/order, then round the FP64 result once to BF16.
    fn simulate(fx: *const Fixture, a: std.mem.Allocator, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        if (fx.kind != .final) return error.NoSumsToSimulate;
        const w = fx.shape.n;
        const R = fx.shape.rows;
        const m = try a.alloc(f64, 2 * w);
        defer a.free(m);
        const terms = m[0..w];
        const base = m[w..];
        for (0..R) |r| {
            const x = fx.words[r * 4 * w ..];
            const qr = fx.q(r);
            for (0..w) |k| {
                base[k] = ref.widen(ref.rne64(@as(f64, qr[0]) * ref.widen(x[k]) + @as(f64, qr[1]) * ref.widen(x[w + k]) + @as(f64, qr[2]) * ref.widen(x[2 * w + k]) + @as(f64, qr[3]) * ref.widen(x[3 * w + k])));
                terms[k] = base[k] * base[k];
            }
            const rn = 1.0 / @sqrt(rows_.sumIn(acc, order, terms) / @as(f64, @floatFromInt(w)) + eps);
            for (0..w) |k| out[r * w + k] = ref.widen(ref.rne64(base[k] * rn * fx.nwAt(k)));
        }
    }

    const Judged = struct { n: usize, planted: usize, off_model: usize, off_fp64: usize, first: ?usize };

    /// Both tests at once: equal to the model's bits, and within the FP64 bound (classes equal where FP64 is not).
    fn judge(fx: *Fixture, a: std.mem.Allocator, got: []const u16) !Judged {
        const n = fx.count();
        const buf = try a.alloc(f64, 2 * n);
        defer a.free(buf);
        const exp = try a.alloc(u16, n);
        defer a.free(exp);
        const want = buf[0..n];
        const allow = buf[n..];
        fx.reference(want);
        fx.bounds(want, allow);
        fx.model(exp);
        var j: Judged = .{ .n = n, .planted = 0, .off_model = 0, .off_fp64 = 0, .first = null };
        for (want, allow, exp, got, 0..) |wv, l, e, gw, i| {
            if (!std.math.isFinite(wv)) j.planted += 1;
            var bad = !ref.same(gw, e);
            if (bad) j.off_model += 1;
            const gv: f64 = ref.widen(gw);
            const ok = if (std.math.isFinite(wv)) (gv == wv or @abs(gv - wv) <= l) else c.classOf(gv) == c.classOf(wv);
            if (!ok) {
                j.off_fp64 += 1;
                bad = true;
            }
            if (bad and j.first == null) j.first = i;
        }
        return j;
    }
};

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fx: Fixture,
    name: []const u8,
    control: ?g.Gate = null,
    x: u64 = 0,
    dptr: u64 = 0,
    pre_ptr: u64 = 0,
    nw_ptr: u64 = 0,
    out: u64 = 0,
    msg: [192]u8 = undefined,

    pub fn spec(t: *const Case) c.Spec {
        return specOf(t.fx.kind, t.name);
    }

    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Case);

    pub fn deinit(t: *Case) void {
        t.fx.free(t.rig.gpu.gpa);
        t.rig.deinit();
    }

    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!t.fx.admits(shape)) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        try t.fx.build(a, shape, fill, seed);
        const w = shape.n;
        const R = shape.rows;
        if (t.fx.kind == .final) {
            t.x = try t.rig.input(std.mem.sliceAsBytes(t.fx.words));
            t.pre_ptr = try t.rig.input(std.mem.sliceAsBytes(t.fx.pre));
            t.nw_ptr = try t.rig.input(t.fx.nw);
            t.out = try t.rig.output(R * w * 2);
        } else {
            t.dptr = try t.rig.input(std.mem.sliceAsBytes(t.fx.d));
            t.x = try t.rig.output(t.fx.count() * 2); // in place: the copies, then the tap right after them
            try t.upload();
        }
        return true;
    }

    fn upload(t: *Case) !void {
        const b: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.x, .len = t.fx.count() * 2 };
        try b.upload(0, std.mem.sliceAsBytes(t.fx.words));
    }

    pub fn run(t: *Case) !void {
        if (t.fx.kind != .final) try t.upload(); // in place: every run starts from the prepared bytes
        const w: c_int = @intCast(t.fx.shape.n);
        const R: c_int = @intCast(t.fx.shape.rows);
        var args: cuda.Args = .{};
        switch (t.fx.kind) {
            .stream => {
                const tap: u64 = if (t.fx.tap) t.x + @as(u64, t.fx.shape.rows) * 4 * t.fx.shape.n * 2 else 0;
                args.add(StreamArgs{ .x = t.x, .xs = 4 * w, .tap = tap, .ts = w, .dir = t.dptr, .alpha = t.fx.alpha, .R = R, .width = w });
            },
            .writer => args.add(WriterArgs{ .g = t.x, .gr = @as(i64, R) * w, .gs = w, .dir = t.dptr, .alpha = t.fx.alpha, .R = R, .width = w, .world = @intCast(t.fx.world) }),
            .final => args.add(FinalArgs{ .x = t.x, .xs = 4 * w, .pre = t.pre_ptr, .nw = t.nw_ptr, .out = t.out, .eps = eps, .nwbf16 = @intFromBool(t.fx.nw16), .R = R, .width = w }),
        }
        const shared: u32 = if (t.fx.kind == .final) @intCast(2 * t.fx.shape.n) else 0;
        const block: u32 = @intCast(ref.nt);
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.fx.shape.rows) }, .block = .{ .x = block }, .shared = shared }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Case) usize {
        return t.fx.count();
    }

    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        t.fx.reference(want);
    }

    /// Preserve exact stress allowances (the final's gate 4); ordinary bounds are the fixture's.
    pub fn bounds(t: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        if (t.fx.stress) return;
        t.fx.bounds(want, allow);
    }

    pub fn simulate(t: *Case, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        try t.fx.simulate(t.rig.gpu.gpa, acc, order, out);
    }

    pub fn check(t: *Case, at: g.Gate) !?[]const u8 {
        const a = t.rig.gpu.gpa;
        const n = t.fx.count();
        const out = try t.rig.outputBytes();
        const got = try a.alloc(u16, n);
        defer a.free(got);
        for (got, 0..) |*v, i| v.* = std.mem.readInt(u16, out[2 * i ..][0..2], .little);
        const j = try t.fx.judge(a, got);
        if (at == .nonfinite and (j.planted == 0 or j.planted == n))
            return std.fmt.bufPrint(&t.msg, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ j.planted, n }) catch "no proof";
        if (j.off_model + j.off_fp64 == 0) return null;
        return std.fmt.bufPrint(&t.msg, "{d} of {d} words differ from the fp32 order model and {d} leave the FP64 bound, first at {?d}", .{ j.off_model, n, j.off_fp64, j.first }) catch "outputs differ";
    }
};

/// One registration: the production kernels and each control's fold and writer, in the order they run (every control
/// after the kernels it breaks). `sum` breaks only the fold's tap, `rank_order` only the writer's ranks.
const Reg = struct { name: []const u8, kind: Kind, tap: bool = false, world: usize = 1, cancel: bool = false, nw16: bool = false, ctl: ?Ctl = null };

const regs = [_]Reg{
    .{ .name = "cuda dsv41 glp add fold", .kind = .stream },
    .{ .name = "cuda dsv41 glp add fold with tap", .kind = .stream, .tap = true },
    .{ .name = "cuda dsv41 glp add writer, 3 ranks", .kind = .writer, .world = 3 },
    .{ .name = "cuda dsv41 glp add writer, 3 cancelling ranks", .kind = .writer, .world = 3, .cancel = true },
    .{ .name = "cuda dsv41 glp add final, fp32 weights", .kind = .final },
    .{ .name = "cuda dsv41 glp add final, bf16 weights", .kind = .final, .nw16 = true },
    .{ .name = "control: dsv41 glp add fold tap sum", .kind = .stream, .tap = true, .ctl = .sum },
    .{ .name = "control: dsv41 glp add fold half", .kind = .stream, .ctl = .half },
    .{ .name = "control: dsv41 glp add writer half", .kind = .writer, .world = 2, .ctl = .half },
    .{ .name = "control: dsv41 glp add fold drop_tail", .kind = .stream, .ctl = .drop_tail },
    .{ .name = "control: dsv41 glp add writer drop_tail", .kind = .writer, .world = 3, .ctl = .drop_tail },
    .{ .name = "control: dsv41 glp add fold drop_nonfinite", .kind = .stream, .ctl = .drop_nonfinite },
    .{ .name = "control: dsv41 glp add writer drop_nonfinite", .kind = .writer, .world = 2, .ctl = .drop_nonfinite },
    .{ .name = "control: dsv41 glp add writer rank_order", .kind = .writer, .world = 3, .cancel = true, .ctl = .rank_order },
    .{ .name = "control: dsv41 glp add fold truncate", .kind = .stream, .ctl = .truncate },
    .{ .name = "control: dsv41 glp add writer truncate", .kind = .writer, .world = 2, .ctl = .truncate },
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    prod: cuda.Module,
    ctl: cuda.Module,
    cases: [regs.len]Case = undefined,
    made: usize = 0,

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var prod = try cuda.Module.load(gpu.d, images.image("dsv41_glp_add") orelse return error.UnknownContractImage);
        errdefer prod.unload();
        var ctl = try cuda.Module.load(gpu.d, images.image("dsv41_glp_add_controls") orelse return error.UnknownContractImage);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .prod = prod, .ctl = ctl };
        errdefer set.release();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (regs) |r| {
            var sym: [64:0]u8 = undefined;
            const symbol = try std.fmt.bufPrintSentinel(&sym, "dsv41_glp_add_{s}{s}{s}", .{ @tagName(r.kind), if (r.ctl != null) "_" else "", if (r.ctl) |v| @tagName(v) else "" }, 0);
            const f = try (if (r.ctl != null) set.ctl else set.prod).function(symbol);
            set.cases[set.made] = .{
                .rig = rig,
                .f = f,
                .fx = .{ .kind = r.kind, .tap = r.tap, .world = r.world, .nw16 = r.nw16, .cancel = r.cancel },
                .name = r.name,
                .control = if (r.ctl) |v| v.gate() else null,
            };
            set.made += 1;
            try list.append(a, set.cases[set.made - 1].kernel());
        }
        std.debug.assert(set.made == regs.len);
        return set;
    }

    fn release(set: *Set) void {
        for (set.cases[0..set.made]) |*x| x.deinit();
    }

    pub fn deinit(set: *Set) void {
        set.release();
        set.ctl.unload();
        set.prod.unload();
        set.gpa.destroy(set);
    }
};

const sum_orders = [_]rows_.Order{ .sequential, .pairwise, .strided };

// Host witness: the fixtures tell every planted fault from production, with no device. Each fault is emulated by
// host arithmetic of the fault the controls image plants (glp_controls.cu); the emulation is judged by the same
// comparison the device cases use, at the fixture, the fills and the gate its control is registered with.

fn trunc16(f: f32) u16 {
    if (f != f) return ref.nan16;
    return @intCast(@as(u32, @bitCast(f)) >> 16);
}

/// What the kernel of `ctl` (null: production) would write for the fixture's inputs.
fn emulate(fx: *const Fixture, ctl: ?Ctl, got: []u16) void {
    const w = fx.shape.n;
    const R = fx.shape.rows;
    fx.model(got);
    const k = ctl orelse return;
    for (0..R) |r| switch (fx.kind) {
        .stream => {
            const x = fx.words[r * 4 * w ..];
            const y = got[r * 4 * w ..];
            switch (k) {
                .sum => if (fx.tap) {
                    for (0..w) |col| got[R * 4 * w + r * w + col] = ref.rne32(ref.widen(y[col]) + ref.widen(y[w + col]) + ref.widen(y[2 * w + col]) + ref.widen(y[3 * w + col]));
                },
                .half => ref.foldRow(x[0 .. 4 * w], fx.d, fx.alpha, .{ .edit_acc = .f16 }, y[0 .. 4 * w], null),
                .drop_tail => for (0..4) |cc| {
                    y[cc * w + w - 1] = x[cc * w + w - 1];
                },
                .drop_nonfinite => for (0..4 * w) |i| {
                    const xv = ref.widen(x[i]);
                    y[i] = ref.rne32((if (std.math.isFinite(xv)) xv else 0) + fx.alpha * fx.d[i % w]);
                },
                .rank_order => unreachable, // the fold has no ranks
                .truncate => for (0..4 * w) |i| {
                    y[i] = trunc16(ref.widen(x[i]) + fx.alpha * fx.d[i % w]);
                },
            }
        },
        .writer => {
            const y = got[r * w ..];
            for (0..w) |col| {
                var s = ref.widen(fx.words[r * w + col]);
                for (1..fx.world) |rk| s += ref.widen(fx.words[rk * R * w + r * w + col]);
                var rev = ref.widen(fx.words[(fx.world - 1) * R * w + r * w + col]);
                var rj = fx.world - 1;
                while (rj > 0) {
                    rj -= 1;
                    rev += ref.widen(fx.words[rj * R * w + r * w + col]);
                }
                const add = fx.alpha * fx.d[col];
                y[col] = switch (k) {
                    .sum => unreachable, // the writer has no tap
                    .half => y[col], // set below, as a whole row
                    .drop_tail => if (col == w - 1) ref.rne32(s) else y[col],
                    .drop_nonfinite => blk: {
                        const o = ref.widen(ref.rne32(s));
                        break :blk ref.rne32((if (std.math.isFinite(o)) o else 0) + add);
                    },
                    .rank_order => ref.rne32(ref.widen(ref.rne32(rev)) + add),
                    .truncate => trunc16(ref.widen(ref.rne32(s)) + add), // only the edit's store: the rank sum's round is the canonical pack
                };
            }
            if (k == .half) {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..fx.world) |rk| ranks[rk] = fx.words[rk * R * w + r * w ..][0..w];
                ref.writerRow(fx.world, ranks[0..fx.world], fx.d, fx.alpha, .{ .edit_acc = .f16 }, y[0..w]);
            }
        },
        .final => unreachable,
    };
}

test "the registered fixtures tell production from each planted fault on the host" {
    const a = std.testing.allocator;
    var production: usize = 0;
    var red: usize = 0;
    for (regs) |r| {
        if (r.kind == .final) continue;
        var fx: Fixture = .{ .kind = r.kind, .tap = r.tap, .world = r.world, .cancel = r.cancel };
        defer fx.free(a);
        const gate_fill: g.Fill = if (r.ctl) |v| (if (v.gate() == .nonfinite) .nonfinite else .random) else .random;
        for (edit_served) |shape| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
            if (fill == .nonfinite and shape.rows < 5) continue; // gate 1's shape has 5 to 8 rows
            const shaped: c.Shape = if (fill == .nonfinite) .{ .rows = 5, .n = shape.n, .k = shape.k } else shape;
            try fx.build(a, shaped, fill, 0x676c70 + shape.n);
            const got = try a.alloc(u16, fx.count());
            defer a.free(got);
            emulate(&fx, null, got);
            const clean = try fx.judge(a, got);
            if (clean.off_model + clean.off_fp64 != 0) {
                std.debug.print("production {s} at {d}x{d}, {t} fill: {d} off the model, {d} off FP64 (first {?d})\n", .{ r.name, shaped.rows, shaped.n, fill, clean.off_model, clean.off_fp64, clean.first });
                return error.ProductionNotGreen;
            }
            production += 1;
            const v = r.ctl orelse continue;
            if (fill != gate_fill) continue; // each control is held at its own gate's fill
            emulate(&fx, v, got);
            const bad = try fx.judge(a, got);
            const seen = if (v.gate() == .nonfinite) bad.off_fp64 > 0 else bad.off_model + bad.off_fp64 > 0;
            if (!seen) {
                std.debug.print("control {s} at {d}x{d} stays green on the host fixture\n", .{ r.name, shaped.rows, shaped.n });
                return error.ControlNotRed;
            }
            red += 1;
        };
    }
    try std.testing.expect(production > 0 and red > 0);
}

test "the cancelling writer fixture is the only one that tells the rank order" {
    const a = std.testing.allocator;
    var plain: Fixture = .{ .kind = .writer, .world = 3 };
    defer plain.free(a);
    var cancelling: Fixture = .{ .kind = .writer, .world = 3, .cancel = true };
    defer cancelling.free(a);
    var told: [2]bool = undefined;
    for ([_]*Fixture{ &plain, &cancelling }, &told) |fx, *t| {
        try fx.build(a, .{ .rows = 4, .n = 1032, .k = 1032 }, .random, 0x77);
        const got = try a.alloc(u16, fx.count());
        defer a.free(got);
        emulate(fx, .rank_order, got);
        const j = try fx.judge(a, got);
        t.* = j.off_model + j.off_fp64 > 0;
    }
    try std.testing.expect(!told[0] or told[1]);
    try std.testing.expect(told[1]);
}

// Host witnesses separate both reduced types and FP16 RMS sums from exact FP32 at the declared width.
test "gate 4 stress fixture of the final: squares exact in fp32 under 2^24, overflowing fp16 in every order" {
    const w = 5120;
    for ([_]f64{ 16 * 16, 57 * 57 }) |square| {
        const terms: [w]f64 = @splat(square);
        const total = square * w;
        try std.testing.expect(total < 0x1p24 and total > 65504);
        for (sum_orders) |o| {
            try std.testing.expectEqual(total, rows_.sumIn(.f32, o, &terms));
            try std.testing.expect(!std.math.isFinite(rows_.sumIn(.f16, o, &terms))); // rn reads 0: the output misses by all of it
        }
    }
    const sp = specOf(.final, "witness");
    try std.testing.expect(1 / c.boundExact(sp, 1) >= 100); // an output of 0 against any finite want
}

test "the final's stress fixture is exact in the model: the host final equals its simulation in every fp32 order" {
    const a = std.testing.allocator;
    var fx: Fixture = .{ .kind = .final };
    defer fx.free(a);
    try fx.build(a, .{ .rows = 2, .n = 5120, .k = 5120 }, .stress, 0x55);
    const out = try a.alloc(f64, fx.count());
    defer a.free(out);
    const want = try a.alloc(f64, fx.count());
    defer a.free(want);
    fx.reference(want);
    for (sum_orders) |o| {
        try fx.simulate(a, .f32, o, out);
        for (out, want) |p, q| try std.testing.expect(@abs(p - q) <= c.boundExact(specOf(.final, "stress"), q));
    }
}

test "the registered cases' device-side code and the set's registration compile" {
    // never taken (no device in a host test), but analyzed: a reference to a function does not analyze its body
    var device: ?check_.Gpu = null;
    _ = &device;
    if (device) |dev| {
        const a = std.testing.allocator;
        var s = try cuda.Stream.init(dev.d, false);
        defer s.deinit();
        var list: std.ArrayList(g.Kernel) = .empty;
        defer list.deinit(a);
        const set = try Set.init(dev, s, a, &list);
        defer set.deinit();
        var t: g.Tally = .{};
        try g.runAll(a, list.items, 1, &t);
    }
    try std.testing.expectEqual(@as(usize, 16), regs.len);
}
