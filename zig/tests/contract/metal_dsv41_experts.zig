//! DeepSeek V4.1's routed experts under the contract on Metal: the gate/up with its epilogue, Kimi's down, controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_experts_ref.zig");
const pipe = @import("metal_dsv41_experts_pipe.zig");

const c = contract.spec;
const g = contract.gates;
const ex = tf.dsv41_experts;
const topk = ref.topk;
const backing_experts = 12;
const base = "dsv41_experts ";

pub const Which = enum { up, down };
/// What a registration varies: the kernel, the real fixture, the full E8M0 range, a fault; `coded`: the decode gate/up.
pub const Variant = struct { which: Which, fixture: bool = false, wide: bool = false, ctl: u8 = 0, coded: bool = false };

/// Kimi's text edited to plant a fault in the down kernels' chains (controls 1-3); the rest use a TF_CTL define.
fn edits(ctl: u8) []const [2][]const u8 {
    const x2 = "const float2 x2 = float2(*(threadgroup const bfloat2*)(x + j));";
    const x2f = "float2 x2 = float2(*(threadgroup const bfloat2*)(x + j));";
    const fma_x = "p[v][r] = fma(x2.x, w[v][j], p[v][r]);";
    const fma_y = "p[v][r] = fma(x2.y, w[v][j], p[v][r]);";
    const store = "if (c == 0 && uint(r) < rn) out[(first + r0 + r) * a.N + n] = bfloat(v);";
    return switch (ctl) {
        1 => &.{.{ x2, x2f ++ " if (!isfinite(x2.x)) x2.x = 0.0f; if (!isfinite(x2.y)) x2.y = 0.0f;" }},
        2 => &.{.{ "for (uint g = c; g < K / 32; g += 8) {", "for (uint g = c; g + 1 < K / 32; g += 8) {" }},
        3 => &.{ .{ fma_x, "p[v][r] = float(half(fma(x2.x, w[v][j], p[v][r])));" }, .{ fma_y, "p[v][r] = float(half(fma(x2.y, w[v][j], p[v][r])));" } },
        4 => &.{.{ store, store ++ " if (c == 0 && n == 0 && first + r0 + r == 0) out[a.rows_max * a.topk * a.N] = bfloat(1.0f);" }},
        else => &.{},
    };
}

const real = [_]c.Shape{ .{ .rows = 1, .n = 2304, .k = 5120 }, .{ .rows = 4, .n = 2304, .k = 5120 } };
const real_down = [_]c.Shape{ .{ .rows = 1, .n = 5120, .k = 2304 }, .{ .rows = 4, .n = 5120, .k = 2304 } };

/// First a small shape at the widest K (gates 1, 2 and 4 take it), then every row count, then V4.1's own sizes.
fn servedOf(comptime which: Which) [18]c.Shape {
    const n: usize = if (which == .up) 96 else 160;
    const k: usize = if (which == .up) 5120 else 2304;
    var out: [18]c.Shape = undefined;
    for (1..17) |r| out[r - 1] = .{ .rows = r, .n = n, .k = k };
    out[16] = (if (which == .up) real else real_down)[0];
    out[17] = (if (which == .up) real else real_down)[1];
    return out;
}
const served_up = servedOf(.up);
const served_down = servedOf(.down);
const fixture_only = [_]c.Shape{ref.fixture.shape};

fn depthOf(s: c.Shape) usize {
    return ref.depth(s.k);
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    const served: []const c.Shape = if (v.fixture) &fixture_only else if (v.which == .up) &served_up else &served_down;
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthOf,
        .out = .bf16,
        .nan = if (v.which == .up) .{ .stated = "a non-finite value in a row's activation makes all of the row's outputs NaN" } else .propagate,
        .tiles = if (v.fixture) .{ .{}, .{}, .{} } else .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = served,
        .sums = if (v.fixture) .{ .f16 = 0, .bf16 = 0, .why = "the fixture's K is 256: its sums cannot reach fp16's limits; the other down registrations prove them" } else .{},
    };
}

const Refs = struct { ids: rig_.Ref, weights: rig_.Ref, slots: rig_.Ref, nslots: rig_.Ref, pairs: rig_.Ref, where: rig_.Ref };

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: ex.Pipelines,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    real_fixture: bool = false,
    plan: ?ref.Plan = null,
    m1: ?ref.Mats = null,
    m3: ?ref.Mats = null,
    x: []u16 = &.{},
    nom: []f32 = &.{},
    spread: []f64 = &.{},
    mag: []f64 = &.{},
    backing: [4]mtl.Buffer = undefined,
    nback: usize = 0,
    xr: rig_.Ref = undefined,
    table: rig_.Ref = undefined,
    rt: Refs = undefined,
    out: rig_.Ref = undefined,
    xq: rig_.Ref = undefined,
    xqs: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = undefined };
        const head = if (v.ctl != 0) try std.fmt.allocPrint(a, "#define TF_CTL {d}\n", .{v.ctl}) else "";
        defer if (head.len > 0) a.free(head);
        const text = try ex.source(a, head, if (v.which == .down) edits(v.ctl) else &.{});
        defer a.free(text);
        p.pipes = try ex.Pipelines.loadSource(device, text);
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.plan) |q| q.deinit();
        if (p.m1) |m| m.deinit();
        if (p.m3) |m| m.deinit();
        for ([_][]const u8{ std.mem.sliceAsBytes(p.x), std.mem.sliceAsBytes(p.nom), std.mem.sliceAsBytes(p.spread), std.mem.sliceAsBytes(p.mag) }) |s| if (s.len > 0) p.a.free(s);
        p.plan = null;
        p.m1 = null;
        p.m3 = null;
        p.x = &.{};
        p.nom = &.{};
        p.spread = &.{};
        p.mag = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    fn pairs(p: *const Proj) usize {
        return p.shape.rows * topk;
    }

    fn call(p: *const Proj) ex.Call {
        return if (p.v.which == .up) .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n) } else .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.n), .n = @intCast(p.shape.k) };
    }

    fn address(r: rig_.Ref) u64 {
        return r.buf.gpuAddress() + r.off;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const up = p.v.which == .up;
        if (shape.rows == 0 or shape.rows > 16 or shape.k % 32 != 0 or shape.n % 32 != 0 or shape.k == 0 or shape.n == 0 or shape.k > (if (up) @as(usize, 5120) else 3072)) return false;
        p.shape = shape;
        p.fill = fill;
        p.real_fixture = p.v.fixture and fill == .random and std.meta.eql(shape, ref.fixture.shape);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const a = p.a;
        p.plan = try ref.Plan.init(a, shape.rows, backing_experts, rng);
        const np = p.pairs();
        p.x = try a.alloc(u16, (if (up) shape.rows else np) * shape.k);
        p.m1 = try ref.Mats.init(a, backing_experts, shape.n, shape.k);
        if (up) p.m3 = try ref.Mats.init(a, backing_experts, shape.n, shape.k);
        if (fill == .stress) {
            for ([_]?ref.Mats{ p.m1, p.m3 }) |m| if (m) |mm| mm.stress(rng);
            ref.stressX(p.x, if (up) shape.rows else np, shape.k, up, rng);
        } else {
            for ([_]?ref.Mats{ p.m1, p.m3 }) |m| if (m) |mm| mm.fill(rng, p.v.wide);
            ref.fillX(p.x, rng);
            if (p.v.wide) p.wideX(rng);
            if (fill == .nonfinite) ref.plantNonfinite(p.x, if (up) shape.rows else np, shape.k);
            if (p.real_fixture) try ref.fixture.load(p.m1.?, p.x);
        }
        try p.upload();
        return true;
    }

    /// Full-range scales: each K group's activation is sized so its terms stay normal floats and its sum finite.
    fn wideX(p: *Proj, rng: std.Random) void {
        const m = p.m1.?;
        for (0..p.x.len / m.k) |r| for (0..m.k / 32) |grp| {
            const e = m.srow(0, 0)[grp];
            const f: f32 = ref.pow2(if (e >= 128) -40 else 0);
            for (0..32) |j| p.x[r * m.k + 32 * grp + j] = c.toBf16((1 + rng.float(f32)) * f * if (rng.boolean()) @as(f32, 1) else -1);
        };
        for (1..m.n * m.e) |row| {
            const first = m.scales[0 .. m.k / 32];
            @memcpy(m.scales[row * m.k / 32 ..][0 .. m.k / 32], first);
        }
    }

    fn upload(p: *Proj) !void {
        const up = p.v.which == .up;
        const pl = p.plan.?;
        const m1 = p.m1.?;
        const w1p = try p.rig.input(m1.codes);
        const w1s = try p.rig.input(m1.scales);
        var refs = [4]rig_.Ref{ w1p, w1s, w1p, w1s };
        if (up) {
            refs[2] = try p.rig.input(p.m3.?.codes);
            refs[3] = try p.rig.input(p.m3.?.scales);
        }
        for (refs, 0..) |r, i| p.backing[i] = r.buf;
        p.nback = 4;
        var table: [backing_experts]ex.ExpertPtrs = undefined;
        const per = m1.n * m1.k / 2;
        for (&table, 0..) |*t, e| t.* = .{
            .w1p = address(refs[0]) + e * per,
            .w1s = address(refs[1]) + e * per / 16,
            .w3p = address(refs[2]) + e * per,
            .w3s = address(refs[3]) + e * per / 16,
            .w2p = address(refs[0]) + e * per,
            .w2s = address(refs[1]) + e * per / 16,
        };
        p.table = try p.rig.input(std.mem.sliceAsBytes(&table));
        p.xr = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.rt = .{
            .ids = try p.rig.input(std.mem.sliceAsBytes(pl.ids)),
            .weights = try p.rig.input(std.mem.sliceAsBytes(pl.weights)),
            .slots = try p.rig.input(std.mem.sliceAsBytes(pl.slots)),
            .nslots = try p.rig.input(std.mem.sliceAsBytes(&pl.nslots)),
            .pairs = try p.rig.input(std.mem.sliceAsBytes(pl.pairs)),
            .where = try p.rig.input(std.mem.sliceAsBytes(pl.where)),
        };
        p.out = try p.rig.output(p.count() * 2);
        if (p.v.coded) {
            p.xq = try p.rig.output(p.shape.rows * p.shape.k * 2);
            p.xqs = try p.rig.output(p.shape.rows * p.shape.k / 8);
        }
    }

    fn at(r: rig_.Ref) ex.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }

    fn route(p: *const Proj) ex.Route {
        const r = p.rt;
        return .{ .ids = at(r.ids), .weights = at(r.weights), .slots = at(r.slots), .nslots = at(r.nslots), .pairs = at(r.pairs), .where = at(r.where) };
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        ex.use(pass.e, p.backing[0..p.nback]);
        if (p.v.coded) {
            ex.quant(&p.pipes, pass.e, at(p.xr), at(p.xq), at(p.xqs), p.call());
            ex.upCoded(&p.pipes, pass.e, at(p.xq), at(p.xqs), at(p.table), p.route(), at(p.out), p.call());
        } else if (p.v.which == .up) ex.up(&p.pipes, pass.e, at(p.xr), at(p.table), p.route(), at(p.out), p.call()) else ex.down(&p.pipes, pass.e, at(p.xr), at(p.table), p.route(), at(p.out), p.call());
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.pairs() * p.shape.n;
    }

    fn blocks(a: std.mem.Allocator, n: usize) ![]usize {
        const all = try a.alloc(usize, n / 32);
        for (all, 0..) |*b, i| b.* = i;
        return all;
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const a = p.a;
        const n = p.count();
        const nom = try a.alloc(f32, n);
        p.nom = nom;
        p.spread = try a.alloc(f64, n);
        p.mag = try a.alloc(f64, n);
        const all = try blocks(a, p.shape.n);
        defer a.free(all);
        if (p.v.which == .up) {
            @memset(p.spread, 0);
            try ref.upAct(a, p.x, p.plan.?, p.m1.?, p.m3.?, all, .{ .act = nom, .spread = p.spread });
            for (want, mag, nom, 0..) |*w, *m, v, i| {
                w.* = v;
                m.* = @abs(v);
                p.mag[i] = m.*;
            }
            return;
        }
        const ex_f = try a.alloc(f64, n);
        defer a.free(ex_f);
        const act = try a.alloc(f32, p.x.len);
        defer a.free(act);
        for (act, p.x) |*y, h| y.* = c.fromBf16(h);
        try ref.downOut(act, p.plan.?, p.m1.?, all_cols(p), nom, ex_f, p.mag);
        @memcpy(mag, p.mag);
        if (!p.real_fixture) {
            for (want, nom) |*w, v| w.* = v;
            return;
        }
        var one: [64]f64 = undefined;
        try ref.npyF64(ref.fixture.ref_npy, &one);
        for (0..p.pairs()) |pos| @memcpy(want[pos * 64 ..][0..64], &one);
    }

    var cols_cache: [5120]usize = blk: {
        @setEvalBranchQuota(20000);
        var v: [5120]usize = undefined;
        for (&v, 0..) |*y, i| y.* = i;
        break :blk v;
    };

    fn all_cols(p: *const Proj) []const usize {
        return cols_cache[0..p.shape.n];
    }

    /// Up must sit inside exp's spread; Kimi's down must equal the oracle's bytes (the fixture keeps float64's bound).
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.real_fixture) return;
        if (p.v.which == .up) @memcpy(allow, p.spread) else @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = p.a;
        const m1 = p.m1.?;
        const k = p.shape.k;
        const t = try a.alloc(f64, 2 * k);
        defer a.free(t);
        if (p.v.which == .down) {
            const xv = try a.alloc(f32, k);
            defer a.free(xv);
            for (0..p.pairs()) |pos| {
                for (xv, p.x[pos * k ..][0..k]) |*y, h| y.* = c.fromBf16(h);
                const ex_i = p.plan.?.expertAt(pos);
                for (0..p.shape.n) |j| {
                    ref.terms(m1.row(ex_i, j), m1.srow(ex_i, j), xv, null, k, t);
                    out[pos * p.shape.n + j] = ref.bf(@floatCast(contract.rows.sumIn(acc, order, t[0..k])));
                }
            }
            return;
        }
        const qv = try a.alloc(f32, k);
        defer a.free(qv);
        const qs = try a.alloc(f32, k / 32);
        defer a.free(qs);
        for (0..p.pairs()) |pos| {
            const pr = p.plan.?.pairs[pos];
            const ex_i = p.plan.?.ids[pr];
            ref.quantRow(p.x[pr / topk * k ..][0..k], qv, qs);
            for (0..p.shape.n / 32) |b| {
                var v: [32]f32 = undefined;
                for (&v, 0..) |*y, j| {
                    const col = 32 * b + j;
                    ref.terms(m1.row(ex_i, col), m1.srow(ex_i, col), qv, qs, k, t[0..k]);
                    ref.terms(p.m3.?.row(ex_i, col), p.m3.?.srow(ex_i, col), qv, qs, k, t[k..]);
                    const gs: f32 = @floatCast(contract.rows.sumIn(acc, order, t[0..k]));
                    const us: f32 = @floatCast(contract.rows.sumIn(acc, order, t[k..]));
                    y.* = ref.swiglu(gs, us, p.plan.?.weights[pr], 1);
                }
                var q: [32]f32 = undefined;
                ref.quantBlock(&v, &q);
                for (q, 0..) |y, j| out[pos * p.shape.n + 32 * b + j] = y;
            }
        }
    }
};

/// Every registered variant, then the pipeline and timing probes that run after the gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .which = .up }, .name = base ++ "up: fp8 activation, clamped SwiGLU" },
        .{ .v = .{ .which = .up, .ctl = 1 }, .name = base ++ "control: up non-finite inputs flushed" },
        .{ .v = .{ .which = .up, .ctl = 2 }, .name = base ++ "control: up bf16 step before act_quant dropped" },
        .{ .v = .{ .which = .up, .ctl = 3 }, .name = base ++ "control: up sums rounded to fp16" },
        .{ .v = .{ .which = .up, .ctl = 4 }, .name = base ++ "control: up writes past its output" },
        .{ .v = .{ .which = .up, .coded = true }, .name = base ++ "up coded: act_quant once, then the gate/up a row at a time" },
        .{ .v = .{ .which = .up, .coded = true, .ctl = 1 }, .name = base ++ "control: up coded non-finite inputs flushed" },
        .{ .v = .{ .which = .up, .coded = true, .ctl = 2 }, .name = base ++ "control: up coded bf16 step before act_quant dropped" },
        .{ .v = .{ .which = .up, .coded = true, .ctl = 3 }, .name = base ++ "control: up coded sums rounded to fp16" },
        .{ .v = .{ .which = .up, .coded = true, .ctl = 4 }, .name = base ++ "control: up coded writes past its output" },
        .{ .v = .{ .which = .down }, .name = base ++ "down: Kimi's chains a slot of one row or of more, at V4.1 shapes" },
        .{ .v = .{ .which = .down, .wide = true }, .name = base ++ "down: full E8M0 range" },
        .{ .v = .{ .which = .down, .fixture = true }, .name = base ++ "down: release expert fixture" },
        .{ .v = .{ .which = .down, .ctl = 1 }, .name = base ++ "control: down non-finite inputs flushed" },
        .{ .v = .{ .which = .down, .ctl = 2 }, .name = base ++ "control: down last K group dropped" },
        .{ .v = .{ .which = .down, .ctl = 3 }, .name = base ++ "control: down sums rounded to fp16" },
        .{ .v = .{ .which = .down, .ctl = 4 }, .name = base ++ "control: down writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    pub fn pipeline(s: *Suite, seed: u64) !usize {
        return pipe.run(s.a, s.device, s.queue, seed);
    }

    pub fn timing(s: *Suite, io: std.Io) !void {
        return pipe.timing(s.a, s.device, s.queue, io);
    }
};
