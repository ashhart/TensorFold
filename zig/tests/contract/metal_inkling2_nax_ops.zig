//! Gates 1-4 contract probes and planted controls for Inkling 2 M5 NAX route_scores, loop_blend, and rel_table.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.dense_units;
const controls = @import("metal_inkling2_nax_ops_controls.zig");

pub const Kind = enum { route_16, route_64, loop_blend, rel_table };

const route16_shapes = [_]c.Shape{ .{ .rows = 1, .n = 128, .k = 2304 }, .{ .rows = 5, .n = 128, .k = 2304 }, .{ .rows = 16, .n = 128, .k = 2304 } };
const route64_shapes = [_]c.Shape{ .{ .rows = 1, .n = 128, .k = 2304 }, .{ .rows = 17, .n = 128, .k = 2304 }, .{ .rows = 65, .n = 128, .k = 2304 } };
const blend_shapes = [_]c.Shape{ .{ .rows = 1, .n = 2304, .k = 64 }, .{ .rows = 5, .n = 2304, .k = 64 }, .{ .rows = 17, .n = 2304, .k = 64 } };
const rel_shapes = [_]c.Shape{ .{ .rows = 1, .n = 1024, .k = 16 }, .{ .rows = 17, .n = 1024, .k = 16 }, .{ .rows = 24, .n = 1024, .k = 16 } };

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .route_16 => .route_16,
        .route_64 => .route_64,
        .loop_blend => .loop_blend,
        .rel_table => .rel_table,
    };
}

fn depthRoute(s: c.Shape) usize {
    return s.k / 128 * contract.attention_range.mma_step + 8;
}
fn depthBlend(_: c.Shape) usize {
    return 4 * contract.attention_range.mma_step + 16;
}
fn depthRel(_: c.Shape) usize {
    return contract.attention_range.mma_step + 4;
}

pub fn spec(k: Kind, control: bool) c.Spec {
    const base = if (control) mutation(k).name() else switch (k) {
        .route_16 => "metal inkling2 dense_units tf_ink2_route_scores_units_16",
        .route_64 => "metal inkling2 dense_units tf_ink2_route_scores_units_64",
        .loop_blend => "metal inkling2 dense_units tf_ink2_loop_blend_units",
        .rel_table => "metal inkling2 dense_units tf_ink2_rel_table_units",
    };
    return switch (k) {
        .route_16 => .{ .name = base, .accum = .f32, .depth = depthRoute, .out = .f32, .served = &route16_shapes },
        .route_64 => .{ .name = base, .accum = .f32, .depth = depthRoute, .out = .f32, .served = &route64_shapes },
        .loop_blend => .{ .name = base, .accum = .f32, .depth = depthBlend, .ops = 32, .mids = 0, .out = .f32, .served = &blend_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "rank-64 tanh blend uses FP32 NAX products and four-step rounding" } },
        .rel_table => .{ .name = base, .accum = .f32, .depth = depthRel, .ops = 4, .mids = 0, .out = .bf16, .served = &rel_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "16-element BF16 product rounded once to BF16" } },
    };
}

fn hr(r: rig_.Ref) host.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    kind: Kind,
    ctl: controls.Mutation,
    shape: c.Shape = undefined,
    b0: []u16 = &.{},
    b1: []u16 = &.{},
    b2: []u16 = &.{},
    f0: []f32 = &.{},
    f1: []f32 = &.{},
    refs: [5]rig_.Ref = undefined,
    out_ref: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const mut = if (control) mutation(kind) else .none;
        const text = try controls.apply(a, mut);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = try host.Pipelines.loadSource(device, text), .kind = kind, .ctl = mut };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.b0);
        p.a.free(p.b1);
        p.a.free(p.b2);
        p.a.free(p.f0);
        p.a.free(p.f1);
        p.b0 = &.{};
        p.b1 = &.{};
        p.b2 = &.{};
        p.f0 = &.{};
        p.f1 = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.ctl != .none), .ctx = p, .vt = &vt, .control = p.ctl.gate() };
    }
    const vt = blk: {
        var table = g.hooks(Probe);
        table.check = null;
        break :blk table;
    };

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.clear();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const n = shape.n;
        const k = shape.k;
        switch (p.kind) {
            .route_16, .route_64 => {
                if (n != 128 or k != 2304 or rows == 0) return false;
                if (p.kind == .route_16 and rows > 16) return false;
                p.b0 = try p.a.alloc(u16, rows * k);
                p.b1 = try p.a.alloc(u16, n * k);
                if (fill == .stress) {
                    const h = k / 2;
                    for (0..rows) |r| for (0..h) |i| {
                        const v: f32 = @floatFromInt(2 * (128 + rng.uintLessThan(u32, 127)));
                        p.b0[r * k + i] = c.toBf16(v);
                        p.b0[r * k + h + i] = c.toBf16(v + if (i < 8) @as(f32, 2) else 0);
                    };
                    for (0..n) |j| for (0..h) |i| {
                        const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 8));
                        p.b1[j * k + i] = c.toBf16(q);
                        p.b1[j * k + h + i] = c.toBf16(-q);
                    };
                } else {
                    for (p.b0) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.4));
                    for (p.b1) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.4));
                }
                if (fill == .nonfinite) {
                    p.b0[0] = c.toBf16(std.math.nan(f32));
                    p.b1[0] = c.toBf16(std.math.inf(f32));
                }
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.b0));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.b1));
                p.out_ref = try p.rig.output(rows * n * 4);
            },
            .loop_blend => {
                if (n != 2304 or k != 64 or rows == 0) return false;
                p.f0 = try p.a.alloc(f32, rows * n);
                p.f1 = try p.a.alloc(f32, rows * n);
                p.b0 = try p.a.alloc(u16, rows * k);
                p.b1 = try p.a.alloc(u16, n * k);
                p.b2 = try p.a.alloc(u16, n * k);
                if (fill == .stress) {
                    @memset(p.f0, 1.0);
                    @memset(p.f1, 1.0);
                    for (p.b0) |*v| v.* = c.toBf16(128.0);
                    for ([_][]u16{ p.b1, p.b2 }) |w| for (0..n) |d| for (0..k / 2) |col| {
                        const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                        w[d * k + col] = c.toBf16(q);
                        w[d * k + col + k / 2] = c.toBf16(if (col == 0) 1.0 / 128.0 - q else -q);
                    };
                } else {
                    for (p.f0) |*v| v.* = @floatCast(rng.floatNorm(f64) * 0.5);
                    for (p.f1) |*v| v.* = @floatCast(rng.floatNorm(f64) * 0.5);
                    for (p.b0) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.2));
                    for (p.b1) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.2));
                    for (p.b2) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.2));
                }
                if (fill == .nonfinite) {
                    p.f0[0] = std.math.nan(f32);
                    p.f1[1] = std.math.inf(f32);
                    p.b0[0] = c.toBf16(std.math.nan(f32));
                }
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.f0));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.f1));
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.b0));
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.b1));
                p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.b2));
                p.out_ref = try p.rig.output(rows * n * 4);
            },
            .rel_table => {
                if (n != 1024 or k != 16 or rows == 0) return false;
                p.b0 = try p.a.alloc(u16, rows * k);
                p.b1 = try p.a.alloc(u16, k * n);
                for (p.b0) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.5));
                for (p.b1) |*v| v.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 0.5));
                if (fill == .nonfinite) {
                    p.b0[0] = c.toBf16(std.math.nan(f32));
                    p.b1[0] = c.toBf16(std.math.inf(f32));
                }
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.b0));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.b1));
                p.out_ref = try p.rig.output(rows * n * 2);
            },
        }
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const rows: u32 = @intCast(p.shape.rows);
        switch (p.kind) {
            .route_16 => {
                pass.e.setPipeline(p.pipes.route_scores_16);
                for ([_]rig_.Ref{ p.refs[0], p.refs[1], p.out_ref }, 0..) |b, i| pass.e.setBuffer(b.buf, b.off, i);
                pass.e.setValue(host.DenseArgs{ .rows = rows, .n = 128, .k = 2304, .ldy = 128 }, 3);
                pass.e.dispatchGroups(mtl.Size.of(4, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
            },
            .route_64 => {
                pass.e.setPipeline(p.pipes.route_scores_64);
                for ([_]rig_.Ref{ p.refs[0], p.refs[1], p.out_ref }, 0..) |b, i| pass.e.setBuffer(b.buf, b.off, i);
                pass.e.setValue(host.DenseArgs{ .rows = rows, .n = 128, .k = 2304, .ldy = 128 }, 3);
                pass.e.dispatchGroups(mtl.Size.of(4, (rows + 63) / 64, 1), mtl.Size.of(256, 1, 1));
            },
            .loop_blend => host.encodeLoopBlend(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.out_ref), rows, 2304),
            .rel_table => host.encodeRelTable(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.out_ref), rows),
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const rows = p.shape.rows;
        const n = p.shape.n;
        const k = p.shape.k;
        switch (p.kind) {
            .route_16, .route_64 => for (0..rows) |r| for (0..n) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..k) |col| {
                    const term = @as(f64, c.fromBf16(p.b0[r * k + col])) * c.fromBf16(p.b1[j * k + col]);
                    sum += term;
                    abs += @abs(term);
                }
                want[r * n + j] = sum;
                mag[r * n + j] = abs;
            },
            .loop_blend => for (0..rows) |r| for (0..n) |d| {
                var fdot: f64 = 0;
                var gdot: f64 = 0;
                var fabs: f64 = 0;
                var gabs: f64 = 0;
                for (0..k) |col| {
                    const z: f64 = c.fromBf16(p.b0[r * k + col]);
                    const ft = @as(f64, c.fromBf16(p.b1[d * k + col])) * z;
                    const gt = @as(f64, c.fromBf16(p.b2[d * k + col])) * z;
                    fdot += ft;
                    gdot += gt;
                    fabs += @abs(ft);
                    gabs += @abs(gt);
                }
                const f = std.math.tanh(fdot);
                const gv = std.math.tanh(gdot);
                const hv: f64 = p.f0[r * n + d];
                const mv: f64 = p.f1[r * n + d];
                want[r * n + d] = (1.0 - f) * hv + gv * mv;
                mag[r * n + d] = (1.0 + fabs) * @abs(hv) + (1.0 + gabs) * @abs(mv) + 1.0;
            },
            .rel_table => for (0..rows) |r| for (0..n) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..k) |col| {
                    const term = @as(f64, c.fromBf16(p.b0[r * k + col])) * c.fromBf16(p.b1[col * n + j]);
                    sum += term;
                    abs += @abs(term);
                }
                want[r * n + j] = sum;
                mag[r * n + j] = abs;
            },
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const rows = p.shape.rows;
        const n = p.shape.n;
        const k = p.shape.k;
        const t = try p.a.alloc(f64, k);
        defer p.a.free(t);
        switch (p.kind) {
            .route_16, .route_64 => for (0..rows) |r| for (0..n) |j| {
                for (0..k) |col| t[col] = @as(f64, c.fromBf16(p.b0[r * k + col])) * c.fromBf16(p.b1[j * k + col]);
                out[r * n + j] = contract.rows.sumIn(acc, order, t);
            },
            .loop_blend => for (0..rows) |r| for (0..n) |d| {
                for (0..k) |col| t[col] = @as(f64, c.fromBf16(p.b1[d * k + col])) * c.fromBf16(p.b0[r * k + col]);
                const f = std.math.tanh(contract.rows.sumIn(acc, order, t));
                for (0..k) |col| t[col] = @as(f64, c.fromBf16(p.b2[d * k + col])) * c.fromBf16(p.b0[r * k + col]);
                const gv = std.math.tanh(contract.rows.sumIn(acc, order, t));
                out[r * n + d] = (1.0 - f) * @as(f64, p.f0[r * n + d]) + gv * @as(f64, p.f1[r * n + d]);
            },
            .rel_table => for (0..rows) |r| for (0..n) |j| {
                for (0..k) |col| t[col] = @as(f64, c.fromBf16(p.b0[r * k + col])) * c.fromBf16(p.b1[col * n + j]);
                out[r * n + j] = contract.rows.sumIn(acc, order, t);
            },
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [8]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const label = if (control) mutation(kind).name() else spec(kind, false).name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, kind, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
    }
};
