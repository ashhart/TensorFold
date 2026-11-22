//! Inkling 2 Mini's routed experts on the tensor units: the same FP64 decode, dot and activation as the M3 entries.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const bf = contract.bf16_round.bf16;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const controls = @import("metal_inkling2_units_controls.zig");
const src = @import("inkling2_metal");

pub const Kind = enum { nv_gu, nv_down, mx_gu, mx_down };

/// The columns the kernels fix for each product, and the experts `group` lists.
const gu_k = 2304;
const down_k = 1152;
const experts = 128;
const topk = 2;

fn symbol(k: Kind) []const u8 {
    return switch (k) {
        .nv_gu => "tf_ink2_nvfp4_gu_units",
        .nv_down => "tf_ink2_nvfp4_down_units",
        .mx_gu => "tf_ink2_mxfp8_gu_units",
        .mx_down => "tf_ink2_mxfp8_down_units",
    };
}

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .nv_gu => .nv_gu_phi,
        .nv_down => .nv_down_phi,
        .mx_gu => .mx_gu_scale,
        .mx_down => .mx_down_scale,
    };
}

fn isNv(k: Kind) bool {
    return k == .nv_gu or k == .nv_down;
}
fn isGu(k: Kind) bool {
    return k == .nv_gu or k == .mx_gu;
}
/// An op is 16 columns: the shared bound on its roundings a step, then the eight partial sums and the epilogue.
fn depth(s: c.Shape) usize {
    return s.k / 16 * contract.attention_range.mma_step + 8 + 3;
}

pub fn spec(k: Kind, control: bool) c.Spec {
    const base = if (control) mutation(k).name() else switch (k) {
        .nv_gu => "metal inkling2 units tf_ink2_nvfp4_gu_units",
        .nv_down => "metal inkling2 units tf_ink2_nvfp4_down_units",
        .mx_gu => "metal inkling2 units tf_ink2_mxfp8_gu_units",
        .mx_down => "metal inkling2 units tf_ink2_mxfp8_down_units",
    };
    return .{
        .name = base,
        .accum = .f32,
        .depth = depth,
        .ops = if (isGu(k)) 12 else 3,
        .mids = if (isGu(k)) 3 else 0,
        .out = .bf16,
        // An expert's pairs go 32 a band (two 16-pair halves): 20 crosses 16 and 40 crosses 32.
        .tiles = .{ .{ .size = 32 }, .{}, .{} },
        .served = if (isGu(k))
            &.{ .{ .rows = 2, .n = 16, .k = gu_k }, .{ .rows = 5, .n = 32, .k = gu_k }, .{ .rows = 20, .n = 16, .k = gu_k }, .{ .rows = 40, .n = 16, .k = gu_k } }
        else
            &.{ .{ .rows = 2, .n = 32, .k = down_k }, .{ .rows = 5, .n = 64, .k = down_k }, .{ .rows = 20, .n = 32, .k = down_k }, .{ .rows = 40, .n = 32, .k = down_k } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and explicit BF16 nodes" },
    };
}

const Sim = struct { acc: c.Accum, order: g.Order };
const Sample = struct { value: f64, magnitude: f64 };
const Dot = struct { value: f64, magnitude: f64 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    kind: Kind,
    ctl: controls.Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u8 = &.{},
    scales: []u8 = &.{},
    amax: []f32 = &.{},
    denominator: []f32 = &.{},
    ids: []i32 = &.{},
    refs: [7]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (control) mutation(kind) else .none);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, src.experts);
        errdefer pipes.deinit();
        try pipes.loadUnits(device, text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .kind = kind, .ctl = if (control) mutation(kind) else .none };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.a.free(p.scales);
        p.a.free(p.amax);
        p.a.free(p.denominator);
        p.a.free(p.ids);
        p.x = &.{};
        p.w = &.{};
        p.scales = &.{};
        p.amax = &.{};
        p.denominator = &.{};
        p.ids = &.{};
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

    fn call(p: *const Probe) host.Call {
        return .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n), .experts = experts, .topk = topk, .format = if (isNv(p.kind)) .nvfp4 else .mxfp8 };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = p.kind;
        if (shape.rows == 0 or shape.rows > 40) return false;
        if (isGu(k) and (shape.k != gu_k or shape.n == 0 or shape.n > 32 or shape.n % 16 != 0)) return false;
        if (!isGu(k) and (shape.k != down_k or shape.n == 0 or shape.n > 64 or shape.n % 32 != 0)) return false;
        p.clear();
        p.shape = shape;
        const nv = isNv(k);
        const routes = shape.rows * topk;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, if (isGu(k)) shape.rows * shape.k else routes * shape.k);
        for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
        p.ids = try p.a.alloc(i32, routes);
        // Expert 2 takes a pair of every row and experts 0 and 1 share the other, so one expert's pairs pass 16.
        for (p.ids, 0..) |*id, i| id.* = @intCast(if (i % 2 == 0) 2 else (i / 2) % 2);
        const matrices: usize = experts * (if (isGu(k)) 2 * shape.n else shape.n);
        p.w = try p.a.alloc(u8, matrices * (if (nv) shape.k / 2 else shape.k));
        p.scales = try p.a.alloc(u8, matrices * (shape.k / @as(usize, if (nv) 16 else 32)));
        for (p.w) |*v| v.* = rng.int(u8);
        // Weights small enough that the gate and up sums stay in the activation's live range.
        for (p.scales, 0..) |*v, i| v.* = if (nv) @intCast(0x30 + (i % 17)) else @intCast(112 + (i % 5));
        if (!nv) for (p.w) |*v| if (v.* & 0x7f == 0x7f) {
            v.* = (v.* & 0x80) | 0x7e;
        };
        if (nv) {
            p.amax = try p.a.alloc(f32, experts);
            for (p.amax, 0..) |*v, i| v.* = 2688.125 + @as(f32, @floatFromInt(i % 4)) * 0.375;
            p.denominator = try p.a.alloc(f32, 1);
            p.denominator[0] = 2695.75 * 64;
        }
        if (fill == .stress) {
            for (p.x, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= p.x.len / 2) @as(f32, -1) else 1));
            @memset(p.scales, if (nv) 0x38 else 127);
        }
        if (fill == .nonfinite and p.x.len >= 3) {
            p.x[0] = c.toBf16(std.math.nan(f32));
            p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
            p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
        }
        const out = try p.rig.output(p.count() * 2);
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.refs[1] = try p.rig.input(p.w);
        p.refs[2] = try p.rig.input(p.scales);
        if (nv) {
            p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.amax));
            p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.denominator));
            p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[6] = out;
        } else {
            p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[4] = out;
        }
        return true;
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const nv = isNv(p.kind);
        const ids = hr(p.refs[if (nv) 5 else 3]);
        const out = hr(p.refs[if (nv) 6 else 4]);
        try host.group(&p.pipes, e, ids, p.call());
        e.barrier();
        try host.units(&p.pipes, e, !isGu(p.kind), hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), if (nv) .{ .amax = hr(p.refs[3]), .denominator = hr(p.refs[4]) } else null, out, p.call());
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * topk * p.shape.n;
    }

    fn e2(code: u8) f64 {
        const v = ([_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[code & 7];
        return if (code & 8 != 0) -v else v;
    }
    fn e4(code: u8) f64 {
        const exponent = (code >> 3) & 15;
        const mantissa = code & 7;
        if (exponent == 15 and mantissa == 7) return std.math.nan(f64);
        const v = if (exponent == 0) @as(f64, @floatFromInt(mantissa)) / 512 else std.math.ldexp(@as(f64, 8 + @as(u16, mantissa)), @as(i32, exponent) - 10);
        return if (code & 0x80 != 0) -v else v;
    }
    fn weight(p: *const Probe, expert: usize, row: usize, col: usize) f64 {
        const k = p.shape.k;
        const rows_per_expert: usize = if (isGu(p.kind)) 2 * p.shape.n else p.shape.n;
        const matrix_row = expert * rows_per_expert + row;
        if (!isNv(p.kind)) {
            const value = p.w[matrix_row * k + col];
            const scale = p.scales[matrix_row * (k / 32) + col / 32];
            return bf(std.math.ldexp(e4(value), @as(i32, scale) - 127));
        }
        const codes = p.w[matrix_row * (k / 2) + col / 2];
        const code = if (col % 2 == 0) codes & 15 else codes >> 4;
        const scale = p.scales[matrix_row * (k / 16) + col / 16];
        const factor = bf(p.amax[expert]) / bf(p.denominator[0]);
        return bf(@as(f32, @floatCast(e2(code) * e4(scale) * factor)));
    }

    fn dot(p: *const Probe, expert: usize, row: usize, x: []const u16, sim: ?Sim) Dot {
        var terms: [gu_k]f64 = undefined;
        var magnitude: f64 = 0;
        for (0..p.shape.k) |col| {
            const term = p.weight(expert, row, col) * @as(f64, c.fromBf16(x[col]));
            terms[col] = term;
            magnitude += @abs(term);
        }
        return .{ .value = if (sim) |s| rows.sumIn(s.acc, s.order, terms[0..p.shape.k]) else blk: {
            var sum: f64 = 0;
            for (terms[0..p.shape.k]) |term| sum += term;
            break :blk sum;
        }, .magnitude = magnitude };
    }

    fn evaluate(p: *const Probe, at: usize, sim: ?Sim) Sample {
        const pair = at / p.shape.n;
        const n = at % p.shape.n;
        const token = pair / topk;
        const expert: usize = @intCast(p.ids[pair]);
        const x = if (isGu(p.kind)) p.x[token * p.shape.k ..][0..p.shape.k] else p.x[pair * p.shape.k ..][0..p.shape.k];
        if (!isGu(p.kind)) {
            const d = p.dot(expert, n, x, sim);
            return .{ .value = bf(d.value), .magnitude = d.magnitude };
        }
        const gd = p.dot(expert, 2 * n, x, sim);
        const ud = p.dot(expert, 2 * n + 1, x, sim);
        const gate = bf(gd.value);
        const up = bf(ud.value);
        const left = 4 * std.math.tanh(gate / 4) * (1 / (1 + @exp(-gate)));
        return .{ .value = bf(left * (25 * std.math.tanh(up / 25))), .magnitude = gd.magnitude + ud.magnitude };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*v, *m, i| {
            const sample = p.evaluate(i, null);
            v.* = sample.value;
            m.* = sample.magnitude;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| v.* = p.evaluate(i, .{ .acc = acc, .order = order }).value;
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

    /// The entries exist only where the GPU has tensor units; elsewhere the M3 kernels serve every width.
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

test "contract labels resolve to the four units entries" {
    for (std.enums.values(Kind)) |kind| {
        const name = spec(kind, false).name;
        try std.testing.expect(std.mem.startsWith(u8, name, "metal inkling2 units "));
        try std.testing.expect(std.mem.indexOf(u8, src.experts_units, symbol(kind)) != null);
    }
}
