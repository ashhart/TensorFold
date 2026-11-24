//! Inkling 2 Mini M3 contracts: independent FP64 decode/dot/activation and ascending-expert combine.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const bf = contract.bf16_round.bf16;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const controls = @import("metal_inkling2_experts_controls.zig");
const src = @import("inkling2_metal");

pub const Kind = enum { nv_gu, nv_down, mx_gu, mx_down, nv_decode, mx_decode, combine };

fn symbol(k: Kind) []const u8 {
    return switch (k) {
        .nv_gu => "tf_ink2_nvfp4_gu",
        .nv_down => "tf_ink2_nvfp4_down",
        .mx_gu => "tf_ink2_mxfp8_gu",
        .mx_down => "tf_ink2_mxfp8_down",
        .nv_decode => "tf_ink2_nvfp4_quantize",
        .mx_decode => "tf_ink2_mxfp8_quantize",
        .combine => "tf_ink2_combine",
    };
}

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .nv_gu => .nv_gu_up,
        .nv_down => .nv_down_phi,
        .mx_gu => .mx_gu_up,
        .mx_down => .mx_down_scale,
        .nv_decode => .nv_decode_nibble,
        .mx_decode => .mx_decode_scale,
        .combine => .combine_slot_order,
    };
}

fn isNv(k: Kind) bool {
    return k == .nv_gu or k == .nv_down or k == .nv_decode;
}
fn isGu(k: Kind) bool {
    return k == .nv_gu or k == .mx_gu;
}
fn isDown(k: Kind) bool {
    return k == .nv_down or k == .mx_down;
}
fn isDecode(k: Kind) bool {
    return k == .nv_decode or k == .mx_decode;
}
fn depth(s: c.Shape) usize {
    return s.k + 3;
}

pub fn spec(k: Kind, control: bool) c.Spec {
    const base = if (control) mutation(k).name() else switch (k) {
        .nv_gu => "metal inkling2 experts tf_ink2_nvfp4_gu",
        .nv_down => "metal inkling2 experts tf_ink2_nvfp4_down",
        .mx_gu => "metal inkling2 experts tf_ink2_mxfp8_gu",
        .mx_down => "metal inkling2 experts tf_ink2_mxfp8_down",
        .nv_decode => "metal inkling2 experts tf_ink2_nvfp4_quantize",
        .mx_decode => "metal inkling2 experts tf_ink2_mxfp8_quantize",
        .combine => "metal inkling2 experts tf_ink2_combine",
    };
    return .{
        .name = base,
        .accum = if (isDecode(k)) .none else .f32,
        .depth = depth,
        .ops = if (isGu(k)) 12 else if (isDecode(k)) 1 else 3,
        .mids = if (isGu(k)) 3 else 0,
        .out = .bf16,
        .served = switch (k) {
            .nv_gu, .mx_gu => &.{ .{ .rows = 2, .n = 4, .k = 32 }, .{ .rows = 4, .n = 8, .k = 64 } },
            .nv_down, .mx_down => &.{ .{ .rows = 2, .n = 8, .k = 32 }, .{ .rows = 4, .n = 12, .k = 64 } },
            .nv_decode, .mx_decode => &.{ .{ .rows = 2, .n = 32, .k = 32 }, .{ .rows = 3, .n = 64, .k = 64 } },
            .combine => &.{ .{ .rows = 2, .n = 8, .k = 8 }, .{ .rows = 4, .n = 12, .k = 8 } },
        },
        .sums = if (k == .combine) .{ .f16 = 0, .bf16 = 100, .why = "bounded cancellation is exact in fp32 but loses its residual in bf16" } else .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and explicit BF16 nodes" },
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
    weights: []f32 = &.{},
    refs: [7]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (control) mutation(kind) else .none);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = try host.Pipelines.loadSource(device, text), .kind = kind, .ctl = if (control) mutation(kind) else .none };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.a.free(p.scales);
        p.a.free(p.amax);
        p.a.free(p.denominator);
        p.a.free(p.ids);
        p.a.free(p.weights);
        p.x = &.{};
        p.w = &.{};
        p.scales = &.{};
        p.amax = &.{};
        p.denominator = &.{};
        p.ids = &.{};
        p.weights = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.ctl != .none), .ctx = p, .vt = if (p.kind == .combine and p.ctl == .combine_slot_order) &vt_exact else &vt, .control = p.ctl.gate() };
    }
    const vt = blk: {
        var table = g.hooks(Probe);
        table.check = null;
        break :blk table;
    };
    const vt_exact = g.hooks(Probe);

    fn call(p: *const Probe) host.Call {
        return .{ .rows = @intCast(p.shape.rows), .k = @intCast(if (p.kind == .combine) 32 else p.shape.k), .n = @intCast(p.shape.n), .experts = if (p.kind == .combine) 8 else 4, .topk = if (p.kind == .combine) 8 else 2, .format = if (isNv(p.kind)) .nvfp4 else .mxfp8 };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = p.kind;
        if (shape.rows == 0 or shape.rows > 8) return false;
        if (isGu(k) and (shape.n > 8 or shape.k > 64 or shape.k % 32 != 0)) return false;
        if (isDown(k) and (shape.n > 12 or shape.k > 64 or shape.k % 32 != 0)) return false;
        if (isDecode(k) and (shape.n != shape.k or shape.k > 64 or shape.k % 32 != 0)) return false;
        if (k == .combine and (shape.n > 12 or shape.k != 8)) return false;
        p.clear();
        p.shape = shape;
        const nv = isNv(k);
        const topk: usize = if (k == .combine) 8 else 2;
        const routes = shape.rows * topk;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (!isDecode(k)) {
            const nx = if (isGu(k)) shape.rows * shape.k else if (isDown(k)) routes * shape.k else routes * shape.n;
            p.x = try p.a.alloc(u16, nx);
            for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
        }
        p.ids = try p.a.alloc(i32, routes);
        const combine_ids = [_]i32{ 2, 0, 1, 7, 3, 6, 4, 5 };
        // Eight distinct, unsorted routes exercise ascending-expert accumulation.
        for (p.ids, 0..) |*id, i| id.* = if (k == .combine) combine_ids[i % 8] else @intCast(if (i % 2 == 0) 2 else (i / 2) % 2);
        const matrices: usize = if (isGu(k)) 8 * shape.n else if (isDown(k)) 4 * shape.n else shape.rows;
        if (k != .combine) {
            p.w = try p.a.alloc(u8, matrices * (if (nv) shape.k / 2 else shape.k));
            p.scales = try p.a.alloc(u8, matrices * (shape.k / @as(usize, if (nv) 16 else 32)));
            for (p.w) |*v| v.* = rng.int(u8);
            for (p.scales, 0..) |*v, i| v.* = if (nv) @intCast(0x30 + (i % 17)) else @intCast(125 + (i % 5));
            if (!nv) for (p.w) |*v| if (v.* & 0x7f == 0x7f) {
                v.* = (v.* & 0x80) | 0x7e;
            };
            if (k == .mx_decode) {
                p.w[0] = 0x30;
                p.scales[0] = 255; // Decodes to 2^127 without first forming an overflowing 2^128.
            }
        }
        if (nv) {
            p.amax = try p.a.alloc(f32, if (isDecode(k)) 1 else 4);
            for (p.amax, 0..) |*v, i| v.* = 2688.125 + @as(f32, @floatFromInt(i)) * 0.375;
            p.denominator = try p.a.alloc(f32, 1);
            p.denominator[0] = 2695.75;
        }
        if (k == .combine) p.weights = try p.a.alloc(f32, routes);
        if (fill == .stress and !isDecode(k)) {
            for (p.x, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= p.x.len / 2) @as(f32, -1) else 1));
            if (k != .combine) @memset(p.scales, if (nv) 0x38 else 127);
        }
        if (k == .combine) {
            const values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
            for (p.x, 0..) |*v, i| v.* = c.toBf16(if (fill == .stress) 1 else values[i % 8]);
            if (fill == .stress) {
                const terms = [_]f32{ 1024, 1, 1, 1, 1, 1, 1, -1029 };
                // Map expert indices back to route slots; the ascending sum is 1024 + six 1s - 1029.
                for (0..p.shape.rows) |token| for (0..8) |expert| for (0..8) |slot| if (p.ids[token * 8 + slot] == expert) {
                    p.weights[token * 8 + slot] = terms[expert];
                };
            } else if (p.ctl == .combine_slot_order) {
                const control_weights = [_]f32{ 0x1p127, 0x1p126, -0x1p127 / 3.0, 0.25, -0.2, 1.0 / 3.0, -2.0 / 7.0, 0.375 };
                for (p.weights, 0..) |*v, i| v.* = control_weights[i % 8];
                // Only this planted control overflows route order; ascending experts remain finite.
            } else {
                const ordinary_weights = [_]f32{ 0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1 };
                for (p.weights, 0..) |*v, i| v.* = ordinary_weights[i % 8];
            }
        }
        if (fill == .nonfinite) {
            if (!isDecode(k) and p.x.len >= 3) {
                p.x[0] = c.toBf16(std.math.nan(f32));
                p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
                p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
            } else if (isDecode(k) and p.scales.len > 1) {
                if (nv) p.scales[0] = 0x7f else p.w[0] = 0x7f;
            }
        }
        const out = try p.rig.output(p.count() * 2);
        if (k == .combine) {
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.weights));
            p.refs[3] = out;
        } else if (isDecode(k)) {
            p.refs[0] = try p.rig.input(p.w);
            p.refs[1] = try p.rig.input(p.scales);
            if (nv) {
                p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.amax));
                p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.denominator));
                p.refs[4] = out;
            } else p.refs[2] = out;
        } else {
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
        if (isDecode(p.kind)) {
            if (nv) try host.nvfp4Quantize(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), @intCast(p.shape.rows), @intCast(p.shape.k)) else try host.mxfp8Quantize(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), @intCast(p.shape.rows), @intCast(p.shape.k));
        } else if (isGu(p.kind)) {
            if (nv) try host.nvfp4Gu(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.refs[5]), hr(p.refs[6]), p.call()) else try host.mxfp8Gu(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), p.call());
        } else if (isDown(p.kind)) {
            if (nv) try host.nvfp4Down(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.refs[5]), hr(p.refs[6]), p.call()) else try host.mxfp8Down(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), p.call());
        } else try host.combine(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), p.call());
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return if (isGu(p.kind) or isDown(p.kind)) p.shape.rows * 2 * p.shape.n else p.shape.rows * p.shape.n;
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
        const nv = isNv(p.kind);
        const k = p.shape.k;
        const rows_per_expert: usize = if (isGu(p.kind)) 2 * p.shape.n else p.shape.n;
        const matrix_row = if (isDecode(p.kind)) row else expert * rows_per_expert + row;
        if (!nv) {
            const value = p.w[matrix_row * k + col];
            const scale = p.scales[matrix_row * (k / 32) + col / 32];
            return bf(std.math.ldexp(e4(value), @as(i32, scale) - 127));
        }
        const codes = p.w[matrix_row * (k / 2) + col / 2];
        const code = if (col % 2 == 0) codes & 15 else codes >> 4;
        const scale = p.scales[matrix_row * (k / 16) + col / 16];
        const factor = bf(p.amax[if (isDecode(p.kind)) 0 else expert]) / bf(p.denominator[0]);
        return bf(@as(f32, @floatCast(e2(code) * e4(scale) * factor)));
    }

    fn dot(p: *const Probe, expert: usize, row: usize, x: []const u16, sim: ?Sim) Dot {
        var terms: [64]f64 = undefined;
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
        if (isDecode(p.kind)) return .{ .value = p.weight(0, at / p.shape.k, at % p.shape.k), .magnitude = 0 };
        if (p.kind == .combine) {
            const token = at / p.shape.n;
            const col = at % p.shape.n;
            var terms: [8]f64 = undefined;
            var magnitude: f64 = 0;
            var made: usize = 0;
            for (0..8) |expert| for (0..8) |slot| if (p.ids[token * 8 + slot] == @as(i32, @intCast(expert))) {
                const value = p.weights[token * 8 + slot] * @as(f64, c.fromBf16(p.x[(token * 8 + slot) * p.shape.n + col]));
                terms[made] = value;
                made += 1;
                magnitude += @abs(value);
            };
            const sum = if (sim) |s| rows.sumIn(s.acc, s.order, terms[0..made]) else blk: {
                var total: f64 = 0;
                for (terms[0..made]) |term| total += term;
                break :blk total;
            };
            return .{ .value = bf(sum), .magnitude = magnitude };
        }
        const pair = at / p.shape.n;
        const n = at % p.shape.n;
        const token = pair / 2;
        const expert: usize = @intCast(p.ids[pair]);
        const x = if (isGu(p.kind)) p.x[token * p.shape.k ..][0..p.shape.k] else p.x[pair * p.shape.k ..][0..p.shape.k];
        if (isDown(p.kind)) {
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
    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes) return null;
        const bytes = try p.rig.outputBytes();
        const want = try p.a.alloc(f64, p.count());
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, p.count());
        defer p.a.free(mag);
        try p.reference(want, mag);
        for (want, 0..) |w, i| if (c.read(.bf16, bytes, i) != w) return "route-slot output differs from the ascending-expert reference";
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [14]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const label = if (control) mutation(kind).name() else spec(kind, false).name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, kind, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
    }
};

test "contract labels resolve to the seven shipped entries" {
    for (std.enums.values(Kind)) |kind| {
        const name = spec(kind, false).name;
        try std.testing.expect(std.mem.startsWith(u8, name, "metal inkling2 experts "));
        try std.testing.expect(std.mem.indexOf(u8, src.experts, symbol(kind)) != null);
    }
}
