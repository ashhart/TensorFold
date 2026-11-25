//! Inkling 2 Mini's routed experts on simdgroup matrices: each output equals its scalar kernel's, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const controls = @import("metal_inkling2_mma_controls.zig");
const src = @import("inkling2_metal");

pub const Kind = enum { nv_gu, nv_down, mx_gu, mx_down };

const experts = 128;
const topk = 2;

fn symbol(k: Kind) []const u8 {
    return switch (k) {
        .nv_gu => "tf_ink2_nvfp4_gu_mma",
        .nv_down => "tf_ink2_nvfp4_down_mma",
        .mx_gu => "tf_ink2_mxfp8_gu_mma",
        .mx_down => "tf_ink2_mxfp8_down_mma",
    };
}

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .nv_gu => .nv_gu_phi,
        .nv_down => .nv_down_order,
        .mx_gu => .mx_gu_scale,
        .mx_down => .mx_down_round,
    };
}

fn isNv(k: Kind) bool {
    return k == .nv_gu or k == .nv_down;
}
fn isGu(k: Kind) bool {
    return k == .nv_gu or k == .mx_gu;
}
fn depth(_: c.Shape) usize {
    return 0;
}

const gu_shapes = [_]c.Shape{ .{ .rows = 2, .n = 32, .k = 2304 }, .{ .rows = 13, .n = 64, .k = 2304 }, .{ .rows = 45, .n = 64, .k = 2304 } };
const down_shapes = [_]c.Shape{ .{ .rows = 2, .n = 64, .k = 1152 }, .{ .rows = 13, .n = 128, .k = 1152 }, .{ .rows = 45, .n = 128, .k = 1152 } };

pub fn spec(k: Kind, control: bool) c.Spec {
    const base = if (control) mutation(k).name() else switch (k) {
        .nv_gu => "metal inkling2 mma tf_ink2_nvfp4_gu_mma",
        .nv_down => "metal inkling2 mma tf_ink2_nvfp4_down_mma",
        .mx_gu => "metal inkling2 mma tf_ink2_mxfp8_gu_mma",
        .mx_down => "metal inkling2 mma tf_ink2_mxfp8_down_mma",
    };
    return .{
        .name = base,
        .accum = .none,
        .depth = depth,
        .out = .bf16,
        .replaces = "the scalar kernels tf_ink2_{nvfp4,mxfp8}_{gu,down}, on the same weights, pairs and rows",
        .nan = .{ .stated = "every output equals the scalar kernel's: a non-finite sum has the same class" },
        // One expert takes a pair of every row, so 45 rows cross a band of 32 pairs and end it at 13 of them.
        .tiles = .{ .{ .size = 32 }, .{ .size = if (isGu(k)) 32 else 64, .step = if (isGu(k)) 32 else 64, .min = if (isGu(k)) 32 else 64 }, .{} },
        .served = if (isGu(k)) &gu_shapes else &down_shapes,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "the scalar kernels' fp32 sums are the baseline, bit for bit" },
    };
}

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
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (control) mutation(kind) else .none);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, src.experts);
        errdefer pipes.deinit();
        try pipes.loadMma(device, text);
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
    const vt = g.hooks(Probe);

    fn call(p: *const Probe) host.Call {
        return .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n), .experts = experts, .topk = topk, .format = if (isNv(p.kind)) .nvfp4 else .mxfp8 };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = p.kind;
        const unit: usize = if (isGu(k)) 32 else 64;
        if (shape.rows == 0 or shape.rows > 64 or shape.n == 0 or shape.n % unit != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        p.clear();
        p.shape = shape;
        const nv = isNv(k);
        const routes = shape.rows * topk;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, if (isGu(k)) shape.rows * shape.k else routes * shape.k);
        for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
        p.ids = try p.a.alloc(i32, routes);
        // Expert 2 takes a pair of every row and experts 0 and 1 share the other, so one expert's pairs pass a band.
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
        if (!isGu(k)) p.plant();
        if (nv) {
            p.amax = try p.a.alloc(f32, experts);
            for (p.amax, 0..) |*v, i| v.* = 2688.125 + @as(f32, @floatFromInt(i % 4)) * 0.375;
            p.denominator = try p.a.alloc(f32, 1);
            p.denominator[0] = 2695.75 * 64;
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
        p.baseline = try p.rig.scratch(p.count() * 2);
        return true;
    }

    /// Huge terms that cancel (+2^20, -2^20) in blocks of every matrix and route leave what each order of adding lost.
    fn plant(p: *Probe) void {
        const nv = isNv(p.kind);
        const unit: usize = if (nv) 16 else 32;
        const bytes_per_block: usize = if (nv) 8 else 32;
        const blocks = p.shape.k / unit;
        const big = c.toBf16(1048576);
        const up = [_]usize{ 1, 5, 9, 13, 17, 21 };
        const down = [_]usize{ 3, 7, 11, 15, 19, 23 };
        for (0..p.x.len / p.shape.k) |row| for (up ++ down) |blk| {
            p.x[row * p.shape.k + blk * unit] = big;
            p.x[row * p.shape.k + blk * unit + 1] = big;
        };
        const matrices = p.scales.len / blocks;
        for (0..matrices) |m| for (up, down) |a, b| {
            for ([_]usize{ a, b }) |blk| p.scales[m * blocks + blk] = if (nv) 0x38 else 127;
            const first = m * blocks * bytes_per_block;
            // One FP4 byte holds two columns of 1.0 or -1.0; an FP8 byte holds one.
            const plus: u8 = if (nv) 0x22 else 0x38;
            const minus: u8 = if (nv) 0xaa else 0xb8;
            for (0..if (nv) 1 else 2) |i| {
                p.w[first + a * bytes_per_block + i] = plus;
                p.w[first + b * bytes_per_block + i] = minus;
            }
        };
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    /// The products into `out`: on the simdgroup matrices, or as the scalar kernels they replace.
    fn launch(p: *Probe, out: rig_.Ref, scalar: bool) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const nv = isNv(p.kind);
        const ids = hr(p.refs[if (nv) 5 else 3]);
        const x = hr(p.refs[0]);
        const w = hr(p.refs[1]);
        const scales = hr(p.refs[2]);
        const globals: ?host.Globals = if (nv) .{ .amax = hr(p.refs[3]), .denominator = hr(p.refs[4]) } else null;
        if (scalar) {
            switch (p.kind) {
                .nv_gu => try host.nvfp4Gu(&p.pipes, e, x, w, scales, globals.?.amax, globals.?.denominator, ids, hr(out), p.call()),
                .nv_down => try host.nvfp4Down(&p.pipes, e, x, w, scales, globals.?.amax, globals.?.denominator, ids, hr(out), p.call()),
                .mx_gu => try host.mxfp8Gu(&p.pipes, e, x, w, scales, ids, hr(out), p.call()),
                .mx_down => try host.mxfp8Down(&p.pipes, e, x, w, scales, ids, hr(out), p.call()),
            }
        } else {
            try host.group(&p.pipes, e, ids, p.call());
            e.barrier();
            try host.mma(&p.pipes, e, !isGu(p.kind), x, w, scales, globals, hr(out), p.call());
        }
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.launch(p.refs[if (isNv(p.kind)) 6 else 4], false);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.launch(p.baseline, true);
        return p.rig.bytes(p.baseline, p.count() * 2);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * topk * p.shape.n;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Non-finite inputs: every output has the scalar kernel's class, and a finite one its bytes.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        try p.run();
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        for (0..p.count()) |i| {
            const a = c.fromBf16(std.mem.readInt(u16, got[2 * i ..][0..2], .little));
            const b = c.fromBf16(std.mem.readInt(u16, want[2 * i ..][0..2], .little));
            if (std.math.isNan(a) and std.math.isNan(b)) continue;
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return "an output differs from the scalar kernel's";
        }
        return null;
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

    /// Every supported Metal GPU runs the simdgroup-matrix experts from 16 rows.
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

test "contract labels resolve to the simdgroup-matrix entries" {
    for (std.enums.values(Kind)) |kind| {
        const name = spec(kind, false).name;
        try std.testing.expect(std.mem.startsWith(u8, name, "metal inkling2 mma "));
        try std.testing.expect(std.mem.indexOf(u8, src.experts_mma, symbol(kind)) != null);
    }
}
