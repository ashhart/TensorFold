//! Inkling 2 Mini's prompt projections on the tensor units: an FP64 dot of BF16 values, stored as BF16.
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

/// The columns the kernel takes: the hidden width, and the query width `wo` reads.
const widths = [_]usize{ 2304, 3072 };
pub const name = "metal inkling2 units tf_ink2_dense_units";

fn depth(s: c.Shape) usize {
    return s.k + 3;
}

pub fn spec(control: bool) c.Spec {
    return .{
        .name = if (control) controls.Mutation.dense_store.name() else name,
        .accum = .f32,
        .depth = depth,
        .ops = 3,
        .out = .bf16,
        // Rows go 16 a band and outputs 32 a tile; 24 and 90 outputs leave the last tile part full.
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 64, .k = 2304 }, .{ .rows = 8, .n = 24, .k = 2304 }, .{ .rows = 20, .n = 90, .k = 3072 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and one BF16 store" },
    };
}

const Sim = struct { acc: c.Accum, order: g.Order };
const Sample = struct { value: f64, magnitude: f64 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    control: bool,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    refs: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Probe {
        const text = try controls.apply(a, if (control) .dense_store else .none);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, src.experts);
        errdefer pipes.deinit();
        try pipes.loadUnits(device, text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .control = control };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.x = &.{};
        p.w = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.control), .ctx = p, .vt = &vt, .control = if (p.control) controls.Mutation.dense_store.gate() else null };
    }
    const vt = blk: {
        var table = g.hooks(Probe);
        table.check = null;
        break :blk table;
    };

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 40 or shape.n == 0 or shape.n > 128 or std.mem.indexOfScalar(usize, &widths, shape.k) == null) return false;
        p.clear();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, shape.rows * shape.k);
        p.w = try p.a.alloc(u16, shape.n * shape.k);
        for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
        for (p.w) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
        if (fill == .stress) {
            // Whole numbers times a power of two: every product and partial sum is exact.
            for (p.x, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= p.x.len / 2) @as(f32, -1) else 1));
            for (p.w, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0x1p-20);
        }
        if (fill == .nonfinite and p.x.len >= 3) {
            p.x[0] = c.toBf16(std.math.nan(f32));
            p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
            p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.refs[2] = try p.rig.output(p.count() * 2);
        return true;
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        try host.denseUnits(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), @intCast(p.shape.rows), @intCast(p.shape.n), @intCast(p.shape.k));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    fn evaluate(p: *const Probe, at: usize, sim: ?Sim) Sample {
        const row = at / p.shape.n;
        const v = at % p.shape.n;
        const k = p.shape.k;
        var store: [3072]f64 = undefined;
        const terms = store[0..k];
        var magnitude: f64 = 0;
        for (terms, 0..) |*term, d| {
            term.* = @as(f64, c.fromBf16(p.w[v * k + d])) * @as(f64, c.fromBf16(p.x[row * k + d]));
            magnitude += @abs(term.*);
        }
        const sum = if (sim) |s| rows.sumIn(s.acc, s.order, terms) else blk: {
            var total: f64 = 0;
            for (terms) |term| total += term;
            break :blk total;
        };
        return .{ .value = bf(sum), .magnitude = magnitude };
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
    probes: [2]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// The entry exists only where the GPU has tensor units; elsewhere a prompt's rows take the shared matvec.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for ([_]bool{ false, true }) |control| {
            const label = spec(control).name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        }
    }
};

test "the dense entry resolves to its symbol in the units source" {
    try std.testing.expect(std.mem.startsWith(u8, name, "metal inkling2 units "));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.experts_units, "tf_ink2_dense_units("));
}
