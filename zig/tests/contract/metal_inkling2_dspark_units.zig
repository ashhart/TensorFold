//! Mini DSpark's tensor-unit block attention against float64: the chunk partials with their fold, and the fold alone.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_dspark_host;
const controls = @import("metal_inkling2_dspark_units_controls.zig");

const Q = host.query_width;
const KV = host.kv_width;
const D = 128;
const pairs = 8 * 24;
const local_ring = 1152;
const scale: f64 = @as(f32, 0x1.6a09e6p-4);
const log2e: f64 = @as(f32, 1.4426950408889634);

pub const Kind = enum { attend, merge };

pub fn label(kind: Kind) []const u8 {
    return switch (kind) {
        .attend => "metal inkling2 dspark_units tf_ink2_dspark_attn_units",
        .merge => "metal inkling2 dspark_units tf_ink2_dspark_attn_merge",
    };
}

fn noSums(_: c.Shape) usize {
    return 1;
}
fn hr(b: rig_.Ref) host.Ref {
    return .{ .buf = b.buf, .off = b.off };
}
fn bf(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

// n is the committed length and k twice the first kept position, plus one on a sliding layer.
const attend_shapes = [_]c.Shape{
    .{ .rows = 8, .n = 0, .k = 0 },
    .{ .rows = 8, .n = 1, .k = 0 },
    .{ .rows = 8, .n = 3, .k = 5 },
    .{ .rows = 8, .n = 300, .k = 140 },
    .{ .rows = 8, .n = 700, .k = 0 },
    .{ .rows = 8, .n = 1025, .k = 1 },
    .{ .rows = 8, .n = 1025, .k = 7 },
    .{ .rows = 8, .n = 5000, .k = 1 },
};
// n is the chunks folded after the block's own keys.
const merge_shapes = [_]c.Shape{ .{ .rows = 8, .n = 0, .k = 0 }, .{ .rows = 8, .n = 1, .k = 0 }, .{ .rows = 8, .n = 5, .k = 0 } };

fn shapes(kind: Kind) []const c.Shape {
    return if (kind == .attend) &attend_shapes else &merge_shapes;
}

/// One entry: its launch on planted inputs, then every output against the float64 sum and its stated bound.
pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    units: host.Units,
    kind: Kind,
    control: ?g.Gate,
    shape: c.Shape = undefined,
    nonfinite: bool = false,
    f: [7][]f32 = @splat(&.{}),
    refs: [6]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const text = try controls.apply(a, if (control) controls.of(kind) else .none);
        defer a.free(text);
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .units = try host.Units.load(device, text), .kind = kind, .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Probe) void {
        t.rig.deinit();
        t.arena.deinit();
        t.units.deinit();
    }
    pub fn kernel(t: *Probe) g.Kernel {
        return .{ .ctx = t, .vt = &vt, .control = t.control, .spec = .{
            .name = if (t.control != null) controls.of(t.kind).name() else label(t.kind),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .nan = .{ .stated = "a NaN in a query makes its (row, head)'s 128 outputs NaN and no other pair's" },
            .served = shapes(t.kind),
        } };
    }
    const vt = g.hooks(Probe);

    fn local(t: *const Probe) bool {
        return t.shape.k % 2 != 0;
    }
    fn ring(t: *const Probe) usize {
        return if (t.local()) local_ring else std.mem.alignForward(usize, @max(t.shape.n, 1), 128);
    }
    fn span(t: *const Probe) struct { first: usize, chunks: usize } {
        const s = host.unitsSpan(@intCast(t.shape.n), t.local(), @intCast(t.shape.k / 2));
        return .{ .first = s.first, .chunks = s.chunks };
    }

    fn floats(t: *Probe, n: usize, rng: std.Random, width: f32) ![]f32 {
        const values = try t.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = bf((rng.float(f32) - 0.5) * width);
        return values;
    }
    fn input(t: *Probe, values: []const f32) !rig_.Ref {
        const bits = try t.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return t.rig.input(std.mem.sliceAsBytes(bits));
    }
    /// The direction every query leans to: a key along it weighs in every pair's sum.
    fn lean(d: usize) f32 {
        return if ((d * 2654435761) >> 7 & 1 == 0) 1 else -1;
    }
    fn plant(row: []f32, weight: f32, rng: std.Random) void {
        for (row, 0..) |*v, d| v.* = bf(weight * lean(d % D) + (rng.float(f32) - 0.5) * 0.02);
    }

    pub fn prepare(t: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        var admitted = false;
        for (shapes(t.kind)) |served| admitted = admitted or std.meta.eql(shape, served);
        if (!admitted) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        t.nonfinite = fill == .nonfinite;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Queries are wide and lean one way: scores spread over several nats, and a planted key weighs in every sum.
        t.f[0] = try t.floats(8 * Q, rng, 48);
        for (t.f[0], 0..) |*v, i| v.* = bf(v.* + 16 * lean(i % D));
        if (t.nonfinite) t.f[0][0] = std.math.nan(f32);
        t.f[1] = try t.floats(8 * KV, rng, 0.5);
        t.f[2] = try t.floats(8 * KV, rng, 0.5);
        plant(t.f[1][3 * KV ..][0..KV], 0.05, rng);
        for (0..3) |j| t.refs[j] = try t.input(t.f[j]);
        t.out = try t.rig.output(8 * Q * 2);
        if (t.kind == .merge) {
            const slots = pairs * shape.n;
            t.f[3] = try t.arena.allocator().alloc(f32, slots * 2 + slots * D);
            for (0..slots) |s| {
                const mass = 1 + rng.float(f32) * 29;
                t.f[3][2 * s] = 8 + rng.float(f32) * 10;
                t.f[3][2 * s + 1] = mass;
                for (t.f[3][slots * 2 + s * D ..][0..D]) |*v| v.* = (rng.float(f32) - 0.5) * 0.5 * mass;
            }
            t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.f[3]));
            return true;
        }
        const size = t.ring();
        const sp = t.span();
        t.f[3] = try t.floats(size * KV, rng, 0.5);
        t.f[4] = try t.floats(size * KV, rng, 0.5);
        if (shape.n > sp.first) {
            // The first and last kept keys weigh in every sum; a key just outside them would swamp it if it were read.
            plant(t.f[3][(sp.first % size) * KV ..][0..KV], 0.05, rng);
            plant(t.f[3][((shape.n - 1) % size) * KV ..][0..KV], 0.05, rng);
            const room = shape.n - sp.first < size;
            if (sp.first > 0 and room) {
                plant(t.f[3][((sp.first - 1) % size) * KV ..][0..KV], 0.5, rng);
                @memset(t.f[4][((sp.first - 1) % size) * KV ..][0..KV], 0.25);
            }
            if (room) {
                plant(t.f[3][(shape.n % size) * KV ..][0..KV], 0.5, rng);
                @memset(t.f[4][(shape.n % size) * KV ..][0..KV], 0.25);
            }
        }
        t.refs[3] = try t.input(t.f[3]);
        t.refs[4] = try t.input(t.f[4]);
        t.refs[5] = try t.rig.scratch(@max(host.unitsBytes(sp.chunks), 16));
        return true;
    }

    pub fn run(t: *Probe) !void {
        const pass = t.rig.begin();
        const R = t.refs;
        switch (t.kind) {
            .attend => host.attendUnits(&t.units, pass.e, hr(R[0]), hr(R[1]), hr(R[2]), hr(R[3]), hr(R[4]), hr(R[5]), hr(t.out), @intCast(t.shape.n), @intCast(t.ring()), t.local(), @intCast(t.shape.k / 2)),
            .merge => host.mergeUnits(&t.units, pass.e, hr(R[0]), hr(R[1]), hr(R[2]), hr(R[3]), hr(t.out), .{ .committed = 0, .ring = 64, .first = 0, .chunk = host.units_chunk, .chunks = @intCast(t.shape.n) }),
        }
        try t.rig.finish(pass);
    }

    pub fn count(_: *Probe) usize {
        return 8 * Q;
    }
    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    fn score(t: *const Probe, group: usize, keys: []const f32) f64 {
        var dot: f64 = 0;
        for (t.f[0][group * D ..][0..D], keys[0..D]) |x, y| dot += @as(f64, x) * y;
        return dot * scale;
    }

    /// Softmax over the kept context and the block's eight keys, in float64 from the bf16 inputs.
    fn attendReference(t: *Probe, group: usize, want: *[D]f64, mag: *[D]f64) !void {
        const kv = (group % 24) / 3;
        const size = t.ring();
        const first = t.span().first;
        const kept = t.shape.n - first;
        const scores = try t.a.alloc(f64, kept + 8);
        defer t.a.free(scores);
        var peak = -std.math.inf(f64);
        for (scores, 0..) |*s, j| {
            s.* = if (j < kept) t.score(group, t.f[3][((first + j) % size) * KV + kv * D ..]) else t.score(group, t.f[1][(j - kept) * KV + kv * D ..]);
            peak = @max(peak, s.*);
        }
        var den: f64 = 0;
        for (scores) |s| den += @exp(s - peak);
        want.* = @splat(0);
        mag.* = @splat(0);
        for (scores, 0..) |s, j| {
            const p = @exp(s - peak) / den;
            const values = if (j < kept) t.f[4][((first + j) % size) * KV + kv * D ..][0..D] else t.f[2][(j - kept) * KV + kv * D ..][0..D];
            for (values, want, mag) |v, *w, *m| {
                w.* += p * v;
                m.* += p * @abs(v);
            }
        }
    }

    /// The block's keys at their powers of two, then each chunk's sums at its own peak, in float64.
    fn mergeReference(t: *Probe, group: usize, want: *[D]f64, mag: *[D]f64) void {
        const kv = (group % 24) / 3;
        const slots = pairs * t.shape.n;
        var bs: [8]f64 = undefined;
        var top = -std.math.inf(f64);
        for (&bs, 0..) |*s, j| {
            s.* = t.score(group, t.f[1][j * KV + kv * D ..]) * log2e;
            top = @max(top, s.*);
        }
        for (0..t.shape.n) |chunk| top = @max(top, t.f[3][2 * (group * t.shape.n + chunk)]);
        var den: f64 = 0;
        want.* = @splat(0);
        mag.* = @splat(0);
        for (bs, 0..) |s, j| {
            const w = std.math.pow(f64, 2, s - top);
            den += w;
            for (t.f[2][j * KV + kv * D ..][0..D], want, mag) |v, *o, *m| {
                o.* += w * v;
                m.* += w * @abs(v);
            }
        }
        for (0..t.shape.n) |chunk| {
            const at = group * t.shape.n + chunk;
            const w = std.math.pow(f64, 2, @as(f64, t.f[3][2 * at]) - top);
            den += w * t.f[3][2 * at + 1];
            for (t.f[3][slots * 2 + at * D ..][0..D], want, mag) |v, *o, *m| {
                o.* += w * v;
                m.* += w * @abs(v);
            }
        }
        for (want, mag) |*o, *m| {
            o.* /= den;
            m.* /= den;
        }
    }

    /// A weight is a bf16, within 2^-8 of the softmax's above and below the line, and so is the output: 3 * 2^-8.
    const divisor = 64;

    pub fn check(t: *Probe, _: g.Gate) !?[]const u8 {
        const bytes = t.rig.bytes(t.out, 8 * Q * 2);
        var want: [D]f64 = undefined;
        var mag: [D]f64 = undefined;
        for (0..pairs) |group| {
            const poisoned = t.nonfinite and group == 0;
            if (!poisoned) switch (t.kind) {
                .attend => try t.attendReference(group, &want, &mag),
                .merge => t.mergeReference(group, &want, &mag),
            };
            for (0..D) |d| {
                const i = group * D + d;
                const got: f64 = c.fromBf16(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
                if (poisoned) {
                    if (!std.math.isNan(got)) return try std.fmt.bufPrint(&t.msg, "output {d} is {e} where its query holds a NaN", .{ i, got });
                    continue;
                }
                const bound = mag[d] / divisor;
                if (!(@abs(got - want[d]) <= bound)) return try std.fmt.bufPrint(&t.msg, "output {d} is {e} where float64 gives {e}: {d:.2} of its bound", .{ i, got, want[d], @abs(got - want[d]) / bound });
            }
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }

    /// The entries exist only where the GPU has tensor units; elsewhere the block attends on the scalar kernel.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const name = if (control) controls.of(kind).name() else label(kind);
            if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) continue;
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = try Probe.init(s.a, s.device, s.queue, kind, control);
            errdefer probe.deinit();
            try s.probes.append(s.a, probe);
            try s.kernels.append(s.a, probe.kernel());
        };
    }
};
