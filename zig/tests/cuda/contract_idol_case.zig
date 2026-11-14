//! A test of one Idol kernel under the contract, from its `Op`: inputs, launch and float64 reference with ranges.
const std = @import("std");
const cuda = @import("cuda");
const idol = @import("idol_cuda");
const rig_ = @import("contract_rig.zig");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const g = r.g;

/// The gates' hooks over an Op: its kinds, out, in_place (input the output starts as), spec, sizes, fill, launch, ref.
pub fn Case(comptime Op: type) type {
    const n = Op.kinds.len;
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        k: idol.Kernels,
        name: []const u8,
        /// The Op's own form (a norm's mode, the bias's GELU), passed to each of its functions.
        variant: u32 = 0,
        control: ?g.Gate = null,
        shape: c.Shape = undefined,
        fill: g.Fill = .random,
        vals: [n][]f64 = @splat(&.{}),
        dev: [n]u64 = @splat(0),
        out: u64 = 0,
        ends: r.Ends,

        pub fn deinit(t: *Self) void {
            t.free();
            t.ends.deinit();
            t.rig.deinit();
        }

        fn free(t: *Self) void {
            for (&t.vals) |*v| {
                t.rig.gpu.gpa.free(v.*);
                v.* = &.{};
            }
        }

        pub fn kernel(t: *Self) g.Kernel {
            return .{ .spec = Op.spec(t.name, t.variant), .ctx = t, .vt = &vt, .control = t.control };
        }
        const vt = g.hooks(Self);

        pub fn prepare(t: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (!Op.accepts(shape)) return false;
            t.rig.reset();
            t.free();
            t.shape = shape;
            t.fill = fill;
            const a = t.rig.gpu.gpa;
            for (&t.vals, Op.sizes(shape)) |*v, len| {
                v.* = try a.alloc(f64, len);
                @memset(v.*, 0);
            }
            var prng = std.Random.DefaultPrng.init(seed);
            Op.fill(prng.random(), fill, shape, t.variant, &t.vals);
            for (&t.vals, &t.dev, Op.kinds) |v, *d, kind| {
                const bytes = try encode(a, kind, v);
                defer a.free(bytes);
                d.* = try t.rig.input(bytes);
            }
            t.out = try t.rig.output(Op.count(shape) * @as(usize, if (Op.out == .f16) 2 else 4));
            return true;
        }

        /// The values as the device holds them (each rounded to its type), and their bytes.
        fn encode(a: std.mem.Allocator, kind: r.Kind, v: []f64) ![]u8 {
            const size: usize = if (kind == .f16) 2 else 4;
            const out = try a.alloc(u8, v.len * size);
            for (v, 0..) |*x, i| switch (kind) {
                .f16 => {
                    const hv: f16 = @floatCast(@as(f32, @floatCast(x.*)));
                    std.mem.writeInt(u16, out[2 * i ..][0..2], @bitCast(hv), .little);
                    x.* = hv;
                },
                .f32 => {
                    const fv: f32 = @floatCast(x.*);
                    std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(fv), .little);
                    x.* = fv;
                },
            };
            return out;
        }

        pub fn run(t: *Self) !void {
            if (Op.in_place) |i| {
                const bytes = t.count() * @as(usize, if (Op.out == .f16) 2 else 4);
                try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.out, .len = bytes }).copyFrom(0, t.dev[i], bytes, t.rig.s.handle);
            }
            try Op.launch(&t.k, t.rig.s, t.dev, t.out, t.shape, t.variant);
            try t.rig.s.synchronize();
        }

        pub fn count(t: *Self) usize {
            return Op.count(t.shape);
        }

        fn constVals(t: *Self) [n][]const f64 {
            var v: [n][]const f64 = undefined;
            for (&v, t.vals) |*o, x| o.* = x;
            return v;
        }

        pub fn reference(t: *Self, want: []f64, mag: []f64) !void {
            try t.ends.size(t.count());
            try Op.ref(t.rig.gpu.gpa, t.shape, t.variant, t.constVals(), .{ .range = t.fill == .stress }, &t.ends, want, mag);
        }

        pub fn bounds(t: *Self, want: []const f64, _: []const f64, allow: []f64) !void {
            t.ends.bounds(want, allow);
        }

        pub fn intervals(t: *Self, low: []f64, high: []f64) !void {
            t.ends.intervals(low, high);
        }

        pub fn simulate(t: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            const a = t.rig.gpu.gpa;
            const mag = try a.alloc(f64, out.len);
            defer a.free(mag);
            var scratch: r.Ends = .{ .a = a };
            defer scratch.deinit();
            try scratch.size(t.count());
            try Op.ref(a, t.shape, t.variant, t.constVals(), .{ .sim = .{ .acc = acc, .order = order } }, &scratch, out, mag);
        }
    };
}

/// The usual registration of a test: its rig on the stream, the kernels (or a control's copy), name and form.
pub fn make(comptime Op: type, rig: rig_.Rig, k: idol.Kernels, name: []const u8, variant: u32, control: ?g.Gate) Case(Op) {
    return .{ .rig = rig, .k = k, .name = name, .variant = variant, .control = control, .ends = .{ .a = rig.gpu.gpa } };
}
