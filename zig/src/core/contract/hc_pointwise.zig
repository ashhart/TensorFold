//! Branch formulas have exact BF16 references; output gates have binary64 bounds for their stated rounding nodes.
const std = @import("std");
const h = @import("../hc_pointwise.zig");
const q = @import("../quant/quant.zig");
pub const Control = enum(u2) { production, nan, repeat, tail };
pub const Variant = struct { op: h.Op, rounding: h.Rounding, streams: c_int = 1, paired: bool = false };
pub const variants = [_]Variant{
    .{ .op = .branch, .rounding = .native, .streams = 2 },        .{ .op = .branch, .rounding = .hf, .streams = 2 },
    .{ .op = .branch, .rounding = .native, .streams = 4 },        .{ .op = .branch, .rounding = .hf, .streams = 4 },
    .{ .op = .branch, .rounding = .native, .streams = 8 },        .{ .op = .branch, .rounding = .hf, .streams = 8 },
    .{ .op = .output_gate, .rounding = .native },                 .{ .op = .output_gate, .rounding = .hf },
    .{ .op = .output_gate, .rounding = .native, .paired = true }, .{ .op = .output_gate, .rounding = .hf, .paired = true },
};
fn value(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}
fn rounded(v: f64) f64 {
    return value(q.convert.bf16(@floatCast(v)));
}
fn sigmoid(g: f64) f64 {
    const e = @exp(-@abs(g));
    return (if (g < 0) e else @as(f64, 1)) / (1 + e);
}
pub const Fixture = struct {
    p: h.Params,
    x: []u16,
    b: []u16,
    g: []u16,
    pub fn init(a: std.mem.Allocator, p: h.Params, fill: anytype, seed: u64) !Fixture {
        const sizes = try h.sizes(p);
        var rng = std.Random.DefaultPrng.init(seed);
        const x = try a.alloc(u16, sizes[0] / 2);
        const b = try a.alloc(u16, sizes[1] / 2);
        const g = try a.alloc(u16, sizes[2] / 2);
        for ([_][]u16{ x, b, g }) |values| {
            for (values) |*v| v.* = q.convert.bf16(@as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -64, 64))) / 8);
        }
        if (fill == .nonfinite) {
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x0000 }, 0..) |bits, i| if (i < x.len) {
                x[i] = bits;
            };
            if (b.len > 3) b[3] = 0x7fc0;
        }
        x[x.len - 1] = 0x4100;
        return .{ .p = p, .x = x, .b = b, .g = g };
    }
    fn factors(f: Fixture, i: usize) struct { x: f64, b: f64, g: f64 } {
        const d: usize = @intCast(f.p.hidden);
        const s: usize = @intCast(f.p.streams);
        const gd: usize = @intCast(f.p.gate_dim);
        const bi = if (f.p.op == .branch) (i / (d * s)) * d + i % d else if (gd > 0) (i / d) * @as(usize, @intCast(f.p.gate_stride)) + ((i % d) / gd) * (2 * gd) + gd + (i % d) % gd else i;
        return .{ .x = value(f.x[i]), .b = value(f.b[bi]), .g = value(f.g[if (f.p.op == .branch) (i / (d * s)) * s + (i / d) % s else 0]) };
    }
    pub fn reference(f: Fixture, want: []f64, mag: []f64) void {
        for (want, mag, 0..) |*w, *m, i| {
            const v = f.factors(i);
            if (f.p.op == .branch) {
                const product = v.b * v.g;
                w.* = rounded(if (f.p.round_mid == .hf) v.x + rounded(product) else @as(f32, @floatCast(v.x + product)));
            } else w.* = v.x * (if (f.p.round_mid == .hf) rounded(sigmoid(v.b)) else sigmoid(v.b));
            m.* = @abs(w.*);
        }
    }
};
pub fn Case(comptime contract: type, comptime B: type) type {
    const c = contract.spec;
    const g = contract.gates;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        variant: Variant,
        control: Control = .production,
        label: [160]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repetition: c_int = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*arena| arena.deinit();
        }
        fn depth(_: c.Shape) usize {
            return 0;
        }
        pub fn kernel(s: *Self) g.Kernel {
            const label = std.fmt.bufPrint(&s.label, "{s} HC pointwise {t} {t} streams{d} paired{} {t}", .{ B.name, s.variant.op, s.variant.rounding, s.variant.streams, s.variant.paired, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
            }, .spec = .{
                .name = label,
                .accum = .none,
                .depth = depth,
                .ops = 32,
                .out = .bf16,
                .in_place = true,
                .tiles = .{ .{ .size = 8 }, if (s.variant.paired) .{ .size = 256, .step = 256, .min = 256 } else .{ .size = 256 }, .{} },
                .served = if (s.variant.paired) &.{ .{ .rows = 4, .n = 512, .k = 1 }, .{ .rows = 16, .n = 6144, .k = 1 } } else &.{ .{ .rows = 4, .n = 257, .k = 1 }, .{ .rows = 16, .n = 5120, .k = 1 } },
                .sums = .{ .why = "Each output has one scalar FMA or product, with no iterative accumulation." },
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(s: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.n > 6144 or shape.k != 1) return false;
            if (s.variant.paired and shape.n % 256 != 0) return false;
            s.rig.reset();
            if (s.arena) |*arena| arena.deinit();
            s.arena = .init(s.a);
            s.repetition = 0;
            s.fixture = try Fixture.init(s.arena.?.allocator(), .{ .rows = @intCast(shape.rows), .hidden = @intCast(shape.n), .streams = s.variant.streams, .op = s.variant.op, .round_mid = s.variant.rounding, .gate_stride = if (s.variant.paired) @intCast(2 * shape.n + 16) else 0, .gate_dim = if (s.variant.paired) 256 else 0 }, fill, seed);
            try s.backend.bind(&s.rig, s.fixture);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.x.len;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture.p, s.control, s.repetition);
            s.repetition += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            s.fixture.reference(want, mag);
        }
        pub fn bounds(s: *Self, want: []const f64, mag: []const f64, allow: []f64) !void {
            const p = s.fixture.p;
            for (want, mag, allow, 0..) |w, m, *limit, i| {
                if (p.op == .branch) {
                    limit.* = 0;
                    continue;
                }
                limit.* = c.bound(s.kernel().spec, .{ .rows = @intCast(p.rows), .n = @intCast(p.hidden), .k = 1 }, w, m);
                if (p.round_mid == .hf) {
                    const v = s.fixture.factors(i);
                    const sig = sigmoid(v.b);
                    const delta = @abs(sig) * 12 * 0x1p-24 + 0x1p-148;
                    const mid = rounded(sig);
                    const err = @abs(v.x) * @max(@abs(rounded(@max(@as(f64, 0), sig - delta)) - mid), @abs(rounded(@min(@as(f64, 1), sig + delta)) - mid));
                    limit.* += err * (1 + 0x1p-8);
                }
            }
        }
    };
}
