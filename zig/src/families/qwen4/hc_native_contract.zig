//! Native HC contracts bind the real geometry, stored BF16 nodes, F32 norm weights and sum partials.
const std = @import("std");
const h = @import("hc_layout.zig");
const ct = @import("core").contract;
const c = ct.spec;
const act = @import("hc_act_reference.zig");
pub const Control = @import("hc_control_source.zig").Control;
fn bf(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}
fn sig(x: f64) f64 {
    const e = @exp(-@abs(x));
    return (if (x < 0) e else @as(f64, 1)) / (1 + e);
}
pub const Fixture = struct {
    geometry: h.Geometry,
    op: h.Op,
    p: h.Call,
    inputs: [3][]align(16) u8 = @splat(&.{}),
    count: usize,
    output_bytes: usize,
    pub fn init(a: std.mem.Allocator, geometry: h.Geometry, op: h.Op, shape: c.Shape, fill: ct.gates.Fill, seed: u64) !Fixture {
        const p: h.Call = .{ .rows = shape.rows, .width = shape.n };
        const sizes = try h.sizes(geometry, p, op);
        const count: usize = if (op == .sum) 1 else if (op == .norm) 3 else 2;
        var f: Fixture = .{ .geometry = geometry, .op = op, .p = p, .count = count, .output_bytes = sizes[count] };
        var rng = std.Random.DefaultPrng.init(seed);
        for (f.inputs[0..count], 0..) |*bytes, i| {
            bytes.* = try a.alignedAlloc(u8, .@"16", sizes[i]);
            if (op == .norm and i > 0) {
                for (std.mem.bytesAsSlice(f32, bytes.*)) |*v| v.* = 0.5 + rng.random().float(f32);
            } else for (std.mem.bytesAsSlice(u16, bytes.*), 0..) |*v, j| {
                const x: f32 = if (fill == .stress) @floatFromInt(1 + rng.random().uintLessThan(u32, 64)) else rng.random().float(f32) * 16 - 8;
                v.* = c.toBf16(x);
                if (fill == .nonfinite and j < 3) v.* = ([_]u16{ 0x7fc0, 0x7f80, 0xff80 })[j];
            }
        }
        if (fill == .stress and op == .mix) {
            const up = std.mem.bytesAsSlice(u16, f.inputs[0]);
            const x = std.mem.bytesAsSlice(u16, f.inputs[1]);
            @memset(up, c.toBf16(16));
            for (x, 0..) |*v, i| v.* = c.toBf16(([_]f32{ 4096, 1, -4096, 1 })[(i / geometry.hidden) % geometry.streams]);
        }
        return f;
    }
    fn input(f: Fixture, i: usize, at: usize) f64 {
        return c.read(if (f.op == .norm and i > 0) .f32 else .bf16, f.inputs[i], at);
    }
    pub fn reference(f: Fixture, want: []f64, mag: []f64, acc: c.Accum, order: ct.rows.Order) void {
        const g = f.geometry;
        const wide = g.hidden * g.streams;
        for (want, mag, 0..) |*out, *magnitude, i| switch (f.op) {
            .sum => {
                const stream = i % g.streams;
                const tile = (i / g.streams) % (g.hidden / 256);
                const row = i / (g.streams * (g.hidden / 256));
                var terms: [256]f64 = undefined;
                for (&terms, 0..) |*v, d| {
                    const x = f.input(0, row * wide + stream * g.hidden + tile * 256 + d);
                    v.* = x * x;
                }
                out.* = ct.rows.sumIn(acc, order, &terms);
                magnitude.* = ct.rows.sumIn(.none, .sequential, &terms);
            },
            .norm => {
                const row = i / wide;
                const stream = (i % wide) / g.hidden;
                var ss: f64 = 0;
                for (0..g.hidden / 256) |j| ss += f.input(1, (row * (g.hidden / 256) + j) * g.streams + stream);
                out.* = f.input(0, i) * f.input(2, i % wide) / @sqrt(ss / @as(f64, @floatFromInt(g.hidden)) + f.p.eps);
                magnitude.* = @abs(out.*);
            },
            .mix => {
                const row = i / g.hidden;
                const d = i % g.hidden;
                var terms: [8]f64 = undefined;
                magnitude.* = 0;
                for (terms[0..g.streams], 0..) |*v, s| {
                    const at = row * wide + s * g.hidden + d;
                    v.* = bf(bf(sig(f.input(0, at))) * f.input(1, at));
                    magnitude.* += @abs(v.*) / @as(f64, @floatFromInt(g.streams));
                }
                out.* = ct.rows.sumIn(acc, order, terms[0..g.streams]) / @as(f64, @floatFromInt(g.streams));
            },
            .act => {
                const x = f.input(0, i);
                out.* = act.point(x, f.input(1, i));
                magnitude.* = @abs(out.*);
            },
        };
    }
};
pub fn Case(comptime contract: type, comptime B: type) type {
    const gates = contract.gates;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        op: h.Op,
        control: Control = .production,
        label: [128]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repetition: u32 = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*arena| arena.deinit();
        }
        fn sumDepth(_: c.Shape) usize {
            return 40;
        }
        fn normDepth(shape: c.Shape) usize {
            return shape.n / 256;
        }
        fn mixDepth(shape: c.Shape) usize {
            return shape.k;
        }
        fn noDepth(_: c.Shape) usize {
            return 0;
        }
        pub fn kernel(s: *Self) gates.Kernel {
            const label = std.fmt.bufPrint(&s.label, "{s} Qwen4 native HC {t} {t}", .{ if (@hasDecl(B, "name")) B.name else "metal", s.op, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = if (s.op == .act) &vt else &scalar_vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = label,
                .accum = if (s.op == .act) .none else .f32,
                .out = if (s.op == .sum) .f32 else .bf16,
                .depth = switch (s.op) {
                    .sum => sumDepth,
                    .norm => normDepth,
                    .mix => mixDepth,
                    .act => noDepth,
                },
                .ops = if (s.op == .norm) 8 else if (s.op == .sum) 1 else 12,
                .tiles = if (s.op == .act) .{ .{ .size = 16 }, .{ .size = 256 }, .{} } else .{ .{ .size = 16 }, .{}, .{} },
                .served = if (s.op == .act) &.{ .{ .rows = 1, .n = 257, .k = 1 }, .{ .rows = 128, .n = 14336, .k = 1 } } else &.{ .{ .rows = 1, .n = 5120, .k = 4 }, .{ .rows = 128, .n = 5120, .k = 4 } },
                .sums = if (s.op == .norm or s.op == .act) .{ .f16 = 0, .bf16 = 0, .why = "Norm's at most20 positive partials do not separate reduced sums beyond its BF16 output bound here; act is scalar." } else .{},
            } };
        }
        const vt = gates.hooks(Self);
        const scalar_vt = blk: {
            var hooks = vt;
            hooks.bounds = null;
            hooks.intervals = null;
            break :blk hooks;
        };
        pub fn prepare(s: *Self, shape: c.Shape, fill: gates.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128) return false;
            if (s.op == .act) {
                if (shape.n == 0 or shape.n > 14336 or shape.k != 1) return false;
            } else if (shape.n != 5120 or shape.k != 4) return false;
            s.rig.reset();
            if (s.arena) |*arena| arena.deinit();
            s.arena = .init(s.a);
            s.repetition = 0;
            s.fixture = try Fixture.init(s.arena.?.allocator(), .{ .hidden = 5120, .streams = 4 }, s.op, shape, fill, seed);
            try s.backend.bind(&s.rig, s.fixture);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.output_bytes / @as(usize, if (s.op == .sum) 4 else 2);
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.op, s.fixture.p, s.repetition);
            s.repetition += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            s.fixture.reference(want, mag, .none, .sequential);
        }
        pub fn bounds(s: *Self, want: []const f64, _: []const f64, allow: []f64) !void {
            for (want, allow, 0..) |w, *limit, i| {
                const range = act.interval(s.fixture.input(0, i), s.fixture.input(1, i));
                limit.* = @max(@abs(w - range.low), @abs(w - range.high));
            }
        }
        pub fn intervals(s: *Self, low: []f64, high: []f64) !void {
            for (low, high, 0..) |*l, *hgh, i| {
                const range = act.interval(s.fixture.input(0, i), s.fixture.input(1, i));
                l.* = range.low;
                hgh.* = range.high;
            }
        }
        pub fn simulate(s: *Self, acc: c.Accum, order: contract.rows.Order, out: []f64) !void {
            const mag = try s.arena.?.allocator().alloc(f64, out.len);
            s.fixture.reference(out, mag, acc, order);
        }
    };
}
