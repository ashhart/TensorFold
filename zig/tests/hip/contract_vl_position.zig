//! Position resize and border sampling through independent FP64 geometry, interpolation and accumulation references.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;

pub const Which = enum { horizontal, vertical, area, position };
fn horizontalDepth(s: c.Shape) usize {
    return (2 * ((128 + s.n - 1) / s.n) + 2) * 2 + 1;
}
fn verticalDepth(s: c.Shape) usize {
    return (2 * ((128 + s.rows - 1) / s.rows) + 2) * 2 + 1;
}
fn areaDepth(_: c.Shape) usize {
    return 128 * 128 + 2;
}
fn positionDepth(_: c.Shape) usize {
    return 8;
}

pub fn spec(w: Which) c.Spec {
    return .{
        .name = switch (w) {
            .horizontal => "hip tf_vl_resize horizontal AA",
            .vertical => "hip tf_vl_resize vertical AA",
            .area => "hip tf_vl_resize adaptive area",
            .position => "hip tf_vl_position content border sampling",
        },
        .accum = .f32,
        .depth = switch (w) {
            .area => areaDepth,
            .position => positionDepth,
            .horizontal => horizontalDepth,
            .vertical => verticalDepth,
        },
        .ops = 16,
        .mids = if (w == .position) 1 else 0,
        .out = if (w == .position) .bf16 else .f32,
        .served = switch (w) {
            .horizontal => &.{ .{ .rows = 128, .n = 104, .k = 1152 }, .{ .rows = 128, .n = 17, .k = 8 } },
            .vertical => &.{ .{ .rows = 113, .n = 104, .k = 1152 }, .{ .rows = 17, .n = 65, .k = 8 } },
            .area => &.{ .{ .rows = 1, .n = 73, .k = 1152 }, .{ .rows = 113, .n = 1, .k = 8 } },
            .position => &.{ .{ .rows = 128, .n = 104, .k = 1152 }, .{ .rows = 8, .n = 8, .k = 8 }, .{ .rows = 1, .n = 4, .k = 8 } },
        },
        .sums = .{ .exact = w == .area },
        .tiles = .{ .{ .size = 16 }, .{ .size = 4 }, .{ .size = 256 } },
    };
}

const Term = struct { value: f64, weight: f64, index: usize };
pub const Position = struct {
    rig: Rig,
    kernels: hip.vision.Kernels,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    input: [2]usize = undefined,
    output: [2]usize = undefined,
    content: [2]usize = undefined,
    values: []f64 = &.{},
    patches: []u16 = &.{},
    classes: []u16 = &.{},
    ptrs: [4]u64 = @splat(0),
    stress: bool = false,
    const vt = g.hooks(Position);

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, stream: hip.Stream, kernels: hip.vision.Kernels, which: Which) Position {
        return .{ .rig = .{ .a = a, .r = r, .s = stream }, .kernels = kernels, .which = which, .arena = .init(a) };
    }
    pub fn deinit(p: *Position) void {
        p.rig.deinit();
        p.arena.deinit();
    }
    pub fn kernel(p: *Position) g.Kernel {
        return .{ .spec = spec(p.which), .ctx = p, .vt = &vt };
    }
    pub fn count(p: *Position) usize {
        return (p.output[0] * p.output[1] + @as(usize, if (p.which == .position) 10 else 0)) * p.shape.k;
    }

    pub fn prepare(p: *Position, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.n > 128 or shape.k == 0 or shape.k > 4096) return false;
        if (p.which == .area and shape.rows != 1 and shape.n != 1 and fill != .nonfinite) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        p.stress = fill == .stress;
        p.output = .{ if (p.which == .area and shape.n > 1) 1 else shape.rows, shape.n };
        p.content = .{ shape.rows * 16 - 7, shape.n * 16 - 5 };
        p.input = switch (p.which) {
            .horizontal => .{ shape.rows, 128 },
            .vertical => .{ 128, shape.n },
            .area => .{ 128, 128 },
            .position => .{ shape.rows, shape.n },
        };
        p.values = try a.alloc(f64, p.input[0] * p.input[1] * shape.k);
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (p.values, 0..) |*v, i| v.* = if (fill == .stress) (if ((i / shape.k / @as(usize, if (p.which == .vertical) p.input[1] else 1)) % 2 == 0) @as(f64, 32768) else -32766) else random.float(f64) * 4 - 2;
        // Non-finites affect a known channel only, leaving all other channels finite.
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, i| {
            p.values[i * shape.k] = v;
        };
        const f = try a.alloc(f32, p.values.len);
        for (f, p.values) |*v, *wide| {
            v.* = @floatCast(wide.*);
            wide.* = v.*;
        }
        if (fill == .stress and (p.which == .position or p.which == .vertical)) {
            // Large cancelling midpoint contributions distinguish FP32 from both reduced accumulators.
            @memset(p.values, 0);
            var terms_buffer: [256]Term = undefined;
            const list = p.terms(0, &terms_buffer, true);
            for (list, 0..) |t, i| if (t.weight != 0) {
                const product: f64 = if (i < list.len / 2) 131583 else -131072;
                p.values[t.index] = @as(f32, @floatCast(product / t.weight));
            };
            for (f, p.values) |*v, wide| v.* = @floatCast(wide);
        }
        p.ptrs[0] = try p.rig.input(std.mem.sliceAsBytes(f));
        p.patches = try a.alloc(u16, shape.rows * shape.n * shape.k);
        p.classes = try a.alloc(u16, 10 * shape.k);
        @memset(p.patches, 0);
        for (p.classes, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(i % 17)) / 8);
        p.ptrs[1] = try p.rig.input(std.mem.sliceAsBytes(p.patches));
        p.ptrs[2] = try p.rig.input(std.mem.sliceAsBytes(p.classes));
        p.ptrs[3] = try p.rig.output(p.count() * if (p.which == .position) @as(usize, 2) else 4);
        return true;
    }

    pub fn run(p: *Position) !void {
        if (p.which == .position) {
            try p.kernels.position(p.rig.s, .{ .ptr = p.ptrs[1], .kind = .bf16 }, p.ptrs[2], p.ptrs[0], .{ .ptr = p.ptrs[3], .kind = .bf16 }, p.output, p.content, p.shape.k, 10);
        } else try p.kernels.resize(p.rig.s, .{ .ptr = p.ptrs[0], .kind = .f32 }, p.ptrs[3], p.input, p.output, p.shape.k, switch (p.which) {
            .horizontal => .horizontal,
            .vertical => .vertical,
            else => .area,
        });
        try p.rig.s.synchronize();
    }

    fn rounded(v: f64, active: bool) f64 {
        return if (active) @as(f32, @floatCast(v)) else v;
    }

    fn coordinate32(index: usize, pixels: usize, size: usize) f64 {
        @setFloatMode(.strict);
        const start: f32 = @floatFromInt(index * 16);
        const extent = std.math.clamp(@as(f32, @floatFromInt(pixels)) - start, 0, 16);
        const normalized = @min(start + 0.5 * extent, @as(f32, @floatFromInt(pixels))) / @as(f32, @floatFromInt(pixels));
        const grid = 2 * normalized - 1;
        return std.math.clamp(((grid + 1) * @as(f32, @floatFromInt(size)) - 1) * 0.5, 0, @as(f32, @floatFromInt(size - 1)));
    }

    fn coordinate(index: usize, pixels: usize, size: usize) f64 {
        const start: f64 = @floatFromInt(index * 16);
        const remaining = std.math.clamp(@as(f64, @floatFromInt(pixels)) - start, 0, 16);
        const normalized = @min(start + remaining / 2, @as(f64, @floatFromInt(pixels))) / @as(f64, @floatFromInt(pixels));
        return std.math.clamp(normalized * @as(f64, @floatFromInt(size)) - 0.5, 0, @as(f64, @floatFromInt(size - 1)));
    }

    fn terms(p: *Position, index: usize, buffer: []Term, geometry32: bool) []Term {
        const ch = index % p.shape.k;
        const x = index / p.shape.k % p.output[1];
        const y = index / p.shape.k / p.output[1];
        var n: usize = 0;
        if (p.which == .position) {
            const px = if (geometry32) coordinate32(x, p.content[1], p.input[1]) else coordinate(x, p.content[1], p.input[1]);
            const py = if (geometry32) coordinate32(y, p.content[0], p.input[0]) else coordinate(y, p.content[0], p.input[0]);
            const x0: usize = @intFromFloat(@floor(px));
            const y0: usize = @intFromFloat(@floor(py));
            const sx = px - @as(f64, @floatFromInt(x0));
            const sy = py - @as(f64, @floatFromInt(y0));
            for (0..2) |dy| for (0..2) |dx| {
                const at = (@min(y0 + dy, p.input[0] - 1) * p.input[1] + @min(x0 + dx, p.input[1] - 1)) * p.shape.k + ch;
                const wx = if (dx == 0) rounded(1 - sx, geometry32) else sx;
                const wy = if (dy == 0) rounded(1 - sy, geometry32) else sy;
                buffer[n] = .{ .value = p.values[at], .index = at, .weight = rounded(wx * wy, geometry32) };
                n += 1;
            };
        } else if (p.which == .area) {
            const y0 = y * p.input[0] / p.output[0];
            const y1 = ((y + 1) * p.input[0] + p.output[0] - 1) / p.output[0];
            const x0 = x * p.input[1] / p.output[1];
            const x1 = ((x + 1) * p.input[1] + p.output[1] - 1) / p.output[1];
            for (y0..y1) |yy| for (x0..x1) |xx| {
                buffer[n] = .{ .value = p.values[(yy * p.input[1] + xx) * p.shape.k + ch], .index = (yy * p.input[1] + xx) * p.shape.k + ch, .weight = 1 / @as(f64, @floatFromInt((y1 - y0) * (x1 - x0))) };
                n += 1;
            };
        } else {
            const axis: usize = if (p.which == .horizontal) 1 else 0;
            const scale = rounded(@as(f64, @floatFromInt(p.input[axis])) / @as(f64, @floatFromInt(p.output[axis])), geometry32);
            const support = @max(scale, 1);
            const center = rounded((@as(f64, @floatFromInt(if (axis == 1) x else y)) + 0.5) * scale, geometry32);
            const inv = rounded(1 / support, geometry32);
            const first: usize = @intFromFloat(@max(@trunc(center - support + 0.5), 0));
            const end: usize = @intFromFloat(@min(@trunc(center + support + 0.5), @as(f64, @floatFromInt(p.input[axis]))));
            var sum: f64 = 0;
            for (first..end) |j| {
                const distance = @abs(rounded(rounded(rounded(@as(f64, @floatFromInt(j)) - center, geometry32) + 0.5, geometry32) * inv, geometry32));
                sum = rounded(sum + @max(rounded(1 - distance, geometry32), 0), geometry32);
            }
            for (first..end) |j| {
                const at = if (axis == 1) (y * p.input[1] + j) else (j * p.input[1] + x);
                const distance = @abs(rounded(rounded(rounded(@as(f64, @floatFromInt(j)) - center, geometry32) + 0.5, geometry32) * inv, geometry32));
                buffer[n] = .{ .value = p.values[at * p.shape.k + ch], .index = at * p.shape.k + ch, .weight = rounded(@max(rounded(1 - distance, geometry32), 0) / sum, geometry32) };
                n += 1;
            }
        }
        return buffer[0..n];
    }

    fn outputs(p: *Position, want: []f64, mag: ?[]f64, sim: ?struct { c.Accum, g.Order }) !void {
        const a = p.arena.allocator();
        const buffer = try a.alloc(Term, 128 * 128);
        const sums = try a.alloc(f64, buffer.len);
        for (want, 0..) |*w, i| {
            if (p.which == .position and i < p.classes.len) {
                w.* = c.fromBf16(p.classes[i]);
                if (mag) |m| m[i] = 0;
                continue;
            }
            const terms_ = p.terms(if (p.which == .position) i - p.classes.len else i, buffer, sim != null);
            var exact: f64 = 0;
            var magnitude: f64 = 0;
            for (terms_, 0..) |t, j| {
                sums[j] = t.value * t.weight;
                exact += sums[j];
                magnitude += @abs(sums[j]);
            }
            w.* = if (sim) |how| core.contract.rows.sumIn(how[0], how[1], sums[0..terms_.len]) else exact;
            if (mag) |m| m[i] = magnitude;
        }
    }
    /// Bound fixed FP32 geometry displacement from ideal coordinates separately from the FP32 dot's error.
    pub fn bounds(p: *Position, want: []const f64, mag: []const f64, allow: []f64) !void {
        const a = p.arena.allocator();
        const buffer = try a.alloc(Term, 128 * 128);
        const s = spec(p.which);
        for (allow, want, mag, 0..) |*limit, exact, magnitude, i| {
            if (p.which == .area) {
                limit.* = if (p.stress) c.boundExact(s, exact) else c.bound(s, p.shape, exact, magnitude);
                continue;
            }
            if (p.which == .position and i < p.classes.len) {
                limit.* = 0;
                continue;
            }
            const list = p.terms(if (p.which == .position) i - p.classes.len else i, buffer, true);
            var value: f64 = 0;
            var absolute: f64 = 0;
            for (list) |t| {
                value += t.value * t.weight;
                absolute += @abs(t.value * t.weight);
            }
            limit.* = @abs(value - exact) + c.bound(s, p.shape, value, absolute);
        }
    }
    pub fn reference(p: *Position, want: []f64, mag: []f64) !void {
        return p.outputs(want, mag, null);
    }
    pub fn simulate(p: *Position, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.outputs(out, null, .{ acc, order });
    }
};
