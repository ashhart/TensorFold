//! GLM-OCR pointwise, layout and rotary launches through independent formulas, guards and repeated calls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const Ops = @import("glm_ocr_ops").Ops;
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
extern "c" fn erf(f64) f64;

pub const Which = enum { bias, fill, cast, add, silu, gelu, merge, filter, split, vision_rope, text_rope };
fn depth(_: c.Shape) usize {
    return 0;
}
fn served(comptime which: Which) []const c.Shape {
    return switch (which) {
        .filter => &.{ .{ .rows = 1, .n = 4096, .k = 1 }, .{ .rows = 1536, .n = 4096, .k = 1 } },
        .merge, .split => &.{ .{ .rows = 8, .n = 1024, .k = 1 }, .{ .rows = 20, .n = 1024, .k = 1 } },
        .vision_rope => &.{ .{ .rows = 8, .n = 1024, .k = 2 }, .{ .rows = 24, .n = 1024, .k = 6 } },
        .text_rope => &.{ .{ .rows = 1, .n = 2048, .k = 128 }, .{ .rows = 17, .n = 1024, .k = 128 } },
        else => &.{ .{ .rows = 1, .n = 1536, .k = 1 }, .{ .rows = 17, .n = 1024, .k = 1 }, .{ .rows = 4, .n = 9216, .k = 1 }, .{ .rows = 1, .n = 59392, .k = 1 } },
    };
}
fn spec(comptime which: Which) c.Spec {
    return .{ .name = "hip GLM-OCR " ++ @tagName(which), .accum = .none, .depth = depth, .ops = switch (which) {
        .silu, .gelu => 8,
        .vision_rope, .text_rope => 12,
        else => 1,
    }, .mids = if (which == .silu) 1 else 0, .out = .bf16, .served = served(which), .tiles = if (which == .filter or which == .vision_rope or which == .text_rope or which == .merge or which == .split)
        .{ .{}, .{}, .{} }
    else
        .{ .{ .size = 16 }, .{ .size = 256 }, .{} } };
}
fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub const Pointwise = struct {
    const vt = g.hooks(Pointwise);
    rig: Rig,
    ops: *const Ops,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    y: []f64 = &.{},
    inverse: [64]f32 = @splat(0),
    positions: [][3]i32 = &.{},
    dx: u64 = 0,
    dy: u64 = 0,
    di: u64 = 0,
    dp: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, stream: hip.Stream, ops: *const Ops, which: Which) Pointwise {
        return .{ .rig = .{ .a = a, .r = r, .s = stream }, .ops = ops, .which = which, .arena = .init(a) };
    }
    pub fn deinit(p: *Pointwise) void {
        p.rig.deinit();
        p.arena.deinit();
    }
    pub fn kernel(p: *Pointwise) g.Kernel {
        const specification = switch (p.which) {
            inline else => |which| comptime spec(which),
        };
        return .{ .spec = specification, .ctx = p, .vt = &vt };
    }
    pub fn count(p: *Pointwise) usize {
        return p.shape.rows * p.shape.n * @as(usize, if (p.which == .split) 3 else 1);
    }
    pub fn prepare(p: *Pointwise, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.rows > 6144 or shape.n > 59392) return false;
        switch (p.which) {
            .filter => if (shape.rows > 1536 or shape.n != 4096) return false,
            .merge, .split => if (shape.rows % 4 != 0 or shape.n != 1024) return false,
            .vision_rope => if (shape.k == 0 or shape.k % 2 != 0 or shape.rows % (2 * shape.k) != 0 or shape.n != 1024) return false,
            .text_rope => if ((shape.n != 1024 and shape.n != 2048) or shape.k != 128) return false,
            else => {},
        }
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        p.shape = shape;
        const a = p.arena.allocator();
        const size = p.count();
        p.x = try a.alloc(f64, size);
        p.y = try a.alloc(f64, @max(size, shape.n));
        var prng = std.Random.DefaultPrng.init(seed);
        for (p.x) |*v| {
            const raw = (prng.random().float(f32) * 2 - 1) * 4;
            v.* = if (p.which == .cast) raw else bf(raw);
        }
        for (p.y) |*v| v.* = bf(prng.random().float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            const plants = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (plants, 0..) |v, i| {
                p.x[@min(i * @max(size / 5, 1), size - 1)] = v;
                p.y[@min(i, p.y.len - 1)] = v;
            }
        }
        p.dx = try p.upload(a, p.x, p.which == .cast);
        p.dy = try p.upload(a, p.y, false);
        p.out = try p.rig.output(size * 2);
        if (p.which == .vision_rope or p.which == .text_rope) {
            const frequencies: usize = if (p.which == .vision_rope) 16 else 64;
            for (p.inverse[0..frequencies], 0..) |*v, i| {
                const power: f32 = @floatCast(std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(frequencies))));
                v.* = 1 / power;
            }
            p.di = try p.rig.input(std.mem.sliceAsBytes(p.inverse[0..frequencies]));
            p.positions = try a.alloc([3]i32, shape.rows);
            for (p.positions, 0..) |*pos, i| pos.* = .{ @intCast(131000 + i), @intCast(2 * i + 3), @intCast(3 * i + 7) };
            p.dp = try p.rig.input(std.mem.sliceAsBytes(p.positions));
        }
        return true;
    }
    fn upload(p: *Pointwise, a: std.mem.Allocator, values: []const f64, wide: bool) !u64 {
        const bytes = try a.alloc(u8, values.len * @as(usize, if (wide) 4 else 2));
        for (values, 0..) |v, i| if (wide) {
            std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little);
        } else std.mem.writeInt(u16, bytes[i * 2 ..][0..2], c.toBf16(@floatCast(v)), .little);
        return p.rig.input(bytes);
    }
    pub fn run(p: *Pointwise) !void {
        const s = p.rig.s;
        const n = p.count();
        if (p.which == .bias or p.which == .gelu or p.which == .vision_rope or p.which == .text_rope) {
            const dst: hip.DeviceBuffer = .{ .r = p.rig.r, .ptr = @ptrFromInt(p.out), .len = n * 2 };
            try dst.copyFromAsync(0, p.dx, n * 2, s);
        }
        switch (p.which) {
            .bias => try p.ops.addBias(s, p.out, p.dy, p.shape.rows, p.shape.n),
            .fill => try p.ops.fillBias(s, p.out, p.dy, p.shape.rows, p.shape.n),
            .cast => try p.ops.castBf16(s, p.dx, p.out, n),
            .add => try p.ops.add(s, p.dx, p.dy, p.out, n),
            .silu => try p.ops.siluMul(s, p.dx, p.dy, p.out, n),
            .gelu => try p.ops.gelu(s, p.out, n),
            .merge => try p.ops.merge(s, p.dx, p.out, p.shape.rows, p.shape.n),
            .filter => try p.ops.channelsLastFilter(s, p.dx, p.out, p.shape.rows),
            .split => try p.ops.splitQkv(s, p.dx, p.out, p.out + n / 3 * 2, p.out + n / 3 * 4, p.shape.rows, p.shape.n),
            .vision_rope => try p.ops.visionRope(s, p.out, p.di, p.shape.rows, p.shape.k),
            .text_rope => try p.ops.textRope(s, p.out, p.di, p.dp, p.shape.rows, p.shape.n / 128),
        }
        try s.synchronize();
    }
    pub fn reference(p: *Pointwise, want: []f64, mag: []f64) !void {
        const width = if (p.which == .filter) @as(usize, 1024) else p.shape.n;
        for (want, mag, 0..) |*out, *m, i| {
            const x = p.x[i];
            out.* = switch (p.which) {
                .bias => x + p.y[i % width],
                .fill => p.y[i % width],
                .cast => x,
                .add => x + p.y[i],
                .silu => bf(x / (1 + @exp(-x))) * p.y[i],
                .gelu => 0.5 * x * (1 + erf(x / @sqrt(@as(f64, 2)))),
                .merge => p.x[(i / (4 * width) * 4 + i % 4) * width + (i / 4) % width],
                .filter => p.x[i / (4 * width) * 4 * width + (i % width) * 4 + (i / width) % 4],
                .split => blk: {
                    const plane = p.shape.rows * width;
                    const row = (i % plane) / width;
                    break :blk p.x[row * 3 * width + i / plane * width + i % width];
                },
                .vision_rope, .text_rope => p.rotary(i),
            };
            m.* = @abs(out.*);
        }
    }
    fn rotary(p: *Pointwise, i: usize) f64 {
        if (p.which == .vision_rope) {
            const j = i % 64;
            const pair = j % 32;
            const row = i / 1024;
            const cell = row / 4;
            const gw = p.shape.k;
            const position = if (pair < 16) cell / (gw / 2) * 2 + (row % 4) / 2 else cell % (gw / 2) * 2 + row % 2;
            const phase: f32 = @as(f32, @floatFromInt(position)) * p.inverse[pair % 16];
            const co = @cos(@as(f64, phase));
            const si = @sin(@as(f64, phase));
            const other = p.x[if (j < 32) i + 32 else i - 32];
            return p.x[i] * co + (if (j < 32) -other else other) * si;
        }
        const pair = (i % 128) / 2;
        const row = i / p.shape.n;
        const axis: usize = if (pair < 16) 0 else if (pair < 40) 1 else 2;
        const phase: f32 = @as(f32, @floatFromInt(p.positions[row][axis])) * p.inverse[pair];
        const co = bf(@cos(@as(f64, phase)));
        const si = bf(@sin(@as(f64, phase)));
        const other = p.x[if (i % 2 == 0) i + 1 else i - 1];
        return bf(p.x[i] * co) + bf((if (i % 2 == 0) -other else other) * si);
    }
    pub fn bounds(p: *Pointwise, want: []const f64, mag: []const f64, allow: []f64) !void {
        const specification = switch (p.which) {
            inline else => |which| spec(which),
        };
        for (want, mag, allow, 0..) |w, m, *out, i| {
            out.* = c.bound(specification, p.shape, w, m);
            if (p.which == .gelu) {
                const x = @abs(p.x[i]);
                // FP32 argument error crosses erf through its bounded derivative; erf itself rounds once.
                out.* += 0.5 * x * 0x1p-24 * (1 + x * @sqrt(2.0 / std.math.pi) * @exp(-0.5 * x * x));
            }
            if (p.which == .vision_rope or p.which == .text_rope) {
                const other = if (p.which == .text_rope) (if (i % 2 == 0) i + 1 else i - 1) else (if (i % 64 < 32) i + 32 else i - 32);
                const scale = @abs(p.x[i]) + @abs(p.x[other]);
                out.* += scale * (if (p.which == .text_rope) @as(f64, 4 * 0x1p-8) else @as(f64, 12 * 0x1p-24));
            }
        }
    }
};
