//! Shared projection contract fixtures, independent binary64 arithmetic and backend-owned guarded calls.
const std = @import("std");
const wo = @import("../weight_only.zig");
const quant = @import("../quant/quant.zig");
pub const Control = enum(u3) { production, nan, repeat, tail, sums };
pub const Variant = struct { format: wo.Format, group: c_int = 64, scale_kind: wo.Scale = .bf16 };
pub const variants = blk: {
    var out: [21]Variant = undefined;
    out[0] = .{ .format = .bf16 };
    out[1] = .{ .format = .fp8 };
    out[2] = .{ .format = .nvfp4 };
    var i: usize = 3;
    for ([_]wo.Format{ .mlx4, .mlx8 }) |format| for ([_]c_int{ 32, 64, 128 }) |group| for (std.enums.values(wo.Scale)) |kind| {
        out[i] = .{ .format = format, .group = group, .scale_kind = kind };
        i += 1;
    };
    break :blk out;
};
fn meta(bytes: []u8, i: usize, value: f32, kind: wo.Scale) void {
    switch (kind) {
        .bf16 => put(bytes, i, value),
        .f16 => std.mem.writeInt(u16, bytes[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(value))), .little),
        .f32 => std.mem.writeInt(u32, bytes[4 * i ..][0..4], @bitCast(value), .little),
    }
}

fn bf(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}
fn put(bytes: []u8, i: usize, value: f32) void {
    std.mem.writeInt(u16, bytes[2 * i ..][0..2], quant.convert.bf16(value), .little);
}
pub const Fixture = struct {
    p: wo.Params,
    x: []u16,
    weight: []u8,
    scales: []u8,
    biases: []u8,
    pub fn init(a: std.mem.Allocator, p: wo.Params, fill: anytype, seed: u64) !Fixture {
        const sizes = try wo.validate(p);
        const x = try a.alloc(u16, sizes.x / 2);
        const weight = try a.alloc(u8, sizes.weight);
        const scales = try a.alloc(u8, sizes.scales);
        const biases = try a.alloc(u8, sizes.biases);
        const out: Fixture = .{ .p = p, .x = x, .weight = weight, .scales = scales, .biases = biases };
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const k: usize = @intCast(p.k);
        for (x) |*v| v.* = quant.convert.bf16(@as(f32, @floatFromInt(random.intRangeAtMost(i32, -8, 8))) / 8);
        switch (p.format) {
            .bf16 => for (0..weight.len / 2) |i| put(weight, i, @as(f32, @floatFromInt(random.intRangeAtMost(i32, -4, 4))) / 8),
            .fp8 => for (weight) |*v| {
                v.* = random.uintLessThan(u8, 0x60);
            },
            .nvfp4 => {
                for (weight) |*v| v.* = random.int(u8);
                @memset(scales, 0x30);
            },
            .mlx4, .mlx8 => {
                for (weight) |*v| v.* = random.int(u8);
                for (0..scales.len / @as(usize, if (p.scale_kind == .f32) 4 else 2)) |i| {
                    meta(scales, i, 1.0 / 32.0, p.scale_kind);
                    meta(biases, i, -0.125, p.scale_kind);
                }
            },
        }
        // A tail is large enough to remain visible after the final BF16 rounding.
        for (0..@intCast(p.rows)) |row| x[row * k + k - 1] = 0x4200;
        if (fill == .stress) {
            @memset(x, 0x4780);
            for (0..@intCast(p.rows)) |row| {
                x[row * k] = quant.convert.bf16(66048);
                for (k / 2..k) |i| x[row * k + i] |= 0x8000;
            }
            switch (p.format) {
                .bf16 => for (0..weight.len / 2) |i| put(weight, i, 1),
                .fp8 => @memset(weight, 0x38),
                .nvfp4 => {
                    @memset(weight, 0x22);
                    @memset(scales, 0x38);
                },
                .mlx4, .mlx8 => {
                    @memset(weight, if (p.format == .mlx4) 0x11 else 1);
                    for (0..scales.len / @as(usize, if (p.scale_kind == .f32) 4 else 2)) |i| {
                        meta(scales, i, 1, p.scale_kind);
                        meta(biases, i, 0, p.scale_kind);
                    }
                },
            }
        } else if (fill == .nonfinite) {
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 }, 0..) |value, row|
                if (row < p.rows) {
                    x[row * k] = value;
                };
            if (p.rows > 3 and k > 1) x[3 * k + 1] = 0xff80;
        }
        return out;
    }
    fn scale(bytes: []const u8, i: usize, kind: wo.Scale) f64 {
        return switch (kind) {
            .bf16 => bf(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)),
            .f16 => @as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little))),
            .f32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little))),
        };
    }
    /// Magnitude includes the affine components separately, so cancellation cannot hide dequantization error.
    pub fn decoded(f: Fixture, col: usize, i: usize) struct { value: f64, magnitude: f64 } {
        const k: usize = @intCast(f.p.k);
        const at = col * k + i;
        const value: f64 = switch (f.p.format) {
            .bf16 => bf(std.mem.readInt(u16, f.weight[2 * at ..][0..2], .little)),
            .fp8 => quant.fp8.e4m3(f.weight[at]) * @as(f64, f.p.global_scale),
            .nvfp4 => blk: {
                const code: u4 = @truncate(f.weight[at / 2] >> @as(u3, @intCast((at % 2) * 4)));
                break :blk quant.nvfp4.e2m1(code) * quant.fp8.e4m3(f.scales[col * (k / 16) + i / 16]) * @as(f64, f.p.global_scale);
            },
            .mlx4, .mlx8 => {
                const bits: usize = if (f.p.format == .mlx4) 4 else 8;
                const code = (f.weight[at * bits / 8] >> @as(u3, @intCast((at * bits) % 8))) & (if (bits == 4) @as(u8, 15) else 255);
                const group = col * (k / @as(usize, @intCast(f.p.group))) + i / @as(usize, @intCast(f.p.group));
                const first = @as(f64, @floatFromInt(code)) * scale(f.scales, group, f.p.scale_kind);
                const last = scale(f.biases, group, f.p.scale_kind);
                return .{ .value = first + last, .magnitude = @abs(first) + @abs(last) };
            },
        };
        return .{ .value = value, .magnitude = @abs(value) };
    }
    pub fn reference(f: Fixture, want: []f64, magnitude: []f64) void {
        const k: usize = @intCast(f.p.k);
        const n: usize = @intCast(f.p.n);
        for (want, magnitude, 0..) |*w, *m, out| {
            w.* = 0;
            m.* = 0;
            for (0..k) |i| {
                const weight = f.decoded(out % n, i);
                const x = bf(f.x[(out / n) * k + i]);
                w.* += x * weight.value;
                m.* += @abs(x) * weight.magnitude;
            }
        }
    }
};

const served = [_]struct { rows: usize, n: usize, k: usize }{
    .{ .rows = 3, .n = 7, .k = 128 }, .{ .rows = 1, .n = 4, .k = 20480 }, .{ .rows = 8, .n = 5120, .k = 128 },
};

/// B owns real uploads and launches; recording tests may substitute B, never production family execution.
pub fn Case(comptime contract: type, comptime B: type) type {
    const c = contract.spec;
    const g = contract.gates;
    return struct {
        const Self = @This();
        fn depth(shape: c.Shape) usize {
            // The longest path across groups32/64/128: per-lane FMAs, shuffle adds, totals and dequantization.
            return (shape.k + 31) / 32 + 10;
        }
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        format: wo.Format,
        control: Control = .production,
        group: c_int = 64,
        scale_kind: wo.Scale = .bf16,
        label: [160]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repetition: c_int = 0,
        stress: bool = false,
        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            if (self.arena) |*arena| arena.deinit();
        }
        pub fn kernel(self: *Self) g.Kernel {
            const label = std.fmt.bufPrint(&self.label, "{s} weight-only {t} g{d} {t} {t}", .{ B.name, self.format, self.group, self.scale_kind, self.control }) catch unreachable;
            const step: usize = switch (self.format) {
                .mlx4, .mlx8 => @intCast(self.group),
                .nvfp4 => 16,
                else => 1,
            };
            return .{ .ctx = self, .vt = &vt, .control = switch (self.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = label,
                .accum = .f32,
                .out = .bf16,
                .depth = depth,
                .tiles = .{ .{ .size = 8 }, .{ .size = 4 }, .{ .size = @max(64, step), .step = step, .min = step } },
                .served = comptime blk: {
                    var shapes: [served.len]c.Shape = undefined;
                    for (served, 0..) |s, i| shapes[i] = .{ .rows = s.rows, .n = s.n, .k = s.k };
                    const held = shapes;
                    break :blk &held;
                },
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            const step: usize = switch (self.format) {
                .mlx4, .mlx8 => @intCast(self.group),
                .nvfp4 => 16,
                else => 1,
            };
            if (shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.n > 20480 or shape.k < step or shape.k > 20480 or shape.k % step != 0) return false;
            self.rig.reset();
            if (self.arena) |*arena| arena.deinit();
            self.arena = .init(self.a);
            self.repetition = 0;
            self.stress = fill == .stress;
            self.fixture = try Fixture.init(self.arena.?.allocator(), .{ .rows = @intCast(shape.rows), .n = @intCast(shape.n), .k = @intCast(shape.k), .format = self.format, .group = self.group, .scale_kind = self.scale_kind }, fill, seed);
            try self.backend.bind(&self.rig, self.fixture);
            return true;
        }
        pub fn count(self: *Self) usize {
            return @as(usize, @intCast(self.fixture.p.rows)) * @as(usize, @intCast(self.fixture.p.n));
        }
        pub fn run(self: *Self) !void {
            try self.backend.launch(&self.rig, self.fixture.p, self.control, self.repetition);
            self.repetition += 1;
        }
        pub fn reference(self: *Self, want: []f64, magnitude: []f64) !void {
            self.fixture.reference(want, magnitude);
        }
        pub fn bounds(self: *Self, want: []const f64, mag: []const f64, allow: []f64) !void {
            const p = self.fixture.p;
            const shape: c.Shape = .{ .rows = @intCast(p.rows), .n = @intCast(p.n), .k = @intCast(p.k) };
            for (want, mag, allow) |w, m, *limit| {
                limit.* = (if (self.stress) c.boundExact(self.kernel().spec, w) else c.bound(self.kernel().spec, shape, w, m)) + @as(f64, @floatFromInt(shape.k)) * 4 * 0x1p-149;
            }
        }
        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            const k: usize = @intCast(self.fixture.p.k);
            const n: usize = @intCast(self.fixture.p.n);
            const terms = try self.a.alloc(f64, k);
            defer self.a.free(terms);
            for (out, 0..) |*v, at| {
                for (terms, 0..) |*term, i| term.* = bf(self.fixture.x[(at / n) * k + i]) * self.fixture.decoded(at % n, i).value;
                v.* = contract.rows.sumIn(acc, order, terms);
            }
        }
    };
}
