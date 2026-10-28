//! MiniMax-M3.1-Flash's glue under the contract: float64 at the kernels' rounding points, and the range fp32 allows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const iv = @import("contract_interval.zig");
pub const unit = iv.unit;
pub const libm_ulps = iv.libm_ulps;
pub const Iv = iv.Iv;
pub const Sim = iv.Sim;
pub const Sums = iv.Sums;

/// A kernel's outputs in float64 against its own ranges: K uploads, launches and solves; this holds the hooks.
pub fn Gated(comptime K: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        k: K,
        exact: bool = false,
        cache: []Iv = &.{},
        fresh: bool = false,

        pub fn deinit(self: *Self) void {
            const a = self.rig.gpu.gpa;
            a.free(self.cache);
            self.k.free(a);
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.k.spec(), .ctx = self, .vt = &vt, .control = self.k.control() };
        }

        const vt = blk: {
            var v = g.hooks(Self);
            if (!@hasDecl(K, "sums")) v.simulate = null;
            break :blk v;
        };

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            self.fresh = false;
            self.exact = fill == .stress;
            var prng = std.Random.DefaultPrng.init(seed);
            return self.k.prepare(self.rig.gpu.gpa, &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
            try self.rig.s.synchronize();
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        fn solved(self: *Self) ![]const Iv {
            if (self.fresh) return self.cache;
            const a = self.rig.gpu.gpa;
            a.free(self.cache);
            self.cache = &.{};
            self.cache = try a.alloc(Iv, self.count());
            try self.k.solve(a, .{ .exact = self.exact }, self.cache);
            self.fresh = true;
            return self.cache;
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            for (want, mag, try self.solved()) |*w, *m, v| {
                w.* = v.m;
                m.* = 0;
            }
        }

        /// The farthest end of the output's range from float64's value: a kernel inside its range is inside its bound.
        pub fn bounds(self: *Self, want: []const f64, _: []const f64, allow: []f64) !void {
            for (allow, want, try self.solved()) |*l, w, v| l.* = if (std.math.isFinite(w)) @max(0, @max(w - v.lo, v.hi - w)) + 0x1p-140 else std.math.floatMax(f64);
        }

        pub fn intervals(self: *Self, low: []f64, high: []f64) !void {
            for (low, high, try self.solved()) |*l, *h, v| {
                l.* = v.lo;
                h.* = v.hi;
            }
        }

        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            const a = self.rig.gpu.gpa;
            const tmp = try a.alloc(Iv, out.len);
            defer a.free(tmp);
            try self.k.solve(a, .{ .sim = .{ .acc = acc, .order = order }, .exact = true }, tmp);
            for (out, tmp) |*o, v| o.* = v.m;
        }
    };
}

/// A kernel whose outputs are ids or exact words, checked by its own stated rule: K uploads, launches and checks.
pub fn Checked(comptime K: type) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        k: K,
        msg: [200]u8 = undefined,

        pub fn deinit(self: *Self) void {
            self.k.free(self.rig.gpu.gpa);
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.k.spec(), .ctx = self, .vt = &vt, .control = self.k.control() };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            var prng = std.Random.DefaultPrng.init(seed);
            return self.k.prepare(self.rig.gpu.gpa, &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
            try self.rig.s.synchronize();
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        pub fn reference(_: *Self, want: []f64, mag: []f64) !void {
            @memset(want, 0);
            @memset(mag, 0);
        }

        pub fn check(self: *Self, _: g.Gate) !?[]const u8 {
            const a = self.rig.gpu.gpa;
            const got = try a.dupe(u8, try self.rig.outputBytes());
            defer a.free(got);
            return self.k.check(a, got, &self.msg);
        }
    };
}

/// No sums: the kernel compares or moves values.
pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// `n` bf16 values uniform in [lo, hi).
pub fn bfs(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
    return v;
}

/// `n` fp32 values uniform in [lo, hi).
pub fn f32s(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]f32 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = lo + rng.float(f32) * (hi - lo);
    return v;
}

/// A whole number in [lo, hi], as fp32.
pub fn int(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(lo + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo + 1)))));
}

pub fn input(rig: *rig_.Rig, x: anytype) !u64 {
    return rig.input(std.mem.sliceAsBytes(x));
}

/// Bytes put back where the device reads and writes them in place, before a launch.
pub fn restore(rig: *rig_.Rig, ptr: u64, bytes: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = rig.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

pub fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub fn readF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

pub fn readI32(bytes: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, bytes[4 * i ..][0..4], .little);
}

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);
