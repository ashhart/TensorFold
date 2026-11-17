//! ZONOS3's Metal contract kit: sources built as the metallib, a rig addressed as the family's, hooks, float64 ranges.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;

const iv = @import("metal_zonos3_interval.zig");
pub const unit = iv.unit;
pub const Iv = iv.Iv;
pub const Sim = iv.Sim;
pub const Sums = iv.Sums;

/// The family's four libraries, by the file a gate 5 label names ("zonos3 <file> <function>").
pub const File = enum { backbone, codec, encoder, teacher };

/// A library's text with z3_common.h inlined.
pub fn source(a: std.mem.Allocator, file: File) ![]u8 {
    const z = sources.zonos3;
    const text = switch (file) {
        .backbone => z.backbone,
        .codec => z.codec,
        .encoder => z.encoder,
        .teacher => z.teacher,
    };
    return std.mem.replaceOwned(u8, a, text, "#include \"z3_common.h\"", z.common);
}

/// The metallib's options at run time: safe math, precise fp32 functions, MLX's MSL version; contraction off by pragma.
pub fn options() mtl.CompileOptions {
    var o = mtl.CompileOptions.mlx();
    o.math_functions = .precise;
    return o;
}

/// The four libraries, compiled once; every registration's pipelines come from them.
pub const Libs = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    libs: [4]mtl.Library,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) !Libs {
        var libs: [4]mtl.Library = undefined;
        var made: usize = 0;
        errdefer for (libs[0..made]) |l| l.deinit();
        for (std.enums.values(File), 0..) |f, i| {
            const text = try source(a, f);
            defer a.free(text);
            libs[i] = try mtl.Library.fromSource(device, text, options());
            made += 1;
        }
        return .{ .a = a, .device = device, .libs = libs };
    }

    pub fn deinit(l: *Libs) void {
        for (l.pipes.items) |p| p.deinit();
        l.pipes.deinit(l.a);
        for (l.libs) |x| x.deinit();
    }

    pub fn pipe(l: *Libs, file: File, name: []const u8) !mtl.Pipeline {
        const p = try mtl.Pipeline.init(l.device, l.libs[@backingInt(file)], name, false);
        errdefer p.deinit();
        try l.pipes.append(l.a, p);
        return p;
    }
};

const slot_shift = 40;
const off_mask = (@as(u64, 1) << slot_shift) - 1;

/// One call's guarded buffers, at ((index + 1) << 40) + offset as in gpu.zig; each launch in its own pass, waited for.
pub const Rig = struct {
    base: rig_.Rig,
    refs: std.ArrayList(rig_.Ref) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) Rig {
        return .{ .base = .{ .a = a, .device = device, .queue = queue } };
    }

    pub fn gpa(r: *const Rig) std.mem.Allocator {
        return r.base.a;
    }

    pub fn deinit(r: *Rig) void {
        r.refs.deinit(r.base.a);
        r.base.deinit();
    }

    pub fn reset(r: *Rig) void {
        r.refs.clearRetainingCapacity();
        r.base.reset();
    }

    fn handle(r: *Rig, ref: rig_.Ref) !u64 {
        try r.refs.append(r.base.a, ref);
        return @as(u64, r.refs.items.len) << slot_shift;
    }

    pub fn input(r: *Rig, data: []const u8) !u64 {
        return r.handle(try r.base.input(data));
    }

    /// An output of `len` bytes, all 0xff: an element no launch writes reads as NaN.
    pub fn output(r: *Rig, len: usize) !u64 {
        return r.handle(try r.base.output(len));
    }

    pub fn scratch(r: *Rig, len: usize) !u64 {
        return r.handle(try r.base.scratch(len));
    }

    /// The buffer and byte offset an address names.
    pub fn at(r: *const Rig, ptr: u64) rig_.Ref {
        const ref = r.refs.items[@intCast((ptr >> slot_shift) - 1)];
        return .{ .buf = ref.buf, .off = ref.off + @as(usize, @intCast(ptr & off_mask)) };
    }

    /// `len` bytes at an address (every launch has finished when `go` returns).
    pub fn bytes(r: *const Rig, ptr: u64, len: usize) []u8 {
        const ref = r.at(ptr);
        return ref.buf.contents()[ref.off..][0..len];
    }

    pub fn zero(r: *Rig, ptr: u64, len: usize) !void {
        @memset(r.bytes(ptr, len), 0);
    }

    /// Bytes put back where the device reads and writes them in place, before a launch.
    pub fn put(r: *Rig, ptr: u64, data: []const u8) !void {
        @memcpy(r.bytes(ptr, data.len), data);
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.base.outputBytes();
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.base.inputBytes();
    }

    pub fn intact(r: *Rig) !bool {
        return r.base.intact();
    }

    pub fn overlap(r: *Rig) bool {
        return r.base.overlap();
    }

    /// One dispatch as gpu.zig's goShared encodes it: buffers 0.., then `args`, `shared` threadgroup bytes; waited for.
    pub fn go(r: *Rig, p: mtl.Pipeline, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype, shared: usize) !void {
        const pass = r.base.begin();
        const e = pass.e;
        if (shared > 0) e.setThreadgroupMemory(std.mem.alignForward(usize, shared, 16), 0);
        e.setPipeline(p);
        for (bufs, 0..) |a, i| {
            const ref = r.at(a);
            e.setBuffer(ref.buf, ref.off, i);
        }
        e.setBytes(std.mem.asBytes(&args), bufs.len);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
        try r.base.finish(pass);
    }
};

/// A kernel's outputs in float64 against its own ranges: K uploads, launches and solves; this holds the hooks.
pub fn Gated(comptime K: type) type {
    return struct {
        const Self = @This();
        rig: Rig,
        k: K,
        exact: bool = false,
        cache: []Iv = &.{},
        fresh: bool = false,

        pub fn deinit(self: *Self) void {
            const a = self.rig.gpa();
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
            return self.k.prepare(self.rig.gpa(), &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        fn solved(self: *Self) ![]const Iv {
            if (self.fresh) return self.cache;
            const a = self.rig.gpa();
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
            const a = self.rig.gpa();
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
        rig: Rig,
        k: K,
        msg: [200]u8 = undefined,

        pub fn deinit(self: *Self) void {
            self.k.free(self.rig.gpa());
            self.rig.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.k.spec(), .ctx = self, .vt = &vt, .control = self.k.control() };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            self.rig.reset();
            var prng = std.Random.DefaultPrng.init(seed);
            return self.k.prepare(self.rig.gpa(), &self.rig, shape, fill, prng.random());
        }

        pub fn run(self: *Self) !void {
            try self.k.run(&self.rig);
        }

        pub fn count(self: *Self) usize {
            return self.k.count();
        }

        pub fn reference(_: *Self, want: []f64, mag: []f64) !void {
            @memset(want, 0);
            @memset(mag, 0);
        }

        pub fn check(self: *Self, _: g.Gate) !?[]const u8 {
            const a = self.rig.gpa();
            const got = try a.dupe(u8, try self.rig.outputBytes());
            defer a.free(got);
            return self.k.check(a, got, &self.msg);
        }
    };
}

/// The runner's list and its --only text: a registration joins the list when its name holds the text.
pub const List = struct {
    a: std.mem.Allocator,
    kernels: *std.ArrayList(g.Kernel),
    only: ?[]const u8,

    pub fn add(l: List, k: g.Kernel) !void {
        if (l.only) |needle| if (std.mem.indexOf(u8, k.spec.name, needle) == null) return;
        try l.kernels.append(l.a, k);
    }
};

/// No sums: the kernel compares or moves values.
pub fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn input(rig: *Rig, x: anytype) !u64 {
    return rig.input(std.mem.sliceAsBytes(x));
}

pub fn restore(rig: *Rig, ptr: u64, data: []const u8) !void {
    return rig.put(ptr, data);
}

pub fn int(x: usize) i32 {
    return @intCast(x);
}

pub fn long(x: usize) i64 {
    return @intCast(x);
}

/// The value m with the range [lo, hi] widened by `rel` of each end.
pub fn wide(m: f64, lo: f64, hi: f64, rel: f64) Iv {
    return Iv.spread(m, lo, hi, rel);
}

/// An increasing precise function of a range, within `ulps` of its value.
pub fn rising(comptime f: fn (f64) f64, a: Iv, ulps: f64) Iv {
    return wide(f(a.m), f(a.lo), f(a.hi), ulps * 0x1p-23);
}

fn tanh64(x: f64) f64 {
    return std.math.tanh(x);
}

/// precise::tanh, within its 5 ulps.
pub fn tanh(a: Iv) Iv {
    return rising(tanh64, a, iv.tanh_ulps);
}

/// z3_log1p(u), u >= 0: log(w) * (u / (w - 1)), w = 1 + u, or u if w rounds to 1; within 2 log ulps and 4 roundings.
pub fn log1p(u: Iv) Iv {
    const f = struct {
        fn of(x: f64) f64 {
            return std.math.log1p(x);
        }
    }.of;
    const rel = (iv.log_ulps * 2 + 4) * unit;
    return wide(f(u.m), f(u.lo), f(u.hi), rel);
}

/// z3_softplus: x past 20, else z3_log1p(exp(x)).
pub fn softplus(a: Iv) Iv {
    if (a.m > 20) return a;
    return log1p(a.exp());
}

/// z3_silu: x / (1 + exp(-x)).
pub fn silu(v: Iv) Iv {
    return v.div(Iv.of(1).add(v.neg().exp()));
}

/// The square of a range: never below zero.
pub fn square(v: Iv) Iv {
    const lo = @min(@abs(v.lo), @abs(v.hi));
    const hi = @max(@abs(v.lo), @abs(v.hi));
    return wide(v.m * v.m, if (v.lo <= 0 and v.hi >= 0) 0 else lo * lo, hi * hi, unit);
}

/// rms_scale: 1 / sqrt(ss / width + eps), each step correctly rounded.
pub fn rmsScale(ss: Iv, width: usize, eps: f32) Iv {
    return Iv.of(1).div(ss.div(Iv.of(@floatFromInt(width))).add(Iv.of(eps)).sqrt());
}

/// `n` fp32 values: uniform in [-span, span), or gate 4's whole numbers in [-32, 32].
pub fn values(a: std.mem.Allocator, rng: std.Random, n: usize, fill: g.Fill, span: f32) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x) |*v| v.* = if (fill == .stress) wholeIn(rng, -32, 32) else (rng.float(f32) * 2 - 1) * span;
    return x;
}

/// A whole number in [lo, hi], as fp32.
pub fn wholeIn(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(lo + @as(i32, @intCast(rng.uintLessThan(u32, @intCast(hi - lo + 1)))));
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

pub fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);

/// NaN, +Inf and -Inf at known places of rows [rows][n]: a row each, row 3 holding both infinities.
pub fn plant(x: []f32, rows: usize, n: usize) void {
    const where = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 0 }, .{ 3, n - 1 } };
    const put = [_]f32{ nan, inf, -inf, inf, -inf };
    for (where, put) |p, v| if (p[0] < rows) {
        x[p[0] * n + p[1]] = v;
    };
}

/// The magnitude gate 4's sums pass through: past fp16's range, and where bf16 steps by 1,024.
pub const big: f32 = 0x1p17;

/// Gate 4's terms: odd numbers to 255 and up to 16 of +-2^17 summing to zero; fp32 sums of 32,768 stay exact.
pub fn stress(x: []f32, rng: std.Random) void {
    for (x) |*v| v.* = @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
    const pairs = @min(16, (x.len / 2) & ~@as(usize, 1));
    for (0..pairs) |j| x[2 * j] = if (j % 2 == 0) big else -big;
}

/// `stress` along every `stride`-th value from `first`: a strided axis of a larger buffer.
pub fn stressAxis(x: []f32, first: usize, stride: usize, count: usize, rng: std.Random) void {
    var tmp: [64]f32 = undefined;
    const head = @min(count, tmp.len);
    stress(tmp[0..head], rng);
    for (0..count) |i| x[first + i * stride] = if (i < head) tmp[i] else @floatFromInt(1 + 2 * rng.uintLessThan(u32, 128));
}

pub fn readF32(data: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, data[4 * i ..][0..4], .little));
}

pub fn readI32(data: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, data[4 * i ..][0..4], .little);
}

/// `scores` to softmax weights in place, exp(score - max), and their fp32 sum; gate 4's zero scores give 1s.
pub fn softmaxParts(s: Sums, scores: []Iv, depth: usize, scratch: []f64) Iv {
    if (s.exact) {
        for (scores) |*e| e.* = Iv.of(1);
    } else {
        var top = scores[0];
        for (scores[1..]) |v| {
            // fmax: a NaN gives way
            if (std.math.isNan(top.m) or v.m > top.m) top.m = v.m;
            top.lo = @max(top.lo, v.lo);
            top.hi = @max(top.hi, v.hi);
        }
        for (scores) |*e| e.* = e.sub(top).exp();
    }
    return s.sum(scores, depth, scratch);
}

/// 1 when a control's fault is on.
pub fn one(on: bool) usize {
    return @intFromBool(on);
}
