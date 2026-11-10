//! fn_exl3_split_f16 under the contract: an fp16 row as bf16 hi = bf16(x) and lo = bf16(x - hi).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;

const block = 256;
const patterns = 1 << 16;

/// A planted fault: the control poses as the kernel and must go red at the gate it names.
pub const Control = enum {
    /// A kernel that flushes NaN and the infinities to zero before it splits (gate 1).
    flush,
    /// A grid rounded down, so the partial last block is never launched (gate 3: whole blocks pass, every tail fails).
    tail,
    /// hi written over x, the kernel's own input (gate 2).
    over,
};

fn none(_: c.Shape) usize {
    return 0;
}

/// Shape: rows x n fp16 values (k is 1), outputs hi then lo, no sums; the last two shapes end in a partial block.
pub fn spec(control: ?Control) c.Spec {
    return .{
        .name = if (control) |x| switch (x) {
            .flush => "control: flashnext_exl3 fn_exl3_split_f16 flushing NaN and Inf to zero",
            .tail => "control: flashnext_exl3 fn_exl3_split_f16 launching whole blocks only",
            .over => "control: flashnext_exl3 fn_exl3_split_f16 writing hi over its input",
        } else "cuda flashnext_exl3 fn_exl3_split_f16 (fp16 row to bf16 hi + lo)",
        .accum = .none,
        .depth = none,
        .ops = 1, // x - hi in fp32 (exact here: at most 3 bits remain)
        .out = .bf16,
        .nan = .propagate, // as float64 does on x - hi: NaN gives NaN twice, and +-Inf gives hi = +-Inf and lo = NaN
        .tiles = .{ .{}, .{ .size = block, .step = 1, .min = 1 }, .{} },
        .served = &.{
            .{ .rows = 1, .n = 2560, .k = 1 },
            .{ .rows = 128, .n = 2560, .k = 1 },
            .{ .rows = 1, .n = 1000, .k = 1 },
            .{ .rows = 1, .n = patterns + 1, .k = 1 },
        },
    };
}

/// Planted at every shape's start and, reversed, its end: zeros, subnormal and normal limits, 1, Inf and NaNs.
pub const edges = [_]u16{
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7bff, 0xfbff, 0x3c00, 0x3c01,
    0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x7c01, 0xfdff,
};

/// Non-finite fill: NaN and both infinities in the first elements, the rest finite.
const planted = [_]u16{ 0x7e00, 0x7c00, 0xfc00, 0xfe01, 0x7c01 };

fn isFinite(bits: u16) bool {
    return bits & 0x7c00 != 0x7c00;
}

/// The odd stride makes i -> i * stride + start a bijection on 16 bits: n = 65536 holds every pattern once.
fn pattern(i: usize, start: u16) u16 {
    return @truncate(i *% 40503 +% start);
}

/// The float64 reference: hi = bf16(x), lo = bf16(x - hi), each rounded once in float64; NaN and Inf follow.
pub fn split(x: u16) [2]f64 {
    const v: f64 = @as(f16, @bitCast(x));
    const hi = bfr.bf16(v);
    return .{ hi, bfr.bf16(v - hi) };
}

pub const Split = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: ?Control = null,
    arena: ?std.heap.ArenaAllocator = null,
    total: usize = 0,
    x: []u16 = &.{}, // the fp16 bits as the reference reads them
    hi: u64 = 0,
    lo: u64 = 0,
    xd: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Split) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Split);

    pub fn kernel(self: *Split) g.Kernel {
        const gate: ?g.Gate = if (self.control) |x| switch (x) {
            .flush => .nonfinite,
            .tail => .shapes,
            .over => .repeat,
        } else null;
        return .{ .spec = spec(self.control), .ctx = self, .vt = &vt, .control = gate };
    }

    /// Random fill spreads all 65,536 patterns (n = 65,536 holds each once) with the edges at both ends.
    pub fn prepare(self: *Split, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != 1) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const start = prng.random().int(u16);
        self.total = shape.rows * shape.n;
        self.x = try a.alloc(u16, self.total);
        for (self.x, 0..) |*v, i| {
            v.* = pattern(i, start);
            if (fill == .nonfinite and !isFinite(v.*)) v.* &= ~@as(u16, 0x0400);
        }
        if (fill == .nonfinite) {
            for (planted, 0..) |bits, i| if (i < self.total) {
                self.x[i] = bits;
            };
        } else for (edges, 0..) |bits, i| if (i < self.total) {
            self.x[i] = bits;
            self.x[self.total - 1 - i] = bits;
        };
        const sent = try a.dupe(u16, self.x);
        if (self.control == .flush) for (sent) |*v| {
            if (!isFinite(v.*)) v.* = 0;
        };
        self.xd = try self.rig.input(std.mem.sliceAsBytes(sent));
        self.hi = try self.rig.output(self.total * 2);
        self.lo = try self.rig.output(self.total * 2);
        return true;
    }

    pub fn run(self: *Split) !void {
        const blocks = if (self.control == .tail) self.total / block else (self.total + block - 1) / block;
        if (blocks != 0) {
            var args: cuda.Args = .{};
            args.add(self.xd);
            args.add(if (self.control == .over) self.xd else self.hi);
            args.add(self.lo);
            args.add(@as(c_longlong, @intCast(self.total)));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = block } }, self.rig.s, &args);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Split) usize {
        return 2 * self.total;
    }

    /// hi then lo for every element; no sums, so no magnitude.
    pub fn reference(self: *Split, want: []f64, mag: []f64) !void {
        for (self.x, 0..) |x, i| {
            const r = split(x);
            want[i] = r[0];
            want[self.total + i] = r[1];
        }
        @memset(mag, 0);
    }

    /// Bit-exact at gates 1 and 3 (a half-ulp bound would let hi be one ulp off), any NaN counting, and hi + lo == x.
    pub fn check(self: *Split, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var nonfinite: usize = 0;
        for (self.x, 0..) |x, i| {
            const got = [2]u16{
                std.mem.readInt(u16, out[2 * i ..][0..2], .little),
                std.mem.readInt(u16, out[2 * (self.total + i) ..][0..2], .little),
            };
            const want = split(x);
            if (!isFinite(x)) nonfinite += 1;
            for (got, want, [_][]const u8{ "hi", "lo" }) |bits, w, name| {
                const ok = if (std.math.isNan(w)) std.math.isNan(bfr.value(bits)) else bits == bfr.bits(@floatCast(w));
                if (!ok) return std.fmt.bufPrint(&self.msg, "x {x:0>4} at {d}: {s} is {x:0>4} where float64 gives {e}", .{ x, i, name, bits, w }) catch "mismatch";
            }
            const sum: f64 = @as(f16, @bitCast(x));
            if (isFinite(x) and @as(f64, bfr.value(got[0])) + @as(f64, bfr.value(got[1])) != sum)
                return std.fmt.bufPrint(&self.msg, "x {x:0>4} at {d}: hi + lo is not x", .{ x, i }) catch "hi + lo differs from x";
        }
        if (at == .nonfinite and (nonfinite == 0 or nonfinite == self.total)) return "the planted values make no proof";
        return null;
    }
};

/// The module and the kernel's registrations: itself and its three controls.
pub const Set = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    cases: [4]Split,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const self = try a.create(Set);
        errdefer a.destroy(self);
        self.a = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3);
        errdefer self.mod.unload();
        const f = try self.mod.function("fn_exl3_split_f16");
        const controls = [_]?Control{ null, .flush, .tail, .over };
        for (&self.cases, controls) |*x, ctl| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .control = ctl };
        for (&self.cases) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn deinit(self: *Set) void {
        for (&self.cases) |*x| x.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};

test "the served shapes hold every fp16 pattern and a partial last block, and the bijection spreads small shapes" {
    const t = std.testing;
    const s = spec(null);
    try t.expect(s.served[3].n >= patterns and s.served[3].n % block != 0 and s.served[2].n % block != 0);
    var seen = try t.allocator.alloc(bool, patterns);
    defer t.allocator.free(seen);
    @memset(seen, false);
    for (0..patterns) |i| seen[pattern(i, 0x1234)] = true;
    for (seen) |v| try t.expect(v);
    var spread: usize = 0;
    for (0..255) |i| spread += @intFromBool(pattern(i, 7) & 0x7c00 != pattern(i + 1, 7) & 0x7c00);
    try t.expect(spread > 200);
}

test "the float64 reference splits every fp16 pattern: hi + lo is x when x is finite, Inf gives a NaN lo" {
    const t = std.testing;
    for (0..patterns) |i| {
        const x: u16 = @intCast(i);
        const r = split(x);
        const v: f64 = @as(f16, @bitCast(x));
        // the kernel's fp32 path and the float64 reference round the same
        const f: f32 = @floatCast(v);
        const hi = bfr.bits(f);
        const lo = bfr.bits(f - bfr.value(hi));
        if (std.math.isNan(v)) {
            try t.expect(std.math.isNan(r[0]) and std.math.isNan(r[1]) and std.math.isNan(bfr.value(hi)));
        } else if (std.math.isInf(v)) {
            try t.expect(r[0] == v and std.math.isNan(r[1]) and std.math.isNan(bfr.value(lo)));
        } else {
            try t.expectEqual(v, r[0] + r[1]);
            try t.expectEqual(bfr.bits(@floatCast(r[0])), hi);
            try t.expectEqual(bfr.bits(@floatCast(r[1])), lo);
        }
    }
}

test "the edges are the boundaries: zeros, subnormals, normals, the greatest finite, infinities, NaNs" {
    const t = std.testing;
    var finite: usize = 0;
    for (edges) |x| {
        if (isFinite(x)) finite += 1;
        const r = split(x);
        if (isFinite(x)) try t.expectEqual(@as(f64, @as(f16, @bitCast(x))), r[0] + r[1]);
    }
    try t.expectEqual(@as(usize, 12), finite);
    try t.expect(planted.len <= 5 and !isFinite(planted[0]) and !isFinite(planted[4]));
}
