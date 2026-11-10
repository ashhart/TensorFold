//! Native FP4 prefill through the production wrapper, checked against independently quantized float64 products.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Rig = @import("contract_rig.zig").Rig;
const Native = cuda.nvfp4.native_prompt.Native;

pub const Fault = enum { none, guard, missing_tail, dropped_nonfinite, narrow_sum };
pub const Config = struct { out: cuda.qmmf.Out, fault: Fault = .none };
pub const entries = [_]Config{
    .{ .out = .bf16 },                             .{ .out = .f32 },
    .{ .out = .f32, .fault = .guard },             .{ .out = .f32, .fault = .missing_tail },
    .{ .out = .f32, .fault = .dropped_nonfinite }, .{ .out = .f32, .fault = .narrow_sum },
};

pub fn run(gpu: @import("check.zig").Gpu) !void {
    if (!cuda.nvfp4.native_prompt.supported(try gpu.ctx.capability())) return error.UnsupportedNativeNvfp4Prefill;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: [entries.len]Case = undefined;
    var kernels: [entries.len]g.Kernel = undefined;
    for (&cases, &kernels, entries) |*item, *kernel_, config| {
        item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = config };
        kernel_.* = item.kernel();
    }
    defer for (&cases) |*item| item.deinit();
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, &kernels, 0x6b63, &tally);
    const limits = try checkLimits(gpu);
    std.debug.print("SUMMARY native NVFP4: {d} kernels pass, {d} fail, {d} controls caught, {d} missed, {d} limit problems\n", .{ tally.passed, tally.failed, tally.caught, tally.missed, limits });
    if (tally.problems() + limits != 0) return error.TestFailed;
}

fn checkLimits(gpu: @import("check.zig").Gpu) !usize {
    const registry = core.contract.registry;
    const sm = try gpu.ctx.capability();
    const index: usize = if (sm == 120) 3 else if (sm == 121) 4 else return error.UnsupportedNativeNvfp4Prefill;
    const release = try core.contract.cubin.releaseFor(gpu.gpa, cuda.kernels.nvfp4_prompt, sm);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.nvfp4_prompt);
    defer module.unload();
    var problems: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "nvfp4_prompt")) continue;
        const symbol = try gpu.gpa.dupeSentinel(u8, record.launch.symbol, 0);
        defer gpu.gpa.free(symbol);
        const f = try module.function(symbol);
        const regs = try f.attribute(.num_regs);
        const shared = try f.attribute(.shared_size_bytes);
        const threads = try f.attribute(.max_threads_per_block);
        const expected = registry.usageFor(record, release)[index];
        const ok = regs == expected.regs and shared == expected.static and threads >= record.launch.threads;
        problems += @intFromBool(!ok);
        std.debug.print("{s} native NVFP4 limits {s}: regs={d} shared={d} threads={d}\n", .{ if (ok) "PASS" else "FAIL", record.launch.symbol, regs, shared, threads });
    }
    return problems;
}

const served = [_]c.Shape{
    .{ .rows = 65, .n = 129, .k = 192 },
    .{ .rows = 17, .n = 65, .k = 64 },
    .{ .rows = 4096, .n = 1, .k = 64 },
    .{ .rows = 1, .n = 17408, .k = 5120 },
    .{ .rows = 1, .n = 5120, .k = 17408 },
};

fn depth(s: c.Shape) usize {
    return s.k;
}

pub fn spec(config: Config) c.Spec {
    const name = switch (config.fault) {
        .none => if (config.out == .f32) "cuda native nvfp4 prompt f32" else "cuda native nvfp4 prompt bf16",
        .guard => "cuda native nvfp4 CONTROL guard write",
        .missing_tail => "cuda native nvfp4 CONTROL omitted final row",
        .dropped_nonfinite => "cuda native nvfp4 CONTROL erased nonfinite input",
        .narrow_sum => "cuda native nvfp4 CONTROL bf16 accumulator replay",
    };
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = if (config.out == .f32) .f32 else .bf16,
        .nan = .{ .stated = "any nonfinite activation makes its quantized group NaN and all its row's outputs NaN" },
        .tiles = .{ .{ .size = 64 }, .{ .size = 128 }, .{ .size = 64, .step = 64, .min = 64 } },
        .served = &served,
    };
}

/// E4M3's finite values, zero, and its two NaN codes, decoded without the production quantization helpers.
fn decodedBits(bits: u8) f64 {
    const code = bits & 127;
    if (code == 127) return std.math.nan(f64);
    const exponent: i32 = @intCast(code >> 3);
    const mantissa: f64 = @floatFromInt(code & 7);
    const value = if (exponent == 0) std.math.ldexp(mantissa, -9) else std.math.ldexp(8 + mantissa, exponent - 10);
    return if (bits & 128 != 0) -value else value;
}

const decoded_codes: [256]f64 = blk: {
    @setEvalBranchQuota(100000);
    var values: [256]f64 = undefined;
    for (&values, 0..) |*value, bits| value.* = decodedBits(@intCast(bits));
    break :blk values;
};
fn e4m3(bits: u8) f64 {
    return decoded_codes[bits];
}

fn e2m1(bits: u8) f64 {
    const values = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const value = values[bits & 7];
    return if (bits & 8 != 0) -value else value;
}

fn nearestScale(value: f32) u8 {
    if (std.math.isInf(value)) return 126;
    var best: u8 = 0;
    var distance = @abs(@as(f64, value));
    for (1..127) |candidate| {
        const diff = @abs(e4m3(@intCast(candidate)) - value);
        if (diff < distance or (diff == distance and candidate & 1 == 0)) {
            best = @intCast(candidate);
            distance = diff;
        }
    }
    return best;
}

fn nearestCode(value: f32) u8 {
    const sign: u8 = if (@as(u32, @bitCast(value)) >> 31 != 0) 8 else 0;
    if (std.math.isInf(value)) return sign | 7;
    var best: u8 = 0;
    var distance = @abs(@as(f64, value));
    for (1..8) |candidate| {
        const diff = @abs(e2m1(@intCast(candidate)) - @abs(@as(f64, value)));
        if (diff < distance or (diff == distance and candidate & 1 == 0)) {
            best = @intCast(candidate);
            distance = diff;
        }
    }
    return sign | best;
}

/// Float64 enumeration chooses codes; the explicit FP32 boundaries are the declared quantizer's arithmetic.
fn quantizedGroup(input: []const u16, inverse: f32, out: []f64) void {
    std.debug.assert(input.len == 16 and out.len == 16);
    var maximum: f32 = 0;
    for (input) |bits| {
        const value = c.fromBf16(bits);
        if (!std.math.isFinite(value)) {
            @memset(out, std.math.nan(f64));
            return;
        }
        maximum = @max(maximum, @abs(value));
    }
    const scaled: f32 = maximum * inverse;
    const divided: f32 = scaled / 6;
    const scale: f32 = @floatCast(e4m3(nearestScale(divided)));
    for (input, out) |bits, *value| {
        const normalized: f32 = c.fromBf16(bits) * inverse;
        const divided_value: f32 = if (scale == 0) 0 else normalized / scale;
        value.* = e2m1(nearestCode(divided_value)) * scale;
    }
}

fn put(codes: []u8, at: usize, value: u8) void {
    const byte = &codes[at / 2];
    byte.* = if (at & 1 == 0) (byte.* & 0xf0) | value else (byte.* & 15) | value << 4;
}

pub const Case = struct {
    rig: Rig,
    config: Config,
    native: ?Native = null,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    quantized: []f64 = &.{},
    codes: []u8 = &.{},
    scales: []u8 = &.{},
    weight: cuda.qmmf.Weight = undefined,
    dx: u64 = 0,
    out: u64 = 0,
    fill: g.Fill = .random,

    fn clear(self: *Case) void {
        self.rig.s.synchronize() catch {};
        if (self.native) |*native| native.unload();
        self.native = null;
        self.rig.reset();
        if (self.arena) |*arena| arena.deinit();
        self.arena = null;
    }

    pub fn deinit(self: *Case) void {
        self.clear();
        self.rig.deinit();
    }

    fn release(ctx: *anyopaque) void {
        const self: *Case = @ptrCast(@alignCast(ctx));
        self.clear();
    }

    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.release = release;
        break :blk hooks;
    };

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.config), .ctx = self, .vt = &vt, .control = switch (self.config.fault) {
            .none => null,
            .guard => .repeat,
            .missing_tail => .shapes,
            .dropped_nonfinite => .nonfinite,
            .narrow_sum => .sums,
        } };
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 4096 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0)
            return false;
        self.clear();
        self.arena = .init(self.rig.gpu.gpa);
        errdefer self.clear();
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.fill = fill;
        const n = shape.n;
        const k = shape.k;
        const stride = k + 64;
        self.x = try a.alloc(u16, shape.rows * stride);
        self.quantized = try a.alloc(f64, shape.rows * k);
        self.codes = try a.alloc(u8, n * k / 2);
        self.scales = try a.alloc(u8, n * k / 16);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(self.x, c.toBf16(std.math.nan(f32)));
        for (0..shape.rows) |row| {
            for (0..k) |column| self.x[row * stride + column] = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        }
        rng.bytes(self.codes);
        for (self.scales) |*value| value.* = 0x28 + rng.uintLessThan(u8, 0x30);
        const activation_scale: f32 = if (fill == .stress) 1 else 0.001261393;
        const weight_scale: f32 = if (fill == .stress) 1 else 0.003109455;
        if (fill == .stress) {
            self.fillStress();
        } else if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |value, row| {
                if (row < shape.rows) self.x[row * stride + row] = c.toBf16(value);
            }
            if (shape.rows > 3) {
                self.x[3 * stride] = c.toBf16(std.math.inf(f32));
                self.x[3 * stride + k - 1] = c.toBf16(-std.math.inf(f32));
            }
        } else {
            @memset(self.x[0..16], 0);
            if (shape.rows > 1) {
                self.x[stride] = 0x8000;
                self.x[stride + 1] = 1;
                self.x[stride + 2] = 0x8001;
            }
        }
        const inverse: f32 = 1.0 / activation_scale;
        for (0..shape.rows) |row| for (0..k / 16) |group| {
            quantizedGroup(self.x[row * stride + group * 16 ..][0..16], inverse, self.quantized[row * k + group * 16 ..][0..16]);
        };
        const npad = cuda.nvfp4.padded(n);
        const words = try a.alloc(u32, npad * k / 8);
        const packed_scales = try a.alloc(u8, npad * k / 16);
        cuda.nvfp4.packWords(words, self.codes, n, k);
        cuda.nvfp4.packScales(packed_scales, self.scales, n, k);
        self.weight = cuda.nvfp4.weight(try self.rig.input(std.mem.sliceAsBytes(words)), try self.rig.input(packed_scales), weight_scale, n, k);
        self.weight.act = activation_scale;
        if (self.config.fault == .dropped_nonfinite and fill == .nonfinite) {
            const erased = try a.dupe(u16, self.x);
            for (erased) |*bits| {
                if (!std.math.isFinite(c.fromBf16(bits.*))) bits.* = 0;
            }
            self.dx = try self.rig.input(std.mem.sliceAsBytes(erased));
        } else self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.out = try self.rig.output(self.count() * c.size(spec(self.config).out));
        self.native = try Native.load(self.rig.gpu.d, shape.rows, k);
        const bytes = self.native.?.scratch.len;
        const guarded = try self.rig.scratch(bytes);
        self.native.?.scratch.free();
        self.native.?.scratch = .{ .d = self.rig.gpu.d, .ptr = guarded, .len = bytes, .borrowed = true };
        return true;
    }

    pub fn run(self: *Case) !void {
        const rows = if (self.config.fault == .missing_tail and self.shape.rows > 1)
            self.shape.rows - 1
        else
            self.shape.rows;
        try self.native.?.matmul(self.rig.s, self.dx, self.shape.k + 64, rows, self.weight, self.out, self.config.out);
        try self.rig.s.synchronize();
        if (self.config.fault == .guard) {
            const target: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out - 1, .len = 1, .borrowed = true };
            try target.upload(0, &.{0});
        }
        if (self.config.fault == .narrow_sum) {
            const values = try self.arena.?.allocator().alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, values);
            const bits = try self.arena.?.allocator().alloc(u32, self.count());
            for (bits, values) |*bit, value| bit.* = @bitCast(@as(f32, @floatCast(value)));
            const target: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = bits.len * 4, .borrowed = true };
            try target.upload(0, std.mem.sliceAsBytes(bits));
        }
    }

    pub fn count(self: *const Case) usize {
        return self.shape.rows * self.shape.n;
    }

    fn fillStress(self: *Case) void {
        const k = self.shape.k;
        const stride = k + 64;
        @memset(self.scales, 0x38);
        for (0..self.shape.rows) |row| {
            @memset(self.x[row * stride ..][0..k], c.toBf16(192));
            self.x[row * stride] = c.toBf16(16);
            self.x[row * stride + k / 2] = c.toBf16(16);
        }
        for (0..self.shape.n) |column| for (0..k) |i| {
            put(self.codes, column * k + i, if (i < k / 2) 6 else if (i == k / 2) 13 else 14);
        };
    }

    fn terms(self: *const Case, row: usize, column: usize, products: []f64) void {
        const k = self.shape.k;
        for (products, 0..) |*product, i| {
            const byte = self.codes[column * (k / 2) + i / 2];
            const code = if (i & 1 == 0) byte & 15 else byte >> 4;
            const scale = self.scales[column * (k / 16) + i / 16];
            product.* = self.quantized[row * k + i] * e2m1(code) * e4m3(scale);
        }
    }

    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        const products = try self.arena.?.allocator().alloc(f64, self.shape.k);
        const alpha: f32 = self.weight.act * self.weight.scale;
        for (want, magnitude, 0..) |*value, *mag, i| {
            self.terms(i / self.shape.n, i % self.shape.n, products);
            value.* = 0;
            mag.* = 0;
            for (products) |term| {
                value.* += term;
                mag.* += @abs(term);
            }
            value.* *= alpha;
            mag.* *= @abs(alpha);
        }
    }

    pub fn simulate(self: *Case, accum: c.Accum, order: g.Order, out: []f64) !void {
        const products = try self.arena.?.allocator().alloc(f64, self.shape.k);
        const alpha: f32 = self.weight.act * self.weight.scale;
        for (out, 0..) |*value, i| {
            self.terms(i / self.shape.n, i % self.shape.n, products);
            value.* = core.contract.rows.sumIn(accum, order, products) * alpha;
        }
    }
};

test "FP4 reference has independent nearest-even codes and scales" {
    try std.testing.expectEqual(@as(u8, 0), nearestCode(0.25));
    try std.testing.expectEqual(@as(u8, 2), nearestCode(0.75));
    try std.testing.expectEqual(@as(u8, 6), nearestCode(5));
    try std.testing.expectEqual(@as(u8, 14), nearestCode(-5));
    try std.testing.expectEqual(@as(u8, 0), nearestScale(0x1p-10));
    try std.testing.expectEqual(@as(u8, 2), nearestScale(3 * 0x1p-10));
    try std.testing.expectEqual(@as(u8, 126), nearestScale(1000));
    try std.testing.expectEqual(@as(f64, 448), e4m3(126));
    try std.testing.expect(std.math.isNan(e4m3(127)));
}

test "FP4 reference zero groups and nonfinite groups have explicit semantics" {
    var input: [16]u16 = @splat(0);
    var out: [16]f64 = undefined;
    quantizedGroup(&input, 128, &out);
    for (out) |value| try std.testing.expectEqual(@as(f64, 0), value);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        input[7] = c.toBf16(bad);
        quantizedGroup(&input, 128, &out);
        for (out) |value| try std.testing.expect(std.math.isNan(value));
    }
}

test "FP4 reference stress discriminates half and bf16 sums in all contract orders" {
    const a = std.testing.allocator;
    var fixture: Case = .{ .rig = undefined, .config = .{ .out = .f32 }, .arena = .init(a), .shape = .{ .rows = 1, .n = 1, .k = 17408 } };
    defer fixture.arena.?.deinit();
    const arena = fixture.arena.?.allocator();
    const k = fixture.shape.k;
    fixture.x = try arena.alloc(u16, k + 64);
    fixture.quantized = try arena.alloc(f64, k);
    fixture.codes = try arena.alloc(u8, k / 2);
    fixture.scales = try arena.alloc(u8, k / 16);
    fixture.weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 1, .k = @intCast(k), .npad = 128, .scale = 1, .act = 1 };
    fixture.fillStress();
    for (0..k / 16) |group|
        quantizedGroup(fixture.x[group * 16 ..][0..16], 1, fixture.quantized[group * 16 ..][0..16]);
    var want: [1]f64 = undefined;
    var magnitude: [1]f64 = undefined;
    var got: [1]f64 = undefined;
    try fixture.reference(&want, &magnitude);
    try std.testing.expectEqual(@as(f64, 16), want[0]);
    for ([_]cuda.qmmf.Out{ .f32, .bf16 }) |out| {
        const allow = c.boundExact(spec(.{ .out = out }), want[0]);
        for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
            try fixture.simulate(.f32, order, &got);
            try std.testing.expectEqual(want[0], got[0]);
            for ([_]c.Accum{ .f16, .bf16 }) |accum| {
                try fixture.simulate(accum, order, &got);
                const margin = if (!std.math.isFinite(got[0])) std.math.inf(f64) else @abs(got[0] - want[0]) / allow;
                try std.testing.expect(margin >= 100);
            }
        }
    }
}

test "native NVFP4 reference decoded table preserves every bit pattern" {
    for (0..256) |i| {
        const want = decodedBits(@intCast(i));
        const got = e4m3(@intCast(i));
        if (std.math.isNan(want)) try std.testing.expect(std.math.isNan(got)) else try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
    }
}
