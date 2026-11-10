//! Native prompt FP8 products, checked against independently quantized FP64 operands through their real wrapper.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Rig = @import("contract_rig.zig").Rig;
const Native = cuda.fp8_prefill.Native;
pub const Kind = enum { calibrated, row, affine4 };
pub const Fault = enum { none, guard, missing_tail, dropped_nonfinite, narrow_sum };
pub const Config = struct { kind: Kind, out: cuda.qmmf.Out, fault: Fault = .none };
pub const entries = [_]Config{
    .{ .kind = .calibrated, .out = .bf16 },                             .{ .kind = .calibrated, .out = .f32 },
    .{ .kind = .row, .out = .bf16 },                                    .{ .kind = .row, .out = .f32 },
    .{ .kind = .affine4, .out = .bf16 },                                .{ .kind = .affine4, .out = .f32 },
    .{ .kind = .calibrated, .out = .f32, .fault = .guard },             .{ .kind = .row, .out = .f32, .fault = .missing_tail },
    .{ .kind = .calibrated, .out = .f32, .fault = .dropped_nonfinite }, .{ .kind = .row, .out = .f32, .fault = .dropped_nonfinite },
    .{ .kind = .affine4, .out = .f32, .fault = .dropped_nonfinite },    .{ .kind = .calibrated, .out = .f32, .fault = .narrow_sum },
    .{ .kind = .row, .out = .f32, .fault = .narrow_sum },               .{ .kind = .affine4, .out = .f32, .fault = .narrow_sum },
};
const served = [_]c.Shape{
    .{ .rows = 65, .n = 129, .k = 192 },   .{ .rows = 17, .n = 65, .k = 64 },
    .{ .rows = 129, .n = 127, .k = 256 },  .{ .rows = 4096, .n = 1, .k = 64 },
    .{ .rows = 3, .n = 3, .k = 17408 },    .{ .rows = 1, .n = 17408, .k = 5120 },
    .{ .rows = 1, .n = 5120, .k = 17408 },
};
fn depth(s: c.Shape) usize {
    return s.k + 8;
}
fn name(comptime kind: Kind, config: Config) []const u8 {
    const prefix = "cuda native fp8 " ++ @tagName(kind);
    return switch (config.fault) {
        .none => if (config.out == .f32) prefix ++ " f32" else prefix ++ " bf16",
        .guard => prefix ++ " CONTROL guard write",
        .missing_tail => prefix ++ " CONTROL omitted final row",
        .dropped_nonfinite => prefix ++ " CONTROL erased nonfinite input",
        .narrow_sum => prefix ++ " CONTROL bf16 accumulator replay",
    };
}
pub fn spec(config: Config) c.Spec {
    return .{
        .name = switch (config.kind) {
            .calibrated => name(.calibrated, config),
            .row => name(.row, config),
            .affine4 => name(.affine4, config),
        },
        .accum = .f32,
        .depth = depth,
        .ops = if (config.kind == .affine4) 4 else 2,
        .out = if (config.out == .f32) .f32 else .bf16,
        .nan = .{ .stated = if (config.kind == .calibrated)
            "E4M3 SATFINITE clamps infinities and finite overflow; NaN input propagates through its row"
        else
            "any nonfinite activation makes its quantized group or row NaN and all its row outputs NaN" },
        .tiles = .{ .{ .size = 64 }, .{ .size = 128 }, .{ .size = 128, .step = 64, .min = 64 } },
        .served = &served,
    };
}
pub fn run(gpu: @import("check.zig").Gpu) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedNativeFp8Prefill;
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
    try g.runAll(gpu.gpa, &kernels, 0xf8a8, &tally);
    try refusals(gpu, stream);
    const limits = try checkLimits(gpu);
    std.debug.print("SUMMARY native FP8: {d} kernels pass, {d} fail, {d} controls caught, {d} missed, {d} limit problems\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed, limits,
    });
    if (tally.problems() + limits != 0) return error.TestFailed;
}

pub const Cases = struct {
    items: [entries.len]Case = undefined,
    initialized: usize = 0,
    pub fn append(self: *Cases, list: *std.ArrayList(g.Kernel), gpu: @import("check.zig").Gpu, stream: cuda.Stream, sm: u32) !void {
        if (sm != 120 and sm != 121) return;
        for (&self.items, entries) |*item, config| {
            item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = config };
            self.initialized += 1;
            try list.append(gpu.gpa, item.kernel());
        }
    }
    pub fn deinit(self: *Cases) void {
        for (self.items[0..self.initialized]) |*item| item.deinit();
    }
};

fn checkLimits(gpu: @import("check.zig").Gpu) !usize {
    const registry = core.contract.registry;
    const sm = try gpu.ctx.capability();
    const index: usize = if (sm == 120) 3 else if (sm == 121) 4 else return error.UnsupportedNativeFp8Prefill;
    const release = try core.contract.cubin.releaseFor(gpu.gpa, cuda.kernels.fp8_prefill, sm);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.fp8_prefill);
    defer module.unload();
    var problems: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "fp8_prefill")) continue;
        const symbol = try gpu.gpa.dupeSentinel(u8, record.launch.symbol, 0);
        defer gpu.gpa.free(symbol);
        const f = try module.function(symbol);
        const regs = try f.attribute(.num_regs);
        const shared = try f.attribute(.shared_size_bytes);
        const threads = try f.attribute(.max_threads_per_block);
        const expected = registry.usageFor(record, release)[index];
        const ok = regs == expected.regs and shared == expected.static and threads >= record.launch.threads;
        problems += @intFromBool(!ok);
        std.debug.print("{s} native FP8 limits {s}: regs={d} shared={d} threads={d}\n", .{
            if (ok) "PASS" else "FAIL", record.launch.symbol, regs, shared, threads,
        });
    }
    return problems;
}

/// IEEE E4M3 finite values and NaNs, decoded without the production conversion routines.
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

/// FP64 enumeration chooses nearest-even codes after the declared FP32 scale arithmetic.
fn encode(value: f32) u8 {
    const sign: u8 = if (@as(u32, @bitCast(value)) >> 31 != 0) 128 else 0;
    if (std.math.isNan(value)) return sign | 127;
    if (std.math.isInf(value)) return sign | 126;
    var best: u8 = 0;
    var distance = @abs(@as(f64, value));
    for (1..127) |candidate| {
        const diff = @abs(e4m3(@intCast(candidate)) - @abs(@as(f64, value)));
        if (diff < distance or (diff == distance and candidate & 1 == 0)) {
            best = @intCast(candidate);
            distance = diff;
        }
    }
    return sign | best;
}
fn dynamic(input: []const u16, out: []u8) f32 {
    var maximum: f32 = 0;
    for (input) |bits| {
        const value = c.fromBf16(bits);
        if (!std.math.isFinite(value)) {
            @memset(out, 127);
            return std.math.nan(f32);
        }
        maximum = @max(maximum, @abs(value));
    }
    const scale: f32 = maximum / 448;
    for (input, out) |bits, *code| {
        const value = c.fromBf16(bits);
        code.* = encode(if (scale == 0) value else value / scale);
    }
    return scale;
}
fn inputAt(i: usize) usize {
    const order = [_]u8{ 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15 };
    return (i / 16) * 16 + order[i % 16];
}

pub const Case = struct {
    rig: Rig,
    config: Config,
    native: ?Native = null,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    x: []u16 = &.{},
    q: []u8 = &.{},
    ascales: []f32 = &.{},
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    columns: []f32 = &.{},
    weight: cuda.qmmf.Weight = undefined,
    affine: cuda.qlinear.Affine4 = undefined,
    dx: u64 = 0,
    out: u64 = 0,
    fn stride(self: *const Case) usize {
        return self.shape.k + @as(usize, if (self.config.kind == .calibrated) 3 else 0);
    }
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
        if (shape.rows == 0 or shape.rows > 4096 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.clear();
        self.arena = .init(self.rig.gpu.gpa);
        errdefer self.clear();
        self.shape = shape;
        self.fill = fill;
        try self.host(self.arena.?.allocator(), seed);
        const a = self.arena.?.allocator();
        const n = shape.n;
        const k = shape.k;
        const kg = k / 64;
        const npad = cuda.fp8.padded(n);
        if (self.config.kind == .affine4) {
            const words = try a.alloc(u32, npad * k / 8);
            @import("w4a8_ref.zig").pack(words, self.codes, n, k);
            const scales = try a.alloc(u16, npad * kg);
            const biases = try a.alloc(u16, npad * kg);
            @memset(scales, 0);
            @memset(biases, 0);
            for (0..n) |column| for (0..kg) |group| {
                scales[group * npad + column] = self.scales[column * kg + group];
                biases[group * npad + column] = self.biases[column * kg + group];
            };
            self.affine = .{ .w = try self.rig.input(std.mem.sliceAsBytes(words)), .s = try self.rig.input(std.mem.sliceAsBytes(scales)), .b = try self.rig.input(std.mem.sliceAsBytes(biases)), .n = n, .k = k, .npad = npad };
        } else {
            const packed_codes = try a.alloc(u8, npad * k);
            cuda.fp8.packCodes(packed_codes, self.codes, n, k);
            self.weight.codes = try self.rig.input(packed_codes);
            if (self.config.kind == .row) {
                const tiles = try a.alloc(f32, npad * kg);
                for (0..npad / 64) |tile| for (0..kg) |group| for (0..64) |lane| {
                    const column = tile * 64 + lane;
                    tiles[(tile * kg + group) * 64 + lane] = if (column < n) self.columns[column] else 1;
                };
                self.weight.scales = try self.rig.input(std.mem.sliceAsBytes(tiles));
            }
        }
        const source = try a.dupe(u16, self.x);
        if (self.config.fault == .dropped_nonfinite) for (source) |*bits| {
            if (!std.math.isFinite(c.fromBf16(bits.*))) bits.* = 0;
        };
        self.dx = try self.rig.input(std.mem.sliceAsBytes(source));
        self.out = try self.rig.output(self.count() * c.size(spec(self.config).out));
        self.native = try Native.load(self.rig.gpu.d, shape.rows, k);
        const bytes = self.native.?.scratch.len;
        const guarded = try self.rig.scratch(bytes);
        self.native.?.scratch.free();
        self.native.?.scratch = .{ .d = self.rig.gpu.d, .ptr = guarded, .len = bytes, .borrowed = true };
        return true;
    }
    fn host(self: *Case, a: std.mem.Allocator, seed: u64) !void {
        const s = self.shape;
        const kg = s.k / 64;
        self.x = try a.alloc(u16, s.rows * self.stride());
        self.q = try a.alloc(u8, s.rows * s.k);
        self.ascales = try a.alloc(f32, s.rows * kg);
        self.codes = try a.alloc(u8, s.n * s.k);
        self.scales = try a.alloc(u16, s.n * kg);
        self.biases = try a.alloc(u16, s.n * kg);
        self.columns = try a.alloc(f32, s.n);
        self.weight = .{ .mode = if (self.config.kind == .row) .fp8g else .fp8, .codes = 0, .scales = 0, .n = @intCast(s.n), .k = @intCast(s.k), .npad = @intCast(cuda.fp8.padded(s.n)), .act = 0.017578125, .scale = if (self.fill == .stress) 0.013671875 else 0.013719 };
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(self.x, c.toBf16(std.math.nan(f32)));
        for (0..s.rows) |row| for (0..s.k) |i| {
            self.x[row * self.stride() + i] = c.toBf16((rng.float(f32) * 2 - 1) * 3);
        };
        for (self.codes) |*code| code.* = if (self.config.kind == .affine4) rng.uintLessThan(u8, 16) else encode((rng.float(f32) * 2 - 1) * 12);
        for (self.scales, self.biases, 0..) |*scale, *bias, i| {
            scale.* = c.toBf16(0.03125 * @as(f32, @floatFromInt(1 + i % 7)));
            bias.* = c.toBf16(-0.125 * @as(f32, @floatFromInt(1 + i % 3)));
        }
        for (self.columns, 0..) |*column, i| column.* = 0.013719 * (1 + @as(f32, @floatFromInt(i % 7)) / 32);
        if (self.fill == .stress) self.stress() else if (self.fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |bad, row| {
                if (row < s.rows) self.x[row * self.stride() + row] = c.toBf16(bad);
            }
            if (s.rows > 3) {
                self.x[3 * self.stride()] = c.toBf16(std.math.inf(f32));
                self.x[3 * self.stride() + s.k - 1] = c.toBf16(-std.math.inf(f32));
            }
        } else {
            @memset(self.x[0..s.k], 0);
            if (s.rows > 1) {
                self.x[self.stride()] = 0x8000;
                self.x[self.stride() + 1] = 1;
                self.x[self.stride() + 2] = 0x8001;
                self.x[self.stride() + 3] = c.toBf16(1.0625);
                self.x[self.stride() + 4] = c.toBf16(1.1875);
            }
            if (s.rows > 2) @memset(self.x[2 * self.stride() ..][0..64], 1);
            if (self.config.kind == .calibrated and s.rows > 3) self.x[3 * self.stride()] = 0x7f7f;
        }
        self.quantize();
    }
    fn stress(self: *Case) void {
        const s = self.shape;
        @memset(self.columns, self.weight.scale);
        @memset(self.scales, c.toBf16(1));
        @memset(self.biases, c.toBf16(-15));
        for (0..s.rows) |row| for (0..s.k) |i| {
            const value: f32 = if (self.config.kind == .affine4) (if (i % 2 == 0) 448 else 1) else (if (i == 0 or i == s.k / 2) @as(f32, 448) else 192) * self.weight.act;
            self.x[row * self.stride() + i] = c.toBf16(value);
        };
        for (0..s.n) |column| for (0..s.k) |i| {
            self.codes[column * s.k + i] = if (self.config.kind == .affine4) (if (i % 2 == 0) 15 else 14) else encode(if (i < s.k / 2) @as(f32, 4) else if (i == s.k / 2 + 1) -3 else -4);
        };
    }
    fn quantize(self: *Case) void {
        const k = self.shape.k;
        const kg = k / 64;
        const inverse: f32 = 1 / self.weight.act;
        for (0..self.shape.rows) |row| {
            const input = self.x[row * self.stride() ..][0..k];
            const out = self.q[row * k ..][0..k];
            if (self.config.kind == .row) {
                const scale = dynamic(input, out);
                @memset(self.ascales[row * kg ..][0..kg], scale);
            } else if (self.config.kind == .affine4) {
                for (0..kg) |group| self.ascales[row * kg + group] =
                    dynamic(input[group * 64 ..][0..64], out[group * 64 ..][0..64]);
            } else {
                for (input, out) |bits, *code| code.* = encode(c.fromBf16(bits) * inverse);
            }
        }
    }
    pub fn count(self: *const Case) usize {
        return self.shape.rows * self.shape.n;
    }
    pub fn run(self: *Case) !void {
        const rows = self.shape.rows - @as(usize, @intFromBool(self.config.fault == .missing_tail and self.shape.rows > 1));
        switch (self.config.kind) {
            .calibrated => try self.native.?.matmul(self.rig.s, self.dx, self.stride(), rows, self.weight, self.out, self.config.out),
            .row => try self.native.?.rowScaled(self.rig.s, self.dx, rows, self.weight, self.out, self.config.out),
            .affine4 => try self.native.?.affine4(self.rig.s, self.dx, rows, self.affine, self.out, self.config.out),
        }
        try self.rig.s.synchronize();
        if (self.config.fault == .none) try self.checkQuantized();
        if (self.config.fault == .guard) {
            const target: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out - 1, .len = 1, .borrowed = true };
            try target.upload(0, &.{0});
        }
        if (self.config.fault == .narrow_sum) {
            const a = self.arena.?.allocator();
            const values = try a.alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, values);
            const bits = try a.alloc(u32, self.count());
            for (bits, values) |*bit, value| bit.* = @bitCast(@as(f32, @floatCast(value)));
            const target: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = bits.len * 4, .borrowed = true };
            try target.upload(0, std.mem.sliceAsBytes(bits));
        }
    }
    fn checkQuantized(self: *Case) !void {
        const k = self.shape.k;
        const ptr = self.native.?.scratch.ptr;
        const bytes = try self.rig.bytes(ptr, self.q.len);
        for (0..self.shape.rows) |row| for (0..k) |slot| {
            const want = self.q[row * k + inputAt(slot)];
            const got = bytes[row * k + slot];
            if (want != got and !(want & 127 == 127 and got & 127 == 127)) return error.Fp8PromptCodeMismatch;
        };
        if (self.config.kind == .calibrated) return;
        const len = if (self.config.kind == .row) self.shape.rows else self.ascales.len;
        const scales = try self.rig.bytes(ptr + self.q.len, len * 4);
        for (0..len) |i| {
            const got: f32 = @bitCast(std.mem.readInt(u32, scales[i * 4 ..][0..4], .little));
            const want = self.ascales[if (self.config.kind == .row) i * (k / 64) else i];
            if (got != want and !(std.math.isNan(got) and std.math.isNan(want))) return error.Fp8PromptScaleMismatch;
        }
        if (self.config.kind != .affine4) return;
        const sums = try self.rig.bytes(ptr + self.q.len + len * 4, len * 4);
        for (0..len) |group| {
            var want: f64 = 0;
            for (self.q[group * 64 ..][0..64]) |code| want += e4m3(code);
            const got: f32 = @bitCast(std.mem.readInt(u32, sums[group * 4 ..][0..4], .little));
            if (got != want and !(std.math.isNan(got) and std.math.isNan(want))) return error.Fp8PromptSumMismatch;
        }
    }
    fn products(self: *const Case, row: usize, column: usize, out: []f64) void {
        for (out, 0..) |*term, i| term.* = e4m3(self.q[row * self.shape.k + i]) *
            (if (self.config.kind == .affine4) @as(f64, @floatFromInt(self.codes[column * self.shape.k + i])) else e4m3(self.codes[column * self.shape.k + i]));
    }
    fn alpha(self: *const Case, row: usize, column: usize) f32 {
        return if (self.config.kind == .row) self.ascales[row * (self.shape.k / 64)] * self.columns[column] else self.weight.act * self.weight.scale;
    }
    fn evaluate(self: *const Case, acc: c.Accum, order: g.Order, row: usize, column: usize, products_: []const f64) f64 {
        if (self.config.kind != .affine4) return core.contract.rows.sumIn(acc, order, products_) * self.alpha(row, column);
        const kg = self.shape.k / 64;
        var total: f64 = 0;
        for (0..kg) |group| {
            const at = column * kg + group;
            const dot = core.contract.rows.sumIn(acc, order, products_[group * 64 ..][0..64]);
            var sum: f64 = 0;
            for (self.q[(row * kg + group) * 64 ..][0..64]) |code| sum += e4m3(code);
            const term = (c.fromBf16(self.scales[at]) * dot + c.fromBf16(self.biases[at]) * sum) * self.ascales[row * kg + group];
            total = rounded(acc, total + rounded(acc, term));
        }
        return total;
    }
    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        const products_ = try self.arena.?.allocator().alloc(f64, self.shape.k);
        for (want, magnitude, 0..) |*value_, *mag, i| {
            const row = i / self.shape.n;
            const column = i % self.shape.n;
            self.products(row, column, products_);
            value_.* = self.evaluate(.none, .sequential, row, column, products_);
            mag.* = 0;
            for (products_, 0..) |term, j| {
                if (self.config.kind == .affine4) {
                    const group = j / 64;
                    const at = column * (self.shape.k / 64) + group;
                    mag.* += (@abs(term * c.fromBf16(self.scales[at])) +
                        @abs(e4m3(self.q[row * self.shape.k + j]) * c.fromBf16(self.biases[at]))) *
                        @abs(self.ascales[row * (self.shape.k / 64) + group]);
                } else mag.* += @abs(term * self.alpha(row, column));
            }
        }
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const products_ = try self.arena.?.allocator().alloc(f64, self.shape.k);
        for (out, 0..) |*value_, i| {
            const row = i / self.shape.n;
            const column = i % self.shape.n;
            self.products(row, column, products_);
            value_.* = self.evaluate(acc, order, row, column, products_);
        }
    }
};
fn rounded(acc: c.Accum, value: f64) f64 {
    return switch (acc) {
        .none => value,
        .f32 => @as(f32, @floatCast(value)),
        .f16 => @as(f16, @floatCast(value)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(value))),
    };
}
fn refusals(gpu: @import("check.zig").Gpu, stream: cuda.Stream) !void {
    var native = try Native.load(gpu.d, 4, 192);
    defer native.unload();
    const w: cuda.qmmf.Weight = .{ .mode = .fp8, .codes = 0, .scales = 0, .n = 1, .k = 64, .npad = 128, .act = 1 };
    for ([_]usize{ 0, 5 }) |m| try std.testing.expectError(error.InvalidFp8PromptShape, native.matmul(stream, 0, 64, m, w, 0, .f32));
    for ([_]u32{ 0, 65, 256 }) |k| {
        var bad = w;
        bad.k = k;
        try std.testing.expectError(error.InvalidFp8PromptShape, native.matmul(stream, 0, k, 1, bad, 0, .f32));
    }
    try std.testing.expectError(error.InvalidFp8PromptShape, native.matmul(stream, 0, 63, 1, w, 0, .f32));
    for ([_]f32{ 0, -1, std.math.nan(f32), std.math.inf(f32), 0x1p-149 }) |scale| {
        var bad = w;
        bad.act = scale;
        try std.testing.expectError(error.InvalidFp8PromptScale, native.matmul(stream, 0, 64, 1, bad, 0, .f32));
    }
    try std.testing.expectError(error.InvalidFp8PromptScale, native.rowScaled(stream, 0, 1, w, 0, .f32));
    var bad = w;
    bad.mode = .fp8g;
    try std.testing.expectError(error.InvalidFp8PromptScale, native.matmul(stream, 0, 64, 1, bad, 0, .f32));
}

test "FP8 prompt reference enumerates nearest-even codes and SATFINITE exceptions independently" {
    for ([_]f32{ 0x1p-10, 3 * 0x1p-10, 1.0625, 1.1875, 1000, std.math.inf(f32), -std.math.inf(f32) }, [_]u8{ 0, 2, 56, 58, 126, 126, 254 }) |value, want| try std.testing.expectEqual(want, encode(value));
    try std.testing.expectEqual(@as(u8, 128), encode(-@as(f32, 0)));
    try std.testing.expect(std.math.isNan(e4m3(encode(std.math.nan(f32)))));
    var input: [64]u16 = @splat(0);
    var out: [64]u8 = undefined;
    try std.testing.expectEqual(@as(f32, 0), dynamic(&input, &out));
    for (out) |code| try std.testing.expectEqual(@as(u8, 0), code);
    @memset(&input, 1);
    try std.testing.expectEqual(@as(f32, c.fromBf16(1) / 448), dynamic(&input, &out));
    for (out) |code| try std.testing.expectEqual(@as(u8, 126), code);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        input[13] = c.toBf16(bad);
        try std.testing.expect(std.math.isNan(dynamic(&input, &out)));
        for (out) |code| try std.testing.expect(std.math.isNan(e4m3(code)));
    }
}
test "FP8 prompt stress rejects genuine half and bf16 sums in every order and both output formats" {
    for (std.enums.values(Kind)) |kind| {
        var fixture: Case = .{ .rig = undefined, .config = .{ .kind = kind, .out = .f32 }, .arena = .init(std.testing.allocator), .shape = .{ .rows = 1, .n = 1, .k = 17408 }, .fill = .stress };
        defer fixture.arena.?.deinit();
        try fixture.host(fixture.arena.?.allocator(), 1);
        var want: [1]f64 = undefined;
        var magnitude: [1]f64 = undefined;
        var got: [1]f64 = undefined;
        try fixture.reference(&want, &magnitude);
        try std.testing.expectEqual(if (kind == .affine4) @as(f64, -8704) else 0.046142578125, want[0]);
        for ([_]cuda.qmmf.Out{ .f32, .bf16 }) |out| {
            const allow = c.boundExact(spec(.{ .kind = kind, .out = out }), want[0]);
            for (std.enums.values(g.Order)) |order| {
                try fixture.simulate(.f32, order, &got);
                try std.testing.expectEqual(want[0], got[0]);
                for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                    try fixture.simulate(acc, order, &got);
                    const margin = if (!std.math.isFinite(got[0])) std.math.inf(f64) else @abs(got[0] - want[0]) / allow;
                    try std.testing.expect(margin >= 100);
                }
            }
        }
    }
}

test "FP8 prompt contracts instantiate their real hooks and refuse invalid shapes before device calls" {
    for (std.enums.values(Kind)) |kind| {
        var fixture: Case = .{ .rig = undefined, .config = .{ .kind = kind, .out = .f32 } };
        const kernel_ = fixture.kernel();
        for ([_]c.Shape{
            .{ .rows = 0, .n = 1, .k = 64 }, .{ .rows = 4097, .n = 1, .k = 64 },
            .{ .rows = 1, .n = 0, .k = 64 }, .{ .rows = 1, .n = 1, .k = 0 },
            .{ .rows = 1, .n = 1, .k = 65 },
        }) |shape| try std.testing.expect(!try kernel_.vt.prepare(kernel_.ctx, shape, .random, 1));
    }
}

test "FP8 prompt launch records cover every export and refuse unmeasured compiler versions" {
    const registry = core.contract.registry;
    var found: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "fp8_prefill")) continue;
        found += 1;
        try std.testing.expectEqual(@as(u32, 120), record.launch.min_sm);
        try std.testing.expectEqual(@as(u32, 256), record.launch.threads);
        try std.testing.expectEqual(registry.none, registry.usageFor(record, null));
        try std.testing.expectEqual(registry.none, registry.usageFor(record, .{ .major = 13, .minor = 3 }));
        try std.testing.expectEqual(record.usage, registry.usageFor(record, .{ .major = 13, .minor = 5 }));
    }
    try std.testing.expectEqual(@as(usize, 9), found);
}

test "FP8 prompt flag and precision requests are explicit" {
    try std.testing.expect(!try cuda.fp8_prefill.selected(null));
    try std.testing.expect(!try cuda.fp8_prefill.selected("0"));
    try std.testing.expect(try cuda.fp8_prefill.selected("1"));
    for ([_][]const u8{ "", "true", "2", "01" }) |raw|
        try std.testing.expectError(error.InvalidFastPrefill, cuda.fp8_prefill.selected(raw));
    try cuda.fp8_prefill.compatible(true, false, false, false);
    try cuda.fp8_prefill.compatible(false, true, true, true);
    try std.testing.expectError(error.IncompatibleFastPrefill, cuda.fp8_prefill.compatible(true, true, false, false));
    try std.testing.expectError(error.IncompatibleFastPrefill, cuda.fp8_prefill.compatible(true, false, true, false));
    try std.testing.expectError(error.IncompatibleFastPrefill, cuda.fp8_prefill.compatible(true, false, false, true));
}

test "FP8 prompt decoded table preserves every independently decoded bit pattern" {
    for (0..256) |i| {
        const want = decodedBits(@intCast(i));
        const got = e4m3(@intCast(i));
        if (std.math.isNan(want)) try std.testing.expect(std.math.isNan(got)) else try std.testing.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
    }
}
