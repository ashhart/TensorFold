//! Both W4A8 kernels registered separately, with natural-order FP64 references and guarded independent inputs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const ref = @import("w4a8_ref.zig");
const Rig = @import("contract_rig.zig").Rig;
pub const Kind = enum { quantize, prefill };
pub const Fault = enum { none, guard, wrong_code, wrong_scale, wrong_sum, narrow_sum, narrow_epilogue };

pub fn plantExceptional(x: []u16) void {
    const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (values, x[0..values.len]) |v, *out| out.* = c.toBf16(v);
}

const served = [_]c.Shape{
    .{ .rows = 129, .n = 129, .k = 256 },  .{ .rows = 127, .n = 127, .k = 64 },
    .{ .rows = 128, .n = 128, .k = 128 },  .{ .rows = 17, .n = 65, .k = 192 },
    .{ .rows = 4096, .n = 1, .k = 64 },    .{ .rows = 4, .n = 1, .k = 32768 },
    .{ .rows = 1, .n = 6144, .k = 5120 },  .{ .rows = 1, .n = 5120, .k = 6144 },
    .{ .rows = 1, .n = 17408, .k = 5120 }, .{ .rows = 1, .n = 5120, .k = 17408 },
};
fn sumDepth(_: c.Shape) usize {
    return 6;
}
fn projectionDepth(s: c.Shape) usize {
    return s.k / 32 + 6;
}
pub fn spec(kind: Kind) c.Spec {
    return .{
        .name = if (kind == .quantize) "cuda w4a8 tf_w4a8_quantize" else "cuda w4a8 tf_w4a8_prefill",
        .accum = .f32,
        .depth = if (kind == .quantize) sumDepth else projectionDepth,
        .out = if (kind == .quantize) .f32 else .bf16,
        .ops = if (kind == .quantize) 0 else 8,
        .nan = .{ .stated = "exceptional-input policy is unsettled; gate 1 deliberately fails closed" },
        .tiles = .{ .{ .size = 128 }, .{ .size = 128 }, .{ .size = 256, .step = 64, .min = 64 } },
        .served = &served,
    };
}
pub const Case = struct {
    rig: Rig,
    kind: Kind,
    kernels: cuda.w4a8.Kernels,
    fault: Fault = .none,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    x: []u16 = &.{},
    xq: []i8 = &.{},
    xa: []f32 = &.{},
    xs: []f32 = &.{},
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    dx: u64 = 0,
    dq: u64 = 0,
    da: u64 = 0,
    ds: u64 = 0,
    out: u64 = 0,
    weight: cuda.qlinear.Affine4 = undefined,
    msg: [200]u8 = undefined,
    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.arena.deinit();
    }
    pub fn kernel(self: *Case) g.Kernel {
        var registration = spec(self.kind);
        registration.name = switch (self.fault) {
            .none => registration.name,
            .guard => "cuda w4a8 CONTROL changed guard byte",
            .wrong_code => if (self.kind == .quantize) "cuda w4a8 CONTROL changed quantized code" else "cuda w4a8 CONTROL changed weight code",
            .wrong_scale => "cuda w4a8 CONTROL changed group scale",
            .wrong_sum => "cuda w4a8 CONTROL changed group sum",
            .narrow_sum => "cuda w4a8 CONTROL bf16 group sums",
            .narrow_epilogue => "cuda w4a8 CONTROL bf16 epilogue sums",
        };
        return .{ .spec = registration, .ctx = self, .vt = &vt, .control = switch (self.fault) {
            .none => null,
            .guard => .repeat,
            .narrow_sum, .narrow_epilogue => .sums,
            else => .shapes,
        } };
    }
    const vt = g.hooks(Case);
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 4096 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.fill = fill;
        const a = self.arena.allocator();
        const kg = shape.k / 64;
        self.x = try a.alloc(u16, shape.rows * (shape.k + 64));
        self.xq = try a.alloc(i8, shape.rows * shape.k);
        self.xa = try a.alloc(f32, shape.rows * kg);
        self.xs = try a.alloc(f32, shape.rows * kg);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x, 0..) |*h, i| h.* = c.toBf16(if (fill == .stress) (if (i % 2 == 0) @as(f32, 65536) else 1) else (rng.float(f32) * 2 - 1) * (if (i % 97 == 0) @as(f32, 16) else 1));
        if (fill == .random) for (0..shape.rows) |row| {
            const first = row * (shape.k + 64);
            // Zero, signed extremes, nearest-even ties and finite reciprocal-overflow boundaries all reach the device.
            const top: f32 = switch (row % 5) {
                0 => 0,
                1 => 127,
                2 => 0x1p-122,
                3 => 0x1p-133,
                else => 1,
            };
            @memset(self.x[first..][0..64], c.toBf16(0));
            self.x[first] = c.toBf16(top);
            self.x[first + 1] = c.toBf16(-top);
            if (row % 5 == 1) {
                self.x[first + 2] = c.toBf16(0.5);
                self.x[first + 3] = c.toBf16(1.5);
            }
        };
        if (fill == .nonfinite) {
            plantExceptional(self.x);
            // No unsupported float-to-int conversion is executed before its policy is specified.
            return true;
        }
        for (0..shape.rows) |row| for (0..kg) |j| {
            const group = try ref.group(self.x[row * (shape.k + 64) + j * 64 ..][0..64]);
            @memcpy(self.xq[row * shape.k + j * 64 ..][0..64], &group.codes);
            self.xa[row * kg + j] = group.scale;
            self.xs[row * kg + j] = @floatCast(group.sum);
        };
        if (self.kind == .quantize) {
            self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
            self.ds = try self.rig.output(shape.rows * kg * 4);
            self.da = try self.rig.output(shape.rows * kg * 4);
            self.dq = try self.rig.output(shape.rows * shape.k);
            return true;
        }
        self.codes = try a.alloc(u8, shape.n * shape.k);
        self.scales = try a.alloc(u16, shape.n * kg);
        self.biases = try a.alloc(u16, shape.n * kg);
        for (self.codes) |*v| v.* = if (fill == .stress) 0 else rng.uintLessThan(u8, 16);
        for (self.scales, self.biases) |*s, *b| {
            s.* = c.toBf16(if (fill == .stress) @as(f32, 0) else 0.125);
            b.* = c.toBf16(if (fill == .stress) @as(f32, 1) else -0.75);
        }
        if (fill == .stress) {
            for (0..shape.rows) |row| for (0..kg) |j| {
                const at = row * (shape.k + 64) + j * 64;
                @memset(self.x[at..][0..64], 0);
                self.x[at] = c.toBf16(switch (j * 4 / kg) {
                    0 => @as(f32, 32768),
                    1, 3 => 1,
                    else => -32768,
                });
                const group = try ref.group(self.x[at..][0..64]);
                @memcpy(self.xq[row * shape.k + j * 64 ..][0..64], &group.codes);
                self.xa[row * kg + j] = group.scale;
                self.xs[row * kg + j] = @floatCast(group.sum);
            };
        }
        const layout = cuda.qlinear.Affine4.layout(shape.n, shape.k);
        const weight_words = try a.alloc(u32, layout.words / 4);
        ref.pack(weight_words, self.codes, shape.n, shape.k);
        const scales = try a.alloc(u16, layout.scales / 2);
        const biases = try a.alloc(u16, layout.scales / 2);
        @memset(scales, 0);
        @memset(biases, 0);
        for (0..shape.n) |col| for (0..kg) |j| {
            scales[j * layout.npad + col] = self.scales[col * kg + j];
            biases[j * layout.npad + col] = self.biases[col * kg + j];
        };
        if (self.fault == .wrong_code) weight_words[0] ^= 0x11111111;
        self.weight = .{ .w = try self.rig.input(std.mem.sliceAsBytes(weight_words)), .s = try self.rig.input(std.mem.sliceAsBytes(scales)), .b = try self.rig.input(std.mem.sliceAsBytes(biases)), .n = shape.n, .k = shape.k, .npad = layout.npad };
        const slots = try a.alloc(i8, self.xq.len);
        for (slots, 0..) |*v, i| v.* = self.xq[i / 32 * 32 + ref.inputAt(i % 32)];
        self.dq = try self.rig.input(std.mem.sliceAsBytes(slots));
        self.da = try self.rig.input(std.mem.sliceAsBytes(self.xa));
        self.ds = try self.rig.input(std.mem.sliceAsBytes(self.xs));
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        return true;
    }
    pub fn run(self: *Case) !void {
        if (self.fill == .nonfinite) return;
        const s = self.shape;
        var args: cuda.launch.Args = .{};
        if (self.kind == .quantize) {
            inline for (.{ self.dx, self.dq, self.da, self.ds, @as(c_int, @intCast(s.k)), @as(c_int, @intCast(s.k + 64)) }) |v| args.add(v);
            try cuda.launch.launch(self.kernels.quantize, .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast((s.k / 64 + 3) / 4) }, .block = .{ .x = 128 } }, self.rig.s, &args);
        } else {
            inline for (.{ self.dq, self.da, self.ds, self.weight.w, self.weight.s, self.weight.b, self.out }) |v| args.add(v);
            const group = @max(1, @min((s.rows + 127) / 128, (12 << 20) / (128 * s.k * 2)));
            inline for (.{ s.rows, s.n, s.k, self.weight.npad, group }) |v| args.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(self.kernels.prefill, .{ .grid = .{ .x = @intCast((s.rows + 127) / 128 * ((s.n + 127) / 128)) }, .block = .{ .x = 256 }, .shared = cuda.w4a8.smem }, self.rig.s, &args);
        }
        try self.rig.s.synchronize();
        if (self.fault == .guard) {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = (if (self.kind == .quantize) self.ds else self.out) - 1, .len = 1, .borrowed = true };
            try b.upload(0, &.{0});
        }
        if (self.kind == .quantize and (self.fault == .wrong_code or self.fault == .wrong_scale or self.fault == .wrong_sum)) {
            const ptr = if (self.fault == .wrong_code) self.dq else if (self.fault == .wrong_scale) self.da else self.ds;
            const data: [4]u8 = if (self.fault == .wrong_code) .{ 1, 0, 0, 0 } else @bitCast(@as(f32, 1024));
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = if (self.fault == .wrong_code) 1 else 4, .borrowed = true };
            try b.upload(0, data[0..b.len]);
        }
        if (self.fault == .narrow_sum or self.fault == .narrow_epilogue) {
            const values = try self.arena.allocator().alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, values);
            const bytes = try self.arena.allocator().alloc(u8, self.count() * c.size(spec(self.kind).out));
            for (values, 0..) |v, i| if (self.kind == .quantize) std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little) else std.mem.writeInt(u16, bytes[i * 2 ..][0..2], c.toBf16(@floatCast(v)), .little);
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = if (self.kind == .quantize) self.ds else self.out, .len = bytes.len, .borrowed = true };
            try b.upload(0, bytes);
        }
    }
    pub fn count(self: *Case) usize {
        return self.shape.rows * (if (self.kind == .quantize) self.shape.k / 64 else self.shape.n);
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const kg = s.k / 64;
        const terms = try self.arena.allocator().alloc(f64, if (self.kind == .quantize) 64 else kg);
        for (want, mag, 0..) |*w, *m, i| {
            if (self.kind == .quantize) {
                const at = (i / kg) * (s.k + 64) + i % kg * 64;
                for (terms, self.x[at..][0..64]) |*v, h| v.* = c.fromBf16(h);
            } else ref.terms(self.xq[i / s.n * s.k ..][0..s.k], self.xa[i / s.n * kg ..][0..kg], self.xs[i / s.n * kg ..][0..kg], self.codes[i % s.n * s.k ..][0..s.k], self.scales[i % s.n * kg ..][0..kg], self.biases[i % s.n * kg ..][0..kg], terms);
            w.* = 0;
            m.* = 0;
            for (terms) |v| {
                w.* += v;
                m.* += @abs(v);
            }
        }
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        const kg = s.k / 64;
        const terms = try self.arena.allocator().alloc(f64, if (self.kind == .quantize) 64 else kg);
        for (out, 0..) |*w, i| {
            if (self.kind == .quantize) {
                const at = (i / kg) * (s.k + 64) + i % kg * 64;
                for (terms, self.x[at..][0..64]) |*v, h| v.* = c.fromBf16(h);
            } else ref.terms(self.xq[i / s.n * s.k ..][0..s.k], self.xa[i / s.n * kg ..][0..kg], self.xs[i / s.n * kg ..][0..kg], self.codes[i % s.n * s.k ..][0..s.k], self.scales[i % s.n * kg ..][0..kg], self.biases[i % s.n * kg ..][0..kg], terms);
            w.* = ref.sumTerms(acc, order, terms);
        }
    }
    pub fn bounds(self: *Case, want: []const f64, mag: []const f64, allow: []f64) !void {
        const s = spec(self.kind);
        for (allow, want, mag, 0..) |*limit, w, m, i| {
            limit.* = if (self.fill == .stress) c.boundExact(s, w) else c.bound(s, self.shape, w, m);
            if (self.kind == .prefill and self.fill != .stress) {
                const kg = self.shape.k / 64;
                for (0..kg) |j| limit.* += 2 * @abs(@as(f64, c.fromBf16(self.scales[(i % self.shape.n) * kg + j]))) * self.xa[(i / self.shape.n) * kg + j];
            }
        }
    }
    pub fn check(self: *Case, at: g.Gate) !?[]const u8 {
        if (at == .nonfinite) return "exceptional inputs planted; their quantization/fallback policy remains unspecified";
        if (self.kind == .quantize) {
            const raw = try self.rig.bytes(self.dq, self.xq.len);
            for (raw, 0..) |got, i| {
                const natural = i / 32 * 32 + ref.inputAt(i % 32);
                const want: u8 = @bitCast(self.xq[natural]);
                if (got != want) return "quantizer codes differ from independent nearest-even reference";
            }
            const scales = try self.rig.bytes(self.da, self.xa.len * 4);
            for (self.xa, 0..) |want, i| {
                const got: f32 = @bitCast(std.mem.readInt(u32, scales[i * 4 ..][0..4], .little));
                if (!(@abs(got - want) <= @abs(want) * 0x1p-21 + 0x1p-149)) return "quantizer scales leave the independent FP32 division bound";
            }
        }
        const n = self.count();
        const a = self.arena.allocator();
        const want = try a.alloc(f64, n);
        const mag = try a.alloc(f64, n);
        const allow = try a.alloc(f64, n);
        try self.reference(want, mag);
        try self.bounds(want, mag, allow);
        const out = try self.rig.bytes(if (self.kind == .quantize) self.ds else self.out, n * c.size(spec(self.kind).out));
        for (want, allow, 0..) |w, limit, i| if (!(@abs(c.read(spec(self.kind).out, out, i) - w) <= limit)) return "output leaves independent FP64 bound";
        return null;
    }
};
