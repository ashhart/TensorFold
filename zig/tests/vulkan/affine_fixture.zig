//! MLX affine 4-bit, group-64, transposed inputs for Vulkan qmm.comp's qmm_f32 build, checked against float64.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const Affine = contract.affine.Affine;

pub const bits = 4;
pub const group = 64;
/// qmm.comp's local_size_x; the shader grid-strides over count = m * n outputs.
pub const local_size_x: u32 = 256;
/// Pads and tails of the float bindings hold this quiet NaN (weights: all codes 15), and every output starts as it.
pub const poison: u32 = 0x7fc5_a5a5;
/// Poisoned elements after each binding's data, so an overrun reads poison rather than past the buffer.
pub const tail = 16;

/// qmm.comp's push constants (std430); qmm reads the fields params() sets, the rest keep MLX's ComputeParams defaults.
pub const Params = extern struct {
    count: u32 = 0,
    operation: u32 = 0,
    lhs_size: u32 = 0,
    rhs_size: u32 = 0,
    reduce_size: u32 = 0,
    output_size: u32 = 0,
    lhs_offset: u32 = 0,
    rhs_offset: u32 = 0,
    output_offset: u32 = 0,
    aux_size: u32 = 0,
    aux_offset: u32 = 0,
    matrix_m: u32 = 0,
    matrix_n: u32 = 0,
    matrix_k: u32 = 0,
    flags: u32 = 0,
    alpha: f32 = 1,
    beta: f32 = 0,
    dims: u32 = 0,
    shape: [4]u32 = .{ 0, 0, 0, 0 },
    in_strides: [4]u32 = .{ 0, 0, 0, 0 },
    out_strides: [4]u32 = .{ 0, 0, 0, 0 },
};

comptime {
    const names = [_][]const u8{ "count", "operation", "lhs_size", "rhs_size", "reduce_size", "output_size", "lhs_offset", "rhs_offset", "output_offset", "aux_size", "aux_offset", "matrix_m", "matrix_n", "matrix_k", "flags", "alpha", "beta", "dims" };
    for (names, 0..) |name, i| if (@offsetOf(Params, name) != 4 * i) @compileError("qmm.comp push constant moved: " ++ name);
    if (@offsetOf(Params, "shape") != 72 or @offsetOf(Params, "in_strides") != 88 or @offsetOf(Params, "out_strides") != 104) @compileError("qmm.comp push constant arrays moved");
    if (@sizeOf(Params) != 120) @compileError("qmm.comp push constants are 120 bytes");
}

/// Base items of each binding's data: lhs_offset, rhs_offset (words), aux_offset, aux_size and output_offset.
pub const Offsets = struct { x: usize = 0, w: usize = 0, scales: usize = 0, biases: usize = 0, out: usize = 0 };

/// random: the contract family's affine fill; cancel: signed scales, bias -8 s and per-group sums that cancel.
pub const Fill = enum { random, cancel };

pub const Case = struct { name: []const u8, shape: c.Shape, fill: Fill = .random, at: Offsets = .{} };

/// Rows 1 and 3, n 8 and 257 (one output past a 256-thread group), k 64 and 128 (one and two groups).
pub const cases = [_]Case{
    .{ .name = "m1 n8 k64", .shape = .{ .rows = 1, .n = 8, .k = 64 } },
    .{ .name = "m3 n257 k128", .shape = .{ .rows = 3, .n = 257, .k = 128 } },
    .{ .name = "m1 n257 k128 offsets", .shape = .{ .rows = 1, .n = 257, .k = 128 }, .at = .{ .x = 5, .w = 3, .scales = 6, .biases = 9, .out = 7 } },
    .{ .name = "m3 n8 k64 cancel", .shape = .{ .rows = 3, .n = 8, .k = 64 }, .fill = .cancel, .at = .{ .x = 1, .w = 2, .scales = 3, .biases = 4, .out = 5 } },
};

/// Roundings an output passes through: s q and + b form its weight, x w its term, then k - 1 in-order fp32 additions.
fn depth(s: c.Shape) usize {
    return s.k + 2;
}

/// qmm_f32 (no defines) only, an operator bring-up gate: no BF16 or model claim; served lists the bring-up cases.
pub const spec: c.Spec = .{ .name = "vulkan qmm_f32 affine q4 g64", .accum = .f32, .depth = depth, .out = .f32, .served = &served };

const served = blk: {
    var s: [cases.len]c.Shape = undefined;
    for (cases, &s) |case, *x| x.* = case.shape;
    break :blk s;
};

/// The worst output: |got - want| over its bound (NaN and infinities count as infinitely far), and the largest error.
pub const Report = struct { worst: f64 = 0, row: usize = 0, col: usize = 0, max_abs: f64 = 0 };

/// One case's five bindings as uploaded, and the float64 value and term magnitude of each output.
pub const Fixture = struct {
    a: std.mem.Allocator,
    case: Case,
    /// Logical codes and bf16 values the float64 reference reads; each bf16 value is exact in the f32 bindings.
    q: Affine,
    /// Binding 0: x [m, k] at at.x.
    x: []f32,
    /// Binding 1: w [n, k / 8] words at at.w, code k of a row in bits 4 (k % 8) of word k / 8.
    w: []u32,
    /// Binding 2: scales [n, k / 64] at at.scales.
    scales: []f32,
    /// Binding 3: biases [n, k / 64] at at.biases.
    biases: []f32,
    /// Binding 4 as it must be uploaded: every element poisoned; outputs [m, n] land at at.out.
    out: []f32,
    want: []f64,
    mag: []f64,

    pub fn init(a: std.mem.Allocator, case: Case, seed: u64) !Fixture {
        const s = case.shape;
        const at = case.at;
        var q = try Affine.init(a, s, bits, group);
        errdefer q.deinit();
        q.form = .dequantized;
        switch (case.fill) {
            .random => try q.fill(.random, seed),
            .cancel => cancel(q, seed),
        }
        const x = try poisoned(a, f32, at.x, q.x.len);
        errdefer a.free(x);
        for (x[at.x..][0..q.x.len], q.x) |*d, v| d.* = c.fromBf16(v);
        const words = s.n * s.k * bits / 32;
        const w = try poisoned(a, u32, at.w, words);
        errdefer a.free(w);
        q.words(w[at.w..][0..words]);
        const scales = try poisoned(a, f32, at.scales, q.scales.len);
        errdefer a.free(scales);
        for (scales[at.scales..][0..q.scales.len], q.scales) |*d, v| d.* = c.fromBf16(v);
        const biases = try poisoned(a, f32, at.biases, q.biases.len);
        errdefer a.free(biases);
        for (biases[at.biases..][0..q.biases.len], q.biases) |*d, v| d.* = c.fromBf16(v);
        const out = try poisoned(a, f32, at.out, s.rows * s.n);
        errdefer a.free(out);
        const want = try a.alloc(f64, s.rows * s.n);
        errdefer a.free(want);
        const mag = try a.alloc(f64, want.len);
        errdefer a.free(mag);
        try q.reference(want, mag);
        for (want) |v| if (!std.math.isFinite(v)) return error.NonFiniteReference;
        return .{ .a = a, .case = case, .q = q, .x = x, .w = w, .scales = scales, .biases = biases, .out = out, .want = want, .mag = mag };
    }

    pub fn deinit(f: Fixture) void {
        f.q.deinit();
        for ([_][]f32{ f.x, f.scales, f.biases, f.out }) |b| f.a.free(b);
        f.a.free(f.w);
        f.a.free(f.want);
        f.a.free(f.mag);
    }

    /// MLX's QuantizedMatmul routing for one transposed, unbatched slice (shape[0] = 1: zero would write nothing).
    pub fn params(f: Fixture) Params {
        const s = f.case.shape;
        const at = f.case.at;
        const count: u32 = @intCast(s.rows * s.n);
        return .{
            .count = count,
            .operation = bits,
            .reduce_size = group,
            .output_size = count,
            .lhs_offset = @intCast(at.x),
            .rhs_offset = @intCast(at.w),
            .output_offset = @intCast(at.out),
            .aux_size = @intCast(at.biases),
            .aux_offset = @intCast(at.scales),
            .matrix_m = @intCast(s.rows),
            .matrix_n = @intCast(s.n),
            .matrix_k = @intCast(s.k),
            .shape = .{ 1, 0, 0, 0 },
        };
    }

    /// Workgroups along x for one dispatch, as MLX's compute_dispatch_group_count.
    pub fn groupCount(f: Fixture) u32 {
        return @intCast((f.want.len + local_size_x - 1) / local_size_x);
    }

    /// got: binding 4 read back whole; guards keep their bits and every output meets its bound, or an error says which.
    pub fn check(f: Fixture, got: []const f32) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        for (got, f.out, 0..) |g, o, i| {
            if (i >= at and i < at + f.want.len) continue;
            if (@as(u32, @bitCast(g)) != @as(u32, @bitCast(o))) return error.GuardClobbered;
        }
        const r = f.measure(got);
        if (!(r.worst <= 1)) return error.OutOfBound;
        return r;
    }

    /// Each output's distance from float64 over its bound; call after check fails to name the worst output.
    pub fn measure(f: Fixture, got: []const f32) Report {
        std.debug.assert(got.len == f.out.len);
        var r: Report = .{};
        for (got[f.case.at.out..][0..f.want.len], f.want, f.mag, 0..) |g, w, m, i| {
            const d = @abs(@as(f64, g) - w);
            const e = if (std.math.isNan(d)) std.math.inf(f64) else d;
            const ratio = e / c.bound(spec, f.case.shape, w, m);
            r.max_abs = @max(r.max_abs, e);
            if (ratio > r.worst) {
                r.worst = ratio;
                r.row = i / f.case.shape.n;
                r.col = i % f.case.shape.n;
            }
        }
        return r;
    }
};

fn poisoned(a: std.mem.Allocator, comptime T: type, base: usize, len: usize) ![]T {
    const buf = try a.alloc(T, base + len + tail);
    @memset(buf, if (T == u32) 0xffff_ffff else @bitCast(poison));
    return buf;
}

/// Bias -8 s (exact) and codes mirrored about 8 across a group's halves; x repeats across them but once a group.
fn cancel(q: Affine, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (q.scales, q.biases, 0..) |*s, *b, i| {
        s.* = c.toBf16((rng.float(f32) * 0.02 + 0.001) * @as(f32, if (i % 2 == 1) -1 else 1));
        b.* = c.toBf16(-8 * c.fromBf16(s.*));
    }
    const h = group / 2;
    for (0..q.codes.len / group) |g| for (0..h) |i| {
        const v = 1 + rng.uintLessThan(u8, 15);
        q.codes[g * group + i] = v;
        q.codes[g * group + h + i] = 16 - v;
    };
    for (0..q.x.len / group) |g| for (0..h) |i| {
        const v = c.toBf16(rng.float(f32) * 4 - 2);
        q.x[g * group + i] = v;
        q.x[g * group + h + i] = if (i == g % h) c.toBf16(rng.float(f32) * 4 - 2) else v;
    };
}

const testing = std.testing;

const Sums = enum { f32_plain, f32_fused, f16 };

/// qmm.comp's transposed loop on the host over the bindings as uploaded: in-order acc += x (s q + b).
fn replay(f: Fixture, sums: Sums, got: []f32) void {
    const s = f.case.shape;
    const at = f.case.at;
    const groups = s.k / group;
    for (0..s.rows) |r| for (0..s.n) |j| {
        var acc: f32 = 0;
        var half: f16 = 0;
        for (0..s.k) |i| {
            const sc = f.scales[at.scales + j * groups + i / group];
            const b = f.biases[at.biases + j * groups + i / group];
            const word = f.w[at.w + j * s.k / 8 + i / 8];
            const q: f32 = @floatFromInt((word >> @intCast(4 * (i % 8))) & 0xf);
            const x = f.x[at.x + r * s.k + i];
            switch (sums) {
                .f32_plain => acc += x * (sc * q + b),
                .f32_fused => acc = @mulAdd(f32, x, @mulAdd(f32, sc, q, b), acc),
                .f16 => half += @as(f16, @floatCast(x * (sc * q + b))),
            }
        }
        got[at.out + r * s.n + j] = if (sums == .f16) half else acc;
    };
}

const Fault = enum { nibbles_swapped, params_strided_by_n, rows_share_row0, bias_dropped };

/// A faulty kernel's binding 4: float64 of the inputs as the fault misreads them, rounded to f32.
fn faulted(f: Fixture, fault: Fault) ![]f32 {
    const s = f.case.shape;
    var q = try Affine.init(f.a, s, bits, group);
    defer q.deinit();
    q.form = .dequantized;
    @memcpy(q.x, f.q.x);
    @memcpy(q.codes, f.q.codes);
    @memcpy(q.scales, f.q.scales);
    @memcpy(q.biases, f.q.biases);
    const groups = s.k / group;
    switch (fault) {
        .nibbles_swapped => for (q.codes, 0..) |*v, i| {
            v.* = f.q.codes[i ^ 1];
        },
        .params_strided_by_n => for (0..s.n) |j| for (0..groups) |g| {
            q.scales[j * groups + g] = f.q.scales[g * s.n + j];
            q.biases[j * groups + g] = f.q.biases[g * s.n + j];
        },
        .rows_share_row0 => for (1..s.rows) |r| @memcpy(q.x[r * s.k ..][0..s.k], f.q.x[0..s.k]),
        .bias_dropped => @memset(q.biases, 0),
    }
    const want = try f.a.alloc(f64, f.want.len);
    defer f.a.free(want);
    const mag = try f.a.alloc(f64, f.want.len);
    defer f.a.free(mag);
    try q.reference(want, mag);
    const got = try f.a.dupe(f32, f.out);
    for (got[f.case.at.out..][0..want.len], want) |*g, w| g.* = @floatCast(w);
    return got;
}

test "vulkan qmm fixture: push constants carry MLX's routing, bases apart and one batch" {
    const f = try Fixture.init(testing.allocator, cases[2], 1);
    defer f.deinit();
    const p = f.params();
    try testing.expectEqual(@as(u32, 257), p.count);
    try testing.expectEqual(p.count, p.output_size);
    try testing.expectEqual(@as(u32, 4), p.operation);
    try testing.expectEqual(@as(u32, 64), p.reduce_size);
    try testing.expectEqual(@as(u32, 0), p.flags);
    try testing.expectEqual([4]u32{ 1, 0, 0, 0 }, p.shape);
    try testing.expectEqualSlices(u32, &[_]u32{ 5, 3, 6, 9, 7 }, &[_]u32{ p.lhs_offset, p.rhs_offset, p.aux_offset, p.aux_size, p.output_offset });
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 257, 128 }, &[_]u32{ p.matrix_m, p.matrix_n, p.matrix_k });
    try testing.expectEqual(@as(u32, 2), f.groupCount());
    const g = try Fixture.init(testing.allocator, cases[1], 1);
    defer g.deinit();
    try testing.expectEqual(@as(u32, 4), g.groupCount());
}

test "vulkan qmm fixture: Q4 words low nibble first, k / 8 words a row from the w base" {
    const a = testing.allocator;
    const q = try Affine.init(a, .{ .rows = 1, .n = 1, .k = 64 }, bits, group);
    defer q.deinit();
    for (q.codes, 0..) |*v, i| v.* = @intCast(i % 16);
    var w: [8]u32 = undefined;
    q.words(&w);
    try testing.expectEqual(@as(u32, 0x7654_3210), w[0]);
    try testing.expectEqual(@as(u32, 0xfedc_ba98), w[1]);
    for (cases) |case| {
        const f = try Fixture.init(a, case, 7);
        defer f.deinit();
        const k = case.shape.k;
        for (0..case.shape.n) |j| for (0..k) |i| {
            const word = f.w[case.at.w + j * k / 8 + i / 8];
            try testing.expectEqual(f.q.codes[j * k + i], @as(u8, @intCast((word >> @intCast(4 * (i % 8))) & 0xf)));
        };
        try testing.expectEqual(@as(u32, 0xffff_ffff), f.w[case.at.w + case.shape.n * k / 8]);
    }
}

test "vulkan qmm fixture: rounded float64 passes; poison, past the bound, NaN, Inf and guard writes fail" {
    const a = testing.allocator;
    const f = try Fixture.init(a, cases[1], 3);
    defer f.deinit();
    const got = try a.dupe(f32, f.out);
    defer a.free(got);
    const at = f.case.at.out;
    try testing.expectError(error.OutOfBound, f.check(got));
    for (got[at..][0..f.want.len], f.want) |*g, w| g.* = @floatCast(w);
    try testing.expect((try f.check(got)).worst <= 1);
    const i = 300;
    const keep = got[at + i];
    got[at + i] = @floatCast(f.want[i] + 4 * c.bound(spec, f.case.shape, f.want[i], f.mag[i]));
    try testing.expectError(error.OutOfBound, f.check(got));
    const r = f.measure(got);
    try testing.expectEqual([2]usize{ 1, 43 }, [2]usize{ r.row, r.col });
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |v| {
        got[at + i] = v;
        try testing.expectError(error.OutOfBound, f.check(got));
    }
    got[at + i] = keep;
    got[at + f.want.len] = 0;
    try testing.expectError(error.GuardClobbered, f.check(got));
    try testing.expectError(error.LengthMismatch, f.check(got[1..]));
}

test "vulkan qmm fixture: in-order f32 sums meet the bound with and without fma, f16 sums do not" {
    const a = testing.allocator;
    for (cases) |case| {
        const f = try Fixture.init(a, case, 13);
        defer f.deinit();
        const got = try a.dupe(f32, f.out);
        defer a.free(got);
        for ([_]Sums{ .f32_plain, .f32_fused }) |sums| {
            replay(f, sums, got);
            _ = try f.check(got);
        }
        replay(f, .f16, got);
        try testing.expectError(error.OutOfBound, f.check(got));
    }
}

test "vulkan qmm fixture: swapped nibbles, param or row strides, dropped bias and wrong bases fail" {
    const a = testing.allocator;
    const plans = [_]struct { Fault, usize }{ .{ .nibbles_swapped, 1 }, .{ .params_strided_by_n, 1 }, .{ .rows_share_row0, 1 }, .{ .bias_dropped, 3 } };
    for (plans) |p| {
        const f = try Fixture.init(a, cases[p[1]], 11);
        defer f.deinit();
        const got = try faulted(f, p[0]);
        defer a.free(got);
        try testing.expectError(error.OutOfBound, f.check(got));
    }
    const f = try Fixture.init(a, cases[2], 11);
    defer f.deinit();
    const got = try a.dupe(f32, f.out);
    defer a.free(got);
    var bad = f;
    bad.case.at.x = 0;
    replay(bad, .f32_plain, got);
    try testing.expectError(error.OutOfBound, f.check(got));
    bad = f;
    bad.case.at.scales = f.case.at.biases;
    bad.case.at.biases = f.case.at.scales;
    replay(bad, .f32_plain, got);
    try testing.expectError(error.OutOfBound, f.check(got));
}

test "vulkan qmm fixture: finite and repeatable for a seed; cancel mixes scale signs and cancels its sums" {
    const a = testing.allocator;
    for (cases) |case| {
        const f = try Fixture.init(a, case, 5);
        defer f.deinit();
        const g = try Fixture.init(a, case, 5);
        defer g.deinit();
        for ([_][2][]const f32{ .{ f.x, g.x }, .{ f.scales, g.scales }, .{ f.biases, g.biases }, .{ f.out, g.out } }) |p| {
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(p[0]), std.mem.sliceAsBytes(p[1]));
        }
        try testing.expectEqualSlices(u32, f.w, g.w);
        try testing.expectEqualSlices(f64, f.want, g.want);
    }
    const f = try Fixture.init(a, cases[3], 5);
    defer f.deinit();
    var signs = [2]bool{ false, false };
    for (f.scales[f.case.at.scales..][0..f.q.scales.len]) |v| signs[@intFromBool(v < 0)] = true;
    try testing.expect(signs[0] and signs[1]);
    for (f.want, f.mag) |w, m| try testing.expect(@abs(w) * 8 < m);
}
