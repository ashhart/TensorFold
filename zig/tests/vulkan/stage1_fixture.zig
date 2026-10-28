//! Stage 1 fixture: BF16 RNE cast vectors, copy/gather layouts, quantized-embedding cases and
//! Q4 affine g64 cases against the contract's FP64 bound. Host-checkable; no device code here.
//! Stage 1 bring-up only: no model-support or performance claim anywhere.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const Affine = contract.affine.Affine;

/// Poisoned elements after each binding's data, so an overrun reads poison rather than past the buffer.
pub const tail = 16;
pub const poison16: u16 = 0xffff;
pub const poison32: u32 = 0xffff_ffff;

pub fn castGuards(readback: []const u16, elements: usize, words_per_element: usize) !void {
    if (words_per_element != 1 and words_per_element != 2) return error.Invalid;
    const data_words = try std.math.mul(usize, elements, words_per_element);
    const total_words = try std.math.add(usize, data_words, tail * words_per_element);
    if (readback.len != total_words) return error.LengthMismatch;
    for (readback[data_words..]) |word| if (word != 0xa5a5) return error.GuardClobbered;
}

pub fn depth(s: c.Shape) usize {
    // Roundings an output passes through: s q and + b form its weight, x w its term, then k - 1 fp32 additions.
    return s.k + 2;
}

/// qmm_bf16 keeps the f32 accumulator and adds one rounding to the bf16 output.
pub const spec: c.Spec = .{ .name = "vulkan qmm_bf16 affine q4 g64", .accum = .f32, .depth = depth, .out = .bf16, .served = &served };

pub const Fill = enum { random, cancel };
pub const Offsets = struct { x: usize = 0, w: usize = 0, scales: usize = 0, biases: usize = 0, out: usize = 0 };
pub const Case = struct { name: []const u8, shape: c.Shape, fill: Fill = .random, at: Offsets = .{} };

/// Small real shapes: one and two rows, one output per 256-thread group and one past it,
/// one and two groups per row, offsets, and a cancel fill. Every m*n is even (word-padded bf16 views).
/// Binding bases sit at 256-byte multiples — the largest minStorageBufferOffsetAlignment Vulkan
/// allows — so the offsets exercise the production aligned-binding contract on any device, not
/// arbitrary offsets; Views guarantee only 4-byte alignment.
pub const cases = [_]Case{
    .{ .name = "m1 n8 k64", .shape = .{ .rows = 1, .n = 8, .k = 64 } },
    .{ .name = "m2 n257 k128", .shape = .{ .rows = 2, .n = 257, .k = 128 } },
    .{ .name = "m2 n256 k128 offsets", .shape = .{ .rows = 2, .n = 256, .k = 128 }, .at = .{ .x = 128, .w = 64, .scales = 256, .biases = 384, .out = 512 } },
    .{ .name = "m3 n8 k64 cancel", .shape = .{ .rows = 3, .n = 8, .k = 64 }, .fill = .cancel, .at = .{ .x = 128, .w = 192, .scales = 256, .biases = 384, .out = 384 } },
};

const served = blk: {
    var s: [cases.len]c.Shape = undefined;
    for (cases, &s) |case, *x| x.* = case.shape;
    break :blk s;
};

/// One f32 input and the exact bf16 bits a round-to-nearest-even cast must store.
pub const Bf16Vec = struct { in: f32, want: u16, why: []const u8 };

/// Ties (to even, both signs), exact grid points, above/below half, negatives, bf16 subnormals and
/// the nonfinite controls. 18 entries; with the 128 randoms the cast views stay word-padded.
pub const bf16_vectors = [_]Bf16Vec{
    .{ .in = 1.0, .want = 0x3f80, .why = "exact" },
    .{ .in = -1.0, .want = 0xbf80, .why = "negative exact" },
    .{ .in = -0.0, .want = 0x8000, .why = "negative zero keeps its sign bit" },
    .{ .in = 1.0 + 0x1p-9, .want = 0x3f80, .why = "below half rounds down" },
    .{ .in = 1.0 + 0x1p-8, .want = 0x3f80, .why = "exact tie, kept lsb 0, ties to even down" },
    .{ .in = 1.0 + 0x1p-7, .want = 0x3f81, .why = "exact grid point, no rounding" },
    .{ .in = 1.0 + 3 * 0x1p-8, .want = 0x3f82, .why = "exact tie, kept lsb 1, ties to even up" },
    .{ .in = 1.0 + 0x1p-8 + 0x1p-20, .want = 0x3f81, .why = "a hair above half rounds up" },
    .{ .in = -(1.0 + 0x1p-8), .want = 0xbf80, .why = "negative tie to even" },
    .{ .in = -(1.0 + 3 * 0x1p-8), .want = 0xbf82, .why = "negative tie, kept lsb 1, ties to even up" },
    .{ .in = 0x1p-133, .want = 0x0001, .why = "one bf16 subnormal unit" },
    .{ .in = 0x1p-126, .want = 0x0080, .why = "128 bf16 subnormal units" },
    .{ .in = 0.75 * 0x1p-133, .want = 0x0001, .why = "three quarters of a unit rounds up to one" },
    .{ .in = 0x1p-134, .want = 0x0000, .why = "half a subnormal unit ties to even zero" },
    .{ .in = 3 * 0x1p-134, .want = 0x0002, .why = "1.5 subnormal units tie to even up" },
    .{ .in = std.math.inf(f32), .want = 0x7f80, .why = "+inf" },
    .{ .in = -std.math.inf(f32), .want = 0xff80, .why = "-inf" },
    .{ .in = std.math.nan(f32), .want = 0x7fc0, .why = "quiet nan keeps its class" },
};

comptime {
    for (bf16_vectors) |v| if (c.toBf16(v.in) != v.want)
        @compileError("bf16 vector disagrees with the contract's RNE cast: " ++ v.why);
}

/// Random u32 words (and a u16 variant) uploaded at fixed offsets with guard tails on both sides.
/// Offsets are 256-byte multiples (the largest minStorageBufferOffsetAlignment Vulkan allows):
/// the copy exercises the production aligned-binding contract, not arbitrary offsets.
pub const copy_words = 9;
pub const copy_half_elements = 22;
pub const copy_u32_src = 64; // words (256 bytes)
pub const copy_u32_dst = 128; // words (512 bytes)
pub const copy_u16_src = 128; // elements (256 bytes)
pub const copy_u16_dst = 256; // elements (512 bytes)

/// A quantized embedding: packed q4 rows plus bf16 scales/biases, gathered by host-chosen ids.
pub const EmbedCase = struct { name: []const u8, tokens: usize, rows: usize, k: usize };

pub const embed_cases = [_]EmbedCase{
    // tokens are even in every case: each gathered token contributes k/64 bf16 scales, and a
    // ragged tail element would be refused by the typed view before any GPU work.
    .{ .name = "6 tok 8 rows k64", .tokens = 6, .rows = 8, .k = 64 },
    .{ .name = "3 tok 4 rows k128 repeats", .tokens = 3, .rows = 4, .k = 128 },
};

/// One embedding case's table as uploaded, the token ids, and the f64 value each output must carry.
pub const EmbedTable = struct {
    a: std.mem.Allocator,
    case: EmbedCase,
    q: Affine,
    /// Packed rows: word [r, i / 8] holds code i of row r in bits 4 * (i % 8).
    w: []u32,
    scales: []u16,
    biases: []u16,
    /// Every id < rows; ids[0] is repeated at ids[1] so a repeated row is always exercised.
    ids: []u32,
    /// want[t * k + i]: the f64 s q + b of token t's row at element i.
    want: []f64,

    pub fn init(a: std.mem.Allocator, case: EmbedCase, seed: u64) !EmbedTable {
        var q = try Affine.init(a, .{ .rows = case.rows, .n = case.rows, .k = case.k }, 4, 64);
        errdefer q.deinit();
        try q.fill(.random, seed);
        const words = case.rows * case.k / 8;
        const w = try a.alloc(u32, words + tail);
        errdefer a.free(w);
        @memset(w, poison32);
        q.words(w[0..words]);
        const scales = try a.alloc(u16, q.scales.len + tail);
        errdefer a.free(scales);
        @memset(scales, poison16);
        @memcpy(scales[0..q.scales.len], q.scales);
        const biases = try a.alloc(u16, q.biases.len + tail);
        errdefer a.free(biases);
        @memset(biases, poison16);
        @memcpy(biases[0..q.biases.len], q.biases);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5eed);
        const rng = prng.random();
        const ids = try a.alloc(u32, case.tokens);
        errdefer a.free(ids);
        for (ids) |*id| id.* = rng.uintLessThan(u32, @intCast(case.rows));
        ids[1 % ids.len] = ids[0]; // a repeated row, always
        const want = try a.alloc(f64, case.tokens * case.k);
        errdefer a.free(want);
        const groups = case.k / 64;
        for (ids, 0..) |row, t| for (0..case.k) |i| {
            const g = @as(usize, @intCast(row)) * groups + i / 64;
            const code: f64 = @floatFromInt(q.codes[@as(usize, @intCast(row)) * case.k + i]);
            want[t * case.k + i] = @as(f64, c.fromBf16(q.scales[g])) * code + @as(f64, c.fromBf16(q.biases[g]));
        };
        return .{ .a = a, .case = case, .q = q, .w = w, .scales = scales, .biases = biases, .ids = ids, .want = want };
    }

    pub fn deinit(t: EmbedTable) void {
        t.q.deinit();
        t.a.free(t.w);
        t.a.free(t.scales);
        t.a.free(t.biases);
        t.a.free(t.ids);
        t.a.free(t.want);
    }
};

/// The worst output: |got - want| over its contract bound (nonfinite counts as infinitely far).
pub const Report = struct { worst: f64 = 0, row: usize = 0, col: usize = 0, max_abs: f64 = 0 };

/// One Q4 case's bindings as uploaded (bf16 as raw bits), and the f64 reference of each output.
pub const Fixture = struct {
    a: std.mem.Allocator,
    case: Case,
    q: Affine,
    /// Binding: x [m, k] bf16 bits at at.x.
    x: []u16,
    /// Binding: w [n, k / 8] words at at.w, code k of row j in bits 4 * (k % 8) of word k / 8.
    w: []u32,
    /// Bindings: scales and biases [n, k / 64] bf16 bits at at.scales / at.biases.
    scales: []u16,
    biases: []u16,
    /// Binding as uploaded: every element poisoned; outputs [m, n] land at at.out.
    out: []u16,
    want: []f64,
    mag: []f64,

    pub fn init(a: std.mem.Allocator, case: Case, seed: u64) !Fixture {
        const s = case.shape;
        const at = case.at;
        var q = try Affine.init(a, s, 4, 64);
        errdefer q.deinit();
        q.form = .dequantized;
        switch (case.fill) {
            .random => try q.fill(.random, seed),
            .cancel => try cancel(q, seed),
        }
        const x = try poisoned16(a, at.x, q.x.len);
        errdefer a.free(x);
        @memcpy(x[at.x..][0..q.x.len], q.x);
        const w = try poisoned32(a, at.w, s.n * s.k / 8);
        errdefer a.free(w);
        q.words(w[at.w..][0 .. s.n * s.k / 8]);
        const scales = try poisoned16(a, at.scales, q.scales.len);
        errdefer a.free(scales);
        @memcpy(scales[at.scales..][0..q.scales.len], q.scales);
        const biases = try poisoned16(a, at.biases, q.biases.len);
        errdefer a.free(biases);
        @memcpy(biases[at.biases..][0..q.biases.len], q.biases);
        const out = try poisoned16(a, at.out, s.rows * s.n);
        errdefer a.free(out);
        const want = try a.alloc(f64, s.rows * s.n);
        errdefer a.free(want);
        const mag = try a.alloc(f64, want.len);
        errdefer a.free(mag);
        try q.reference(want, mag);
        for (want, mag) |expected, magnitude| if (!std.math.isFinite(expected) or !std.math.isFinite(magnitude) or magnitude < 0) return error.NonFiniteReference;
        return .{ .a = a, .case = case, .q = q, .x = x, .w = w, .scales = scales, .biases = biases, .out = out, .want = want, .mag = mag };
    }

    pub fn deinit(f: Fixture) void {
        f.q.deinit();
        f.a.free(f.x);
        f.a.free(f.w);
        f.a.free(f.scales);
        f.a.free(f.biases);
        f.a.free(f.out);
        f.a.free(f.want);
        f.a.free(f.mag);
    }

    /// got: bf16 bits read back whole; guards keep their bits and every output meets its bound.
    pub fn check(f: Fixture, got: []const u16) !Report {
        if (got.len != f.out.len) return error.LengthMismatch;
        const at = f.case.at.out;
        for (got, f.out, 0..) |g, o, i| {
            if (i >= at and i < at + f.want.len) continue;
            if (g != o) return error.GuardClobbered;
        }
        const r = f.measure(got);
        if (!(r.worst <= 1)) return error.OutOfBound;
        return r;
    }

    /// Each output's distance from f64 over its bound; call after check fails to name the worst output.
    pub fn measure(f: Fixture, got: []const u16) Report {
        std.debug.assert(got.len == f.out.len);
        var r: Report = .{};
        for (got[f.case.at.out..][0..f.want.len], f.want, f.mag, 0..) |bits, w, m, i| {
            const g: f64 = c.fromBf16(bits);
            const d = @abs(g - w);
            const e = if (std.math.isNan(d)) std.math.inf(f64) else d;
            const bound = c.bound(spec, f.case.shape, w, m);
            const valid = std.math.isFinite(m) and m >= 0 and std.math.isFinite(bound) and bound > 0;
            const ratio = if (valid) e / bound else std.math.inf(f64);
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

fn poisoned16(a: std.mem.Allocator, base: usize, len: usize) ![]u16 {
    const buf = try a.alloc(u16, base + len + tail);
    @memset(buf, poison16);
    return buf;
}

fn poisoned32(a: std.mem.Allocator, base: usize, len: usize) ![]u32 {
    const buf = try a.alloc(u32, base + len + tail);
    @memset(buf, poison32);
    return buf;
}

/// Bias -8 s (exact) and codes mirrored about 8 across a group's halves; x repeats across them but once a group.
fn cancel(q: Affine, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const group = q.group;
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

test "bf16 vectors agree with the contract's RNE cast at every stated geometry" {
    for (bf16_vectors) |v| try testing.expectEqual(v.want, c.toBf16(v.in));
    try testing.expectEqual(@as(u16, 0x3f81), c.toBf16(1.0 + 0x1p-8 + 0x1p-20)); // a hair above half
}

test "projection fixture bindings meet the largest Vulkan storage alignment" {
    for (cases) |case| {
        for ([_]usize{ case.at.x * 2, case.at.w * 4, case.at.scales * 2, case.at.biases * 2, case.at.out * 2 }) |bytes| {
            try testing.expectEqual(@as(usize, 0), bytes % 256);
        }
    }
}

test "both cast widths check the actual tail after a whole output readback" {
    const elements = bf16_vectors.len + 128;
    inline for (.{ 1, 2 }) |words| {
        var shell: [(elements + tail) * words]u16 = undefined;
        @memset(shell[0 .. elements * words], 0x3f80);
        @memset(shell[elements * words ..], 0xa5a5);
        try castGuards(&shell, elements, words);
        try testing.expectError(error.LengthMismatch, castGuards(shell[0 .. elements * words], elements, words));
        try testing.expectError(error.LengthMismatch, castGuards(shell[0 .. tail * words], elements, words));
        shell[elements * words] ^= 1;
        try testing.expectError(error.GuardClobbered, castGuards(&shell, elements, words));
    }
}

test "embedding tables choose valid, once-repeated ids and finite f64 wants over poison tails" {
    var t = try EmbedTable.init(testing.allocator, embed_cases[1], 7);
    defer t.deinit();
    try testing.expect(t.ids[1] == t.ids[0]);
    for (t.ids) |id| try testing.expect(id < t.case.rows);
    const words = t.case.rows * t.case.k / 8;
    for (t.w[words..]) |g| try testing.expectEqual(poison32, g);
    var nonzero: usize = 0;
    for (t.want) |w| {
        try testing.expect(std.math.isFinite(w));
        nonzero += @intFromBool(w != 0);
    }
    try testing.expect(nonzero > t.want.len / 2);
}

test "q4 fixture: the checker refuses a planted nonfinite output and a clobbered guard" {
    var f = try Fixture.init(testing.allocator, cases[0], 3);
    defer f.deinit();
    const got = try testing.allocator.dupe(u16, f.out);
    defer testing.allocator.free(got);
    for (f.want, 0..) |w, i| got[f.case.at.out + i] = c.toBf16(@floatCast(w));
    _ = try f.check(got);
    got[f.case.at.out] = 0x7f80; // +inf at output 0
    try testing.expectError(error.OutOfBound, f.check(got));
    got[f.case.at.out] = c.toBf16(@floatCast(f.want[0]));
    got[f.case.at.out + f.want.len] ^= 1;
    try testing.expectError(error.GuardClobbered, f.check(got));
}

test "q4 fixture: reference-rounded outputs meet the contract bound, including the cancel fill" {
    for (cases) |case| {
        var f = try Fixture.init(testing.allocator, case, 11);
        defer f.deinit();
        const got = try testing.allocator.dupe(u16, f.out);
        defer testing.allocator.free(got);
        for (f.want, 0..) |w, i| got[f.case.at.out + i] = c.toBf16(@floatCast(w));
        const r = try f.check(got);
        try testing.expect(r.worst <= 1);
    }
}
