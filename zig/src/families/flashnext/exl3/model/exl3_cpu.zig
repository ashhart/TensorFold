//! CPU GEMV for one EXL3 matrix.

const std = @import("std");
const builtin = @import("builtin");
const format = @import("../pack/exl3_format.zig");

/// AVX fp256, compiled only when this target actually has AVX2. No AVX-512.
const avx2_target = builtin.cpu.arch == .x86_64 and
    builtin.zig_backend != .stage2_x86_64 and
    builtin.cpu.has(.x86, .avx2);

const Acc = @Vector(4, f64);

/// fp32 SwiGLU from the v0.6.6 gate-up epilogue (`act_mode` 1).
pub fn swiglu(gate: []const f64, up: []const f64, out: []f64, limit: f64) error{Length}!void {
    if (gate.len != up.len or gate.len != out.len) return error.Length;
    for (gate, up, out) |g, u, *y| {
        const gg = @min(g, limit);
        const uu = @min(@max(u, -limit), limit);
        y.* = (gg / (1.0 + @exp(-gg))) * uu;
    }
}

/// Weighted sum of `weights.len` expert rows.
pub fn combine(rows: []const f64, weights: []const f64, n: usize, out: []f64) error{Length}!void {
    if (n == 0 or out.len != n or weights.len == 0 or rows.len != weights.len * n) return error.Length;
    @memset(out, 0);
    for (weights, 0..) |w, s| {
        const row = rows[s * n ..][0..n];
        for (out, row) |*y, v| y.* += w * v;
    }
}

/// K = tiles_k * 16 and N = tiles_n * 16, both multiples of 128 (one Hadamard block).
pub fn gemv(
    w: format.Width,
    codebook: format.Codebook,
    trellis: []const i16,
    tiles_k: usize,
    tiles_n: usize,
    suh: []const f16,
    svh: []const f16,
    x: []const f64,
    y: []f64,
    scratch: []f64,
) error{Length}!void {
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    if (k % format.hadamard_dim != 0 or n % format.hadamard_dim != 0) return error.Length;
    if (x.len != k or y.len != n or suh.len != k or svh.len != n) return error.Length;
    const cells = k * n;
    if (scratch.len < 3 * cells) return error.Length;

    const weight = scratch[0..cells];
    format.dequantize(w, codebook, trellis, tiles_k, tiles_n, suh, svh, weight, scratch[cells .. 3 * cells]);

    for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |i| sum += x[i] * weight[i * n + j];
        y[j] = sum;
    }
}

/// Bit-exact with `gemv`.
pub fn gemvAvx2(
    w: format.Width,
    codebook: format.Codebook,
    trellis: []const i16,
    tiles_k: usize,
    tiles_n: usize,
    suh: []const f16,
    svh: []const f16,
    x: []const f64,
    y: []f64,
    scratch: []f64,
) error{Length}!void {
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    if (k % format.hadamard_dim != 0 or n % format.hadamard_dim != 0) return error.Length;
    if (x.len != k or y.len != n or suh.len != k or svh.len != n) return error.Length;
    const cells = k * n;
    if (scratch.len < 3 * cells) return error.Length;

    const weight = scratch[0..cells];
    format.dequantize(w, codebook, trellis, tiles_k, tiles_n, suh, svh, weight, scratch[cells .. 3 * cells]);

    // N is a multiple of 128, so the four-column walk covers every column.
    if (comptime avx2_target) {
        accumulateAvx2(weight, x, y);
    } else {
        accumulatePortable(weight, x, y);
    }
}

/// Same four lanes as the AVX2 loop, one left-to-right sum per column.
fn accumulatePortable(weight: []const f64, x: []const f64, y: []f64) void {
    @setFloatMode(.strict);
    const k = x.len;
    const n = y.len;
    var j: usize = 0;
    while (j < n) : (j += 4) {
        var acc: [4]f64 = .{ 0, 0, 0, 0 };
        var i: usize = 0;
        while (i < k) : (i += 1) {
            const xi = x[i];
            const row = i * n + j;
            inline for (0..4) |lane| acc[lane] += xi * weight[row + lane];
        }
        y[j..][0..4].* = acc;
    }
}

fn avx2Load(p: [*]const f64) Acc {
    return asm (
        \\vmovupd (%[p]), %[out]
        : [out] "=x" (-> Acc),
        : [p] "r" (p),
    );
}

fn avx2Broadcast(p: *const f64) Acc {
    return asm (
        \\vbroadcastsd (%[p]), %[out]
        : [out] "=x" (-> Acc),
        : [p] "r" (p),
    );
}

fn avx2Mul(a: Acc, b: Acc) Acc {
    return asm (
        \\vmulpd %[b], %[a], %[out]
        : [out] "=x" (-> Acc),
        : [a] "x" (a),
          [b] "x" (b),
    );
}

/// `out = sum + prod`, the same operand order as `sum += prod`.
fn avx2Add(sum: Acc, prod: Acc) Acc {
    return asm (
        \\vaddpd %[prod], %[sum], %[out]
        : [out] "=x" (-> Acc),
        : [sum] "x" (sum),
          [prod] "x" (prod),
    );
}

fn avx2Store(p: [*]f64, v: Acc) void {
    asm volatile (
        \\vmovupd %[v], (%[p])
        :
        : [v] "x" (v),
          [p] "r" (p),
        : .{ .memory = true });
}

/// ymm lanes are columns j..j+3. vmovupd:
fn accumulateAvx2(weight: []const f64, x: []const f64, y: []f64) void {
    comptime if (builtin.cpu.arch != .x86_64) @compileError("AVX2 GEMV is x86_64 only");
    const k = x.len;
    const n = y.len;
    var j: usize = 0;
    while (j < n) : (j += 4) {
        var acc: Acc = @splat(0);
        var i: usize = 0;
        while (i < k) : (i += 1) {
            const prod = avx2Mul(avx2Broadcast(&x[i]), avx2Load(weight.ptr + i * n + j));
            acc = avx2Add(acc, prod);
        }
        avx2Store(y.ptr + j, acc);
    }
}

const sentinel: f64 = @bitCast(@as(u64, 0x7FF8_0000_0000_0001));

/// 128x128 mul1, width 2 halves. Trellis length is 8*8*Width.tileWords().
const Square = struct {
    allocator: std.mem.Allocator,
    w: format.Width,
    tiles_k: usize,
    tiles_n: usize,
    trellis: []i16,
    suh: []f16,
    svh: []f16,
    x: []f64,
    y: []f64,
    scratch: []f64,

    fn init(allocator: std.mem.Allocator) !Square {
        const w: format.Width = .{ .halves = 2 };
        const tiles_k: usize = 8;
        const tiles_n: usize = 8;
        const k = tiles_k * 16;
        const n = tiles_n * 16;
        const trellis = try allocator.alloc(i16, tiles_k * tiles_n * w.tileWords());
        errdefer allocator.free(trellis);
        for (trellis, 0..) |*word, i| word.* = @bitCast(@as(u16, @truncate(i *% 4099 + 7)));

        const suh = try allocator.alloc(f16, k);
        errdefer allocator.free(suh);
        for (suh) |*v| v.* = 1.0;
        const svh = try allocator.alloc(f16, n);
        errdefer allocator.free(svh);
        for (svh) |*v| v.* = 1.0;

        const x = try allocator.alloc(f64, k);
        errdefer allocator.free(x);
        for (x, 0..) |*v, i| v.* = @floatFromInt(i + 1);

        const y = try allocator.alloc(f64, n);
        errdefer allocator.free(y);
        for (y) |*v| v.* = sentinel;

        const scratch = try allocator.alloc(f64, 3 * k * n);
        errdefer allocator.free(scratch);
        return .{
            .allocator = allocator,
            .w = w,
            .tiles_k = tiles_k,
            .tiles_n = tiles_n,
            .trellis = trellis,
            .suh = suh,
            .svh = svh,
            .x = x,
            .y = y,
            .scratch = scratch,
        };
    }

    fn deinit(self: Square) void {
        self.allocator.free(self.trellis);
        self.allocator.free(self.suh);
        self.allocator.free(self.svh);
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.allocator.free(self.scratch);
    }
};

test "128x128 mul1 gemv matches dequantize then a naive matvec" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    const k = s.tiles_k * 16;
    const n = s.tiles_n * 16;
    try std.testing.expectEqual(@as(usize, 128), k);
    try std.testing.expectEqual(@as(usize, 128), n);
    try std.testing.expectEqual(@as(usize, 8 * 8 * s.w.tileWords()), s.trellis.len);

    try gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, s.y, s.scratch);

    const weight = try gpa.alloc(f64, k * n);
    defer gpa.free(weight);
    const dq_scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(dq_scratch);
    format.dequantize(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, weight, dq_scratch);

    for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |i| sum += s.x[i] * weight[i * n + j];
        try std.testing.expectEqual(@as(u64, @bitCast(sum)), @as(u64, @bitCast(s.y[j])));
    }
}

test "scratch one f64 short returns Length and leaves y untouched" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    for (s.y) |*v| v.* = sentinel;
    const short = s.scratch[0 .. s.scratch.len - 1];
    try std.testing.expectError(error.Length, gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, s.y, short));
    for (s.y) |v| try std.testing.expectEqual(@as(u64, @bitCast(sentinel)), @as(u64, @bitCast(v)));
}

test "swiglu is silu(gate) times up, and a finite limit clamps both sides" {
    var out = [_]f64{ 9, 9 };
    const gate = [_]f64{ 0, 2 };
    const up = [_]f64{ 3, 4 };
    try swiglu(&gate, &up, &out, std.math.inf(f64));
    try std.testing.expectEqual(@as(f64, 0), out[0]);
    const silu2 = 2.0 / (1.0 + @exp(-2.0));
    try std.testing.expectApproxEqAbs(silu2 * 4.0, out[1], 1e-12);

    const wide_gate = [_]f64{2};
    const wide_up = [_]f64{4};
    var clamped = [_]f64{9};
    try swiglu(&wide_gate, &wide_up, &clamped, 1);
    const silu1 = 1.0 / (1.0 + @exp(-1.0));
    try std.testing.expectApproxEqAbs(silu1, clamped[0], 1e-12);

    var held = [_]f64{9};
    try std.testing.expectError(error.Length, swiglu(wide_gate[0..], wide_up[0..], held[0..0], 1));
    try std.testing.expectEqual(@as(f64, 9), held[0]);
}

test "combine is the weighted sum of expert rows" {
    const rows = [_]f64{ 1, 2, 3, 4 };
    const weights = [_]f64{ 0.25, 0.75 };
    var out = [_]f64{ 9, 9 };
    try combine(&rows, &weights, 2, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), out[1], 1e-12);
    try std.testing.expectError(error.Length, combine(rows[0..3], &weights, 2, &out));
}

test "x shorter than K returns Length" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    for (s.y) |*v| v.* = sentinel;
    const short_x = s.x[0 .. s.x.len - 1];
    try std.testing.expectError(error.Length, gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, short_x, s.y, s.scratch));
    for (s.y) |v| try std.testing.expectEqual(@as(u64, @bitCast(sentinel)), @as(u64, @bitCast(v)));
}

fn expectGemvAvx2(allocator: std.mem.Allocator, tiles_k: usize, tiles_n: usize) !void {
    const w: format.Width = .{ .halves = 2 };
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    const trellis = try allocator.alloc(i16, tiles_k * tiles_n * w.tileWords());
    defer allocator.free(trellis);
    for (trellis, 0..) |*word, i| word.* = @bitCast(@as(u16, @truncate(i *% 4099 + 7)));

    const suh = try allocator.alloc(f16, k);
    defer allocator.free(suh);
    for (suh) |*v| v.* = 1.0;
    const svh = try allocator.alloc(f16, n);
    defer allocator.free(svh);
    for (svh) |*v| v.* = 1.0;

    const x = try allocator.alloc(f64, k);
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    const y = try allocator.alloc(f64, n);
    defer allocator.free(y);
    const y2 = try allocator.alloc(f64, n);
    defer allocator.free(y2);
    @memset(y, sentinel);
    @memset(y2, sentinel);

    const scratch = try allocator.alloc(f64, 3 * k * n);
    defer allocator.free(scratch);
    try std.testing.expectEqual(@as(usize, 3 * k * n), scratch.len);

    try gemv(w, .mul1, trellis, tiles_k, tiles_n, suh, svh, x, y, scratch);
    try gemvAvx2(w, .mul1, trellis, tiles_k, tiles_n, suh, svh, x, y2, scratch);
    for (y, y2) |a, b| try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(b)));
}

test "gemvAvx2 matches gemv on the 128x128 mul1 tile" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    const y2 = try gpa.alloc(f64, s.y.len);
    defer gpa.free(y2);
    @memset(s.y, sentinel);
    @memset(y2, sentinel);
    try gemv(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, s.y, s.scratch);
    try gemvAvx2(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, y2, s.scratch);
    for (s.y, y2) |a, b| try std.testing.expectEqual(@as(u64, @bitCast(a)), @as(u64, @bitCast(b)));
}

test "gemvAvx2 matches gemv on synthetic 2560x640 and 640x2560" {
    // 2560 * 640 * 3 f64. One buffer, not a second scratch shape.
    try std.testing.expectEqual(@as(usize, 39_321_600), 3 * 2560 * 640 * @sizeOf(f64));
    const gpa = std.testing.allocator;
    try expectGemvAvx2(gpa, 160, 40);
    try expectGemvAvx2(gpa, 40, 160);
}

test "gemvAvx2 scratch one f64 short returns Length and leaves y untouched" {
    const gpa = std.testing.allocator;
    const s = try Square.init(gpa);
    defer s.deinit();
    for (s.y) |*v| v.* = sentinel;
    const short = s.scratch[0 .. s.scratch.len - 1];
    try std.testing.expectError(error.Length, gemvAvx2(s.w, .mul1, s.trellis, s.tiles_k, s.tiles_n, s.suh, s.svh, s.x, s.y, short));
    for (s.y) |v| try std.testing.expectEqual(@as(u64, @bitCast(sentinel)), @as(u64, @bitCast(v)));
}
