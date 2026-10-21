//! Independent scalar EXL3 oracle over resident bytes; Module(Real) removes intermediates, never a storage boundary.

// EXL3 attribution: docs/recipes/exl3.md; pinned checkpoint provenance in THIRD_PARTY_NOTICES.md.

const std = @import("std");
const math = @import("math.zig");
const exl3 = @import("tensorfold").exl3_projection;
const weights = @import("tensorfold").deepseek_v41_weights;
const format = exl3.format;
const block_size = format.hadamard_dim;
const inverse: f32 = 0.08838834764831845;
const states_count = 65536;

/// Reuse bounded vector/codebook storage for heterogeneous matrices; never allocate decoded W[K,N].
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    rotated: []f64,
    sums: []f64,
    codebooks: []f16,

    pub fn init(gpa: std.mem.Allocator, max_k: usize, max_n: usize) !Scratch {
        if (max_k == 0 or max_n == 0) return error.InvalidReferenceProjectionCapacity;
        const rotated = try gpa.alloc(f64, max_k);
        errdefer gpa.free(rotated);
        const sums = try gpa.alloc(f64, max_n);
        errdefer gpa.free(sums);
        const codebooks = try gpa.alloc(f16, 3 * states_count);
        errdefer gpa.free(codebooks);
        for ([_]format.Codebook{ .inst3, .mcg, .mul1 }, 0..) |book, bi| {
            for (0..states_count) |state| {
                codebooks[bi * states_count + state] = format.codebookValue(book, @intCast(state));
            }
        }
        return .{ .gpa = gpa, .rotated = rotated, .sums = sums, .codebooks = codebooks };
    }

    pub fn deinit(s: *Scratch) void {
        s.gpa.free(s.rotated);
        s.gpa.free(s.sums);
        s.gpa.free(s.codebooks);
        s.* = undefined;
    }
};

fn borrowed(view: exl3.View, bytes: usize) ![]const u8 {
    const len = view.buffer.length();
    if (view.offset % 2 != 0 or view.offset > len or bytes > len - view.offset)
        return error.InvalidReferenceProjectionView;
    const address = view.buffer.contents();
    if (@intFromPtr(address) == 0) return error.NonCpuAccessibleExl3Buffer;
    return address[view.offset..][0..bytes];
}

fn scaleBytes(count: usize, kind: exl3.ScaleFormat) !usize {
    return std.math.mul(usize, if (kind == .packed_signs) count / 16 else count, 2);
}

fn half(bytes: []const u8, i: usize) f32 {
    const value: f16 = @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
    return @floatCast(value);
}

fn scale(bytes: []const u8, i: usize, kind: exl3.ScaleFormat) f32 {
    if (kind == .f16) return half(bytes, i);
    const word = std.mem.readInt(u16, bytes[2 * (i / 16) ..][0..2], .little);
    return if (((word >> @as(u4, @intCast(i % 16))) & 1) != 0) -1.0 else 1.0;
}

/// CPU stream reader consumes MSB-first bits from little-endian int16 pairs, not the GPU indexed decoder.
const Stream = struct {
    bytes: []const u8,
    cursor: usize = 0,
    pending: u32 = 0,
    available: u6 = 0,

    fn take(s: *Stream, count: u5) u16 {
        if (s.available == 0) {
            s.pending = std.mem.readInt(u32, s.bytes[s.cursor..][0..4], .little);
            s.cursor += 4;
            s.available = 32;
        }
        const first: u5 = @intCast(@min(s.available, count));
        const head: u32 = s.pending >> @as(u5, @intCast(32 - @as(u6, first)));
        s.pending <<= first;
        s.available -= first;
        if (first == count) return @intCast(head);
        const rest = count - first;
        s.pending = std.mem.readInt(u32, s.bytes[s.cursor..][0..4], .little);
        s.cursor += 4;
        const tail = s.pending >> @as(u5, @intCast(32 - @as(u6, rest)));
        s.pending <<= rest;
        s.available = 32 - @as(u6, rest);
        return @intCast((head << rest) | tail);
    }
};

/// Circular initial state uses final 16 stream bits; decode only a sixteen-column accumulator and live stream.
fn accumulateTile(comptime halves: u16, tile: []const u8, table: []const f16, x: []const f64, acc: *[16]f64) void {
    @setFloatMode(.strict);
    var stream: Stream = .{ .bytes = tile };
    var state: u16 = @truncate(std.mem.readInt(u32, tile[tile.len - 4 ..][0..4], .little));
    var p: usize = 0;
    while (p < 256) : (p += 1) {
        // Odd half-widths alternate floor(bits),ceil(bits), exactly as format.streamEnds specifies.
        const advance: u4 = @intCast(halves / 2 + (if (halves % 2 == 1) p % 2 else 0));
        state = (state << advance) | stream.take(advance);
        const lane = p / 8;
        const j = p % 8;
        const row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
        const col = lane / 4 + 8 * (j >> 2);
        const weight: f64 = @floatCast(table[state]);
        acc[col] += x[row] * weight;
    }
}

pub fn Module(comptime Real: type) type {
    return struct {
        const Self = @This();
        pub const wr = math.Module(Real).wr;

        fn bf16(value: Real) u16 {
            if (Real == f32) {
                const bits: u32 = @bitCast(value);
                // Preserve a quiet NaN instead of letting rounding turn a tiny NaN payload into infinity.
                if ((bits & 0x7fffffff) > 0x7f800000) return @as(u16, @truncate(bits >> 16)) | 0x0040;
                return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
            }
            return math.Module(Real).b(value);
        }

        /// Seven ordinary Sylvester butterflies, with explicit stage boundaries and normalization afterward.
        fn hadamard(values: *[block_size]Real) void {
            @setFloatMode(.strict);
            var stride: usize = 1;
            while (stride < block_size) : (stride *= 2) {
                var base: usize = 0;
                while (base < block_size) : (base += 2 * stride) {
                    for (0..stride) |j| {
                        const a = values[base + j];
                        const b = values[base + stride + j];
                        values[base + j] = a + b;
                        values[base + stride + j] = a - b;
                    }
                }
            }
        }

        fn prepare(x: []const u16, su: []const u8, kind: exl3.ScaleFormat, rotated: []f64) void {
            @setFloatMode(.strict);
            var base: usize = 0;
            while (base < x.len) : (base += block_size) {
                var values: [block_size]Real = undefined;
                for (&values, 0..) |*value, i| {
                    const input: Real = @floatCast(@as(f32, @bitCast(@as(u32, x[base + i]) << 16)));
                    value.* = input * wr(scale(su, base + i, kind));
                }
                hadamard(&values);
                for (values, 0..) |value, i| {
                    const rounded: f16 = @floatCast(value * wr(inverse));
                    rotated[base + i] = @floatCast(rounded);
                }
            }
        }

        fn product(comptime halves: u16, matrix: exl3.Matrix, trellis: []const u8, table: []const f16, rotated: []const f64, sums: []f64) !void {
            const ks = try matrix.kStride();
            const ns = matrix.nStride();
            const tile_bytes = @as(usize, halves) * 16;
            @memset(sums, 0);
            for (0..matrix.k / 16) |tk| {
                const input = rotated[tk * 16 ..][0..16];
                for (0..matrix.n / 16) |tn| {
                    const at = tk * ks + tn * ns;
                    accumulateTile(halves, trellis[at..][0..tile_bytes], table, input, sums[tn * 16 ..][0..16]);
                }
            }
        }

        fn project(matrix: exl3.Matrix, x: []const u16, output_len: usize, scratch: *Scratch) !void {
            try matrix.validate();
            if (x.len != matrix.k or output_len != matrix.n) return error.InvalidReferenceProjectionLength;
            if (matrix.k > scratch.rotated.len or matrix.n > scratch.sums.len)
                return error.ReferenceProjectionScratchTooSmall;
            const trellis = try borrowed(matrix.trellis, try matrix.storageBytes());
            const su = try borrowed(matrix.suh, try scaleBytes(matrix.k, matrix.suh_format));
            // Check every borrowed view before doing arithmetic, including the output-only scale and optional bias.
            _ = try borrowed(matrix.svh, try scaleBytes(matrix.n, matrix.svh_format));
            if (matrix.bias) |bias| _ = try borrowed(bias, try std.math.mul(usize, matrix.n, 2));
            const rotated = scratch.rotated[0..matrix.k];
            const sums = scratch.sums[0..matrix.n];
            prepare(x, su, matrix.suh_format, rotated);
            const book_index: usize = @intCast(@backingInt(matrix.codebook));
            const table = scratch.codebooks[book_index * states_count ..][0..states_count];
            switch (matrix.width.halves) {
                inline 2...16 => |halves| try product(halves, matrix, trellis, table, rotated, sums),
                else => unreachable,
            }
        }

        fn finish(comptime T: type, matrix: exl3.Matrix, y: []T, sums: []const f64) !void {
            @setFloatMode(.strict);
            const sv = try borrowed(matrix.svh, try scaleBytes(matrix.n, matrix.svh_format));
            const bias = if (matrix.bias) |view| try borrowed(view, try std.math.mul(usize, matrix.n, 2)) else null;
            var base: usize = 0;
            while (base < matrix.n) : (base += block_size) {
                var values: [block_size]Real = undefined;
                // In FP32 one FP64->FP32 boundary replaces GPU FP32 dot-product accumulation; FP64 keeps the wide sum.
                for (&values, 0..) |*value, i| value.* = @floatCast(sums[base + i]);
                hadamard(&values);
                for (values, 0..) |value, i| {
                    var out: Real = (value * wr(inverse)) * wr(scale(sv, base + i, matrix.svh_format));
                    if (bias) |bytes| out += wr(half(bytes, base + i));
                    y[base + i] = if (T == u16) bf16(out) else @floatCast(out);
                }
            }
        }

        /// Kept-dense: quantized rotation and output stages; a wide dot replaces the tile decode.
        fn applyDense(comptime T: type, d: weights.Dense, x: []const u16, y: []T, scratch: *Scratch) !void {
            @setFloatMode(.strict);
            if (x.len != d.k or y.len != d.n) return error.InvalidReferenceProjectionLength;
            if (d.k > scratch.rotated.len or d.n > scratch.sums.len)
                return error.ReferenceProjectionScratchTooSmall;
            const su = try borrowed(.{ .buffer = d.suh.buffer, .offset = d.suh.offset }, d.k * 2);
            const sv = try borrowed(.{ .buffer = d.svh.buffer, .offset = d.svh.offset }, d.n * 2);
            const w = try borrowed(.{ .buffer = d.weight.buffer, .offset = d.weight.offset }, d.n * d.k * 2);
            const rotated = scratch.rotated[0..d.k];
            const sums = scratch.sums[0..d.n];
            prepare(x, su, .f16, rotated);
            for (sums) |*slot| slot.* = 0;
            for (0..d.n) |j| {
                var acc: f64 = 0;
                const wrow = 2 * j * d.k;
                for (0..d.k) |k| {
                    const word = std.mem.readInt(u16, w[wrow + 2 * k ..][0..2], .little);
                    acc += rotated[k] * @as(f64, @as(f32, @bitCast(@as(u32, word) << 16)));
                }
                sums[j] = acc;
            }
            var base: usize = 0;
            while (base < d.n) : (base += block_size) {
                var values: [block_size]Real = undefined;
                for (&values, 0..) |*value, i| value.* = @floatCast(sums[base + i]);
                hadamard(&values);
                for (values, 0..) |value, i| {
                    const out = (value * wr(inverse)) * wr(half(sv, base + i));
                    y[base + i] = if (T == u16) bf16(out) else @floatCast(out);
                }
            }
        }

        /// One BF16 input row and BF16 output row; caller must finish GPU writes before borrowing shared CPU bytes.
        pub fn applyTrellis(matrix: exl3.Matrix, x: []const u16, y: []u16, scratch: *Scratch) !void {
            try project(matrix, x, y.len, scratch);
            try finish(u16, matrix, y, scratch.sums[0..matrix.n]);
        }

        /// The same arithmetic with the module-width output boundary, for compressor projections and logits.
        pub fn applyF32Trellis(matrix: exl3.Matrix, x: []const u16, y: []Real, scratch: *Scratch) !void {
            try project(matrix, x, y.len, scratch);
            try finish(Real, matrix, y, scratch.sums[0..matrix.n]);
        }

        /// Either weight representation, one entry point: trellis bytes or a kept-dense FP16 row.
        pub fn apply(matrix: weights.AnyMatrix, x: []const u16, y: []u16, scratch: *Scratch) !void {
            switch (matrix) {
                .trellis => |m| return applyTrellis(m, x, y, scratch),
                .dense => |d| return applyDense(u16, d, x, y, scratch),
            }
        }

        /// The same union dispatch with the module-width output boundary.
        pub fn applyF32(matrix: weights.AnyMatrix, x: []const u16, y: []Real, scratch: *Scratch) !void {
            switch (matrix) {
                .trellis => |m| return applyF32Trellis(m, x, y, scratch),
                .dense => |d| return applyDense(Real, d, x, y, scratch),
            }
        }
    };
}

const f32_module = Module(f32);
pub const apply = f32_module.apply;
pub const applyF32 = f32_module.applyF32;

test "CPU circular streaming matches scalar format for every supported width and codebook" {
    var scratch = try Scratch.init(std.testing.allocator, 128, 128);
    defer scratch.deinit();
    var words: [128]i16 = undefined;
    for (&words, 0..) |*word, i| word.* = @bitCast(@as(u16, @truncate(i * 40503 + 17329)));
    var x: [16]f64 = undefined;
    for (&x, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i + 1)) * 0.0625;
    var placement: [256]format.Placement = undefined;
    format.tilePlacement(&placement);
    inline for (2..17) |halves| {
        const width: format.Width = .{ .halves = halves };
        const tile = words[0..width.tileWords()];
        var states: [256]u32 = undefined;
        format.tileStates(width, tile, &states);
        for ([_]format.Codebook{ .inst3, .mcg, .mul1 }, 0..) |book, bi| {
            if (width.isHalf() and (book != .mul1 or halves > 7)) continue;
            const table = scratch.codebooks[bi * states_count ..][0..states_count];
            var expected: [16]f64 = @splat(0);
            for (states, placement) |state, where| {
                const weight: f64 = @floatCast(format.codebookValue(book, @intCast(state)));
                expected[where.col] += x[where.row] * weight;
            }
            var actual: [16]f64 = @splat(0);
            accumulateTile(halves, std.mem.sliceAsBytes(tile), table, &x, &actual);
            try std.testing.expectEqualSlices(f64, &expected, &actual);
        }
    }
}

test "FP32 Hadamard then FP16 and BF16 boundaries are deterministic" {
    const f32m = Module(f32);
    var x: [128]u16 = @splat(0);
    x[0] = 0x3f80;
    var signs: [16]u8 = @splat(0);
    signs[0] = 1;
    var rotated: [128]f64 = undefined;
    f32m.prepare(&x, &signs, .packed_signs, &rotated);
    const rounded: f16 = @floatCast(-@as(f32, inverse));
    for (rotated) |value| try std.testing.expectEqual(@as(f64, @floatCast(rounded)), value);
    try std.testing.expectEqual(@as(u16, 0x3f80), f32m.bf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), f32m.bf16(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expectEqual(@as(u16, 0x7fc0), f32m.bf16(@bitCast(@as(u32, 0x7f800001))));
}

test "frozen FP32 rotation and output boundaries match the pre-repair oracle" {
    // Frozen from the repaired sign packing in b25c4a8ce: prepare, Hadamard and narrowing are FP32, so words stay.
    const f32m = Module(f32);
    const inputs = [_]f32{ 1, -1, 0x1p-24, 1 + 0x1p-7, 0x1.8p-24, -0x1p-7, 0x1p-8, 1 - 0x1p-7 };
    var x: [block_size]u16 = undefined;
    for (&x, 0..) |*word, i| word.* = f32m.bf16(inputs[i % inputs.len]);
    var signs: [block_size / 8]u8 = @splat(0);
    for (0..block_size) |i| {
        if (i % 3 == 1) signs[2 * (i / 16) + (i % 16) / 8] |= @as(u8, 1) << @intCast(i % 8);
    }
    var rotated: [block_size]f64 = undefined;
    f32m.prepare(&x, &signs, .packed_signs, &rotated);
    var rotation: [block_size]u16 = undefined;
    for (&rotation, rotated) |*word, value| word.* = @bitCast(@as(f16, @floatCast(value)));
    // The finish output stage: widen the rotation, seven butterflies, the inverse and column scale, BF16.
    var values: [block_size]f32 = undefined;
    for (&values, rotation) |*value, word| value.* = @floatCast(@as(f16, @bitCast(word)));
    f32m.hadamard(&values);
    var narrowed: [block_size]u16 = undefined;
    for (&narrowed, values, 0..) |*word, value, i| {
        const column_scale: f16 = if (i % 2 == 0) 1 else 1 + 0x1p-7;
        word.* = f32m.bf16((value * f32m.wr(inverse)) * @as(f32, @floatCast(column_scale)));
    }
    const frozen_rotation = [_]u16{
        0x3c3d, 0x1cf3, 0xb9b8, 0x3f16, 0x35cd, 0x398d, 0x91a8, 0x3c43,
        0xb5b6, 0x183e, 0x11a8, 0xb5a5, 0xb5a5, 0x11a8, 0x183e, 0xb5b6,
    };
    const frozen_narrowed = [_]u16{
        0x3f80, 0x3f81, 0xb982, 0x3f82, 0x3913, 0xbc00, 0x3b81, 0xbf80,
        0x3f80, 0xbf81, 0x3809, 0x3f82, 0xb8cb, 0x3bfd, 0x3b84, 0x3f80,
    };
    try std.testing.expectEqualSlices(u16, &frozen_rotation, rotation[0..16]);
    try std.testing.expectEqualSlices(u16, &frozen_narrowed, narrowed[0..16]);
}
