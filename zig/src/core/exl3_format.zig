//! The EXL3 trellis format (ExLlamaV3's QTIP-derived quantization) as a backend-neutral
//! reference decoder, per `docs/recipes/exl3.md`. CUDA and Metal readers decode tiles into
//! tensor-core fragments; this module is the correctness oracle they are checked against,
//! written deliberately naive (one bit at a time) so it reads exactly like the spec.
//!
//! A layer with K inputs and N outputs (both multiples of 128; the quantizer pads) stores
//! `trellis` int16 [K/16, N/16, 16 * bits], `suh` fp16 [K], `svh` fp16 [N], an optional
//! fp16 `bias`, and a zero-size marker (`mcg` or `mul1`) naming the codebook. Each tile
//! holds 256 values in a circular bitstream of R = 256 * bits bits; value p's 16-bit state
//! ends at E(p) (exclusive, wrapping past the tile's last bit), read most significant bit
//! first from little-endian pairs of the int16 words:
//!
//!     E(p) = (p + 1) * bits                                  integer bits
//!     E(p) = ((p + 1) * (2 * KA + 1) - ((p + 1) % 2)) / 2    bits = KA + 1/2 (mul1 only)
//!
//! The codebook maps the state to an fp16 value with one rounding:
//!     3inst  x = (s * 89226354 + 64248484) mod 2^32; x = (x & 0x8FFF8FFF) ^ 0x3B603B60;
//!            value = fp16(x & 0xFFFF) + fp16(x >> 16)
//!     mcg    x = s * 0xCBAC1FED mod 2^32, then as 3inst
//!     mul1   x = s * 0x83DCD12D mod 2^32; h = 1024 + the sum of x's four bytes;
//!            value = h * fp16(0x1EEE) + fp16(0xC931)
//!
//! The layer: W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh), H the 128x128 Sylvester Hadamard
//! scaled by 1/sqrt(128), so y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias.

const std = @import("std");

pub const Codebook = enum { inst3, mcg, mul1 };

pub const Inst3Mul: u32 = 89226354;
pub const Inst3Add: u32 = 64248484;
pub const McgMul: u32 = 0xCBAC1FED;
pub const Mul1Mul: u32 = 0x83DCD12D;
pub const Mask: u32 = 0x8FFF8FFF;
pub const Flip: u32 = 0x3B603B60;
pub const Mul1ScaleBits: u16 = 0x1EEE; // fp16 bit pattern
pub const Mul1BiasBits: u16 = 0xC931; // fp16 bit pattern
pub const hadamard_dim = 128;

/// A width in half-bit units: whole widths carry halves = 2 * bits; the mul1 half steps
/// carry halves = 2 * KA + 1.
pub const Width = struct {
    halves: u16,

    /// int16 words per tile: 16 * bits.
    pub fn tileWords(self: Width) usize {
        return @as(usize, self.halves) * 8;
    }
    pub fn isHalf(self: Width) bool {
        return self.halves % 2 == 1;
    }
};

fn fp16(bits_pattern: u16) f64 {
    const h: f16 = @bitCast(bits_pattern);
    return @floatCast(h);
}

/// The fp16 value of one 16-bit state under ``codebook``; exactly one rounding per state.
pub fn codebookValue(codebook: Codebook, state: u16) f16 {
    const s: u64 = state;
    const x: u32 = switch (codebook) {
        .mul1 => @truncate(s * Mul1Mul),
        .mcg => @truncate(s * McgMul),
        .inst3 => @truncate(s * Inst3Mul +% Inst3Add),
    };
    if (codebook == .mul1) {
        const h: f64 = @floatFromInt(1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255));
        // float64 multiply-add, then one rounding to fp16 (the exactness the format relies on)
        return @floatCast(h * fp16(Mul1ScaleBits) + fp16(Mul1BiasBits));
    }
    const masked = (x & Mask) ^ Flip;
    const lo: f64 = fp16(@intCast(masked & 0xFFFF));
    const hi: f64 = fp16(@intCast(masked >> 16));
    return @floatCast(lo + hi); // the float64 sum is exact: one fp16 rounding
}

/// E(p) for p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive).
pub fn streamEnds(w: Width, ends: *[256]u32) void {
    if (!w.isHalf()) {
        for (0..256) |p1| ends[p1] = @intCast((p1 + 1) * w.halves / 2);
    } else {
        const ka: u32 = (w.halves - 1) / 2;
        for (0..256) |p1| {
            ends[p1] = @intCast(((p1 + 1) * (2 * ka + 1) - ((p1 + 1) % 2)) / 2);
        }
    }
}

/// One bit of a tile's circular stream: int16 words in pairs as little-endian 32-bit words,
/// most significant bit of each word first, wrapping past the tile's last bit.
fn streamBit(words: []const i16, index: u32) u32 {
    const w32_index = index / 32;
    const within: u5 = @intCast(31 - (index % 32));
    const lo: u16 = @bitCast(words[w32_index * 2]);
    const hi: u16 = @bitCast(words[w32_index * 2 + 1]);
    const word: u32 = @as(u32, lo) | (@as(u32, hi) << 16);
    return (word >> within) & 1;
}

/// The 16-bit state of each of a tile's 256 values, in stream order, from the tile's words.
pub fn tileStates(w: Width, tile: []const i16, states: *[256]u32) void {
    std.debug.assert(tile.len == w.tileWords());
    var ends: [256]u32 = undefined;
    streamEnds(w, &ends);
    const ring: u32 = @intCast(256 * w.halves / 2);
    for (0..256) |p| {
        const start = (ends[p] + ring - 16) % ring;
        var s: u32 = 0;
        for (0..16) |b| s = (s << 1) | streamBit(tile, (start + @as(u32, @intCast(b))) % ring);
        states[p] = s;
    }
}

pub const Placement = struct { row: u8, col: u8 };

/// (row, column) in the 16x16 tile of each stream value p = 0..255: lane l of a warp holds
/// values 8l..8l+7, exactly its B fragments of the tile's two mma.m16n8k16.
pub fn tilePlacement(placement: *[256]Placement) void {
    for (0..256) |p| {
        const lane = p / 8;
        const j = p % 8;
        placement[p] = .{
            .row = @intCast(2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)),
            .col = @intCast(lane / 4 + 8 * (j >> 2)),
        };
    }
}

/// W_q [K, N] fp16 in the rotated domain, from trellis int16 [K/16, N/16, 16 * bits].
pub fn unpack(w: Width, codebook: Codebook, trellis: []const i16, tiles_k: usize, tiles_n: usize, out: []f16) void {
    var placement: [256]Placement = undefined;
    tilePlacement(&placement);
    const k = tiles_k * 16;
    const words = w.tileWords();
    var tk: usize = 0;
    while (tk < tiles_k) : (tk += 1) {
        var tn: usize = 0;
        while (tn < tiles_n) : (tn += 1) {
            const tile = trellis[(tk * tiles_n + tn) * words ..][0..words];
            var states: [256]u32 = undefined;
            tileStates(w, tile, &states);
            for (0..256) |p| {
                out[(tk * 16 + placement[p].row) * k + tn * 16 + placement[p].col] =
                    codebookValue(codebook, @intCast(states[p]));
            }
        }
    }
}

/// su / sv int16 [n/16] -> fp16 [n] of +1 and -1: bit b of word w set means element 16w + b is -1.
pub fn unpackSigns(packed_words: []const i16, out: []f16) void {
    for (packed_words, 0..) |word16, w| {
        const word: u16 = @bitCast(word16);
        for (0..16) |b| {
            const negative = (word >> @intCast(b)) & 1 == 1;
            out[w * 16 + b] = if (negative) -1.0 else 1.0;
        }
    }
}

fn parityOf(v: usize) usize {
    return @popCount(v) & 1;
}

/// H / sqrt(128) applied to every block of hadamard_dim along ``axis`` of a [rows, cols]
/// f64 matrix, in place. Each Hadamard entry is (+-1) / sqrt(128) rounded once; block
/// products sum in f64 like the numpy reference (BLAS may order the 128 exact sums
/// differently, hence the few-ULP tolerance the dequantize test compares with).
pub fn rotate(x: []f64, rows: usize, cols: usize, axis: usize) void {
    const inv = 1.0 / std.math.sqrt(@as(f64, hadamard_dim));
    if (axis == 0) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            var b: usize = 0;
            while (b < rows / hadamard_dim) : (b += 1) rotateBlock(x, cols, b * hadamard_dim, c, axis, inv);
        }
    } else {
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            var b: usize = 0;
            while (b < cols / hadamard_dim) : (b += 1) rotateBlock(x, cols, r, b * hadamard_dim, axis, inv);
        }
    }
}

fn rotateBlock(x: []f64, cols: usize, base_r: usize, base_c: usize, axis: usize, inv: f64) void {
    var scratch: [hadamard_dim]f64 = undefined;
    for (0..hadamard_dim) |i| {
        var sum: f64 = 0;
        for (0..hadamard_dim) |j| {
            const h = (if (parityOf(i & j) == 1) @as(f64, -1.0) else @as(f64, 1.0)) * inv;
            const v = if (axis == 0) x[(base_r + j) * cols + base_c] else x[base_r * cols + base_c + j];
            sum += h * v;
        }
        scratch[i] = sum;
    }
    for (0..hadamard_dim) |i| {
        if (axis == 0) x[(base_r + i) * cols + base_c] = scratch[i] else x[base_r * cols + base_c + i] = scratch[i];
    }
}

/// The layer's weight W [K, N] f64: diag(suh) @ H_K @ W_q @ H_N @ diag(svh).
/// ``scratch`` must hold 2 * K * N f64.
pub fn dequantize(w: Width, codebook: Codebook, trellis: []const i16, tiles_k: usize, tiles_n: usize, suh: []const f16, svh: []const f16, out: []f64, scratch: []f64) void {
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    const wq_f64 = scratch[0 .. k * n];
    const wq_f16: []f16 = @as([*]f16, @ptrCast(@alignCast(scratch[k * n ..].ptr)))[0 .. k * n];
    unpack(w, codebook, trellis, tiles_k, tiles_n, wq_f16);
    for (wq_f64, wq_f16) |*d, s| d.* = @floatCast(s);
    rotate(wq_f64, k, n, 0);
    for (wq_f64, 0..) |*v, i| v.* *= @floatCast(suh[i / n]);
    rotate(wq_f64, k, n, 1);
    for (wq_f64, 0..) |*v, i| v.* *= @floatCast(svh[i % n]);
    @memcpy(out, wq_f64);
}

// -- the oracle fixture ------------------------------------------------------------
// Generated by the independent numpy reference decoder (see the recipe): 27 codebook x
// width records of random tiles with their exact states, codebook spot values, and one
// complete 128x128 Hadamard block of a real mcg @ 8.00 bpw group (Qwen3.5-2B geometry,
// self_attn.q_proj) with its scales and expected W_q / dequantized block. The file lives
// at zig/tests/fixtures/exl3_oracle.bin and is read relative to the repository root, the
// working directory every `zig build test` step and the recipe's commands run from.

var oracle: []const u8 = &.{};

fn loadOracle(gpa: std.mem.Allocator) !void {
    const io = std.testing.io;
    oracle = try std.Io.Dir.cwd().readFileAlloc(io, "zig/tests/fixtures/exl3_oracle.bin", gpa, .limited(1 << 22));
}

fn u16At(at: usize) u16 {
    return std.mem.readInt(u16, oracle[at..][0..2], .little);
}
fn u32At(at: usize) u32 {
    return std.mem.readInt(u32, oracle[at..][0..4], .little);
}
fn u64At(at: usize) u64 {
    return std.mem.readInt(u64, oracle[at..][0..8], .little);
}

const Record = struct { halves: u16, codebook: Codebook, kt: u32, nt: u32, next: usize };

fn recordAt(start: usize) Record {
    const halves = u16At(start);
    const codebook: Codebook = @fromBackingInt(@intCast(oracle[start + 2]));
    const kt = u32At(start + 4);
    const nt = u32At(start + 8);
    const w: Width = .{ .halves = halves };
    // the fixture stores every tile's trellis words first (k-major), then every tile's states
    return .{ .halves = halves, .codebook = codebook, .kt = kt, .nt = nt, .next = start + 12 + kt * nt * (w.tileWords() * 2 + 256 * 4) };
}

test "the reference decoder reproduces the numpy oracle on every codebook and width" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try loadOracle(arena.allocator());
    try std.testing.expectEqualSlices(u8, "EXL3ORAC", oracle[0..8]);
    const count = u64At(8);
    var at: usize = 16;
    var tiles_checked: usize = 0;
    for (0..count) |_| {
        const rec = recordAt(at);
        const w: Width = .{ .halves = rec.halves };
        const words = w.tileWords();
        const trellis_base = at + 12;
        const states_base = trellis_base + rec.kt * rec.nt * words * 2;
        var tile: [128]i16 = undefined; // 8 bits is the widest: 128 words
        var tk: usize = 0;
        while (tk < rec.kt) : (tk += 1) {
            var tn: usize = 0;
            while (tn < rec.nt) : (tn += 1) {
                const ti = tk * rec.nt + tn;
                const tile_at = trellis_base + ti * words * 2;
                @memcpy(std.mem.sliceAsBytes(tile[0..words]), oracle[tile_at .. tile_at + words * 2]);
                var states: [256]u32 = undefined;
                tileStates(w, tile[0..words], &states);
                const want_at = states_base + ti * 256 * 4;
                for (0..256) |p| {
                    const want = std.mem.readInt(u32, oracle[want_at + p * 4 ..][0..4], .little);
                    try std.testing.expectEqual(want, states[p]);
                }
                tiles_checked += 1;
            }
        }
        at = rec.next;
    }
    try std.testing.expectEqual(@as(usize, 27 * 6), tiles_checked);

    const spots = u32At(at);
    at += 4;
    for (0..spots) |_| {
        const codebook: Codebook = @fromBackingInt(@intCast(u16At(at)));
        const state = u16At(at + 2);
        const want_bits = u16At(at + 4);
        at += 6;
        const got: u16 = @bitCast(codebookValue(codebook, state));
        try std.testing.expectEqual(want_bits, got);
    }
}

test "a real mcg 8.00 bpw block unpacks and dequantizes to the oracle's values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try loadOracle(arena.allocator());
    var at: usize = 16;
    const count = u64At(8);
    for (0..count) |_| at = recordAt(at).next;
    at += 4 + u32At(at) * 6; // past the codebook spots

    const halves = u16At(at);
    const kt = u32At(at + 2);
    const nt = u32At(at + 6);
    const trellis_len = u32At(at + 10);
    at += 14;
    const w: Width = .{ .halves = halves };
    try std.testing.expectEqual(@as(u32, 8), kt);
    try std.testing.expectEqual(@as(u32, 8), nt);

    const gpa = std.testing.allocator;
    const trellis = try gpa.alloc(i16, trellis_len / 2);
    defer gpa.free(trellis);
    @memcpy(std.mem.sliceAsBytes(trellis), oracle[at .. at + trellis_len]);
    at += trellis_len;

    const suh_len = u32At(at);
    at += 4;
    const suh_bits = try gpa.alloc(u16, suh_len / 2);
    defer gpa.free(suh_bits);
    @memcpy(std.mem.sliceAsBytes(suh_bits), oracle[at .. at + suh_len]);
    at += suh_len;
    const svh_len = u32At(at);
    at += 4;
    const svh_bits = try gpa.alloc(u16, svh_len / 2);
    defer gpa.free(svh_bits);
    @memcpy(std.mem.sliceAsBytes(svh_bits), oracle[at .. at + svh_len]);
    at += svh_len;
    const wq_len = u32At(at);
    at += 4;
    const wq_want = try gpa.alloc(u16, wq_len / 2);
    defer gpa.free(wq_want);
    @memcpy(std.mem.sliceAsBytes(wq_want), oracle[at .. at + wq_len]);
    at += wq_len;
    const dq_len = u32At(at);
    at += 4;
    const dq_want = try gpa.alloc(f64, dq_len / 8);
    defer gpa.free(dq_want);
    @memcpy(std.mem.sliceAsBytes(dq_want), oracle[at .. at + dq_len]);

    const k = kt * 16;
    const n = nt * 16;
    const wq = try gpa.alloc(f16, k * n);
    defer gpa.free(wq);
    unpack(w, .mcg, trellis, kt, nt, wq);
    for (wq, wq_want) |got, want| try std.testing.expectEqual(want, @as(u16, @bitCast(got)));

    const suh = std.mem.bytesAsSlice(f16, std.mem.sliceAsBytes(suh_bits));
    const svh = std.mem.bytesAsSlice(f16, std.mem.sliceAsBytes(svh_bits));
    const scratch = try gpa.alloc(f64, 2 * k * n);
    defer gpa.free(scratch);
    const dq = try gpa.alloc(f64, k * n);
    defer gpa.free(dq);
    dequantize(w, .mcg, trellis, kt, nt, suh, svh, dq, scratch);
    // the Hadamard blocks sum in f64 like the reference; BLAS may order the 128 exact
    // products differently, so compare with a few-ULP tolerance instead of bit patterns
    var max_diff: f64 = 0;
    var max_mag: f64 = 0;
    for (dq, dq_want) |got, want| {
        max_diff = @max(max_diff, @abs(got - want));
        max_mag = @max(max_mag, @abs(want));
    }
    try std.testing.expect(max_diff <= 1e-9 * (1.0 + max_mag));
}
