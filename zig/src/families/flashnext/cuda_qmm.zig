//! Group-32 lane matmul: MLX words packed as qmm.pack, then the shared CUDA kernel's 16-row tile.

const std = @import("std");
const cuda = @import("cuda");

pub const group: usize = 32;

const offsets = [8]u32{ 0, 8, 16, 24, 1, 9, 17, 25 };

/// LaneTile<32, 16, 64, 1, 4, 4>::SMEM. One by four warps, four stages.
pub const smem: u32 = 9472;
pub const threads: u32 = 128;

const symbol: [:0]const u8 = "_ZN6tf_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii";

pub const Packed = struct {
    weight: []u8,
    scales: []u8,
    biases: []u8,
    n: usize,
    k: usize,
    npad: usize,

    pub fn deinit(self: *Packed, gpa: std.mem.Allocator) void {
        gpa.free(self.weight);
        gpa.free(self.scales);
        gpa.free(self.biases);
        self.* = undefined;
    }
};

fn nibble(words: []const u8, k8: usize, col: usize, pos: usize) u32 {
    const word_i = pos / 8;
    const shift: u5 = @intCast((pos % 8) * 4);
    const off = (col * k8 + word_i) * 4;
    const word = std.mem.readInt(u32, words[off..][0..4], .little);
    return (word >> shift) & 0xF;
}

/// MLX (n, k/8) words and (n, k/32) scales and biases -> the lane layout. n is padded to 128.
pub fn pack(gpa: std.mem.Allocator, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k8: usize) !Packed {
    if (k8 == 0 or k8 % 4 != 0 or words.len != n * k8 * 4) return error.UnexpectedTensor;
    const k = k8 * 8;
    if (k % group != 0) return error.UnexpectedTensor;
    const kg = k / group;
    if (scales.len != n * kg * 2 or biases.len != scales.len) return error.UnexpectedTensor;
    const npad = (n + 127) / 128 * 128;
    const tiles = npad / 64;
    const weight = try gpa.alloc(u8, tiles * kg * 8 * 32 * 4);
    errdefer gpa.free(weight);
    @memset(weight, 0);
    for (0..n) |col| {
        const tile = col / 64;
        const inner = col % 64;
        const j = inner / 8;
        const r = inner % 8;
        for (0..kg) |g| {
            for (0..4) |c| {
                var bits: u32 = 0;
                for (0..8) |p| {
                    const pos = g * group + 2 * c + offsets[p];
                    bits |= nibble(words, k8, col, pos) << @intCast(p * 4);
                }
                const idx = r * 4 + c;
                const dst = (((tile * kg + g) * 8 + j) * 32 + idx) * 4;
                std.mem.writeInt(u32, weight[dst..][0..4], bits, .little);
            }
        }
    }
    const scales_out = try gpa.alloc(u8, kg * npad * 2);
    errdefer gpa.free(scales_out);
    const biases_out = try gpa.alloc(u8, kg * npad * 2);
    errdefer gpa.free(biases_out);
    @memset(scales_out, 0);
    @memset(biases_out, 0);
    for (0..kg) |g| {
        for (0..n) |col| {
            const src = (col * kg + g) * 2;
            const dst = (g * npad + col) * 2;
            @memcpy(scales_out[dst..][0..2], scales[src..][0..2]);
            @memcpy(biases_out[dst..][0..2], biases[src..][0..2]);
        }
    }
    return .{ .weight = weight, .scales = scales_out, .biases = biases_out, .n = n, .k = k, .npad = npad };
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// One row: each group is `dot(x, q) * scale + xs * bias`, groups added in order, then rounded to bf16.
pub fn dotRow(x: []const u16, xs: []const f32, words: []const u8, scales: []const u8, biases: []const u8, n: usize, k: usize, out: []u16) !void {
    if (k == 0 or k % group != 0 or x.len < k or xs.len < k / group or out.len < n) return error.UnexpectedTensor;
    const kg = k / group;
    const k8 = k / 8;
    if (words.len != n * k8 * 4 or scales.len != n * kg * 2 or biases.len != scales.len) return error.UnexpectedTensor;
    for (0..n) |col| {
        var acc: f32 = 0;
        const word_row = words[col * k8 * 4 ..][0 .. k8 * 4];
        const scale_row = scales[col * kg * 2 ..][0 .. kg * 2];
        const bias_row = biases[col * kg * 2 ..][0 .. kg * 2];
        for (0..kg) |g| {
            var p: f32 = 0;
            for (0..4) |i| {
                const word = std.mem.readInt(u32, word_row[(g * 4 + i) * 4 ..][0..4], .little);
                for (0..8) |j| {
                    const q: f32 = @floatFromInt((word >> @intCast(j * 4)) & 0xF);
                    p += promote(x[g * group + i * 8 + j]) * q;
                }
            }
            const s = promote(std.mem.readInt(u16, scale_row[g * 2 ..][0..2], .little));
            const b = promote(std.mem.readInt(u16, bias_row[g * 2 ..][0..2], .little));
            acc = @mulAdd(f32, xs[g], b, @mulAdd(f32, p, s, acc));
        }
        out[col] = toBf16(acc);
    }
}

/// qmm.cu's 16-row launch for one K slice: x (m, k) bf16 and xs (m, k/32) fp32 into out (m, n) bf16.
pub fn matmul(d: *const cuda.Driver, stream: cuda.Stream, x: u64, xs: u64, weight: u64, scales: u64, biases: u64, out: u64, m: usize, n: usize, k: usize) !void {
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    if (m == 0 or m > 16 or k == 0 or k % group != 0) return error.UnexpectedTensor;
    var module = try cuda.Module.load(d, cuda.kernels.qmm);
    defer module.unload();
    const f = try module.function(symbol);
    try f.allowDynamicShared(smem);
    const bm: usize = 16;
    const bn: usize = 64;
    const rows_t = (m + bm - 1) / bm;
    const cols_t = (n + bn - 1) / bn;
    const sweep = @max(@as(usize, 1), @min(rows_t, (12 << 20) / (bm * k * 2)));
    const npad = (n + 127) / 128 * 128;
    const ldx = k;
    var args: cuda.Args = .{};
    args.add(x);
    args.add(xs);
    args.add(weight);
    args.add(scales);
    args.add(biases);
    args.add(out);
    args.add(@as(u64, 0));
    for ([_]usize{ m, n, k, 1, npad, ldx, sweep }) |v| args.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows_t * cols_t) }, .block = .{ .x = threads }, .shared = smem }, stream, &args);
}

test "one group's first nibble pair lands in the lane word" {
    var words: [16]u8 = @splat(0);
    std.mem.writeInt(u32, words[0..4], 0x21, .little);
    var scale: [2]u8 = undefined;
    std.mem.writeInt(u16, &scale, 0x3f80, .little);
    const bias: [2]u8 = @splat(0);
    var got = try pack(std.testing.allocator, &words, &scale, &bias, 1, 4);
    defer got.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 128), got.npad);
    try std.testing.expectEqual(@as(u32, 0x20001), std.mem.readInt(u32, got.weight[0..4], .little));
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, got.weight[4..8], .little));
    try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, got.scales[0..2], .little));
}

test "scale times the nibble dot plus the group bias rounds to five" {
    var words: [16]u8 = @splat(0);
    std.mem.writeInt(u32, words[0..4], 0x21, .little);
    var scale: [2]u8 = undefined;
    std.mem.writeInt(u16, &scale, 0x3f80, .little);
    var bias: [2]u8 = undefined;
    std.mem.writeInt(u16, &bias, 0x3f80, .little);
    var x: [32]u16 = @splat(0);
    x[0] = 0x3f80;
    x[1] = 0x3f80;
    const xs = [_]f32{2};
    var out: [1]u16 = undefined;
    try dotRow(&x, &xs, &words, &scale, &bias, 1, 32, &out);
    try std.testing.expectEqual(@as(u16, 0x40a0), out[0]);
}
