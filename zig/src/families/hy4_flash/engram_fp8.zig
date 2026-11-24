//! Hy4-flash 100-byte FP8 Engram row format (96 e4m3fn codes from fp64 + fp32 row scale, 4 KiB HY4ENGR1 header).
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const engram = @import("engram.zig");

const Text = config.Text;
const fp8 = core.quant.fp8;

pub const magic: u64 = 0x3152474E45345948; // "HY4ENGR1" little-endian
pub const header_bytes: u64 = 4096;
pub const dim: usize = 96;
pub const row_bytes: usize = dim + 4;
pub const fp8_max: f64 = 448.0;

pub fn fileName(buf: []u8, layer: usize) ![]u8 {
    return std.fmt.bufPrint(buf, "engram_layer{d}.hy4e", .{layer});
}

pub const Header = struct {
    layer: u32,
    layer_id: u32,
    seed: u64,
    rows: u64,
    sizes: [engram.columns]u64,

    pub fn of(text: Text, layer_id: u32) !Header {
        if (layer_id == 0) return error.BadEngramGeometry;
        var key_buf: [10]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{d}", .{layer_id});
        const groups = text.engram_layer_head_vocab_sizes.map.get(key) orelse return error.BadEngramGeometry;
        var sizes: [engram.columns]u64 = undefined;
        var total: u64 = 0;
        var at: usize = 0;
        for (groups) |g| for (g) |prime| {
            if (at >= sizes.len or prime < 2) return error.BadEngramGeometry;
            sizes[at] = prime;
            total = try std.math.add(u64, total, prime);
            at += 1;
        };
        if (at != sizes.len) return error.BadEngramGeometry;
        return .{
            .layer = layer_id - 1,
            .layer_id = layer_id,
            .seed = text.engram_seed,
            .rows = total,
            .sizes = sizes,
        };
    }

    pub fn encode(h: Header) [header_bytes]u8 {
        var b: [header_bytes]u8 = @splat(0);
        const words = [8]u64{ magic, h.layer, h.layer_id, h.seed, h.rows, dim, row_bytes, engram.columns };
        for (words, 0..) |w, i| std.mem.writeInt(u64, b[8 * i ..][0..8], w, .little);
        for (h.sizes, 0..) |s, i| std.mem.writeInt(u64, b[64 + 8 * i ..][0..8], s, .little);
        return b;
    }

    pub fn parse(b: *const [header_bytes]u8, file_size: u64) !Header {
        var w: [8]u64 = undefined;
        for (&w, 0..) |*x, i| x.* = std.mem.readInt(u64, b[8 * i ..][0..8], .little);
        if (w[0] != magic) return error.NotHy4EngramPack;
        if (w[5] != dim or w[6] != row_bytes or w[7] != engram.columns) return error.BadHy4EngramPack;
        if (w[1] + 1 != w[2] or w[2] > std.math.maxInt(u32) or w[4] == 0) return error.BadHy4EngramPack;
        var sizes: [engram.columns]u64 = undefined;
        var sum: u64 = 0;
        for (&sizes, 0..) |*s, i| {
            s.* = std.mem.readInt(u64, b[64 + 8 * i ..][0..8], .little);
            if (s.* == 0) return error.BadHy4EngramPack;
            sum = std.math.add(u64, sum, s.*) catch return error.BadHy4EngramPack;
        }
        if (sum != w[4]) return error.BadHy4EngramPack;
        const payload = std.math.mul(u64, w[4], row_bytes) catch return error.BadHy4EngramPack;
        const want = std.math.add(u64, header_bytes, payload) catch return error.BadHy4EngramPack;
        if (file_size != want) return error.BadHy4EngramPack;
        return .{ .layer = @intCast(w[1]), .layer_id = @intCast(w[2]), .seed = w[3], .rows = w[4], .sizes = sizes };
    }

    pub fn checkConfig(h: Header, text: Text, layer_id: u32) !void {
        const want = try Header.of(text, layer_id);
        if (h.layer_id != want.layer_id or h.seed != want.seed or h.rows != want.rows) return error.UnsupportedEngram;
        if (!std.mem.eql(u64, &h.sizes, &want.sizes)) return error.UnsupportedEngram;
    }
};

/// Exact float32 -> float8_e4m3fn round-to-nearest-even matching PyTorch `.float().to(torch.float8_e4m3fn)`.
pub fn e4m3Rne(x: f32) u8 {
    const bits: u32 = @bitCast(x);
    const sign: u8 = @intCast((bits >> 31) << 7);
    const ab: u32 = bits & 0x7fff_ffff;
    if (ab > 0x43e8_0000) return sign | 0x7f;
    if (ab < 0x3a80_0000) return sign;
    if (ab < 0x3c80_0000) {
        const exp = ab >> 23;
        const sig: u32 = 0x80_0000 | (ab & 0x7f_ffff);
        const shift: u5 = @intCast(141 - exp);
        var q: u32 = sig >> shift;
        const rem: u32 = sig & ((@as(u32, 1) << shift) - 1);
        const half: u32 = @as(u32, 1) << (shift - 1);
        if (rem > half or (rem == half and (q & 1) == 1)) q += 1;
        return sign | @as(u8, @intCast(q));
    }
    const exp = ab >> 23;
    const mant = ab & 0x7f_ffff;
    var q: u32 = ((exp - 120) << 3) | (mant >> 20);
    const rem = mant & 0x0f_ffff;
    if (rem > 0x08_0000 or (rem == 0x08_0000 and (q & 1) == 1)) q += 1;
    return sign | @as(u8, @intCast(q));
}

/// Pack one 96-wide bf16 row into 96 e4m3fn codes + fp32(amax / 448) using fp64 scaling.
pub fn packRow(bf16_bytes: *const [dim * 2]u8, out: *[row_bytes]u8) void {
    var x64: [dim]f64 = undefined;
    var amax: f64 = 1e-12;
    for (0..dim) |j| {
        const u = std.mem.readInt(u16, bf16_bytes[2 * j ..][0..2], .little);
        const v: f64 = @as(f32, @bitCast(@as(u32, u) << 16));
        x64[j] = v;
        const a = @abs(v);
        if (a > amax) amax = a;
    }
    const s64: f64 = amax / fp8_max;
    for (0..dim) |j| {
        const clamped = std.math.clamp(x64[j] / s64, -fp8_max, fp8_max);
        out[j] = e4m3Rne(@floatCast(clamped));
    }
    const s32: f32 = @floatCast(s64);
    std.mem.writeInt(u32, out[dim..][0..4], @bitCast(s32), .little);
}

/// Decode one 100-byte packed row as `code[j] * scale` in fp32.
pub fn unpackRowF32(rec: *const [row_bytes]u8, out: *[dim]f32) void {
    const scale: f32 = @bitCast(std.mem.readInt(u32, rec[dim..][0..4], .little));
    for (0..dim) |j| out[j] = @as(f32, @floatCast(fp8.e4m3(rec[j]))) * scale;
}
