//! Pure M3 layout, decode and activation math shared by the host and its scalar tests.
const std = @import("std");

pub const hidden: u32 = 2304;
pub const intermediate: u32 = 1152;
pub const expert_count: u32 = 128;
pub const top_k: u32 = 8;
pub const grouped_tile_m: u32 = 8;
pub const max_grouped_rows: u32 = 64;
pub const max_grouped_pairs: u32 = max_grouped_rows * top_k;

pub const Format = enum {
    nvfp4,
    mxfp8,

    pub fn block(f: Format) u32 {
        return if (f == .nvfp4) 16 else 32;
    }
    pub fn codeBytes(f: Format, k: u32) u32 {
        return if (f == .nvfp4) k / 2 else k;
    }
};

pub const Call = struct {
    rows: u32,
    k: u32,
    n: u32,
    experts: u32 = expert_count,
    topk: u32 = top_k,
    format: Format,
};

/// Threadgroups of 32 threads, one an output: `blocks` along the outputs by `groups` route pairs (or tokens).
pub const Geometry = struct {
    groups: u32,
    blocks: u32,
    threads: u32 = 32,

    /// The output (block, lane) computes, or null past the row: each output is one thread's serial dot.
    pub fn output(g: Geometry, n: u32, block: u32, lane: u32) ?u32 {
        const at = block * g.threads + lane;
        return if (at < n) at else null;
    }
};

/// Threadgroups of `width` along the outputs, rounded up without overflowing at the largest u32 widths.
fn blocksWidth(n: u32, width: u32) u32 {
    return n / width + @intFromBool(n % width != 0);
}

/// A product's grid: every output of every route pair at once.
pub fn geometry(c: Call) Geometry {
    return .{ .groups = c.rows * c.topk, .blocks = blocksWidth(c.n, 32) };
}

/// The combine's grid: every output of every token at once.
pub fn combineGeometry(c: Call) Geometry {
    return .{ .groups = c.rows, .blocks = blocksWidth(c.n, 128), .threads = 128 };
}

/// Exact tight upper bound on active expert tiles for `rows * topk` pairs across `experts` with `grouped_tile_m`.
pub fn maxTiles(c: Call) u32 {
    const pairs = c.rows * c.topk;
    const active = @min(c.experts, pairs);
    return active + (pairs - active) / grouped_tile_m;
}

/// GPU scratch bytes needed by tf_ink2_expert_group: tile count + packed work items + pair map.
pub fn groupBytes(rows: usize, topk: usize) usize {
    return (1 + 2 * rows * topk) * @sizeOf(u32);
}

/// Grouped product grid: output blocks by worst-case expert tile count; inactive tiles exit on tile_id >= count.
pub fn groupedGeometry(c: Call) Geometry {
    return .{ .groups = maxTiles(c), .blocks = blocksWidth(c.n, 64), .threads = 64 };
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k == 0 or c.n == 0 or c.experts == 0 or c.topk == 0 or c.topk > c.experts)
        return error.BadExpertShape;
    if (c.k % c.format.block() != 0 or (c.format == .nvfp4 and c.k % 2 != 0))
        return error.BadExpertBlock;
}

pub fn checkGrouped(c: Call) !void {
    try check(c);
    const pairs = std.math.mul(u32, c.rows, c.topk) catch return error.BadExpertShape;
    if (pairs > max_grouped_pairs or c.experts > std.math.maxInt(u16))
        return error.BadExpertShape;
}

pub fn formatForLayer(layer: u32) !Format {
    if (layer < 20) return .nvfp4;
    if (layer < 24) return .mxfp8;
    return error.BadExpertLayer;
}

pub fn roundBf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    const rounded: u16 = if (std.math.isNan(v))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    return @bitCast(@as(u32, rounded) << 16);
}

pub fn e2m1(code: u8) f32 {
    const values = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = values[code & 7];
    return if (code & 8 != 0) -v else v;
}

pub fn e4m3(code: u8) f32 {
    const exponent = (code >> 3) & 15;
    const mantissa = code & 7;
    if (exponent == 15 and mantissa == 7) return std.math.nan(f32);
    const value = if (exponent == 0)
        @as(f32, @floatFromInt(mantissa)) / 512.0
    else
        std.math.ldexp(@as(f32, 1) + @as(f32, @floatFromInt(mantissa)) / 8.0, @as(i32, exponent) - 7);
    return if (code & 0x80 != 0) -value else value;
}

pub fn phi(amax: f32, denominator: f32) f32 {
    return roundBf16(amax) / roundBf16(denominator);
}

pub fn nvWeight(codes: u8, column: u32, scale: u8, factor: f32) f32 {
    const code = if (column & 1 != 0) codes >> 4 else codes & 15;
    return roundBf16(e2m1(code) * e4m3(scale) * factor);
}

pub fn mxWeight(value: u8, scale: u8) f32 {
    return roundBf16(std.math.ldexp(e4m3(value), @as(i32, scale) - 127));
}

pub fn softcap(x: f32, cap: f32) f32 {
    return @floatCast(@as(f64, cap) * std.math.tanh(@as(f64, x) / @as(f64, cap)));
}

/// Inputs are the two already-BF16-rounded GU results; the result rounds once at the end.
pub fn swiglu(gate: f32, up: f32) f32 {
    const left = softcap(gate, 4) * (1 / (1 + std.math.exp(-gate)));
    return roundBf16(left * softcap(up, 25));
}

pub const MatrixArgs = extern struct { k: u32, n: u32, topk: u32, experts: u32 };
pub const DecodeArgs = extern struct { rows: u32, k: u32 };
pub const GroupArgs = extern struct { rows: u32, topk: u32, experts: u32, tile_m: u32 = grouped_tile_m };
pub const GroupedMatrixArgs = extern struct { rows: u32, k: u32, n: u32, topk: u32, experts: u32 };

/// Deterministic host grouping: counts[e] = min(n, cap), members[e * cap ..] in ascending pair order.
pub fn groupReference(ids: []const i32, experts: u32, cap: u32, counts: []u32, members: []u32) void {
    const e_len: usize = @intCast(experts);
    @memset(counts[0..e_len], 0);
    for (ids, 0..) |raw, p| {
        if (raw < 0 or raw >= @as(i32, @intCast(experts))) continue;
        const e: usize = @intCast(raw);
        const n = counts[e];
        if (n < cap) members[e * @as(usize, cap) + n] = @intCast(p);
        counts[e] = n + 1;
    }
    for (counts[0..e_len]) |*c| c.* = @min(c.*, cap);
}

pub fn e4m3Bits(code: u8) u32 {
    const exp: u32 = (code >> 3) & 15;
    const mant: u32 = code & 7;
    if (exp == 15 and mant == 7) return 0x7fc0_0000;
    const v: f32 = if (exp != 0)
        @bitCast(((exp + 120) << 23) | (mant << 20))
    else
        @as(f32, @floatFromInt(mant)) / 512.0;
    return @bitCast(if ((code & 0x80) != 0) -v else v);
}

/// The narrow grouping remains at and below 256 pairs.
pub fn wideGroup(c: Call) bool {
    return @as(u64, c.rows) * c.topk > 256;
}
