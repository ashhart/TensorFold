//! Independent FP64 definition: per-element source decoding, with no production extractor or native-half conversion.
const Kind = @import("gguf_tensor.zig").Kind;
pub const kinds = [_]Kind{ .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k, .q3_k, .q2_k };
pub const Geometry = struct { bytes: usize, values: usize, bits: u8, group: usize, halves: []const usize };
pub fn geometry(k: Kind) Geometry {
    return switch (k) {
        .q8_0 => .{ .bytes = 34, .values = 32, .bits = 8, .group = 32, .halves = &.{0} },
        .q4_0 => .{ .bytes = 18, .values = 32, .bits = 4, .group = 32, .halves = &.{0} },
        .q4_1 => .{ .bytes = 20, .values = 32, .bits = 4, .group = 32, .halves = &.{ 0, 2 } },
        .q5_0 => .{ .bytes = 22, .values = 32, .bits = 5, .group = 32, .halves = &.{0} },
        .q5_1 => .{ .bytes = 24, .values = 32, .bits = 5, .group = 32, .halves = &.{ 0, 2 } },
        .q4_k => .{ .bytes = 144, .values = 256, .bits = 4, .group = 32, .halves = &.{ 0, 2 } },
        .q5_k => .{ .bytes = 176, .values = 256, .bits = 5, .group = 32, .halves = &.{ 0, 2 } },
        .q6_k => .{ .bytes = 210, .values = 256, .bits = 6, .group = 16, .halves = &.{208} },
        .q3_k => .{ .bytes = 110, .values = 256, .bits = 3, .group = 16, .halves = &.{108} },
        .q2_k => .{ .bytes = 84, .values = 256, .bits = 2, .group = 16, .halves = &.{ 80, 82 } },
        else => unreachable,
    };
}
pub fn integer(b: []const u8, start: usize, count: usize) u32 {
    var out: u32 = 0;
    for (0..count) |j| {
        const at = start + j;
        const bit = (b[at / 8] >> @intCast(at % 8)) & 1;
        out |= @as(u32, bit) << @intCast(j);
    }
    return out;
}
pub fn half(b: []const u8, at: usize) !f64 {
    const u = @as(u16, b[at]) + 256 * @as(u16, b[at + 1]);
    const e = (u >> 10) & 31;
    const sign = @as(u64, u >> 15) << 63;
    var m: u64 = u & 1023;
    if (e == 31) return error.NonFiniteQuantScale;
    if (e != 0) return @bitCast(sign | ((@as(u64, e) + 1008) << 52) | (m << 42));
    if (m == 0) return @bitCast(sign);
    var exponent: u64 = 1009;
    while (m < 1024) {
        m *= 2;
        exponent -= 1;
    }
    return @bitCast(sign | (exponent << 52) | ((m - 1024) << 42));
}
const Params = struct { d: f64, s: f64 = 1, min: f64 = 0, m: f64 = 0, zero: f64 = 0 };
fn word(b: []const u8, at: usize) u32 {
    return @as(u32, b[at]) + 256 * @as(u32, b[at + 1]) + 65536 * @as(u32, b[at + 2]) + 16777216 * @as(u32, b[at + 3]);
}
fn params(k: Kind, b: []const u8, g: usize) !Params {
    switch (k) {
        .q8_0 => return .{ .d = try half(b, 0), .zero = 128 },
        .q4_0 => return .{ .d = try half(b, 0), .zero = 8 },
        .q5_0 => return .{ .d = try half(b, 0), .zero = 16 },
        .q4_1, .q5_1 => return .{ .d = try half(b, 0), .min = -(try half(b, 2)), .m = 1 },
        .q4_k, .q5_k => {
            const table = b[4..16];
            const sc = if (g < 4) integer(table, 8 * g, 6) else integer(table, 8 * (g + 4), 4) + 16 * integer(table, 8 * (g - 4) + 6, 2);
            const mn = if (g < 4) integer(table, 8 * (g + 4), 6) else integer(table, 8 * (g + 4) + 4, 4) + 16 * integer(table, 8 * g + 6, 2);
            return .{ .d = try half(b, 0), .s = @floatFromInt(sc), .min = try half(b, 2), .m = @floatFromInt(mn) };
        },
        .q6_k => {
            const raw = b[192 + g];
            const s = if (raw < 128) @as(i32, raw) else @as(i32, raw) - 256;
            return .{ .d = try half(b, 208), .s = @floatFromInt(s), .zero = 32 };
        },
        .q3_k => {
            // Independently follow the definition's three-word to four-word transformation.
            const a0 = word(b, 96);
            const a1 = word(b, 100);
            const t = word(b, 104);
            const k1: u32 = 0x03030303;
            const k2: u32 = 0x0f0f0f0f;
            const words = [_]u32{
                (a0 & k2) | (((t >> 0) & k1) << 4),
                (a1 & k2) | (((t >> 2) & k1) << 4),
                ((a0 >> 4) & k2) | (((t >> 4) & k1) << 4),
                ((a1 >> 4) & k2) | (((t >> 6) & k1) << 4),
            };
            const signed: i32 = @as(i32, @intCast((words[g / 4] >> @intCast(8 * (g % 4))) & 255)) - 32;
            return .{ .d = try half(b, 108), .s = @floatFromInt(signed), .zero = 4 };
        },
        .q2_k => return .{ .d = try half(b, 80), .s = @floatFromInt(integer(b, g * 8, 4)), .min = try half(b, 82), .m = @floatFromInt(integer(b, g * 8 + 4, 4)) },
        else => unreachable,
    }
}
pub fn coefficients(k: Kind, b: []const u8, g: usize) !struct { s: f64, b: f64 } {
    const p = try params(k, b, g);
    return .{ .s = p.d * p.s, .b = -p.d * p.s * p.zero - p.min * p.m };
}
pub fn code(k: Kind, b: []const u8, i: usize) u32 {
    switch (k) {
        .q8_0 => return if (b[i + 2] < 128) @as(u32, b[i + 2]) + 128 else @as(u32, b[i + 2]) - 128,
        .q4_0, .q4_1, .q5_0, .q5_1 => {
            const q_at: usize = switch (k) {
                .q4_0 => 2,
                .q4_1 => 4,
                .q5_0 => 6,
                .q5_1 => 8,
                else => unreachable,
            };
            const low = integer(b, 8 * (q_at + i % 16) + 4 * (i / 16), 4);
            if (k == .q4_0 or k == .q4_1) return low;
            const h_at: usize = if (k == .q5_0) 2 else 4;
            return low + 16 * integer(b, 8 * h_at + i, 1);
        },
        .q4_k, .q5_k => {
            const start: usize = if (k == .q4_k) 16 else 48;
            const low = integer(b, 8 * (start + (i / 64) * 32 + i % 32) + 4 * ((i / 32) % 2), 4);
            return low + if (k == .q5_k) 16 * integer(b, 8 * (16 + i % 32) + i / 32, 1) else 0;
        },
        .q6_k => {
            const n = i / 128;
            const quarter = (i % 128) / 32;
            const lane = i % 32;
            const low_at = n * 64 + lane + (if (quarter == 1 or quarter == 3) @as(usize, 32) else 0);
            return integer(b, low_at * 8 + (if (quarter >= 2) @as(usize, 4) else 0), 4) +
                16 * integer(b, (128 + n * 32 + lane) * 8 + 2 * quarter, 2);
        },
        .q3_k, .q2_k => {
            const start: usize = if (k == .q3_k) 32 else 16;
            const low = integer(b, (start + (i / 128) * 32 + i % 32) * 8 + 2 * ((i / 32) % 4), 2);
            return low + if (k == .q3_k) 4 * integer(b, (i % 32) * 8 + i / 32, 1) else 0;
        },
        else => unreachable,
    }
}
/// Original unshifted expression, deliberately not an evaluation of the affine output.
pub fn value(k: Kind, b: []const u8, i: usize) !f64 {
    const g = geometry(k);
    const p = try params(k, b, i / g.group);
    const q: f64 = @floatFromInt(code(k, b, i));
    return p.d * p.s * (q - p.zero) - p.min * p.m;
}
