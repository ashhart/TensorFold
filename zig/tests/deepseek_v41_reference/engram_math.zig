//! CPU Engram from DeepSeek AI inference/engram.py and model.py, rev 2cba9e42; Module(Real) keeps FP32 or FP64.

const std = @import("std");
const tf = @import("tensorfold");
const math = @import("math.zig");
const projection = @import("projection.zig");
const Engram = tf.deepseek_v41_weights.Engram;
const Config = tf.deepseek_v41_config.Text;
const multipliers = [2][4]u64{
    .{ 76632096046245, 4839876093313, 35959672319349, 73987337458391 },
    .{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 },
};
const primes = [2][24]u32{
    .{ 16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183, 16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321, 16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463 },
    .{ 16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627, 16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769, 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889 },
};
fn word(raw: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, raw[at..][0..4], .little);
}

pub fn Module(comptime Real: type) type {
    return struct {
        const m = math.Module(Real);
        const proj = projection.Module(Real);
        /// Engram gather narrowing: the E4M3 table word times its UE8M0 row scale, rounded to BF16.
        pub fn narrowedProduct(table: Real, scale: Real) u16 {
            return m.b(table * scale);
        }
        const StateImpl = struct {
            map: []const u8,
            history: [3]u32 = .{ 2, 2, 2 },
            gathered: [6144]u16 = undefined,
            kv: [25600]u16 = undefined,

            pub fn init(blob: []const u8) !StateImpl {
                const at = 96 + 2 * 232;
                if (blob.len != at + 129280 * 4 or !std.mem.eql(u8, blob[0..8], "DSENGR01"))
                    return error.InvalidEngramMetadata;
                for ([_]u32{ 129280, 99092, 2, 4, 8, 2 }, 0..) |v, i|
                    if (word(blob, 8 + i * 4) != v) return error.InvalidEngramMetadata;
                for (0..2) |layer| {
                    const base = 96 + layer * 232;
                    if (word(blob, base) != (if (layer == 0) @as(u32, 1) else @as(u32, 14)))
                        return error.InvalidEngramMetadata;
                    var offset: u32 = 0;
                    for (multipliers[layer], 0..) |v, j|
                        if (std.mem.readInt(u64, blob[base + 8 + 8 * j ..][0..8], .little) != v)
                            return error.InvalidEngramMetadata;
                    for (primes[layer], 0..) |v, j| {
                        if (word(blob, base + 40 + 4 * j) != v or word(blob, base + 136 + 4 * j) != offset) return error.InvalidEngramMetadata;
                        offset += v;
                    }
                    if (word(blob, base + 4) != offset) return error.InvalidEngramMetadata;
                }
                const map = blob[at..];
                const expected = [_]u8{ 0xc6, 0x0a, 0x86, 0x32, 0x2e, 0xc1, 0x7b, 0x41, 0x42, 0xbf, 0xef, 0x3c, 0x57, 0xa8, 0xd8, 0x1f, 0xb4, 0x28, 0x55, 0x0c, 0xdf, 0x48, 0x7f, 0x88, 0xa4, 0x32, 0x0c, 0xb5, 0x9f, 0xe4, 0x64, 0x81 };
                var digest: [32]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(map, &digest, .{});
                if (!std.mem.eql(u8, &expected, &digest)) return error.InvalidEngramTokenMap;
                return .{ .map = map };
            }
            pub fn compressed(self: *const StateImpl, id: u32) !u32 {
                if (id >= 129280) return error.InvalidToken;
                return word(self.map, @as(usize, id) * 4);
            }
            pub fn advance(self: *StateImpl, id: u32) void {
                self.history[2] = self.history[1];
                self.history[1] = self.history[0];
                self.history[0] = id;
            }
            pub fn apply(self: *StateImpl, e: Engram, layer: usize, id: u32, h: []u16, c: Config, scratch: *projection.Scratch) !void {
                @setFloatMode(.strict);
                const tokens = [4]u32{ id, self.history[0], self.history[1], self.history[2] };
                var rolling = @as(u64, tokens[0]) * multipliers[layer][0];
                var offset: usize = 0;
                for (1..4) |lookback| {
                    rolling ^= @as(u64, tokens[lookback]) * multipliers[layer][lookback];
                    for (0..8) |head| {
                        const column = (lookback - 1) * 8 + head;
                        const row = offset + @as(usize, @intCast(rolling % primes[layer][column]));
                        for (0..256) |d| self.gathered[column * 256 + d] =
                            narrowedProduct(m.value(e.table, row * 256 + d), m.value(e.scales, row * 8 + d / 32));
                        offset += primes[layer][column];
                    }
                }
                try proj.applyTrellis(e.wkv, &self.gathered, &self.kv, scratch);
                for (0..4) |copy| {
                    var hh: f64 = 0;
                    var kk: f64 = 0;
                    var dot: f64 = 0;
                    for (0..5120) |d| {
                        const at = copy * 5120 + d;
                        const x = m.f(h[at]);
                        const k = m.f(self.kv[at]);
                        const weight: Real = m.value(e.q_weight, at) * m.value(e.k_weight, at);
                        hh += @as(f64, x) * x;
                        kk += @as(f64, k) * k;
                        const weighted: Real = (x * weight) * k;
                        dot += @as(f64, weighted);
                    }
                    const h_mean = m.meanFromWide(hh, 5120);
                    const k_mean = m.meanFromWide(kk, 5120);
                    const h_inverse = m.idealRsqrt(h_mean + m.wr(c.rms_norm_eps));
                    const k_inverse = m.idealRsqrt(k_mean + m.wr(c.rms_norm_eps));
                    const rstd: Real = h_inverse * k_inverse;
                    const dot_wide: Real = @floatCast(dot);
                    const normalized: Real = (dot_wide * rstd) * (1 / @sqrt(@as(Real, 5120)));
                    const gate = m.sigmoid(std.math.copysign(@sqrt(@max(@abs(normalized), @as(Real, 1e-6))), normalized));
                    for (0..5120) |d| {
                        const at = copy * 5120 + d;
                        h[at] = m.b(m.f(h[at]) + gate * m.f(self.kv[20480 + d]));
                    }
                }
            }
        };
        pub const State = StateImpl;
    };
}

/// The retained FP32 instance; every field the caller touches today keeps its name and type.
pub const f32_module = Module(f32);
pub const State = f32_module.State;

test "frozen FP32 Engram gather boundaries match the pre-repair oracle" {
    // Frozen at 53b790082 (pre-repair): FP32 gather words stay; cells pair E4M3 codes and UE8M0 scales.
    const f32m = math.Module(f32);
    const cells = [_][2]u8{
        .{ 56, 254 }, // 1 * 2^127, the largest BF16 exponent
        .{ 64, 254 }, // 2 * 2^127 overflows to infinity
        .{ 56, 253 }, // 1 * 2^126
        .{ 126, 254 }, // 448 * 2^127 overflows to infinity
        .{ 1, 3 }, // 2^-9 * 2^-124 = 2^-133, one BF16 subnormal unit
        .{ 3, 3 }, // 3 * 2^-133
        .{ 7, 2 }, // 7 * 2^-134 = 3.5 units, ties to even 4
        .{ 1, 2 }, // 2 * 2^-135 = 0.5 units, ties to even zero
        .{ 1, 4 }, // 2^-132 = 2 units
        .{ 56, 6 }, // 1 * 2^-121
        .{ 0, 254 }, // zero table word
        .{ 0x80 | 1, 3 }, // -2^-133
        .{ 127, 0 }, // E4M3 NaN table word
        .{ 255, 0 }, // E4M3 NaN table word
    };
    var narrowed: [cells.len]u16 = undefined;
    for (&narrowed, cells) |*narrowed_word, cell| {
        narrowed_word.* = f32_module.narrowedProduct(f32m.e4(cell[0]), f32m.e8(cell[1]));
    }
    const frozen = [_]u16{
        0x7f00, 0x7f80, 0x7e80, 0x7f80, 0x0001, 0x0003, 0x0004,
        0x0000, 0x0002, 0x0300, 0x0000, 0x8001, 0x7fc0, 0x7fc0,
    };
    try std.testing.expectEqualSlices(u16, &frozen, &narrowed);
}
