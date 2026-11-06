//! CPU Engram from DeepSeek AI inference/engram.py and model.py, rev 2cba9e42aa026125f3ed06c6d98c1db82f7ca027.
const std = @import("std");
const tf = @import("tensorfold");
const m = @import("math.zig");
const projection = @import("projection.zig");
const Engram = tf.deepseek_v41_weights.Engram;
const Config = tf.deepseek_v41_config.Config;
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
/// Where a streamed model's Engram rows come from. One call per (layer, row): the reference asks only for the rows a
/// handful of teacher tokens touch, so there is no cache here — the reader's own staging is enough.
pub const RowSource = struct {
    ptr: *anyopaque,
    rowFn: *const fn (p: *anyopaque, layer: usize, row: u32, out: []u16) anyerror!void,
    pub fn row(s: RowSource, layer: usize, id: u32, out: []u16) !void {
        return s.rowFn(s.ptr, layer, id, out);
    }
};
pub const State = struct {
    map: []const u8,
    history: [3]u32 = .{ 2, 2, 2 },
    gathered: [6144]u16 = undefined,
    kv: [25600]u16 = undefined,
    /// The four applied gate scalars, one per copy, kept so a trace can print the value the reference actually used
    /// instead of inferring it from a bf16 hidden state.
    gates: [4]f32 = @splat(0),

    /// Diagnostic faults, set only by a caller that is measuring the gate rather than qualifying a model.
    pub const Faults = struct {
        /// XOR into the first gathered value: 1 flips one bf16 ulp (sensitivity floor), 0x3C00 flips the exponent.
        pub var gather_xor: u16 = 0;
        /// Multiplier on every applied gate scalar.
        pub var gate_scale: f32 = 1;
    };

    pub fn init(blob: []const u8) !State {
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
                if (word(blob, base + 40 + 4 * j) != v or word(blob, base + 136 + 4 * j) != offset)
                    return error.InvalidEngramMetadata;
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
    pub fn compressed(self: *const State, id: u32) !u32 {
        if (id >= 129280) return error.InvalidToken;
        return word(self.map, @as(usize, id) * 4);
    }
    pub fn advance(self: *State, id: u32) void {
        self.history[2] = self.history[1];
        self.history[1] = self.history[0];
        self.history[0] = id;
    }
    /// The Engram's own gate at the chain's precision T. The FP32 contract is unchanged; the FP64 chain keeps the
    /// two 5120-dot statistics and the gate in FP64.
    pub fn apply(self: *State, comptime T: type, e: Engram, layer: usize, id: u32, h: []u16, c: Config, scratch: *projection.Scratch, rows: ?RowSource) !void {
        @setFloatMode(.strict);
        const tokens = [4]u32{ id, self.history[0], self.history[1], self.history[2] };
        var rolling = @as(u64, tokens[0]) * multipliers[layer][0];
        var offset: usize = 0;
        for (1..4) |lookback| {
            rolling ^= @as(u64, tokens[lookback]) * multipliers[layer][lookback];
            for (0..8) |head| {
                const column = (lookback - 1) * 8 + head;
                const row = offset + @as(usize, @intCast(rolling % primes[layer][column]));
                if (e.streamed) {
                    // The tables are not resident; ask the row source for exactly the row the resident path would read.
                    const src = rows orelse return error.ReferenceNeedsResidentEngramTables;
                    try src.row(layer, @intCast(row), self.gathered[column * 256 ..][0..256]);
                } else for (0..256) |d| self.gathered[column * 256 + d] =
                    m.b(m.value(e.table.?, row * 256 + d) * m.value(e.scales.?, row * 8 + d / 32));
                offset += primes[layer][column];
            }
        }
        // Before the projection, so the fault propagates the way a bad row would.
        if (Faults.gather_xor != 0) self.gathered[0] ^= Faults.gather_xor;
        try projection.applyOf(u16, T, .{ .trellis = e.wkv }, &self.gathered, &self.kv, scratch);
        for (0..4) |copy| {
            var hh: f64 = 0;
            var kk: f64 = 0;
            var dot: f64 = 0;
            for (0..5120) |d| {
                const at = copy * 5120 + d;
                const x = m.f(h[at]);
                const k = m.f(self.kv[at]);
                const weight = m.value(e.q_weight, at) * m.value(e.k_weight, at);
                hh += @as(f64, x) * x;
                kk += @as(f64, k) * k;
                const weighted: f32 = (x * weight) * k;
                dot += weighted;
            }
            if (T == f32) {
                const h_mean = m.meanFromWide(hh, 5120);
                const k_mean = m.meanFromWide(kk, 5120);
                const h_inverse = m.idealRsqrt(h_mean + c.eps);
                const k_inverse = m.idealRsqrt(k_mean + c.eps);
                const rstd: f32 = h_inverse * k_inverse;
                const dot_f32: f32 = @floatCast(dot);
                const normalized: f32 = (dot_f32 * rstd) * (1 / @sqrt(@as(f32, 5120)));
                const gate = m.sigmoid(std.math.copysign(@sqrt(@max(@abs(normalized), 1e-6)), normalized)) * Faults.gate_scale;
                self.gates[copy] = gate;
                for (0..5120) |d| {
                    const at = copy * 5120 + d;
                    h[at] = m.b(m.f(h[at]) + gate * m.f(self.kv[20480 + d]));
                }
            } else {
                const h_inverse = 1 / @sqrt(hh / 5120 + @as(f64, c.eps));
                const k_inverse = 1 / @sqrt(kk / 5120 + @as(f64, c.eps));
                const rstd: f64 = h_inverse * k_inverse;
                const normalized: f64 = (dot * rstd) * (1 / @sqrt(@as(f64, 5120)));
                const gate = 1 / (1 + @exp(-std.math.copysign(@sqrt(@max(@abs(normalized), 1e-6)), normalized))) * Faults.gate_scale;
                self.gates[copy] = @floatCast(gate);
                for (0..5120) |d| {
                    const at = copy * 5120 + d;
                    h[at] = m.b64(@as(f64, m.f(h[at])) + gate * @as(f64, m.f(self.kv[20480 + d])));
                }
            }
        }
    }
};
