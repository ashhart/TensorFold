//! Pure Inkling 2 Mini loop launch geometry and independent FP64 arithmetic.
const std = @import("std");

pub const hidden = 2304;
pub const rank = 64;
pub const core_layers = 12;
pub const roles = 4;
pub const message_entries = core_layers * roles;
pub const eps = 1e-6;
pub const message_scale: f64 = @as(f32, @bitCast(@as(u32, 0x3caaaaab)));
pub const RowsArgs = extern struct { rows: u32 };

pub const Role = enum(u8) { q, k, v, mlp };
pub const Kind = enum {
    messages,
    gates,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .messages => "tf_inkling2_loop_messages",
            .gates => "tf_inkling2_loop_gates",
        };
    }
    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            .messages => "metal inkling2 loop messages tf_inkling2_loop_messages",
            .gates => "metal inkling2 loop gates tf_inkling2_loop_gates",
        };
    }
};

pub const Call = struct { rows: usize, n: usize, k: usize };
pub const Geometry = struct { groups: usize, threads: u32 = 256, tg_memory: u32 };

pub fn valid(kind: Kind, c: Call) bool {
    if (c.rows == 0 or c.rows > 256) return false;
    return switch (kind) {
        .messages => c.n == message_entries * hidden and c.k == hidden,
        .gates => c.n == hidden and c.k == rank,
    };
}
pub fn count(kind: Kind, c: Call) usize {
    return switch (kind) {
        .messages => c.rows * message_entries * hidden,
        .gates => c.rows * hidden,
    };
}
pub fn geometry(kind: Kind, c: Call) Geometry {
    return .{
        .groups = switch (kind) {
            .messages => c.rows * message_entries,
            .gates => c.rows,
        },
        .tg_memory = switch (kind) {
            .messages => 144,
            .gates => 80,
        },
    };
}

/// q/mlp and k/v each occupy even/odd rows of their separate 24xH tensors.
pub fn queryRow(role: Role, layer: usize) usize {
    std.debug.assert(layer < core_layers);
    return 2 * layer + switch (role) {
        .q, .k => @as(usize, 0),
        .v, .mlp => 1,
    };
}
pub fn usesQueryMessageTensor(role: Role) bool {
    return role == .q or role == .mlp;
}

/// Round through FP32 and BF16 nearest-even, returning an exactly representable FP64 value.
pub fn roundBf16(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}

pub fn rmsScale(x: []const f32) f64 {
    std.debug.assert(x.len == hidden);
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    return 1 / @sqrt(ss / hidden + eps);
}

/// One of the 48 messages. Banks contain their already-decoded BF16 values as f32.
pub fn message(bank0: []const f32, bank1: []const f32, q_mlp: []const f32, kv: []const f32, layer: usize, role: Role, out: []f64) void {
    std.debug.assert(bank0.len == hidden and bank1.len == hidden and out.len == hidden);
    std.debug.assert(q_mlp.len == 2 * core_layers * hidden and kv.len == q_mlp.len);
    const queries = if (usesQueryMessageTensor(role)) q_mlp else kv;
    const query = queries[queryRow(role, layer) * hidden ..][0..hidden];
    const inverse = [2]f64{ rmsScale(bank0), rmsScale(bank1) };
    var logits: [2]f64 = undefined;
    var peak = -std.math.inf(f64);
    inline for (0..2) |s| {
        const bank = if (s == 0) bank0 else bank1;
        var dot: f64 = 0;
        for (0..hidden) |d| dot += @as(f64, query[d]) * (@as(f64, bank[d]) * inverse[s]);
        logits[s] = dot * message_scale;
        peak = @max(peak, logits[s]);
    }
    var weights: [2]f64 = undefined;
    var den: f64 = 0;
    for (0..2) |s| {
        weights[s] = @exp(logits[s] - peak);
        den += weights[s];
    }
    for (0..hidden) |d| {
        const beta0 = @as(f64, bank0[d]) * inverse[0];
        const beta1 = @as(f64, bank1[d]) * inverse[1];
        out[d] = weights[0] / den * beta0 + weights[1] / den * beta1;
    }
}

/// BF16(concat(rho(h), rho(m))) D projection and the BF16 zeta endpoint, with FP64 sums.
pub fn downProjection(h: []const f32, m: []const f32, down: []const f32, zeta: *[rank]f64) void {
    std.debug.assert(h.len == hidden and m.len == hidden and down.len == rank * 2 * hidden);
    const ih = rmsScale(h);
    const im = rmsScale(m);
    var normalized: [2 * hidden]f64 = undefined;
    for (0..hidden) |d| {
        normalized[d] = roundBf16(@as(f64, h[d]) * ih);
        normalized[hidden + d] = roundBf16(@as(f64, m[d]) * im);
    }
    for (0..rank) |r| {
        var dot: f64 = 0;
        for (0..2 * hidden) |d| dot += @as(f64, down[r * 2 * hidden + d]) * normalized[d];
        zeta[r] = roundBf16(dot);
    }
}

/// The reference's four explicit binary32 blend nodes. FP64 carries values between nodes.
pub fn blend(forget_value: f64, gain_value: f64, h: f64, m: f64) f64 {
    const f: f32 = @floatCast(forget_value);
    const g: f32 = @floatCast(gain_value);
    const state: f32 = @floatCast(h);
    const message_value: f32 = @floatCast(m);
    const keep: f32 = 1.0 - f;
    const left: f32 = keep * state;
    const right: f32 = g * message_value;
    const sum: f32 = left + right;
    return sum;
}

/// Full rank-64 gate. Matrix slices contain their exact BF16 values widened to f32.
pub fn gate(h: []const f32, m: []const f32, down: []const f32, forget: []const f32, gain: []const f32, out: []f64) void {
    std.debug.assert(h.len == hidden and m.len == hidden and out.len == hidden);
    std.debug.assert(forget.len == hidden * rank and gain.len == forget.len);
    var zeta: [rank]f64 = undefined;
    downProjection(h, m, down, &zeta);
    for (0..hidden) |d| {
        var fdot: f64 = 0;
        var gdot: f64 = 0;
        for (0..rank) |r| {
            fdot += @as(f64, forget[d * rank + r]) * zeta[r];
            gdot += @as(f64, gain[d * rank + r]) * zeta[r];
        }
        out[d] = blend(std.math.tanh(fdot), std.math.tanh(gdot), h[d], m[d]);
    }
}
