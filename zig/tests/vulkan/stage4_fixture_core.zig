// Stage 4 fixture core: the consts, the f64 budget and the bf16 gate references the fixture parts share.

const std = @import("std");
const contract = @import("contract");
pub const c = contract.spec;
const gdn_mod = @import("vulkan").gdn;

pub const tail = 16;
pub const poison16: u16 = 0xffff;

pub const Model = gdn_mod.Model;
pub const tokens: u64 = 8;

/// `worst` is |got - want| OVER its budget (a ratio; >1 fails), with `where`.
pub const Report = struct { worst: f64 = 0, where: usize = 0 };

// First failing element, probe print. `want` = HOST oracle-chain ref; sp/exp never stored; negative plants pass null.
pub const Output = enum { soa, g, beta, y, state, snapshot };
pub const Violation = struct {
    index: usize,
    output: Output = .soa,
    want: f64 = 0,
    got: f64 = 0,
    bound: f64 = 0,
    ratio: f64 = 0,
    // The want value's f32 bits where the reference is an f32 scalar (g): shows f32 class (subnormal/underflow) only.
    want_f32_bits: u32 = 0,
};

pub fn violated(v: ?*Violation, output: Output, index: usize, want: f64, got: f64, bound: f64) void {
    if (v) |d| d.* = .{
        .output = output,
        .index = index,
        .want = want,
        .got = got,
        .bound = bound,
        .ratio = if (bound > 0) @abs(got - want) / bound else std.math.inf(f64),
        .want_f32_bits = @bitCast(@as(f32, @floatCast(want))),
    };
}

// budget: depth f32 sum@mag, ops f32 + mids bf16 rounds, 1 store; not sum-of-squares; never enlarge after GPU runs
pub fn budget(depth: f64, ops: f64, mids: f64, want: f64, mag: f64) f64 {
    const u: f64 = 0x1p-24;
    const g = if (depth > 1) depth * u / (1 - depth * u) else 0;
    const value_scale = @max(@abs(want), mag);
    const work = g * mag + (ops * u + mids * 0x1p-8) * value_scale;
    return work + 0x1p-8 * (@abs(want) + work) + 0x1p-133;
}

pub fn sampleDyadic(rng: std.Random, lo: i32, hi: i32) u16 {
    const e = rng.intRangeAtMost(i32, lo, hi);
    const m: f32 = if (rng.boolean()) 1.0 else 1.5;
    return c.toBf16(m * std.math.pow(f32, 2, @floatFromInt(e)) * (if (rng.boolean()) @as(f32, 1) else -1));
}

pub fn sampleDyadicF32(rng: std.Random, lo: i32, hi: i32) f32 {
    const e = rng.intRangeAtMost(i32, lo, hi);
    const m: f32 = if (rng.boolean()) 1.0 else 1.5;
    return m * std.math.pow(f32, 2, @floatFromInt(e)) * (if (rng.boolean()) @as(f32, 1) else -1);
}

pub fn poisonedShell(a: std.mem.Allocator, len: usize) ![]u16 {
    const buf = try a.alloc(u16, len + tail);
    @memset(buf, poison16);
    return buf;
}

pub fn f32Of(pairs: []const u16, word: usize) f64 {
    return @as(f32, @bitCast(@as(u32, pairs[2 * word]) | (@as(u32, pairs[2 * word + 1]) << 16)));
}

// f64 chain shared by split lane-0 and decode raw-gates; omits bf16 rounds (~20x); kept for amplification test
pub fn gateRef(a_in: f64, b_in: f64, a_log: f64, dt: f64) gdn_mod.GateF64 {
    return gdn_mod.gateReferenceF64(a_in, b_in, a_log, dt);
}

// Oracle: s=BF16(a+dt); sp=BF16(max(s,0)+log(1+exp(-|s|))); g f32; beta=BF16(sigmoid(b)); residual = exp/log ulps
pub fn gateRefOracle(a_in: u16, b_in: u16, a_log: f64, dt_in: u16) gdn_mod.GateF64 {
    const a = c.fromBf16(a_in);
    const dt = c.fromBf16(dt_in);
    const s = c.fromBf16(c.toBf16(a + dt));
    const sp_raw: f32 = @max(s, 0.0) + @log(@as(f32, 1.0) + @exp(-@abs(s)));
    const sp = c.fromBf16(c.toBf16(sp_raw));
    // the shader's exp reads the f32 A_log: cast BEFORE the exp
    const ea: f32 = @exp(@as(f32, @floatCast(a_log)));
    const g: f32 = @exp(-ea * sp);
    const beta_raw: f32 = 1.0 / (1.0 + @exp(-c.fromBf16(b_in)));
    return .{ .g = @floatCast(g), .beta = @floatCast(c.fromBf16(c.toBf16(beta_raw))) };
}
