//! The HIP experts' activation factors at every bf16 input, bit-equal to Zig's f32 std.math.tanh and @exp.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_experts_hip");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("contract_inkling2_experts_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const n = K.factor_inputs;

fn depth(_: c.Shape) usize {
    return 0;
}

const nan_rule = "each factor is NaN where Zig's f32 gives NaN, and every other output has Zig's bits";
const served = [_]c.Shape{.{ .rows = 1, .n = 2 * n, .k = 1 }};

/// tf_ink2_act_factors over all 65,536 bf16 patterns: the gate factors, then the up factors, as fp32.
pub const Factors = struct {
    rig: Rig,
    k: *const K.Kernels,
    native: bool = false, // control: the device's own tanhf and expf
    in: u64 = 0,
    gate: u64 = 0,
    up: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Factors) void {
        self.rig.deinit();
    }

    pub fn kernel(self: *Factors) g.Kernel {
        const name = if (self.native) "control: inkling2_experts hip activation factors through the device's own tanhf and expf" else "inkling2_experts hip tf_ink2_act_factors: softcap4(g) * sigmoid(g) and softcap25(u) at all 65,536 bf16 inputs";
        const s: c.Spec = .{ .name = name, .accum = .none, .depth = depth, .out = .f32, .served = &served, .nan = .{ .stated = nan_rule } };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.native) .shapes else null };
    }

    const vt = g.hooks(Factors);

    pub fn count(_: *const Factors) usize {
        return 2 * n;
    }

    /// The inputs are every bf16 pattern whatever the fill: NaNs and both infinities are among them.
    pub fn prepare(self: *Factors, shape: c.Shape, _: g.Fill, _: u64) !bool {
        if (shape.n != 2 * n) return false;
        self.rig.reset();
        var codes: [n]u16 = undefined;
        for (&codes, 0..) |*v, i| v.* = @intCast(i);
        self.in = try self.rig.input(std.mem.sliceAsBytes(&codes));
        self.gate = try self.rig.output(n * 4);
        self.up = try self.rig.output(n * 4);
        return true;
    }

    pub fn run(self: *Factors) !void {
        try K.actFactors(self.k, self.rig.s, self.in, self.gate, self.up, n, self.native);
        try self.rig.s.synchronize();
    }

    fn want(i: usize) f32 {
        const v: f32 = @bitCast(@as(u32, @intCast(i % n)) << 16);
        return if (i < n) ref.gateFactor(v) else ref.upFactor(v);
    }

    pub fn reference(_: *Factors, out: []f64, mag: []f64) !void {
        for (out, mag, 0..) |*o, *m, i| {
            o.* = want(i);
            m.* = @abs(o.*);
        }
    }

    /// Every output's bits against Zig's (NaN as a class); at gate 1, NaN and finite outputs both occur.
    pub fn check(self: *Factors, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var nans: usize = 0;
        var differ: usize = 0;
        var first: usize = 0;
        for (0..2 * n) |i| {
            const w = want(i);
            const got: f32 = @bitCast(std.mem.readInt(u32, out[4 * i ..][0..4], .little));
            if (std.math.isNan(w)) nans += 1;
            if (!ref.same32(got, w)) {
                if (differ == 0) first = i;
                differ += 1;
            }
        }
        if (differ > 0) {
            const which = if (first < n) "gate" else "up";
            const fmt = "{d} of {d} factors differ; first the {s} factor at bf16 0x{x:0>4}: 0x{x:0>8} where Zig gives 0x{x:0>8}";
            const got: u32 = std.mem.readInt(u32, out[4 * first ..][0..4], .little);
            return std.fmt.bufPrint(&self.msg, fmt, .{ differ, 2 * n, which, first % n, got, @as(u32, @bitCast(want(first))) }) catch "a factor differs";
        }
        if (at == .nonfinite and (nans == 0 or nans == 2 * n)) return "the inputs give no mix of NaN and finite factors: no proof";
        return null;
    }
};
