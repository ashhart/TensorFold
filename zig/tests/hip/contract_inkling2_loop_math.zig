//! The HIP loop kernels' f32 tanh, exp and 1 / sqrt against Zig's at sampled inputs and every branch's boundaries.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_loop_hip");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("contract_inkling2_loop_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn depth(_: c.Shape) usize {
    return 0;
}

/// Branch thresholds of musl's tanhf, expf and expm1f as Zig ports them (tanh's expm1 runs at 2|x|, so its half too).
const edges = [_]u32{
    0x3f0c9f54, 0x41200000, 0x3e82c578, 0x00800000, // tanh
    0x42aeac50, 0x42b17218, 0x42cff1b5, 0x3eb17218, 0x3f851592, 0x39000000, // exp
    0x4195b844, 0x42b17180, 0x33000000, 0x4115b844, 0x42317180, 0x3e317218, 0x3f051592, 0x32800000, // expm1
};

/// Each edge one ulp either side, both signs; zeros, infinities, NaNs, subnormals, the largest finite, a few squares.
fn boundaries(out: *std.ArrayList(u32), a: std.mem.Allocator) !void {
    for (edges) |e| for ([_]u32{ e - 1, e, e + 1 }) |v| for ([_]u32{ 0, 0x80000000 }) |s| try out.append(a, v | s);
    const fixed = [_]u32{ 0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0x00000001, 0x007fffff, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0x40800000, 0x45100000, 0x358637bd };
    for (fixed) |v| try out.append(a, v);
    // exp's subnormal outputs, x from -87.3 to -104 every 1/64, where its ldexp rounds
    var x: f32 = -87.25;
    while (x > -104.5) : (x -= 1.0 / 64.0) try out.append(a, @bitCast(x));
}

/// The cell's inputs: the boundaries, all 65,536 bf16 patterns, then seeded random f32 bit patterns.
pub fn inputs(a: std.mem.Allocator, random: usize) ![]f32 {
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);
    try boundaries(&list, a);
    for (0..65536) |i| try list.append(a, @as(u32, @intCast(i)) << 16);
    var prng = std.Random.DefaultPrng.init(0x7a6d);
    for (0..random) |_| try list.append(a, prng.random().int(u32));
    const out = try a.alloc(f32, list.items.len);
    for (out, list.items) |*o, v| o.* = @bitCast(v);
    return out;
}

const ops = [3]K.Math{ .tanh, .exp, .inv_sqrt };
const device = [3]K.Math{ .device_tanh, .device_exp, .device_rsqrt };
const nan_rule = "each output is NaN where Zig's f32 gives NaN, and every other output has Zig's bits";
// one call: the three functions over every input
const served = [_]c.Shape{.{ .rows = 1, .n = 3, .k = 1 }};

/// tf_ink2_loop_math through tanh, exp and 1 / sqrt over the same inputs, outputs in that order.
pub const Math = struct {
    rig: Rig,
    k: *const K.Kernels,
    native: bool = false, // control: the device's own tanhf, expf and rsqrtf
    xs: []f32 = &.{},
    in: u64 = 0,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    const random = 1 << 18;

    pub fn deinit(self: *Math) void {
        self.rig.a.free(self.xs);
        self.rig.deinit();
    }

    pub fn kernel(self: *Math) g.Kernel {
        const name = if (self.native) "control: inkling2_loop hip f32 math through the device's own tanhf, expf and rsqrtf" else "inkling2_loop hip tf_ink2_loop_math: Zig's f32 tanh, exp and 1 / sqrt at every branch edge and 327,680 more";
        const s: c.Spec = .{ .name = name, .accum = .none, .depth = depth, .out = .f32, .served = &served, .nan = .{ .stated = nan_rule } };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.native) .shapes else null };
    }

    const vt = g.hooks(Math);

    pub fn init(self: *Math) !void {
        self.xs = try inputs(self.rig.a, random);
    }

    pub fn count(self: *const Math) usize {
        return 3 * self.xs.len;
    }

    pub fn prepare(self: *Math, shape: c.Shape, _: g.Fill, _: u64) !bool {
        if (shape.n != 3) return false;
        self.rig.reset();
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.xs));
        self.out = try self.rig.output(4 * 3 * self.xs.len);
        return true;
    }

    pub fn run(self: *Math) !void {
        const n: u32 = @intCast(self.xs.len);
        for (if (self.native) device else ops, 0..) |op, i| try K.math(self.k, self.rig.s, self.in, self.out + 4 * i * n, n, op);
        try self.rig.s.synchronize();
    }

    fn want(self: *const Math, i: usize) f32 {
        return ref.zig(ops[i / self.xs.len], self.xs[i % self.xs.len]);
    }

    pub fn reference(self: *Math, out: []f64, mag: []f64) !void {
        for (out, mag, 0..) |*o, *m, i| {
            o.* = self.want(i);
            m.* = @abs(o.*);
        }
    }

    /// Every output's bits against Zig's (NaN as a class); at gate 1, NaN and finite outputs both occur.
    pub fn check(self: *Math, at: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        var nans: usize = 0;
        var differ: usize = 0;
        var first: usize = 0;
        for (0..self.count()) |i| {
            const w = self.want(i);
            const y: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            if (std.math.isNan(w)) nans += 1;
            if (!ref.same32(y, w)) {
                if (differ == 0) first = i;
                differ += 1;
            }
        }
        if (differ > 0) {
            const x: u32 = @bitCast(self.xs[first % self.xs.len]);
            const y = std.mem.readInt(u32, got[4 * first ..][0..4], .little);
            const fmt = "{d} of {d} outputs differ; first {t} at 0x{x:0>8}: 0x{x:0>8} where Zig gives 0x{x:0>8}";
            return std.fmt.bufPrint(&self.msg, fmt, .{ differ, self.count(), ops[first / self.xs.len], x, y, @as(u32, @bitCast(self.want(first))) }) catch "an output differs";
        }
        if (at == .nonfinite and (nans == 0 or nans == self.count())) return "the inputs give no mix of NaN and finite outputs: no proof";
        return null;
    }
};
