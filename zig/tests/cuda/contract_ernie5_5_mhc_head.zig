//! ERNIE 5.5's hc_head (ernie5_5/mhc.cu) under the contract: the last layer's 4 streams into one row, as mhc.head.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_mhc_cuda");
const rig_ = @import("contract_rig.zig");
const mc = @import("contract_ernie5_5_mhc.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const mh = ernie.mhc;

const d = mh.width;
const kw = mh.wide;

/// A planted fault (control entries): ERNIE's pre-side mix RMS, or the mixes without hc_head's scale.
pub const HeadFault = enum { none, ernie_rms, no_scale };

pub const HeadEntry = struct { name: []const u8, symbol: [:0]const u8, fault: HeadFault = .none };

pub const head_entries = [_]HeadEntry{
    .{ .name = "cuda ernie5_5 mhc tf_ernie55_hc_head (the last layer's 4 streams into one row)", .symbol = "tf_ernie55_hc_head" },
    .{ .name = "control: ernie5_5 hc_head with ERNIE's pre-side mix RMS, 1 / (sqrt(mean) + eps)", .symbol = "tf_ctl_ernie55_hc_head_ernie_rms", .fault = .ernie_rms },
    .{ .name = "control: ernie5_5 hc_head without its scale on the mixes", .symbol = "tf_ctl_ernie55_hc_head_no_scale", .fault = .no_scale },
};

/// hc_head's scale, of the order the checkpoint's is.
const head_scale = [1]f32{0.7};

const head_why = "pre is 1/2 under stress, so the mixes' sums cannot move the output; its 4 collapse terms are exact";

/// rows (gates 1 and 4 and the controls take the first: its rows 2 and 5 are tiny), a decode row, a prompt chunk.
pub const head_served = &[_]c.Shape{ .{ .rows = 6, .n = d, .k = kw }, .{ .rows = 1, .n = d, .k = kw }, .{ .rows = 64, .n = d, .k = kw } };

pub fn headSpec(e: HeadEntry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = mc.depth, .ops = 12, .out = .bf16, .served = if (e.fault == .none) head_served else head_served[0..1], .sums = .{ .f16 = 0, .bf16 = 0, .why = head_why } };
}

pub const Head = struct {
    rig: rig_.Rig,
    entry: HeadEntry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    hc_eps: f32 = mc.params.hc_eps,
    x: []u16 = &.{}, // rows of 4 bf16 streams
    fnw: []f32 = &.{}, // [4][16384]
    t: []f64 = &.{}, // scratch: a row's terms, then the row as f64
    base: [mh.streams]f32 = undefined,
    dev: [4]u64 = @splat(0), // x, fn, scale, base
    out: u64 = 0,

    const vt = g.hooks(Head);

    pub fn kernel(self: *Head) g.Kernel {
        return .{ .spec = headSpec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Head) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.fnw);
        a.free(self.t);
        self.x = &.{};
        self.fnw = &.{};
        self.t = &.{};
    }

    pub fn deinit(self: *Head) void {
        self.free();
        self.rig.deinit();
    }

    /// Rows as the pre side's (unit, offset by 8, below 1e-6); stress: pre 1/2 and stream 0 = 2v, so the output is v.
    pub fn prepare(self: *Head, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != kw or shape.n != d) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        self.hc_eps = mc.params.hc_eps;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, shape.rows * kw);
        self.fnw = try a.alloc(f32, mh.streams * kw);
        self.t = try a.alloc(f64, 2 * kw);
        if (fill == .stress) {
            self.hc_eps = 0;
            @memset(self.fnw, 0);
            @memset(&self.base, 0);
            for (0..shape.rows) |r| for (self.x[r * kw ..][0..kw], 0..) |*v, i| {
                v.* = if (i < d) bfr.bits(2 * (@as(f32, @floatFromInt(rng.uintLessThan(u32, 63))) - 31)) else 0;
            };
        } else {
            for (self.fnw) |*f| f.* = (rng.float(f32) * 2 - 1) * 0.02;
            for (&self.base) |*b| b.* = rng.float(f32) * 2 - 1;
            for (0..shape.rows) |r| {
                const m: f32 = if (r % 3 == 2) 1e-6 else 1;
                const off: f32 = if (r % 3 == 1) 8 else 0;
                for (self.x[r * kw ..][0..kw], 0..) |*v, i| v.* = bfr.bits(m * (off + (rng.float(f32) * 2 - 1) * (if (i % 512 == 77) @as(f32, 20) else 1)));
            }
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, 42, std.math.nan(f32) }, .{ 1, d + 7, std.math.inf(f32) }, .{ 2, kw - 1, -std.math.inf(f32) } };
            for (planted) |pl| if (pl[0] < shape.rows) {
                self.x[pl[0] * kw + pl[1]] = bfr.bits(pl[2]);
            };
        }
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), try self.rig.input(std.mem.sliceAsBytes(self.fnw)), try self.rig.input(std.mem.sliceAsBytes(&head_scale)), try self.rig.input(std.mem.sliceAsBytes(&self.base)) };
        self.out = try self.rig.output(shape.rows * d * 2);
        return true;
    }

    pub fn run(self: *Head) !void {
        try ernie.launchHead(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.out, .{ .rows = self.shape.rows }, mc.params.rms_eps, self.hc_eps);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Head) usize {
        return self.shape.rows * d;
    }

    /// Row r's 4 streams as f64, in the upper half of the scratch.
    fn row(self: *Head, r: usize) *[kw]f64 {
        const x = self.t[kw..][0..kw];
        for (x, self.x[r * kw ..][0..kw]) |*v, h| v.* = mc.bf(h);
        return x;
    }

    /// mhc.head in float64; mag: the collapse's terms, and the mixes' |terms| through sigmoid' and the scale.
    pub fn reference(self: *Head, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| {
            const x = self.row(r);
            mh.head(x, self.fnw, head_scale[0], &self.base, mc.params.rms_eps, self.hc_eps, mh.exact, self.t[0..kw], want[r * d ..][0..d]);
            var ss: f64 = 0;
            for (x) |v| ss += v * v;
            const rms = 1 / @sqrt(ss / kw + mc.params.rms_eps);
            var pw: [4]f64 = undefined;
            var dp: [4]f64 = undefined;
            for (0..4) |k| {
                var abs: f64 = 0;
                var sum: f64 = 0;
                for (x, self.fnw[k * kw ..][0..kw]) |v, f| {
                    abs += @abs(v * f);
                    sum += v * f;
                }
                pw[k] = mh.sigmoid(sum * rms * head_scale[0] + self.base[k]) + self.hc_eps;
                dp[k] = 0.25 * head_scale[0] * 1.5 * abs * rms;
            }
            for (mag[r * d ..][0..d], 0..) |*mg, h| {
                var om: f64 = 0;
                for (0..4) |s| om += @abs(pw[s] * x[s * d + h]) + @abs(x[s * d + h]) * dp[s];
                mg.* = om;
            }
        }
    }

    /// mhc.head with every sum in `acc`, in `order`.
    pub fn simulate(self: *Head, acc: c.Accum, order: g.Order, out: []f64) !void {
        const sim: mc.Sim = .{ .acc = acc, .order = order };
        for (0..self.shape.rows) |r| {
            const x = self.row(r);
            mh.head(x, self.fnw, head_scale[0], &self.base, mc.params.rms_eps, self.hc_eps, sim, self.t[0..kw], out[r * d ..][0..d]);
        }
    }
};
