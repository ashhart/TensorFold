//! Qwen3's served projected-head norm, including the unchanged V suffix, guarded in-place and faulted controls.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;
const Args = extern struct { rows: u32, heads: u32, kv_heads: u32, stride: u32, eps: f32 };
const Mutation = enum { none, wrong_mean, hide_nan };
fn depth(_: c.Shape) usize {
    return 9;
}
fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}
fn name(m: Mutation) []const u8 {
    return switch (m) {
        .none => "qwen3 heads q3_head_norm",
        .wrong_mean => "control: qwen3 heads wrong mean",
        .hide_nan => "control: qwen3 heads drops nonfinite",
    };
}
fn text(a: std.mem.Allocator, m: Mutation) ![]u8 {
    const edit: [2][]const u8 = switch (m) {
        .none => return a.dupe(u8, ks.qwen3_heads),
        .wrong_mean => .{ "total / 128.0f + a.eps", "total / 256.0f + a.eps" },
        .hide_nan => .{ "const float v = float(x[t]);", "const float v = isfinite(float(x[t])) ? float(x[t]) : 0.0f;" },
    };
    if (std.mem.count(u8, ks.qwen3_heads, edit[0]) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, ks.qwen3_heads, edit[0], edit[1]);
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mutation: Mutation,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    weights: [256]u16 = undefined,
    out: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: Mutation) !Probe {
        const src = try text(a, mutation);
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try mtl.Pipeline.init(device, lib, "q3_head_norm", false), .mutation = mutation };
    }
    pub fn deinit(p: *Probe) void {
        p.a.free(p.x);
        p.rig.deinit();
        p.pipe.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = switch (p.mutation) {
            .none => null,
            .wrong_mean => .shapes,
            .hide_nan => .nonfinite,
        }, .spec = .{ .name = name(p.mutation), .accum = .f32, .depth = depth, .ops = 3, .mids = 1, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 1, .n = 24, .k = 128 }, .{ .rows = 17, .n = 40, .k = 128 }, .{ .rows = 2048, .n = 40, .k = 128 } }, .sums = .{ .f16 = 50, .bf16 = 0, .why = "BF16 output masks the BF16 accumulator bound; FP16 squares overflow on exact integer stress rows" } } };
    }
    const vt = g.hooks(Probe);
    fn stride(p: *const Probe) usize {
        return (p.shape.n + 8) * 128;
    }
    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.stride();
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k != 128 or (s.n != 24 and s.n != 40)) return false;
        p.a.free(p.x);
        p.x = &.{};
        p.rig.reset();
        p.shape = s;
        p.x = try p.a.alloc(u16, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.x) |*x| x.* = c.toBf16(6 * rng.float(f32) - 3);
        for (&p.weights, 0..) |*w, i| w.* = c.toBf16(if (i < 128) 0.75 else 1.25);
        if (fill == .stress) {
            for (0..s.rows) |r| {
                for (p.x[r * p.stride() ..][0 .. s.n * 128]) |*x| x.* = c.toBf16(400);
            }
        }
        if (fill == .nonfinite) {
            const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (planted, 0..) |v, i| p.x[i * 128 + 13] = c.toBf16(v);
            // K heads see nonfinite independently of the Q heads; V must remain untouched.
            p.x[(s.n - 8) * 128 + 7] = c.toBf16(std.math.inf(f32));
        }
        p.out = try p.rig.output(p.x.len * 2);
        p.w = try p.rig.input(std.mem.sliceAsBytes(&p.weights));
        return true;
    }
    pub fn run(p: *Probe) !void {
        @memcpy(p.rig.bytes(p.out, p.x.len * 2), std.mem.sliceAsBytes(p.x));
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.out.buf, p.out.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        e.setBuffer(p.w.buf, p.w.off + 256, 2);
        e.setValue(Args{ .rows = @intCast(p.shape.rows), .heads = @intCast(p.shape.n - 8), .kv_heads = 8, .stride = @intCast(p.stride()), .eps = 1e-6 }, 3);
        e.dispatchGroups(mtl.Size.of(p.shape.n, p.shape.rows, 1), mtl.Size.of(128, 1, 1));
        try p.rig.finish(pass);
        if (!p.unchangedV()) return error.ProjectedVChanged;
    }
    fn values(p: *Probe, out: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) void {
        for (p.x, out, 0..) |v, *o, i| {
            o.* = c.fromBf16(v);
            if (mag) |m| m[i] = 0;
        }
        for (0..p.shape.rows) |r| for (0..p.shape.n) |h| {
            const at = r * p.stride() + h * 128;
            var terms: [128]f64 = undefined;
            for (&terms, 0..) |*v, i| {
                const x = c.fromBf16(p.x[at + i]);
                v.* = x * x;
            }
            var ss: f64 = 0;
            if (acc) |a| ss = sumIn(a, order, &terms) else {
                for (terms) |v| ss += v;
            }
            const inv = 1.0 / @sqrt(ss / 128.0 + @as(f64, @as(f32, 1e-6)));
            for (out[at..][0..128], 0..) |*v, i| {
                const wi = if (h < p.shape.n - 8) i else 128 + i;
                v.* = bf(bf(c.fromBf16(p.x[at + i]) * inv) * c.fromBf16(p.weights[wi]));
                if (mag) |m| m[at + i] = @abs(v.*) / 2;
            }
        };
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.values(want, mag, null, .sequential);
    }
    /// Bound each rounded endpoint from the FP32 sum tree and rsqrt, before either BF16 rounding.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (p.x, low, high) |v, *lo, *hi| {
            lo.* = c.fromBf16(v);
            hi.* = lo.*;
        }
        const u = 0x1p-24;
        const gamma = c.gamma(9, u);
        for (0..p.shape.rows) |r| for (0..p.shape.n) |h| {
            const at = r * p.stride() + h * 128;
            var ss: f64 = 0;
            for (p.x[at..][0..128]) |v| {
                const x = c.fromBf16(v);
                ss += x * x;
            }
            const inv_lo = (1 - 3 * u) / @sqrt(ss * (1 + gamma) / 128 + @as(f64, @as(f32, 1e-6)));
            const inv_hi = (1 + 3 * u) / @sqrt(ss * (1 - gamma) / 128 + @as(f64, @as(f32, 1e-6)));
            for (0..128) |i| {
                const x = c.fromBf16(p.x[at + i]);
                const w = c.fromBf16(p.weights[(if (h < p.shape.n - 8) @as(usize, 0) else 128) + i]);
                const a = @min(x * inv_lo, x * inv_hi);
                const b = @max(x * inv_lo, x * inv_hi);
                const left = bf(a - @abs(a) * u) * w;
                const right = bf(b + @abs(b) * u) * w;
                low[at + i] = bf(@min(left, right));
                high[at + i] = bf(@max(left, right));
            }
        };
    }
    /// The reference is already BF16-rounded; its discontinuities need the two reachable signed endpoints.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const low = try p.a.alloc(f64, want.len);
        defer p.a.free(low);
        const high = try p.a.alloc(f64, want.len);
        defer p.a.free(high);
        try p.intervals(low, high);
        for (allow, want, low, high) |*a, w, lo, hi| a.* = @max(@abs(w - lo), @abs(hi - w));
    }
    fn unchangedV(p: *Probe) bool {
        const got = p.rig.bytes(p.out, p.x.len * 2);
        for (0..p.shape.rows) |r| {
            const at = r * p.stride() + p.shape.n * 128;
            if (!std.mem.eql(u8, got[at * 2 ..][0 .. 8 * 128 * 2], std.mem.sliceAsBytes(p.x[at..][0 .. 8 * 128])))
                return false;
        }
        return true;
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        p.values(out, null, acc, order);
    }
};
pub const Suite = struct {
    probes: [3]?Probe = @splat(null),
    pub fn deinit(s: *Suite) void {
        for (&s.probes) |*p| if (p.*) |*probe| probe.deinit();
    }
    pub fn collect(s: *Suite, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (std.enums.values(Mutation), 0..) |m, i| {
            if (only) |needle| if (std.mem.indexOf(u8, name(m), needle) == null) continue;
            s.probes[i] = try Probe.init(a, device, queue, m);
            try kernels.append(a, s.probes[i].?.kernel());
        }
    }
};
