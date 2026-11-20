//! SAM3's flash attention tiles (flash.metal) on the tensor units against float64 softmax attention's derived range.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const vt = g.hooks(Probe);
const period = 67;
const u: f64 = 0x1p-24;

const Kernel = struct { name: []const u8, half: bool, d: usize, bk: usize, heads: usize, batch: usize, served: []const c.Shape };

/// The family's calls (attention.zig): FP32 trunk tiles and FP16 full grid at d 64, memory and concept heads at d 32.
const kernels_ = [_]Kernel{
    .{ .name = "sam3_flash_f32_d64_k32", .half = false, .d = 64, .bk = 32, .heads = 16, .batch = 9, .served = &.{
        .{ .rows = 576, .n = 1024, .k = 576 },
        .{ .rows = 65, .n = 1024, .k = 97 },
    } },
    .{ .name = "sam3_flash_f16_d64_k32", .half = true, .d = 64, .bk = 32, .heads = 16, .batch = 1, .served = &.{
        .{ .rows = 5184, .n = 1024, .k = 5184 },
        .{ .rows = 65, .n = 1024, .k = 97 },
    } },
    .{ .name = "sam3_flash_f32_d32_k32", .half = false, .d = 32, .bk = 32, .heads = 8, .batch = 1, .served = &.{
        .{ .rows = 5184, .n = 256, .k = 33 },
        .{ .rows = 33, .n = 256, .k = 5184 },
        .{ .rows = 201, .n = 256, .k = 201 },
        .{ .rows = 201, .n = 256, .k = 33 },
        .{ .rows = 33, .n = 256, .k = 33 },
        .{ .rows = 5183, .n = 256, .k = 5188 },
    } },
    .{ .name = "sam3_flash_f16_d32_k64", .half = true, .d = 32, .bk = 64, .heads = 8, .batch = 1, .served = &.{
        .{ .rows = 5184, .n = 256, .k = 5188 },
        .{ .rows = 5184, .n = 256, .k = 36544 },
        .{ .rows = 65, .n = 256, .k = 97 },
    } },
};

const control_from = "acc[d][e] = acc[d][e] * alpha[e >> 2];";

/// The engine's SAM library; the control keeps each block's accumulator unscaled when the row maximum moves.
pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const raw = try sources.tensor_library.source(a, sources.sam3_source);
    if (!control) return raw;
    defer a.free(raw);
    if (std.mem.count(u8, raw, control_from) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, control_from, "acc[d][e] = acc[d][e];");
}

pub fn limitText(a: std.mem.Allocator, _: []const u8, control: bool) ![]u8 {
    return text(a, control);
}

fn label(i: usize, control: bool) []const u8 {
    return switch (i) {
        inline 0...kernels_.len - 1 => |k| if (control) "control: sam3 flash unscaled accumulator " ++ kernels_[k].name else "metal sam3 flash " ++ kernels_[k].name,
        else => unreachable,
    };
}

fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    const at = if (j < half) j else j - half;
    if (j >= half * 2 or at >= 256) return 0;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    return if (j < half) big + @as(f32, if (at < 8 and at % 2 == 0) 1 else 0) else -big;
}

fn depth(s: c.Shape) usize {
    return 64 + s.k;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kernel: Kernel,
    which: usize,
    control: bool,
    shape: c.Shape = undefined,
    stress: bool = false,
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []f32 = &.{},
    half: []f64 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.which, p.control),
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .out = if (p.kernel.half) .f16 else .f32,
            .served = p.kernel.served,
            .tiles = .{ .{ .size = 64 }, .{}, .{ .size = p.kernel.bk } },
        };
    }

    fn reset(p: *Probe) void {
        for ([_][]f32{ p.q, p.k, p.v }) |x| p.a.free(x);
        p.a.free(p.half);
        p.q = &.{};
        p.k = &.{};
        p.v = &.{};
        p.half = &.{};
        p.rig.reset();
    }

    fn operand(p: *const Probe, x: f32) f32 {
        return if (p.kernel.half) @as(f16, @floatCast(x)) else x;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const kn = p.kernel;
        if (s.rows == 0 or s.k == 0 or s.n != kn.heads * kn.d) return false;
        p.reset();
        p.shape = s;
        p.stress = fill == .stress;
        p.q = try p.a.alloc(f32, p.count());
        p.k = try p.a.alloc(f32, kn.batch * s.k * s.n);
        p.v = try p.a.alloc(f32, p.k.len);
        p.half = try p.a.alloc(f64, p.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..kn.batch) |b| for (0..s.rows) |r| {
            const row = p.q[(b * s.rows + r) * s.n ..][0..s.n];
            if (r < period) {
                for (row) |*x| x.* = if (p.stress) 0 else p.operand(rng.float(f32) * 2 - 1);
            } else @memcpy(row, p.q[(b * s.rows + r % period) * s.n ..][0..s.n]);
        };
        for (0..kn.batch) |b| for (0..s.k) |j| for (0..s.n) |e| {
            const i = (b * s.k + j) * s.n + e;
            p.k[i] = if (p.stress) 0 else p.operand(rng.float(f32) * 2 - 1);
            p.v[i] = if (p.stress) stressValue(j, s.k) else p.operand(rng.float(f32) * 2 - 1);
        };
        if (fill == .nonfinite) {
            p.q[0] = std.math.nan(f32);
            p.q[s.n] = std.math.inf(f32);
            p.q[2 * s.n] = -std.math.inf(f32);
            p.q[3 * s.n] = std.math.inf(f32);
            p.q[3 * s.n + 1] = -std.math.inf(f32);
            p.v[0] = std.math.inf(f32);
            p.v[1] = -std.math.inf(f32);
            p.v[2] = std.math.inf(f32);
            p.v[(if (s.k > 1) s.n else 0) + 2] = -std.math.inf(f32);
            p.v[3] = std.math.nan(f32);
        }
        for ([_][]const f32{ p.q, p.k, p.v }, &p.ins) |values, *input| {
            if (!p.kernel.half) {
                input.* = try p.rig.input(std.mem.sliceAsBytes(values));
                continue;
            }
            const h = try p.a.alloc(f16, values.len);
            defer p.a.free(h);
            for (h, values) |*y, x| y.* = @floatCast(x);
            input.* = try p.rig.input(std.mem.sliceAsBytes(h));
        }
        p.out = try p.rig.output(p.count() * @as(usize, if (kn.half) 2 else 4));
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const kn = p.kernel;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.ins, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.setValue([4]u32{ @intCast(s.rows), @intCast(s.k), @intCast(kn.heads), @intCast(kn.batch) }, 4);
        pass.e.dispatchThreads(mtl.Size.of((s.rows + 63) / 64 * 32, kn.heads * 4, kn.batch), mtl.Size.of(32, 4, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.kernel.batch * p.shape.rows * p.shape.n;
    }

    /// FP32 operands: three 11-bit parts, six products a k16 step; FP16: one product, P rounded to half.
    fn form(p: *const Probe) ar.Form {
        const kn = p.kernel;
        const blocks: f64 = @floatFromInt((p.shape.k + kn.bk - 1) / kn.bk);
        const steps: f64 = @floatFromInt(kn.bk / 16);
        const parts: f64 = if (kn.half) 1 else 6;
        return .{
            .score = @as(f64, @floatFromInt(kn.d / 16)) * parts * ar.mma_step,
            .score_round = 4,
            .fast_exp = false,
            .chunk_exp = false,
            .p_rel = if (kn.half) 0x1p-11 else 0,
            .num = (steps * parts * ar.mma_step + 1) * blocks,
            .den = 3 * blocks + 3,
            .div = 4,
            .exp_ulps = .{ 4 * blocks, 2 * std.math.log2e + 0.5 },
        };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const kn = p.kernel;
        const width = kn.d;
        const kv = try p.a.alloc(f64, 2 * s.k * width + 2 * s.k);
        defer p.a.free(kv);
        const keys = kv[0 .. s.k * width];
        const values = kv[s.k * width ..][0 .. s.k * width];
        const weights = kv[2 * s.k * width ..][0..s.k];
        const tau = kv[2 * s.k * width + s.k ..];
        var query: [64]f64 = undefined;
        const f = p.form();
        for (0..kn.batch) |b| for (0..kn.heads) |h| {
            var vmax: f64 = 0;
            for (0..s.k) |j| for (0..width) |e| {
                const src = (b * s.k + j) * s.n + h * width + e;
                keys[j * width + e] = p.k[src];
                values[j * width + e] = p.v[src];
                if (std.math.isFinite(values[j * width + e])) vmax = @max(vmax, @abs(values[j * width + e]));
            };
            // P's subnormal halves are off by up to 2^-25 each; l >= 1, as the row's largest weight is exp2(0)
            const flush: f64 = if (kn.half) @as(f64, @floatFromInt(s.k)) * 0x1p-25 * vmax * (1 + 0x1p-10) else 0;
            for (0..s.rows) |r| {
                const at = (b * s.rows + r) * s.n + h * width;
                if (r >= period) {
                    const from = (b * s.rows + r % period) * s.n + h * width;
                    @memcpy(want[at..][0..width], want[from..][0..width]);
                    @memcpy(p.half[at..][0..width], p.half[from..][0..width]);
                    continue;
                }
                for (query[0..width], p.q[at..][0..width]) |*x, q| x.* = q;
                ar.head(query[0..width], keys, values, s.k, width, 1 / @sqrt(@as(f64, @floatFromInt(width))), f, weights, tau, want[at..][0..width], p.half[at..][0..width]);
                for (p.half[at..][0..width]) |*x| x.* += flush;
            }
        };
        @memcpy(mag, p.half);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (want, p.half, allow) |w, half, *v| {
            const base = if (p.stress) c.boundExact(p.spec(), w) else half;
            // FP16 kernels store o in half: its rounding on top of the FP32 range
            v.* = if (p.kernel.half and !p.stress) base + 0x1p-11 * (@abs(w) + base) + 0x1p-25 else base;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try p.a.alloc(f64, p.shape.k);
        defer p.a.free(terms);
        for (terms, 0..) |*x, j| x.* = stressValue(j, terms.len);
        @memset(out, contract.rows.sumIn(acc, order, terms) / @as(f64, @floatFromInt(terms.len)));
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [2 * kernels_.len]?*Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        if (!device.tensorUnits()) return;
        for ([_]bool{ false, true }) |control| {
            var wanted = false;
            for (0..kernels_.len) |i| {
                if (only) |needle| if (std.mem.indexOf(u8, label(i, control), needle) == null) continue;
                wanted = true;
            }
            if (!wanted) continue;
            const body = try text(s.a, control);
            defer s.a.free(body);
            const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
            defer lib.deinit();
            for (kernels_, 0..) |kn, i| {
                if (only) |needle| if (std.mem.indexOf(u8, label(i, control), needle) == null) continue;
                const pipe = try mtl.Pipeline.init(device, lib, kn.name, false);
                errdefer pipe.deinit();
                const p = try s.a.create(Probe);
                errdefer s.a.destroy(p);
                try kernels.ensureUnusedCapacity(s.a, 1);
                p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .kernel = kn, .which = i, .control = control };
                s.probes[@as(usize, @intFromBool(control)) * kernels_.len + i] = p;
                kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
            }
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
    }
};
