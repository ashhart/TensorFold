//! Native fused boundary coefficients: guarded production dispatch, independent FP64 split and real GPU controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const mhc = tf.deepseek_v41.mhc;
const c = contract.spec;
const g = contract.gates;
const width = mhc.boundary_width;
const stream_width = 4 * width;
const slices = mhc.boundary_slices;
const parts = 25;
const base = "dsv41-native-boundary ";
const hc_eps: f32 = 1e-6;
const iterations = 20;

// Independent split equations for the supplied partials (unit inverse, scale and zero affine base).
fn split(comptime T: type, projected: [24]T) [24]T {
    @setFloatMode(.strict);
    const eps: T = hc_eps;
    var out: [24]T = undefined;
    for (0..4) |i| {
        out[i] = 1 / (1 + @exp(-projected[i])) + eps;
        out[4 + i] = 2 / (1 + @exp(-projected[4 + i]));
    }
    for (0..4) |r| {
        const off = 8 + r * 4;
        var high: T = -std.math.inf(T);
        for (0..4) |j| high = @max(high, projected[off + j]);
        for (0..4) |j| out[off + j] = @exp(projected[off + j] - high);
        const sum = (out[off] + out[off + 2]) + (out[off + 1] + out[off + 3]);
        for (0..4) |j| out[off + j] = out[off + j] / sum + eps;
    }
    for (0..iterations) |it| {
        if (it > 0) for (0..4) |r| {
            const off = 8 + r * 4;
            const sum = ((out[off] + out[off + 2]) + (out[off + 1] + out[off + 3])) + eps;
            for (0..4) |j| out[off + j] /= sum;
        };
        for (0..4) |j| {
            const off = 8 + j;
            const sum = ((out[off] + out[off + 8]) + (out[off + 4] + out[off + 12])) + eps;
            for (0..4) |r| out[off + r * 4] /= sum;
        }
    }
    return out;
}

fn depth(_: c.Shape) usize {
    return slices + 4 * 20;
}

fn text(a: std.mem.Allocator, ctl: u8) ![]u8 {
    const replacement: [2][]const u8 = switch (ctl) {
        1 => .{ "slice[t] = s;", "slice[t] = isfinite(s) ? s : 0.0f;" },
        2 => .{ "comb_out[r * kCopies * kCopies + lane] = comb;", "comb_out[r * kCopies * kCopies + lane] = e;" },
        3 => .{ "s += PART[(size_t(r) * kSlices + j) * kParts + t];", "s = float(half(s + PART[(size_t(r) * kSlices + j) * kParts + t]));" },
        4 => .{ "comb_out[r * kCopies * kCopies + lane] = comb;", "comb_out[r * kCopies * kCopies + lane + 1] = comb;" },
        else => unreachable,
    };
    const old, const new = replacement;
    if (std.mem.count(u8, sources.deepseek_v41_hc_boundary, old) != 1) return error.MissingBoundaryControlSite;
    return std.mem.replaceOwned(u8, a, sources.deepseek_v41_hc_boundary, old, new);
}

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    ctl: u8,
    name: []const u8,
    production: ?mhc.BoundaryPipelines = null,
    pipe: mtl.Pipeline = undefined,
    unfused: ?mhc.Pipelines = null,
    shape: c.Shape = undefined,
    partials: []f32 = undefined,
    inputs: [6]rig_.Ref = undefined,
    outputs: [3]rig_.Ref = undefined,
    normed: rig_.Ref = undefined,
    mixes: rig_.Ref = undefined,
    inverse: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8, name: []const u8) !Proj {
        var p = Proj{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .ctl = ctl, .name = name };
        if (ctl == 0) {
            p.production = try mhc.BoundaryPipelines.load(device);
            errdefer p.production.?.deinit();
            p.unfused = try mhc.Pipelines.load(device);
        } else {
            const source = try text(a, ctl);
            defer a.free(source);
            var options = mtl.CompileOptions.mlx();
            options.math_functions = .precise;
            const lib = try mtl.Library.fromSource(device, source, options);
            defer lib.deinit();
            p.pipe = try mtl.Pipeline.init(device, lib, "tf_hc_boundary_mix", false);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        if (p.production) |*v| v.deinit() else p.pipe.deinit();
        if (p.unfused) |*v| v.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const hooks = struct {
            const normal = g.hooks(Proj);
            const control = blk: {
                var v = g.hooks(Proj);
                v.replaced = null;
                break :blk v;
            };
        };
        return .{ .spec = .{
            .name = p.name,
            .accum = .f32,
            .out = .f32,
            .depth = depth,
            .ops = 260,
            .served = &.{ .{ .rows = 7, .n = 1, .k = stream_width }, .{ .rows = 1, .n = 1, .k = stream_width }, .{ .rows = 16, .n = 1, .k = stream_width } },
            .sums = .{ .why = "twenty FP32 partials in slice order, then the pinned row/column Sinkhorn butterflies" },
            .replaces = if (p.ctl == 0) "dsv41_hc_split on identical reduced arguments" else null,
        }, .ctx = p, .vt = if (p.ctl == 0) &hooks.normal else &hooks.control, .control = switch (p.ctl) {
            0 => null,
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            else => .repeat,
        } };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        @setFloatMode(.strict);
        if (shape.k != stream_width or shape.n != 1 or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        p.partials = try a.alloc(f32, shape.rows * slices * parts);
        @memset(p.partials, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| for (0..24) |q| {
            for (0..slices) |j| p.partials[(r * slices + j) * parts + q] = (rng.float(f32) * 2 - 1) * 0.25;
            if (fill == .stress) {
                for (0..slices) |j| p.partials[(r * slices + j) * parts + q] = 0;
                p.partials[r * slices * parts + q] = 4096;
                p.partials[(r * slices + 1) * parts + q] = @floatFromInt(1 + (q + r) % 3);
                p.partials[(r * slices + 2) * parts + q] = -4096;
            }
        };
        if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |v, r| if (r < shape.rows) {
                p.partials[r * slices * parts + (r * 5) % 24] = v;
            };
        }
        const zero = try a.alloc(u8, shape.rows * stream_width * 2);
        @memset(zero, 0);
        const scale = [_]f32{ 1, 1, 1 };
        const bias: [24]f32 = @splat(0);
        p.inputs = .{
            try p.rig.input(zero),                          try p.rig.input(std.mem.sliceAsBytes(p.partials)),
            try p.rig.input(std.mem.sliceAsBytes(&bias)),   try p.rig.input(std.mem.sliceAsBytes(&scale)),
            try p.rig.input(zero[0 .. shape.rows * 4 * 4]), try p.rig.input(zero[0 .. width * 2]),
        };
        p.outputs = .{ try p.rig.output(shape.rows * 4 * 4), try p.rig.output(shape.rows * 4 * 4), try p.rig.output(shape.rows * 16 * 4) };
        p.normed = try p.rig.scratch(shape.rows * width * 2);
        const reduced = try a.alloc(f32, shape.rows * 24);
        for (0..shape.rows) |r| for (0..24) |q| {
            var s: f32 = 0;
            for (0..slices) |j| s += p.partials[(r * slices + j) * parts + q];
            reduced[r * 24 + q] = s;
        };
        p.mixes = try p.rig.input(std.mem.sliceAsBytes(reduced));
        const inverse = try a.alloc(f32, shape.rows);
        @memset(inverse, 1);
        p.inverse = try p.rig.input(std.mem.sliceAsBytes(inverse));
        return true;
    }

    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, first: usize, rows: usize) void {
        e.setPipeline(if (p.production) |v| v.boundary else p.pipe);
        inline for (0..6) |i| e.setBuffer(p.inputs[i].buf, p.inputs[i].off + switch (i) {
            0 => first * stream_width * 2,
            1 => first * slices * parts * 4,
            4 => first * 4 * 4,
            else => 0,
        }, i);
        const args = mhc.BoundaryArgs{ .eps = 1, .hc_eps = hc_eps, .iters = iterations };
        e.setBytes(std.mem.asBytes(&args), 6);
        inline for (p.outputs, .{ 4, 4, 16 }, 7..) |v, n, i| e.setBuffer(v.buf, v.off + first * n * 4, i);
        e.setBuffer(p.normed.buf, p.normed.off + first * width * 2, 10);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.shape.rows * 24;
    }

    pub fn replaced(p: *Proj) ![]const u8 {
        const pass = p.rig.begin();
        mhc.encodeSplit(pass.e, &p.unfused.?, view(p.mixes), view(p.inverse), view(p.inputs[3]), view(p.inputs[2]), view(p.outputs[0]), view(p.outputs[1]), view(p.outputs[2]), .{ .rows = @intCast(p.shape.rows), .eps = hc_eps });
        try p.rig.finish(pass);
        return p.rig.outputBytes();
    }

    fn view(v: rig_.Ref) mhc.View {
        return .{ .buffer = v.buf, .offset = v.off };
    }

    fn calculate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) void {
        for (0..p.shape.rows) |r| {
            var projected: [24]f64 = undefined;
            for (&projected, 0..) |*v, q| {
                var terms: [slices]f64 = undefined;
                for (&terms, 0..) |*t, j| t.* = p.partials[(r * slices + j) * parts + q];
                v.* = contract.rows.sumIn(acc, order, &terms);
            }
            if (acc == .none) {
                p.writeMix(f64, split(f64, projected), r, out);
            } else {
                // Gate 4 varies the partial-sum accumulator; the split keeps its FP32 source boundaries.
                var projected32: [24]f32 = undefined;
                for (&projected32, projected) |*v, x| v.* = @floatCast(x);
                p.writeMix(f32, split(f32, projected32), r, out);
            }
        }
    }

    fn writeMix(p: *Proj, comptime T: type, mix: [24]T, r: usize, out: []f64) void {
        inline for (.{ 0, 4, 8 }, .{ 4, 4, 16 }) |offset, n| {
            for (mix[offset..][0..n], 0..) |v, i| out[p.shape.rows * offset + r * n + i] = v;
        }
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        p.calculate(.none, .sequential, want);
        for (mag, want) |*m, v| m.* = @max(1, @abs(v));
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        p.calculate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        const names = [_][]const u8{ base ++ "coefficients tf_hc_boundary_mix", base ++ "control: flush nonfinite partials", base ++ "control: omit Sinkhorn", base ++ "control: fp16 partial accumulator", base ++ "control: comb overrun" };
        for (names, 0..) |name, ctl| {
            if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) continue;
            const p = try a.create(Proj);
            p.* = Proj.init(a, device, queue, @intCast(ctl), name) catch |err| {
                a.destroy(p);
                return err;
            };
            s.probes.append(a, p) catch |err| {
                p.deinit();
                a.destroy(p);
                return err;
            };
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            if (!try p.prepare(.{ .rows = 33, .n = 1, .k = stream_width }, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, 11 }) |step| {
                inline for (p.outputs, .{ 4, 4, 16 }) |v, n|
                    @memset(p.rig.bytes(v, p.shape.rows * n * 4), 0xff);
                const pass = p.rig.begin();
                var first: usize = 0;
                while (first < p.shape.rows) : (first += step)
                    p.encode(pass.e, first, @min(step, p.shape.rows - first));
                try p.rig.finish(pass);
                if (!try p.rig.intact() or !std.mem.eql(u8, whole, try p.rig.outputBytes())) bad += 1;
            }
            std.debug.print("{s} row invariance {s}: 33 rows split at 1,7,11; {d} differences\n", .{ if (bad == 0) "PASS" else "FAIL", p.name, bad });
        }
        return bad;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
