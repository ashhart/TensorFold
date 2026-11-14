const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const vt = g.hooks(Probe);

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const first = std.mem.indexOf(u8, sources.sam3_source, "[[kernel, max_total_threads_per_threadgroup(256)]]\nvoid im_sam_mask_downsample").?;
    const end = std.mem.indexOfPos(u8, sources.sam3_source, first, "// NHWC activations").?;
    const raw = try std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\n", sources.sam3_source[first..end] });
    if (!control) return raw;
    defer a.free(raw);
    if (std.mem.count(u8, raw, "if(conditioning)") != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, "if(conditioning)", "if(false)");
}

fn depth(_: c.Shape) usize {
    return 19;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    conditioning: bool,
    control: bool,
    values: []f32 = &.{},
    weights: [16 * 32 * 9]f32 = undefined,
    biases: [16]f32 = undefined,
    input: rig_.Ref = undefined,
    weight: rig_.Ref = undefined,
    bias: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn label(conditioning: bool, control: bool) []const u8 {
        return if (control) "control: sam3 sparse mask omits conditioning" else if (conditioning)
            "metal sam3 sparse mask conditioned"
        else
            "metal sam3 sparse mask unconditioned";
    }

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = label(p.conditioning, p.control),
            .accum = .f32,
            .out = .f32,
            .depth = depth,
            .served = &.{.{ .rows = 576, .n = 576, .k = 16 }},
        };
    }

    pub fn reset(p: *Probe) void {
        p.a.free(p.values);
        p.values = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 576 or shape.n != 576 or shape.k != 16) return false;
        // This fixed-geometry operator still runs its complete grid when the harness caps gate rows.
        p.reset();
        p.values = try p.a.alloc(f32, 1152 * 1152);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (p.values, 0..) |*value, i| value.* = if (fill == .stress) switch (i % 3) {
            0 => 1023,
            1 => -1022,
            else => 0,
        } else random.float(f32) * 2 - 1;
        for (&p.weights) |*value| value.* = if (fill == .stress) 0 else (random.float(f32) * 2 - 1) / 8;
        for (&p.biases) |*value| value.* = if (fill == .stress) 0 else (random.float(f32) * 2 - 1) / 8;
        if (fill == .stress) {
            for (0..16) |channel| @memset(p.weights[channel * 32 * 9 ..][0..9], 1023);
        } else if (fill == .nonfinite) {
            p.values[0] = std.math.nan(f32);
            p.values[1152 * 100 + 100] = std.math.inf(f32);
            p.values[1152 * 100 + 101] = -std.math.inf(f32);
            p.values[1152 * 500 + 500] = -std.math.inf(f32);
            p.biases[13] = std.math.nan(f32);
            p.biases[14] = std.math.inf(f32);
            p.biases[15] = -std.math.inf(f32);
        } else {
            p.values[0] = @bitCast(@as(u32, 1));
            p.values[p.values.len - 1] = -0x1p-126;
        }
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.values));
        p.weight = try p.rig.input(std.mem.asBytes(&p.weights));
        p.bias = try p.rig.input(std.mem.asBytes(&p.biases));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for ([_]rig_.Ref{ p.input, p.weight, p.bias, p.out }, 0..) |ref, i| pass.e.setBuffer(ref.buf, ref.off, i);
        pass.e.setValue(@as(u32, @intFromBool(p.conditioning)), 4);
        pass.e.dispatchThreads(mtl.Size.of(p.count(), 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(_: *const Probe) usize {
        return 576 * 576 * 16;
    }

    fn terms(p: *const Probe, i: usize) [19]f64 {
        const channel = i % 16;
        const row = i / 16 / 576;
        const col = i / 16 % 576;
        var out: [19]f64 = @splat(0);
        for (0..3) |ky| for (0..3) |kx| {
            const y = @as(isize, @intCast(row * 2 + ky)) - 1;
            const x = @as(isize, @intCast(col * 2 + kx)) - 1;
            if (y < 0 or x < 0 or y >= 1152 or x >= 1152) continue;
            const tap = ky * 3 + kx;
            out[tap * 2] = @as(f64, p.values[@as(usize, @intCast(y * 1152 + x))]) * p.weights[channel * 32 * 9 + tap];
            if (p.conditioning) out[tap * 2 + 1] = p.weights[(channel * 32 + 16) * 9 + tap];
        };
        out[18] = p.biases[channel];
        return out;
    }

    pub fn reference(p: *Probe, want: []f64, magnitude: []f64) !void {
        for (want, magnitude, 0..) |*value, *mag, i| {
            value.* = 0;
            mag.* = 0;
            for (p.terms(i)) |term| {
                value.* += term;
                mag.* += @abs(term);
            }
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*value, i| value.* = contract.rows.sumIn(acc, order, &p.terms(i));
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [3]?Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&s.probes, 0..) |*entry, i| {
            const label = Probe.label(i != 0, i == 2);
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            const source = try text(s.a, i == 2);
            defer s.a.free(source);
            entry.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "im_sam_mask_downsample"), .conditioning = i != 0, .control = i == 2 };
            if (entry.*) |*probe| try kernels.append(s.a, .{ .spec = probe.spec(), .ctx = probe, .vt = &vt, .control = if (i == 2) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (&s.probes) |*entry| if (entry.*) |*p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
        };
    }
};
