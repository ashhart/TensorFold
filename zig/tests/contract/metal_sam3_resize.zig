const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const vt = g.hooks(Probe);
const nonfinite_control_name = "control: sam3 resize lerp exceptional inputs";

pub fn text(a: std.mem.Allocator, control: bool) ![]u8 {
    const first = std.mem.indexOf(u8, sources.sam3_source, "[[kernel, max_total_threads_per_threadgroup(256)]]\nvoid im_sam_resize").?;
    const end = std.mem.indexOfPos(u8, sources.sam3_source, first, "kernel void im_sam_group_norm").?;
    const raw = try std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\n", sources.sam3_source[first..end] });
    if (!control) return raw;
    defer a.free(raw);
    return std.mem.replaceOwned(u8, a, raw, "if(p.w==0)", "if(p.w==1)");
}

pub fn nonfiniteText(a: std.mem.Allocator) ![]u8 {
    const raw = try text(a, false);
    defer a.free(raw);
    const from = "if(any((as_type<uint4>(samples)&0x7fffffffu)>=0x7e800000u))";
    if (std.mem.count(u8, raw, from) != 1) return error.ControlTextNotFound;
    return std.mem.replaceOwned(u8, a, raw, from, "if(false)");
}

fn depth(_: c.Shape) usize {
    return 8;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mode: usize,
    control: bool,
    nonfinite_control: bool = false,
    stress: bool = false,
    shape: c.Shape = undefined,
    values: []f32 = &.{},
    input: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn label(mode: usize, control: bool) []const u8 {
        if (control) return "control: sam3 resize chooses wrong interpolation";
        return switch (mode) {
            0 => "metal sam3 resize bilinear image",
            1 => "metal sam3 resize bilinear masks",
            else => "metal sam3 resize nearest masks",
        };
    }

    fn spec(p: *const Probe) c.Spec {
        return .{
            .name = if (p.nonfinite_control) nonfinite_control_name else label(p.mode, p.control),
            .accum = if (p.mode == 2) .none else .f32,
            .out = .f32,
            .depth = depth,
            .tiles = .{ .{ .size = 256 }, .{ .size = 256 }, .{} },
            .served = if (p.mode == 0) &.{.{ .rows = 1008, .n = 1008, .k = 3 }} else if (p.mode == 1) &.{
                .{ .rows = 432, .n = 768, .k = 1 },
                .{ .rows = 1152, .n = 1152, .k = 1 },
                .{ .rows = 72, .n = 72, .k = 1 },
                .{ .rows = 432, .n = 768, .k = 16 },
                .{ .rows = 1152, .n = 1152, .k = 16 },
                .{ .rows = 72, .n = 72, .k = 16 },
                .{ .rows = 1152, .n = 1152, .k = 32 },
            } else &.{
                .{ .rows = 432, .n = 768, .k = 1 },
                .{ .rows = 1152, .n = 1152, .k = 1 },
                .{ .rows = 72, .n = 72, .k = 256 },
                .{ .rows = 144, .n = 144, .k = 256 },
                .{ .rows = 288, .n = 288, .k = 256 },
            },
        };
    }

    fn geometry(p: *const Probe) [2]usize {
        if (p.stress) return .{ 2 * p.shape.rows, 2 * p.shape.n };
        return if (p.mode == 0) .{ 432, 768 } else if (p.mode == 1 and p.shape.k == 32) .{ 1008, 1008 } else if (p.mode == 2 and p.shape.k == 256) .{ 72, 72 } else .{ 288, 288 };
    }

    pub fn reset(p: *Probe) void {
        p.a.free(p.values);
        p.values = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or switch (p.mode) {
            0 => shape.k != 3,
            1 => shape.k != 1 and shape.k != 16 and shape.k != 32,
            else => shape.k != 1 and shape.k != 256,
        }) return false;
        p.reset();
        p.shape = shape;
        p.stress = fill == .stress;
        const h, const w = p.geometry();
        p.values = try p.a.alloc(f32, h * w * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        for (p.values, 0..) |*value, i| value.* = if (p.stress) (if ((i / shape.k) % 2 == 0) 16257 else -16256) else prng.random().float(f32) * 2 - 1;
        if (fill == .nonfinite) {
            const y: usize = if (p.mode == 2) 0 else @intFromFloat(@max(0, @mulAdd(f32, 0.5, @as(f32, @floatFromInt(h)) / @as(f32, @floatFromInt(shape.rows)), -0.5)));
            const x: usize = if (p.mode == 2) 0 else @intFromFloat(@max(0, @mulAdd(f32, 0.5, @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(shape.n)), -0.5)));
            p.values[(y * w + x) * shape.k] = std.math.nan(f32);
            p.values[(y * w + @min(x + 1, w - 1)) * shape.k] = std.math.inf(f32);
            p.values[(@min(y + 1, h - 1) * w + x) * shape.k] = -std.math.inf(f32);
        } else if (fill == .random) {
            p.values[0] = std.math.floatMax(f32);
            p.values[shape.k] = -std.math.floatMax(f32);
            p.values[w * shape.k] = @bitCast(@as(u32, 1));
        }
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.values));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const h, const w = p.geometry();
        const s = p.shape;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.input.buf, p.input.off, 0);
        pass.e.setBuffer(p.out.buf, p.out.off, 1);
        pass.e.setValue([4]u32{ @intCast(h), @intCast(w), @intCast(s.k), @intFromBool(p.mode != 2) }, 2);
        pass.e.setValue([2]u32{ @intCast(s.rows), @intCast(s.n) }, 3);
        pass.e.setValue([2]f32{ @as(f32, @floatFromInt(h)) / @as(f32, @floatFromInt(s.rows)), @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(s.n)) }, 4);
        pass.e.dispatchThreads(mtl.Size.of(p.count(), 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n * p.shape.k;
    }

    fn terms(p: *const Probe, index: usize) [4]f64 {
        const s = p.shape;
        const h, const w = p.geometry();
        const y = index / s.k / s.n;
        const x = index / s.k % s.n;
        const channel = index % s.k;
        if (p.mode == 2) return .{ p.values[((y * h / s.rows) * w + x * w / s.n) * s.k + channel], 0, 0, 0 };
        // Coordinates retain the explicit FP32 scale and fused center transform.
        const scale_y = @as(f32, @floatFromInt(h)) / @as(f32, @floatFromInt(s.rows));
        const scale_x = @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(s.n));
        const sy = @max(0, @mulAdd(f32, @as(f32, @floatFromInt(y)) + 0.5, scale_y, -0.5));
        const sx = @max(0, @mulAdd(f32, @as(f32, @floatFromInt(x)) + 0.5, scale_x, -0.5));
        const y0 = @min(@as(usize, @intFromFloat(sy)), h - 1);
        const x0 = @min(@as(usize, @intFromFloat(sx)), w - 1);
        const y1 = @min(y0 + 1, h - 1);
        const x1 = @min(x0 + 1, w - 1);
        const fy: f64 = sy - @as(f32, @floatFromInt(y0));
        const fx: f64 = sx - @as(f32, @floatFromInt(x0));
        return .{
            p.values[(y0 * w + x0) * s.k + channel] * (1 - fx) * (1 - fy),
            p.values[(y0 * w + x1) * s.k + channel] * fx * (1 - fy),
            p.values[(y1 * w + x0) * s.k + channel] * (1 - fx) * fy,
            p.values[(y1 * w + x1) * s.k + channel] * fx * fy,
        };
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
    probes: [5]?Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&s.probes, 0..) |*entry, i| {
            const mode = if (i >= 3) 0 else i;
            const label = if (i == 4) nonfinite_control_name else Probe.label(mode, i == 3);
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            const source = if (i == 4) try nonfiniteText(s.a) else try text(s.a, i == 3);
            defer s.a.free(source);
            entry.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, source, "im_sam_resize"), .mode = mode, .control = i >= 3, .nonfinite_control = i == 4 };
            if (entry.*) |*probe| try kernels.append(s.a, .{ .spec = probe.spec(), .ctx = probe, .vt = &vt, .control = if (i == 4) .nonfinite else if (i == 3) .shapes else null });
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
