//! SAM3's arithmetic kernels of layout.metal, masks.metal and convolution.metal against float64.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const squash = @import("metal_sam3_arithmetic_squash.zig");
const conv = @import("metal_sam3_arithmetic_conv.zig");

const u: f64 = 0x1p-24;

pub const File = enum { layout, convolution, masks };

const Edit = struct { function: []const u8, file: File, from: []const u8, to: []const u8 };

/// Each kernel's planted control: one unique edit of its source.
const edits = [_]Edit{
    .{ .function = "sam3_bias_rows", .file = .layout, .from = "bias[i % shape.y]", .to = "bias[(i + 1u) % shape.y]" },
    .{ .function = "sam3_scale", .file = .layout, .from = "dst[i] = src[i] * p.factor;", .to = "dst[i] = src[i];" },
    .{ .function = "sam3_relu", .file = .layout, .from = "max(src[i], 0.0f)", .to = "fabs(src[i])" },
    .{ .function = "sam3_sigmoid", .file = .layout, .from = "fast::exp(-src[i])", .to = "fast::exp(src[i])" },
    .{ .function = "sam3_mask_squash", .file = .masks, .from = "squashed * p.gain + p.offset", .to = "squashed * p.gain" },
    .{ .function = "sam3_depthwise", .file = .convolution, .from = "weight[(channel * k + ky) * k + kx]", .to = "weight[(channel * k + kx) * k + ky]" },
    .{ .function = "sam3_group_norm", .file = .convolution, .from = "half_width > 0u;", .to = "half_width > 1u;" },
};

/// One kernel file of the SAM library, cut from the baseline source at its include (it holds no tensor ops).
fn segment(file: File) ![]const u8 {
    if (file == .masks) return sources.sam3_masks;
    const all = sources.sam3_baseline_source;
    const head = "#include <metal_stdlib>";
    const marker = if (file == .layout) "struct Sam3Scale " else "struct Sam3ConvShape ";
    if (std.mem.count(u8, all, marker) != 1) return error.SamSourceNotFound;
    const at = std.mem.indexOf(u8, all, marker).?;
    const start = std.mem.lastIndexOf(u8, all[0..at], head) orelse return error.SamSourceNotFound;
    return all[start .. std.mem.indexOfPos(u8, all, at, head) orelse all.len];
}

fn editOf(function: []const u8) !Edit {
    for (edits) |e| if (std.mem.eql(u8, e.function, function)) return e;
    return error.UnknownLaunch;
}

/// The file holding `function`, with its planted control applied when asked.
pub fn text(a: std.mem.Allocator, function: []const u8, control: bool) ![]u8 {
    if (std.mem.eql(u8, function, "sam3_copy")) return a.dupe(u8, try segment(.layout));
    const e = try editOf(function);
    const raw = try segment(e.file);
    if (!control) return a.dupe(u8, raw);
    if (std.mem.count(u8, raw, e.from) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, e.from, e.to);
}

pub fn limitText(a: std.mem.Allocator, name: []const u8, control: bool) ![]u8 {
    return text(a, name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..], control);
}

pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, function: []const u8, control: bool) !mtl.Pipeline {
    const body = try text(a, function, control);
    defer a.free(body);
    return rig_.pipeline(device, body, function);
}

const Kind = enum { bias, scale, relu, sigmoid };
const factor: f32 = 0.1;

fn oneRounding(_: c.Shape) usize {
    return 1;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// x + bias in place (conv outputs after the core product), x * 0.1, max(x, 0) and the fast sigmoid.
const Unary = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    control: bool,
    pipe: mtl.Pipeline,
    copy: ?mtl.Pipeline = null,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    b: []f32 = &.{},
    input: rig_.Ref = undefined,
    bias: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn label(kind: Kind, control: bool) []const u8 {
        return switch (kind) {
            .bias => if (control) "control: sam3 arithmetic bias rows reads the next bias" else "metal sam3 arithmetic bias rows",
            .scale => if (control) "control: sam3 arithmetic scale drops its factor" else "metal sam3 arithmetic scale",
            .relu => if (control) "control: sam3 arithmetic relu as fabs" else "metal sam3 arithmetic relu",
            .sigmoid => if (control) "control: sam3 arithmetic sigmoid of -x" else "metal sam3 arithmetic sigmoid",
        };
    }

    fn function(kind: Kind) []const u8 {
        return switch (kind) {
            .bias => "sam3_bias_rows",
            .scale => "sam3_scale",
            .relu => "sam3_relu",
            .sigmoid => "sam3_sigmoid",
        };
    }

    fn spec(p: *const Unary) c.Spec {
        return .{
            .name = label(p.kind, p.control),
            .accum = if (p.kind == .bias) .f32 else .none,
            .depth = if (p.kind == .bias) oneRounding else noSums,
            .out = .f32,
            .in_place = p.kind == .bias,
            .nan = if (p.kind == .relu) .{ .stated = "max(x, 0); a NaN input passes through bit for bit, as float64 keeps it" } else .propagate,
            .tiles = .{ .{ .size = 256 }, .{}, .{} },
            .served = switch (p.kind) {
                // Conv outputs per chunk: patch embedding, 3x3 over 256 channels, 1x1 over 1024, the mask encoder.
                .bias => &.{ .{ .rows = 5184, .n = 1024, .k = 1 }, .{ .rows = 3584, .n = 256, .k = 1 }, .{ .rows = 8192, .n = 256, .k = 1 }, .{ .rows = 16384, .n = 16, .k = 1 } },
                // The memory reader's grid positions times 0.1.
                .scale => &.{.{ .rows = 5184, .n = 256, .k = 1 }},
                // The pixel decoder's group-normed maps and the query MLPs.
                .relu => &.{ .{ .rows = 20736, .n = 256, .k = 1 }, .{ .rows = 82944, .n = 256, .k = 1 }, .{ .rows = 200, .n = 256, .k = 1 }, .{ .rows = 200, .n = 2048, .k = 1 } },
                // Reference boxes and the mask head's object scores.
                .sigmoid => &.{ .{ .rows = 200, .n = 4, .k = 1 }, .{ .rows = 1, .n = 16, .k = 1 } },
            },
        };
    }

    fn reset(p: *Unary) void {
        p.a.free(p.x);
        p.a.free(p.b);
        p.x = &.{};
        p.b = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Unary, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        p.reset();
        p.shape = shape;
        p.x = try p.a.alloc(f32, shape.rows * shape.n);
        p.b = try p.a.alloc(f32, if (p.kind == .bias) shape.n else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const span: f32 = if (p.kind == .sigmoid) 32 else 8;
        for (p.x) |*v| v.* = (rng.float(f32) - 0.5) * span;
        for (p.b) |*v| v.* = (rng.float(f32) - 0.5) * 8;
        if (fill == .stress) {
            // Exact in fp32 in any order; fp16 overflows and bf16 drops the 1.
            @memset(p.x, 0x1p20 + 1);
            @memset(p.b, -0x1p20);
        } else if (fill == .nonfinite) {
            p.x[0] = std.math.nan(f32);
            p.x[1] = std.math.inf(f32);
            p.x[2] = -std.math.inf(f32);
            if (p.kind == .bias and p.b.len > 2) p.b[p.b.len - 1] = -std.math.inf(f32);
        } else {
            p.x[p.x.len - 1] = if (p.kind == .sigmoid) 80 else std.math.floatMax(f32) / 2;
            if (p.x.len > 1) p.x[p.x.len - 2] = if (p.kind == .sigmoid) -80 else -0.0;
        }
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.x));
        if (p.kind == .bias) p.bias = try p.rig.input(std.mem.sliceAsBytes(p.b));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Unary) !void {
        const n: u32 = @intCast(p.count());
        const pass = p.rig.begin();
        const grid = mtl.Size.of(p.count(), 1, 1);
        const group = mtl.Size.of(256, 1, 1);
        if (p.kind == .bias) {
            // The kernel adds in place: restore x from its pristine copy first, so every run sees the same x.
            pass.e.setPipeline(p.copy.?);
            pass.e.setBuffer(p.input.buf, p.input.off, 0);
            pass.e.setBuffer(p.out.buf, p.out.off, 1);
            pass.e.setValue(n, 2);
            pass.e.dispatchThreads(grid, group);
            pass.e.barrier();
            pass.e.setPipeline(p.pipe);
            pass.e.setBuffer(p.out.buf, p.out.off, 0);
            pass.e.setBuffer(p.bias.buf, p.bias.off, 1);
            pass.e.setValue([2]u32{ n, @intCast(p.shape.n) }, 2);
        } else {
            pass.e.setPipeline(p.pipe);
            pass.e.setBuffer(p.input.buf, p.input.off, 0);
            pass.e.setBuffer(p.out.buf, p.out.off, 1);
            if (p.kind == .scale) pass.e.setValue(extern struct { count: u32, factor: f32 }{ .count = n, .factor = factor }, 2) else pass.e.setValue(n, 2);
        }
        pass.e.dispatchThreads(grid, group);
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Unary) usize {
        return p.shape.rows * p.shape.n;
    }

    fn want(p: *const Unary, i: usize) f64 {
        const x: f64 = p.x[i];
        return switch (p.kind) {
            .bias => x + p.b[i % p.shape.n],
            .scale => x * factor,
            .relu => if (std.math.isNan(x)) x else @max(x, 0),
            .sigmoid => 1 / (1 + @exp(-x)),
        };
    }

    pub fn reference(p: *Unary, out: []f64, mag: []f64) !void {
        for (out, mag, 0..) |*w, *m, i| {
            w.* = p.want(i);
            m.* = if (p.kind == .bias) @abs(@as(f64, p.x[i])) + @abs(@as(f64, p.b[i % p.shape.n])) else @abs(w.*);
        }
    }

    /// The sigmoid's fast exp as (40 + 2|x|) ulps (core/attention_range.zig's model), then the add and a 2-ulp divide.
    pub fn bounds(p: *Unary, out: []const f64, _: []const f64, allow: []f64) !void {
        if (p.kind != .sigmoid) return;
        for (out, allow, 0..) |w, *l, i| {
            if (!std.math.isFinite(w)) continue;
            const x: f64 = p.x[i];
            const e = @exp(-x);
            const delta = (40 + 2 * @abs(x)) * 0x1p-23;
            const share = if (std.math.isInf(e)) 1 else e / (1 + e);
            const rel = (delta * share + 5 * u) / (1 - delta - 5 * u);
            l.* = @abs(w) * rel * (1 + 0x1p-20) + 0x1p-149;
        }
    }

    pub fn simulate(p: *Unary, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*w, i| w.* = contract.rows.sumIn(acc, order, &[2]f64{ p.x[i], p.b[i % p.shape.n] });
    }

    fn deinit(p: *Unary) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
        if (p.copy) |pipe| pipe.deinit();
    }
};

/// The relu before NaN passed through, the baseline of every non-NaN output's bits.
const relu_guard = "(as_type<uint>(src[i]) & 0x7fffffffu) > 0x7f800000u ? src[i] : max(src[i], 0.0f)";

fn previousRelu(a: std.mem.Allocator, device: mtl.Device) !mtl.Pipeline {
    const raw = try segment(.layout);
    if (std.mem.count(u8, raw, relu_guard) != 1) return error.ControlTextNotFoundOnce;
    const body = try std.mem.replaceOwned(u8, a, raw, relu_guard, "max(src[i], 0.0f)");
    defer a.free(body);
    return rig_.pipeline(device, body, "sam3_relu");
}

/// relu checked bit for bit: a NaN input's own bits, otherwise max(x, 0) and the previous kernel's bits (signed zeros).
const Relu = struct {
    inner: Unary,
    rig: *rig_.Rig = undefined,
    previous: mtl.Pipeline,
    base: rig_.Ref = undefined,

    pub fn prepare(r: *Relu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try r.inner.prepare(shape, fill, seed)) return false;
        if (fill == .random) for (r.inner.x, 0..) |*x, i| switch (i % 61) {
            3 => x.* = 0.0,
            5 => x.* = -0.0,
            7 => x.* = @bitCast(@as(u32, 0x80000001)),
            else => {},
        };
        if (fill != .stress) r.inner.x[r.inner.x.len / 2] = @bitCast(@as(u32, 0xffc12345));
        @memcpy(r.rig.bytes(r.inner.input, r.inner.x.len * 4), std.mem.sliceAsBytes(r.inner.x));
        r.base = try r.rig.scratch(r.count() * 4);
        return true;
    }
    pub fn run(r: *Relu) !void {
        return r.inner.run();
    }
    pub fn count(r: *const Relu) usize {
        return r.inner.count();
    }
    pub fn reference(r: *Relu, out: []f64, mag: []f64) !void {
        return r.inner.reference(out, mag);
    }

    pub fn check(r: *Relu, _: g.Gate) !?[]const u8 {
        const pass = r.rig.begin();
        pass.e.setPipeline(r.previous);
        pass.e.setBuffer(r.inner.input.buf, r.inner.input.off, 0);
        pass.e.setBuffer(r.base.buf, r.base.off, 1);
        pass.e.setValue(@as(u32, @intCast(r.count())), 2);
        pass.e.dispatchThreads(mtl.Size.of(r.count(), 1, 1), mtl.Size.of(256, 1, 1));
        try r.rig.finish(pass);
        const out = r.rig.bytes(r.inner.out, r.count() * 4);
        const base = r.rig.bytes(r.base, r.count() * 4);
        for (r.inner.x, 0..) |x, i| {
            const bits = std.mem.readInt(u32, out[4 * i ..][0..4], .little);
            const got: f32 = @bitCast(bits);
            if (std.math.isNan(x)) {
                if (bits != @as(u32, @bitCast(x))) return "a NaN input does not pass through bit for bit";
            } else if (got != @max(x, 0)) return "an output is not max(x, 0)" else if (bits != std.mem.readInt(u32, base[4 * i ..][0..4], .little))
                return "a non-NaN output's bits differ from the previous relu's";
        }
        return null;
    }
};

const vt_unary = g.hooks(Unary);
const vt_relu = g.hooks(Relu);

pub const Suite = struct {
    a: std.mem.Allocator,
    unary: [6]?*Unary = @splat(null),
    relu: [2]?*Relu = @splat(null),
    squash: squash.Suite = .{ .a = undefined },
    conv: conv.Suite = .{ .a = undefined },

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        s.squash = .{ .a = s.a };
        s.conv = .{ .a = s.a };
        var made: usize = 0;
        for ([_]Kind{ .bias, .scale, .relu, .sigmoid }) |kind| for ([_]bool{ false, true }) |control| {
            if (only) |needle| if (std.mem.indexOf(u8, Unary.label(kind, control), needle) == null) continue;
            try kernels.ensureUnusedCapacity(s.a, 1);
            const inner: Unary = .{
                .a = s.a,
                .rig = .{ .a = s.a, .device = device, .queue = queue },
                .kind = kind,
                .control = control,
                .pipe = try pipeline(s.a, device, Unary.function(kind), control),
                .copy = if (kind == .bias) try pipeline(s.a, device, "sam3_copy", false) else null,
            };
            const gate: ?g.Gate = if (control) .shapes else null;
            if (kind == .relu) {
                const r = try s.a.create(Relu);
                r.* = .{ .inner = inner, .previous = try previousRelu(s.a, device) };
                r.rig = &r.inner.rig;
                s.relu[@intFromBool(control)] = r;
                kernels.appendAssumeCapacity(.{ .spec = r.inner.spec(), .ctx = r, .vt = &vt_relu, .control = gate });
            } else {
                const p = try s.a.create(Unary);
                p.* = inner;
                s.unary[made] = p;
                made += 1;
                kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt_unary, .control = gate });
            }
        };
        try s.squash.collect(device, queue, kernels, only);
        try s.conv.collect(device, queue, kernels, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.unary) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
        for (s.relu) |maybe| if (maybe) |r| {
            r.inner.deinit();
            r.previous.deinit();
            s.a.destroy(r);
        };
        s.squash.deinit();
        s.conv.deinit();
    }
};
