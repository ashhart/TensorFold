//! The qwen27 steering kernel on the GPU: the fused norm's steered row against the float64 reference.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const steer = contract.steer;

/// The kernel the family compiles for the steering launch.
const symbol = "custom_kernel_q27_norm_steer_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float_float_float_uint";

/// The width the family compiles the kernel at.
const width = steer.hidden;

/// The strength the launch carries.
const strength: f32 = 1.0;

/// The eps the norm reads.
const eps: f32 = 1e-6;

/// The rows the frame pads a call to, as the kernel's grid is sized.
fn padded(rows: usize) usize {
    return 16 * ((rows + 15) / 16);
}

/// The groups of a row that write their own partial, four threads to a group.
const xs_groups = steer.threads / 4;

/// The fault a control plants in the kernel's own source: a sum that drops a NaN, a dot summed in half.
const Variant = enum { production, nan, sums };

/// One kernel this suite registers: the mode it serves, the source it is built from and the gate its control breaks.
pub const Entry = struct {
    name: []const u8,
    mode: steer.Mode,
    variant: Variant,
    control: ?g.Gate,
};

/// Every kernel this suite registers: the modes the launch serves, and the controls that break a gate.
pub const entries = [_]Entry{
    .{ .name = "qwen27 steer project", .mode = .project, .variant = .production, .control = null },
    .{ .name = "qwen27 steer add", .mode = .add, .variant = .production, .control = null },
    .{ .name = "qwen27 steer project nan control", .mode = .project, .variant = .nan, .control = .nonfinite },
    .{ .name = "qwen27 steer project sums control", .mode = .project, .variant = .sums, .control = .sums },
};

/// One line of the source replaced.
fn edit(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, text, from) orelse return error.UnknownSource;
    return std.mem.concat(a, u8, &.{ text[0..at], to, text[at + from.len ..] });
}

/// The family's source with the specialization its width needs, and the fault a control plants.
fn source(a: std.mem.Allocator, variant: Variant) ![]u8 {
    const template = tf.qwen27.runtime_sources.norm_steer;
    const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.UnknownSource;
    const definitions = try std.fmt.allocPrint(a, "constexpr int K={d};", .{width});
    defer a.free(definitions);
    const text = try std.mem.concat(a, u8, &.{ template[0..mark], definitions, template[mark + "Q27_SPECIALIZE".len ..] });
    defer a.free(text);
    return switch (variant) {
        .production => a.dupe(u8, text),
        .nan => edit(a, text, "dot += dot_parts[i];", "dot += isnan(dot_parts[i]) ? 0.0f : dot_parts[i];"),
        .sums => edit(a, text, "sd = simd_sum(sd);", "sd = float(half(simd_sum(sd)));"),
    };
}

/// One kernel's device side: the rig's buffers, the reference beside them and the launch.
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mode: steer.Mode,
    spec: c.Spec = undefined,
    exact_sums: bool = false,
    s: ?steer.Steer = null,
    rows: usize = 0,
    groups: usize = 0,
    h: rig_.Ref = undefined,
    residual: rig_.Ref = undefined,
    weight: rig_.Ref = undefined,
    epsilon: rig_.Ref = undefined,
    dims: rig_.Ref = undefined,
    dir: rig_.Ref = undefined,
    ho: rig_.Ref = undefined,
    xo: rig_.Ref = undefined,
    xs: rig_.Ref = undefined,

    const vt = g.hooks(Probe);

    pub fn deinit(p: *Probe) void {
        if (p.s) |s| s.deinit();
        p.rig.deinit();
        p.pipe.deinit();
    }

    pub fn kernel(p: *Probe, e: Entry) g.Kernel {
        var spec = steer.spec(.{ .mode = e.mode, .alpha = strength, .rows = 1, .hidden = width });
        spec.name = e.name;
        p.spec = spec;
        return .{ .spec = spec, .ctx = p, .vt = &vt, .control = e.control };
    }

    /// The buffers for a shape and a fill: the residual stream, the direction and the norm's own inputs.
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        if (shape.n != width or shape.k != width or shape.rows == 0) return false;
        if (p.s) |s| s.deinit();
        p.s = null;
        var s = try steer.Steer.init(p.a, .{ .mode = p.mode, .alpha = strength, .rows = shape.rows, .hidden = width });
        errdefer s.deinit();
        s.fill(fill, seed);
        p.exact_sums = fill == .stress;
        p.rows = shape.rows;
        p.groups = padded(shape.rows);
        const zeros = try p.a.alloc(u16, shape.rows * width);
        defer p.a.free(zeros);
        @memset(zeros, 0);
        p.h = try p.rig.input(std.mem.sliceAsBytes(s.x));
        // The kernel adds the block's output to the residual: the residual is zero, so the row stands alone.
        p.residual = try p.rig.input(std.mem.sliceAsBytes(zeros));
        @memset(zeros, c.toBf16(1));
        p.weight = try p.rig.input(std.mem.sliceAsBytes(zeros));
        const eps_value: f32 = eps;
        p.epsilon = try p.rig.input(std.mem.asBytes(&eps_value));
        const dims = [2]i32{ @intCast(shape.rows), @intCast(p.groups) };
        p.dims = try p.rig.input(std.mem.asBytes(&dims));
        p.dir = try p.rig.input(std.mem.sliceAsBytes(s.dir));
        p.ho = try p.rig.output(shape.rows * width * 2);
        p.xo = try p.rig.scratch(shape.rows * width * 2);
        p.xs = try p.rig.scratch(xs_groups * p.groups * 4);
        p.s = s;
        return true;
    }

    /// The launch the engine makes: a threadgroup a row, the fused norm's threads.
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.h.buf, p.h.off, 0);
        e.setBuffer(p.residual.buf, p.residual.off, 1);
        e.setBuffer(p.weight.buf, p.weight.off, 2);
        e.setBuffer(p.epsilon.buf, p.epsilon.off, 3);
        e.setBuffer(p.dims.buf, p.dims.off, 4);
        e.setBuffer(p.ho.buf, p.ho.off, 5);
        e.setBuffer(p.xo.buf, p.xo.off, 6);
        e.setBuffer(p.xs.buf, p.xs.off, 7);
        e.setBuffer(p.dir.buf, p.dir.off, 8);
        e.setValue(strength, 9);
        e.setValue(@as(u32, @backingInt(p.mode)), 10);
        e.dispatchGroups(mtl.Size.of(1, p.groups, 1), mtl.Size.of(steer.threads, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.rows * width;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.s.?.reference(want, mag);
    }

    /// The bound the gate checks: the exact-sum bound on the stress rows, where the sums are exact.
    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (p.exact_sums) {
            for (allow, want) |*l, w| l.* = c.boundExact(p.spec, w);
            return;
        }
        p.s.?.bounds(want, mag, allow);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.s.?.simulate(acc, order, out);
    }
};

/// The suite's kernels: each variant's library, the probes and the kernels the gates run.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    libs: [3]?mtl.Library = @splat(null),

    fn library(s: *Suite, variant: Variant) !mtl.Library {
        const at: usize = @backingInt(variant);
        if (s.libs[at]) |l| return l;
        const text = try source(s.a, variant);
        defer s.a.free(text);
        const lib = try mtl.Library.fromSource(s.device, text, mtl.CompileOptions.mlx());
        s.libs[at] = lib;
        return lib;
    }

    pub fn add(s: *Suite, e: Entry) !void {
        const lib = try s.library(e.variant);
        const pipe = try mtl.Pipeline.init(s.device, lib, symbol, false);
        errdefer pipe.deinit();
        try s.probes.ensureUnusedCapacity(s.a, 1);
        try s.kernels.ensureUnusedCapacity(s.a, 1);
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .pipe = pipe, .mode = e.mode };
        s.probes.appendAssumeCapacity(p);
        s.kernels.appendAssumeCapacity(p.kernel(e));
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        for (s.libs) |l| if (l) |lib| lib.deinit();
    }
};
