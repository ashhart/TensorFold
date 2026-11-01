// Norm/SwiGLU build with -DUSE_BF16; binary_vec is bf16 by default (no defines); runtime has no CPU fallback.
const std = @import("std");
const dispatch = @import("dispatch.zig");
const memory = @import("memory.zig");
const params = @import("stage1_params");
const tensor = @import("tensor.zig");
const Device = @import("device.zig").Device;
const Commands = @import("commands.zig").Commands;
const Buffer = memory.Buffer;
const Error = dispatch.Error;
const View = tensor.View;

// norm/SwiGLU: -DUSE_BF16; -DASH_QWEN35=Ash8c (build-time); -DPRESERVE_F32 needs shaderDenormPreserveFloat32.
pub const Shaders = struct {
    norm_bf16: []align(4) const u8,
    add_bf16: []align(4) const u8,
    add_preserve_bf16: []align(4) const u8,
    swiglu_bf16: []align(4) const u8,
};

const dummy_bytes: u64 = 256;

/// fast_norm.comp's push block ends at `beta`: 17 words.
const norm_push_bytes: u32 = 68;
/// binary_vec.comp reads only the first nine words, the cast prefix.
const add_push_bytes: u32 = 36;
/// swiglu.comp has its own four-word block.
const SwigluParams = extern struct { count: u32 = 0, gate_offset: u32 = 0, up_offset: u32 = 0, materialize: u32 = 0 };
const swiglu_push_bytes: u32 = @sizeOf(SwigluParams);

comptime {
    if (norm_push_bytes != @offsetOf(params.Params, "beta") + 4) @compileError("fast_norm push block drifted from Params");
    if (add_push_bytes != params.cast_push_bytes) @compileError("binary_vec push block drifted from the cast prefix");
}

fn groups(count: u64, local: u32, max_x: u32) Error![3]u32 {
    const n = std.math.cast(u32, (count + local - 1) / local) orelse return error.TooLarge;
    return .{ @min(n, max_x), 1, 1 };
}

pub const NormMlp = struct {
    dev: *const Device,
    norm_bf16: dispatch.Pipeline,
    add_bf16: dispatch.Pipeline,
    swiglu_bf16: dispatch.Pipeline,
    // Dummy bindings cover unused fast_norm bias 2 and unmaterialized SwiGLU sigmoid/silu 3,4.
    dummy: Buffer,

    /// Builds the three pipelines; the device must have storage16 and int16 for the bf16 paths.
    pub fn init(dev: *const Device, shaders: Shaders) Error!NormMlp {
        if (!dev.features.storage16 or !dev.features.int16) return error.MissingFeature;
        var m: NormMlp = .{
            .dev = dev,
            .norm_bf16 = undefined,
            .add_bf16 = undefined,
            .swiglu_bf16 = undefined,
            .dummy = try Buffer.init(dev, .device, dummy_bytes),
        };
        errdefer m.dummy.deinit();
        m.norm_bf16 = try dispatch.Pipeline.init(dev, shaders.norm_bf16, .{ .bindings = 4, .push_bytes = norm_push_bytes, .writes = 1 << 3 });
        errdefer m.norm_bf16.deinit();
        m.add_bf16 = try dispatch.Pipeline.init(dev, if (dev.v12.shaderDenormPreserveFloat32 != 0) shaders.add_preserve_bf16 else shaders.add_bf16, .{ .bindings = 3, .push_bytes = add_push_bytes, .writes = 1 << 2 });
        errdefer m.add_bf16.deinit();
        m.swiglu_bf16 = try dispatch.Pipeline.init(dev, shaders.swiglu_bf16, .{ .bindings = 5, .push_bytes = swiglu_push_bytes, .writes = 1 << 2 });
        return m;
    }

    /// Destroys the pipelines and the dummy; call only after the last submission using them returned success.
    pub fn deinit(m: *NormMlp) void {
        m.norm_bf16.deinit();
        m.add_bf16.deinit();
        m.swiglu_bf16.deinit();
        m.dummy.deinit();
    }

    fn dummyBinding(m: *const NormMlp) dispatch.Binding {
        return .{ .buffer = &m.dummy, .len = dummy_bytes };
    }

    /// Preserve the omitted contract, precision, and host-interface facts from the original comment.
    pub fn rmsNorm(m: *NormMlp, cmds: *Commands, out: View, x: View, gamma: View, rows: u64, width: u64, eps: f32, timeout_ns: u64) Error!void {
        if (out.kind != .bf16 or x.kind != .bf16 or gamma.kind != .bf16) return error.Invalid;
        const plan = try normPlan(x.elements, out.elements, gamma.elements, rows, width, eps);
        // grid-stride row loop: a group count past the device limit still covers every row
        const g = try groups(plan.rows, 1, m.dev.properties.limits.maxComputeWorkGroupCount[0]);
        try m.norm_bf16.bind(&.{ x.bind(), gamma.bind(), m.dummyBinding(), out.bind() });
        var p: params.Params = .{
            .count = plan.count,
            .lhs_size = plan.width, // not 1: the row-gamma read, per column
            .reduce_size = plan.width,
            .output_size = plan.rows,
        };
        p.alpha = eps;
        try m.norm_bf16.run(cmds, std.mem.asBytes(&p)[0..norm_push_bytes], g, timeout_ns);
    }

    // bf16 add is out-of-place; binary_vec loads 4 elements/8 bytes, so count and operands must be divisible by 4.
    pub fn add(m: *NormMlp, cmds: *Commands, out: View, a: View, b: View, timeout_ns: u64) Error!void {
        if (out.kind != .bf16 or a.kind != .bf16 or b.kind != .bf16) return error.Invalid;
        if (out.elements != a.elements or a.elements != b.elements) return error.Invalid;
        try vec4Guard(out.elements);
        try m.add_bf16.bind(&.{ a.bind(), b.bind(), out.bind() });
        var p: params.Params = .{
            .count = try out.elemCount32(),
            .operation = 0, // add
            .lhs_size = try a.elemCount32(),
            .rhs_size = try b.elemCount32(),
        };
        try m.add_bf16.run(cmds, std.mem.asBytes(&p)[0..add_push_bytes], try groups(p.count, m.add_bf16.local[0], m.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
    }

    // SwiGLU is 4-wide/out-of-place; bindings 3/4 use materialize=0; ASH_QWEN35 rounds once, MLX fused_chain 3 times.
    pub fn swiglu(m: *NormMlp, cmds: *Commands, out: View, gate: View, up: View, timeout_ns: u64) Error!void {
        if (out.kind != .bf16 or gate.kind != .bf16 or up.kind != .bf16) return error.Invalid;
        if (out.elements != gate.elements or gate.elements != up.elements) return error.Invalid;
        try vec4Guard(out.elements);
        try m.swiglu_bf16.bind(&.{ gate.bind(), up.bind(), out.bind(), m.dummyBinding(), m.dummyBinding() });
        const p: SwigluParams = .{ .count = try out.elemCount32() };
        try m.swiglu_bf16.run(cmds, std.mem.asBytes(&p), try groups(p.count, m.swiglu_bf16.local[0], m.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
    }
};

fn cast32(v: u64) Error!u32 {
    return std.math.cast(u32, v) orelse error.TooLarge;
}

fn vec4Guard(elements: u64) Error!void {
    if (elements % 4 != 0) return error.Invalid;
}

/// The RMSNorm's derived geometry, or the named reason it cannot run; pure and host-tested.
pub const NormPlan = struct { rows: u32, width: u32, count: u32 };

pub fn normPlan(x_elements: u64, out_elements: u64, gamma_elements: u64, rows: u64, width: u64, eps: f32) Error!NormPlan {
    if (!std.math.isFinite(eps) or eps <= 0) return error.Invalid;
    if (rows == 0 or width == 0) return error.Invalid;
    const total = std.math.mul(u64, rows, width) catch return error.TooLarge;
    if (x_elements != total or out_elements != total) return error.Invalid;
    // A one-element bf16 gamma cannot pass View.init's word-padding rule, so no scalar form.
    if (gamma_elements != width) return error.Invalid;
    return .{
        .rows = try cast32(rows),
        .width = try cast32(width),
        .count = try cast32(total),
    };
}

const testing = std.testing;

test "normPlan derives the geometry and refuses torn rows, wrong gamma, bad eps and overflow" {
    const plan = try normPlan(4 * 8, 4 * 8, 8, 4, 8, 1e-6);
    try testing.expectEqual(@as(u32, 4), plan.rows);
    try testing.expectEqual(@as(u32, 8), plan.width);
    try testing.expectEqual(@as(u32, 32), plan.count);
    try testing.expectError(error.Invalid, normPlan(4 * 8, 4 * 8 + 1, 8, 4, 8, 1e-6));
    try testing.expectError(error.Invalid, normPlan(4 * 8, 4 * 8, 7, 4, 8, 1e-6));
    try testing.expectError(error.Invalid, normPlan(4 * 8, 4 * 8, 8, 4, 8, 0));
    try testing.expectError(error.Invalid, normPlan(4 * 8, 4 * 8, 8, 4, 8, std.math.inf(f32)));
    try testing.expectError(error.Invalid, normPlan(0, 0, 8, 0, 8, 1e-6));
    try testing.expectError(error.TooLarge, normPlan(1 << 32, 1 << 32, 1 << 32, 1, 1 << 32, 1e-6)); // width past u32
    try testing.expectError(error.TooLarge, normPlan(1 << 32, 1 << 32, 1 << 16, 1 << 16, 1 << 16, 1e-6)); // rows*width past u32
}

test "the four-wide gate refuses ragged element counts" {
    try testing.expectError(error.Invalid, vec4Guard(6));
    try vec4Guard(8);
}
