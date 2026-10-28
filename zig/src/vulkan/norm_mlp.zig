//! Norm/MLP pointwise Vulkan operators: BF16 RMSNorm gamma, residual add and
//! SiLU(gate)*up, over the shared set-0 storage / push-constant pipeline model.
//! Shaders: kernels/vulkan/{fast_norm,binary_vec,swiglu}.comp compiled with
//! -DUSE_BF16 (mlx-omarchy sources, MIT); the runtime never falls back to CPU.
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

/// The compiled shader images NormMlp runs; every field is a real SPIR-V image
/// of the named variant, each built from its .comp with -DUSE_BF16. Compile
/// norm and swiglu with -DASH_QWEN35 (together) for the Ash8c Metal oracle
/// semantics; default is the copied MLX bit-parity form. The host API is
/// identical either way; the variant is fixed at build time, not a runtime option.
pub const Shaders = struct {
    norm_bf16: []align(4) const u8,
    add_bf16: []align(4) const u8,
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
    /// Read-only stand-in for declared bindings a kernel never touches
    /// (fast_norm's unused bias slot 2, swiglu's unmaterialized sigmoid/silu 3 and 4).
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
        m.add_bf16 = try dispatch.Pipeline.init(dev, shaders.add_bf16, .{ .bindings = 3, .push_bytes = add_push_bytes, .writes = 1 << 2 });
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

    /// RMSNorm over rows of width bf16 elements: out = x * rsqrt(mean(x^2) + eps) * gamma,
    /// gamma a per-column bf16 row of width elements. A one-element scalar gamma cannot
    /// pass View.init's word-padding rule, so the host advertises only the row form
    /// (the shader's lhs_size == 1 broadcast stays in the source, unused).
    /// Views bind at their own base, so every shader element offset is zero.
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

    /// Residual add out = a + b on bf16, out of place. binary_vec moves four elements
    /// per 8-byte load, so the count and both operand sizes must be multiples of four.
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

    /// SwiGLU out = silu(gate) * up on bf16, out of place, intermediates not materialized
    /// (bindings 3 and 4 stay on the dummy, materialize = 0). Same four-wide gate as add.
    /// Storage semantics follow the compiled variant: ASH_QWEN35 single-round (Metal oracle)
    /// or default MLX fused_chain three-round form.
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
