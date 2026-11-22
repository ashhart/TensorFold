//! Bit-exact fused decode operators for Inkling 2 Mini: conv+store+residual, q/rel/kv prep, dual loop mix, enter.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ops_mod = @import("metal_ops.zig");
const loop_mod = @import("metal_loop.zig");

pub const View = ops_mod.View;
pub const Tau = ops_mod.Tau;

pub const conv_taps = ops_mod.conv_taps;
pub const conv_ring = ops_mod.conv_ring;
/// Max rows per item where conv history reads and stores occupy disjoint slots in `conv_ring`.
pub const max_fused_conv_rows = conv_ring - (conv_taps - 1);

const ConvResAttnKernelArgs = extern struct {
    rows: u32,
    channels: u32,
    pos0: u32,
    ring: u32,
    has_block: u32,
    save_block: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    divisor: f32,
};

const ConvResMlpKernelArgs = extern struct {
    rows: u32,
    channels: u32,
    pos0: u32,
    ring: u32,
    divisor: f32,
};

const FusedQkvKernelArgs = extern struct {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    dim: u32,
    rel_dim: u32,
    pos0: u32,
    conv_ring: u32,
    kv_ring: u32,
    log_scaled: u32,
    floor_positions: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    qk_threshold: f32,
    rel_threshold: f32,
    alpha: f32,
    eps: f32,
    conv_divisor: f32,
};

const DualLoopMixKernelArgs = extern struct {
    hidden: u32,
    block_stride: u32,
    out_stride: u32,
    partial_pitch: u32,
    blocks0: u32,
    queries0: u32,
    blocks1: u32,
    queries1: u32,
    eps_unweighted: f32,
    scale: f32,
};

const FusedEnterKernelArgs = extern struct {
    hidden: u32,
    rows: u32,
    bank_rows: u32,
    pad: u32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(ConvResAttnKernelArgs) == 36);
    std.debug.assert(@sizeOf(ConvResMlpKernelArgs) == 20);
    std.debug.assert(@sizeOf(FusedQkvKernelArgs) == 68);
    std.debug.assert(@sizeOf(DualLoopMixKernelArgs) == 40);
    std.debug.assert(@sizeOf(FusedEnterKernelArgs) == 16);
}

pub const Error = error{InvalidFusedOp};

pub const ConvResAttnArgs = struct {
    x: View,
    hist: View,
    w: View,
    hin: View,
    block_dst: ?View,
    out: View,
    rows: u32,
    channels: u32,
    pos0: u32,
    ring: u32 = conv_ring,
    has_block: bool,
};

pub const ConvResMlpArgs = struct {
    x: View,
    hist: View,
    w: View,
    h32: View,
    out: View,
    rows: u32,
    channels: u32,
    pos0: u32,
    ring: u32 = conv_ring,
};

pub const QRelPrepArgs = struct {
    q: View,
    q_scale: View,
    rel: View,
    rel_scale: View,
    rows: u32,
    q_heads: u32,
    dim: u32,
    rel_dim: u32,
    qk_threshold: f32,
    rel_threshold: f32,
    tau: ?Tau = null,
};

pub const QkvPrepStaticArgs = struct {
    q: View,
    q_scale: View,
    rel: View,
    rel_scale: View,
    kraw: View,
    vraw: View,
    khist: View,
    vhist: View,
    kw: View,
    vw: View,
    kcache: View,
    vcache: View,
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    dim: u32,
    rel_dim: u32,
    pos0: u32,
    conv_ring: u32 = conv_ring,
    kv_ring: u32,
    qk_threshold: f32,
    rel_threshold: f32,
    eps: f32,
    tau: ?Tau = null,
};

pub const QkvPrepDynamicArgs = struct {
    q: View,
    q_scale: View,
    rel: View,
    rel_scale: View,
    kraw: View,
    vraw: View,
    khist: View,
    vhist: View,
    kw: View,
    vw: View,
    kcache: View,
    vcache: View,
    dynk: View,
    dynv: View,
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    dim: u32,
    rel_dim: u32,
    pos0: u32,
    conv_ring: u32 = conv_ring,
    kv_ring: u32,
    qk_threshold: f32,
    rel_threshold: f32,
    eps: f32,
    tau: ?Tau = null,
};

pub const DualLoopMixArgs = struct {
    blocks0: View,
    partial0: View,
    queries0: [loop_mod.max_queries]View,
    nb0: u32,
    nq0: u32,
    out0: View,
    blocks1: View,
    partial1: View,
    queries1: [3]View,
    nb1: u32,
    nq1: u32,
    out1: View,
    block_stride: u32,
    out_stride: u32,
    rows: u32,
    hidden: u32,
    eps_unweighted: f32 = 1e-6,
};

pub const FusedEnterArgs = struct {
    src_blocks: View,
    bank: View,
    entry_block: View,
    hid: View,
    bank_h: View,
    hidden: u32,
    rows: u32,
    bank_rows: u32,
};

pub const FusedOps = struct {
    conv_res_attn: mtl.Pipeline,
    conv_res_mlp: mtl.Pipeline,
    q_rel_prep: mtl.Pipeline,
    qkv_prep_static: mtl.Pipeline,
    qkv_prep_dynamic: mtl.Pipeline,
    dual_loop_mix: mtl.Pipeline,
    fused_enter: mtl.Pipeline,

    pub fn load(device: mtl.Device) !FusedOps {
        return loadSource(device, sources.inkling2_fused_ops);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !FusedOps {
        // Match metal_ops.zig compile options for conv/qkv/elem kernels and metal_loop.zig (.precise) for loop_mix.
        const ops_lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer ops_lib.deinit();

        var precise_opts = mtl.CompileOptions.mlx();
        precise_opts.math_functions = .precise;
        const loop_lib = try mtl.Library.fromSource(device, text, precise_opts);
        defer loop_lib.deinit();

        const conv_res_attn = try mtl.Pipeline.init(device, ops_lib, "inkling2_conv_store_residual_attn", false);
        errdefer conv_res_attn.deinit();
        const conv_res_mlp = try mtl.Pipeline.init(device, ops_lib, "inkling2_conv_store_residual_mlp", false);
        errdefer conv_res_mlp.deinit();
        const q_rel_prep = try mtl.Pipeline.init(device, ops_lib, "inkling2_fused_q_rel_prep", false);
        errdefer q_rel_prep.deinit();
        const qkv_prep_static = try mtl.Pipeline.init(device, ops_lib, "inkling2_fused_qkv_prep_static", false);
        errdefer qkv_prep_static.deinit();
        const qkv_prep_dynamic = try mtl.Pipeline.init(device, ops_lib, "inkling2_fused_qkv_prep_dynamic", false);
        errdefer qkv_prep_dynamic.deinit();
        const dual_loop_mix = try mtl.Pipeline.init(device, loop_lib, "inkling2_fused_dual_loop_mix", false);
        errdefer dual_loop_mix.deinit();
        const fused_enter = try mtl.Pipeline.init(device, ops_lib, "inkling2_fused_enter", false);
        errdefer fused_enter.deinit();

        return .{
            .conv_res_attn = conv_res_attn,
            .conv_res_mlp = conv_res_mlp,
            .q_rel_prep = q_rel_prep,
            .qkv_prep_static = qkv_prep_static,
            .qkv_prep_dynamic = qkv_prep_dynamic,
            .dual_loop_mix = dual_loop_mix,
            .fused_enter = fused_enter,
        };
    }

    pub fn deinit(o: *FusedOps) void {
        o.conv_res_attn.deinit();
        o.conv_res_mlp.deinit();
        o.q_rel_prep.deinit();
        o.qkv_prep_static.deinit();
        o.qkv_prep_dynamic.deinit();
        o.dual_loop_mix.deinit();
        o.fused_enter.deinit();
    }

    pub fn convStoreResidualAttn(o: *const FusedOps, e: mtl.ComputeEncoder, a: ConvResAttnArgs) !void {
        if (a.rows == 0 or a.channels == 0 or a.ring < conv_taps or a.rows > a.ring - (conv_taps - 1))
            return error.InvalidFusedOp;
        e.setPipeline(o.conv_res_attn);
        bind(e, &.{ a.x, a.hist, a.w, a.hin, a.block_dst orelse a.hin, a.out });
        e.setValue(ConvResAttnKernelArgs{
            .rows = a.rows,
            .channels = a.channels,
            .pos0 = a.pos0,
            .ring = a.ring,
            .has_block = @intFromBool(a.has_block),
            .save_block = @intFromBool(a.block_dst != null),
            .divisor = @floatCast(@sqrt(@as(f64, conv_taps + 1))),
        }, 6);
        e.dispatchThreads(mtl.Size.of(a.channels, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn convStoreResidualMlp(o: *const FusedOps, e: mtl.ComputeEncoder, a: ConvResMlpArgs) !void {
        if (a.rows == 0 or a.channels == 0 or a.ring < conv_taps or a.rows > a.ring - (conv_taps - 1))
            return error.InvalidFusedOp;
        e.setPipeline(o.conv_res_mlp);
        bind(e, &.{ a.x, a.hist, a.w, a.h32, a.out });
        e.setValue(ConvResMlpKernelArgs{
            .rows = a.rows,
            .channels = a.channels,
            .pos0 = a.pos0,
            .ring = a.ring,
            .divisor = @floatCast(@sqrt(@as(f64, conv_taps + 1))),
        }, 5);
        e.dispatchThreads(mtl.Size.of(a.channels, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn qRelPrep(o: *const FusedOps, e: mtl.ComputeEncoder, a: QRelPrepArgs) !void {
        if (a.rows == 0 or a.q_heads == 0 or a.dim == 0 or a.dim > 256 or a.rel_dim == 0 or a.rel_dim > 256)
            return error.InvalidFusedOp;
        const t = a.tau orelse Tau{ .pos0 = 0, .alpha = 0, .floor = 1 };
        e.setPipeline(o.q_rel_prep);
        bind(e, &.{ a.q, a.q_scale, a.rel, a.rel_scale });
        e.setValue(FusedQkvKernelArgs{
            .rows = a.rows,
            .q_heads = a.q_heads,
            .kv_heads = 0,
            .dim = a.dim,
            .rel_dim = a.rel_dim,
            .pos0 = t.pos0,
            .conv_ring = conv_ring,
            .kv_ring = 0,
            .log_scaled = @intFromBool(a.tau != null),
            .floor_positions = t.floor,
            .qk_threshold = a.qk_threshold,
            .rel_threshold = a.rel_threshold,
            .alpha = t.alpha,
            .eps = 0,
            .conv_divisor = 1,
        }, 4);
        e.dispatchGroups(mtl.Size.of(a.q_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn qkvPrepStatic(o: *const FusedOps, e: mtl.ComputeEncoder, a: QkvPrepStaticArgs) !void {
        if (a.rows == 0 or a.q_heads == 0 or a.kv_heads == 0 or a.dim == 0 or a.dim > 256 or a.rel_dim == 0 or a.rel_dim > 256 or
            a.conv_ring < conv_taps or a.rows > a.conv_ring - (conv_taps - 1))
            return error.InvalidFusedOp;
        const t = a.tau orelse Tau{ .pos0 = a.pos0, .alpha = 0, .floor = 1 };
        e.setPipeline(o.qkv_prep_static);
        bind(e, &.{ a.q, a.q_scale, a.rel, a.rel_scale, a.kraw, a.vraw, a.khist, a.vhist, a.kw, a.vw, a.kcache, a.vcache });
        e.setValue(FusedQkvKernelArgs{
            .rows = a.rows,
            .q_heads = a.q_heads,
            .kv_heads = a.kv_heads,
            .dim = a.dim,
            .rel_dim = a.rel_dim,
            .pos0 = a.pos0,
            .conv_ring = a.conv_ring,
            .kv_ring = a.kv_ring,
            .log_scaled = @intFromBool(a.tau != null),
            .floor_positions = t.floor,
            .qk_threshold = a.qk_threshold,
            .rel_threshold = a.rel_threshold,
            .alpha = t.alpha,
            .eps = a.eps,
            .conv_divisor = @floatCast(@sqrt(@as(f64, conv_taps + 1))),
        }, 12);
        e.dispatchGroups(mtl.Size.of(a.q_heads + a.kv_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn qkvPrepDynamic(o: *const FusedOps, e: mtl.ComputeEncoder, a: QkvPrepDynamicArgs) !void {
        if (a.rows == 0 or a.q_heads == 0 or a.kv_heads == 0 or a.dim == 0 or a.dim > 256 or a.rel_dim == 0 or a.rel_dim > 256 or
            a.conv_ring < conv_taps or a.rows > a.conv_ring - (conv_taps - 1))
            return error.InvalidFusedOp;
        const t = a.tau orelse Tau{ .pos0 = a.pos0, .alpha = 0, .floor = 1 };
        e.setPipeline(o.qkv_prep_dynamic);
        bind(e, &.{ a.q, a.q_scale, a.rel, a.rel_scale, a.kraw, a.vraw, a.khist, a.vhist, a.kw, a.vw, a.kcache, a.vcache, a.dynk, a.dynv });
        e.setValue(FusedQkvKernelArgs{
            .rows = a.rows,
            .q_heads = a.q_heads,
            .kv_heads = a.kv_heads,
            .dim = a.dim,
            .rel_dim = a.rel_dim,
            .pos0 = a.pos0,
            .conv_ring = a.conv_ring,
            .kv_ring = a.kv_ring,
            .log_scaled = @intFromBool(a.tau != null),
            .floor_positions = t.floor,
            .qk_threshold = a.qk_threshold,
            .rel_threshold = a.rel_threshold,
            .alpha = t.alpha,
            .eps = a.eps,
            .conv_divisor = @floatCast(@sqrt(@as(f64, 2 * conv_taps + 1))),
        }, 14);
        e.dispatchGroups(mtl.Size.of(a.q_heads + a.kv_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn dualLoopMix(o: *const FusedOps, e: mtl.ComputeEncoder, a: DualLoopMixArgs) !void {
        if (a.rows == 0 or a.hidden == 0 or a.hidden > loop_mod.max_hidden or
            a.nb0 + 1 > loop_mod.max_sources or a.nq0 == 0 or a.nq0 > loop_mod.max_queries or
            a.nb1 + 1 > loop_mod.max_sources or a.nq1 == 0 or a.nq1 > 3)
            return error.InvalidFusedOp;
        e.setPipeline(o.dual_loop_mix);
        bind(e, &.{
            a.blocks0,
            a.partial0,
            a.queries0[0],
            a.queries0[if (a.nq0 > 1) 1 else 0],
            a.queries0[if (a.nq0 > 2) 2 else 0],
            a.queries0[if (a.nq0 > 3) 3 else 0],
            a.out0,
            a.blocks1,
            a.partial1,
            a.queries1[0],
            a.queries1[if (a.nq1 > 1) 1 else 0],
            a.queries1[if (a.nq1 > 2) 2 else 0],
            a.out1,
        });
        e.setValue(DualLoopMixKernelArgs{
            .hidden = a.hidden,
            .block_stride = a.block_stride,
            .out_stride = a.out_stride,
            .partial_pitch = a.hidden,
            .blocks0 = a.nb0,
            .queries0 = a.nq0,
            .blocks1 = a.nb1,
            .queries1 = a.nq1,
            .eps_unweighted = a.eps_unweighted,
            .scale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(a.hidden)))),
        }, 13);
        e.dispatchGroups(mtl.Size.of(a.rows, 2, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn enter(o: *const FusedOps, e: mtl.ComputeEncoder, a: FusedEnterArgs) !void {
        if (a.rows == 0 or a.hidden == 0) return error.InvalidFusedOp;
        e.setPipeline(o.fused_enter);
        bind(e, &.{ a.src_blocks, a.bank, a.entry_block, a.hid, a.bank_h });
        e.setValue(FusedEnterKernelArgs{
            .hidden = a.hidden,
            .rows = a.rows,
            .bank_rows = a.bank_rows,
        }, 5);
        e.dispatchThreads(mtl.Size.of(a.hidden, a.rows + a.bank_rows, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, views: []const View) void {
    for (views, 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
}

// Per-device lazy cache so `metal_forward.zig` can use `FusedOps` without mutating `Model` in `metal_model.zig`.
var cache_lock: std.atomic.Value(u8) = .init(0);
var cached_device: std.atomic.Value(usize) = .init(0);
var cached_ops: FusedOps = undefined;

pub fn forDevice(device: mtl.Device) !*const FusedOps {
    const dev_addr = @intFromPtr(device.id);
    if (cached_device.load(.acquire) == dev_addr) return &cached_ops;
    while (cache_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.store(0, .release);
    if (cached_device.load(.acquire) == dev_addr) return &cached_ops;
    const loaded = try FusedOps.load(device);
    if (cached_device.load(.monotonic) != 0) cached_ops.deinit();
    cached_ops = loaded;
    cached_device.store(dev_addr, .release);
    return &cached_ops;
}
