//! Inkling 2 Mini's looped-core operators on Metal: the pass-to-pass message attention and the message gates
//! (kernels/metal/inkling2/loop.metal). Like metal_ops.zig the caller owns the encoder and every buffer; each op only
//! encodes dispatches in order. A row's bits never depend on the other rows of its call.
//!
//! The reference (transformers' Inkling2LoopMessageAttentionResidual and Inkling2LoopMessageGate), per row:
//!   messages = attention-residual mixes of four learned queries over the previous pass's banked blocks and final hidden
//!              state (each RMS-normed without a weight, softmax scale hidden^-0.5), fp32;
//!   gate(h, m) = (1 - tanh(Wf d)) * h + tanh(Wg d) * m, d = bf16(Wd [rms(h), rms(m)]) rounded to bf16, Wf and Wg the
//!              bf16-stored fp32 gate matrices, then the layer's weighted RMS norm rounds the mix to bf16.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const exl3 = @import("../../core/exl3_projection.zig");

pub const View = exl3.View;

pub const max_sources = 9;
pub const max_queries = 4;
pub const max_rank = 128;
pub const max_gates = 3;
pub const max_hidden = 6144;

const Kernel = enum { mix, gate_down, gate_mix };
const kernel_count = @typeInfo(Kernel).@"enum".field_names.len;

/// Every symbol the library defines, in pipeline order (the registration list the contract's launch records name).
pub const symbols = blk: {
    var out: [kernel_count][:0]const u8 = undefined;
    for (@typeInfo(Kernel).@"enum".field_names, 0..) |name, i| out[i] = "inkling2_loop_" ++ name;
    break :blk out;
};

const MixKernelArgs = extern struct {
    hidden: u32,
    blocks: u32,
    block_stride: u32,
    partial_f32: u32,
    queries: u32,
    out_stride: u32,
    partial_pitch: u32,
    pad: u32 = 0,
    eps_unweighted: f32,
    scale: f32,
};
const GateKernelArgs = extern struct {
    hidden: u32,
    rank: u32,
    rows: u32,
    msg_stride: u32,
    slot0: u32,
    slot1: u32,
    slot2: u32,
    out_stride: u32,
    eps_unweighted: f32,
    eps_norm: f32,
};

comptime {
    std.debug.assert(@sizeOf(MixKernelArgs) == 40 and @sizeOf(GateKernelArgs) == 40);
}

pub const Error = error{InvalidLoopOp};

/// Block attention residual without its trailing norm: sources are `nb` bf16 blocks [nb][rows][hidden] (`block_stride`
/// elements apart) then `partial` [rows][hidden] (fp32 when `partial_f32`, else bf16; rows `partial_ld` apart, 0 = hidden).
/// Query q (fp32 [hidden], `queries[q]`) writes fp32 mixes [rows][hidden] at `out` + q * `out_stride` elements.
pub const MixArgs = struct {
    blocks: View,
    block_stride: u32,
    nb: u32,
    partial: View,
    partial_f32: bool,
    partial_ld: u32 = 0,
    queries: [max_queries]View,
    nq: u32,
    out: View,
    out_stride: u32,
    rows: u32,
    hidden: u32,
    eps_unweighted: f32 = 1e-6,
};

/// Gate g reads its state h[g] (fp32 [rows][hidden]) and the message plane slots[g] of `msg` (fp32, planes `msg_stride`
/// elements apart) and its bf16 down matrix [rank][2 hidden]; down [gate][rows][rank] bf16 receives the rounded outputs.
pub const GateDownArgs = struct {
    h: [max_gates]View,
    msg: View,
    msg_stride: u32,
    slots: [max_gates]u32,
    gates: u32,
    down_w: [max_gates]View,
    down: View,
    rows: u32,
    hidden: u32,
    rank: u32,
    eps_unweighted: f32 = 1e-6,
};

/// The gates' outputs: bf16 [rows][hidden] planes `out_stride` elements apart, each normed by norms[g] (bf16 [hidden]).
/// forget[g] and gain[g] are bf16 [hidden][rank].
pub const GateMixArgs = struct {
    h: [max_gates]View,
    msg: View,
    msg_stride: u32,
    slots: [max_gates]u32,
    gates: u32,
    down: View,
    forget: [max_gates]View,
    gain: [max_gates]View,
    norms: [max_gates]View,
    out: View,
    out_stride: u32,
    rows: u32,
    hidden: u32,
    rank: u32,
    eps: f32,
};

pub const Ops = struct {
    pipes: [kernel_count]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Ops {
        return loadSource(device, sources.inkling2_loop_ops);
    }

    /// The same operators from other text (the contract's planted controls edit the source).
    pub fn loadSource(device: mtl.Device, text: []const u8) !Ops {
        var compile = mtl.CompileOptions.mlx();
        compile.math_functions = .precise;
        const library = try mtl.Library.fromSource(device, text, compile);
        defer library.deinit();
        var o: Ops = undefined;
        var loaded: usize = 0;
        errdefer for (o.pipes[0..loaded]) |p| p.deinit();
        for (symbols, 0..) |name, i| {
            o.pipes[i] = try mtl.Pipeline.init(device, library, name, false);
            loaded += 1;
            if (o.pipes[i].maxThreads() < 256 or o.pipes[i].simdWidth() != 32) return error.LoopLaunchLimit;
        }
        return o;
    }

    pub fn deinit(o: *Ops) void {
        for (o.pipes) |p| p.deinit();
    }

    fn pipe(o: *const Ops, k: Kernel) mtl.Pipeline {
        return o.pipes[@intFromEnum(k)];
    }

    pub fn mix(o: *const Ops, e: mtl.ComputeEncoder, a: MixArgs) !void {
        if (a.rows == 0 or a.nq == 0 or a.nq > max_queries or a.nb >= max_sources or a.hidden == 0 or a.hidden > max_hidden)
            return error.InvalidLoopOp;
        e.setPipeline(o.pipe(.mix));
        bind(e, &.{ a.blocks, a.partial, a.queries[0], a.queries[1], a.queries[2], a.queries[3], a.out });
        e.setValue(MixKernelArgs{
            .hidden = a.hidden,
            .blocks = a.nb,
            .block_stride = a.block_stride,
            .partial_f32 = @intFromBool(a.partial_f32),
            .queries = a.nq,
            .out_stride = a.out_stride,
            .partial_pitch = if (a.partial_ld == 0) a.hidden else a.partial_ld,
            .eps_unweighted = a.eps_unweighted,
            .scale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(a.hidden)))),
        }, 7);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn gateCheck(rows: u32, hidden: u32, rank: u32, gates: u32) !void {
        if (rows == 0 or hidden == 0 or hidden > max_hidden or rank == 0 or rank > max_rank or gates == 0 or gates > max_gates)
            return error.InvalidLoopOp;
    }

    pub fn gateDown(o: *const Ops, e: mtl.ComputeEncoder, a: GateDownArgs) !void {
        try gateCheck(a.rows, a.hidden, a.rank, a.gates);
        e.setPipeline(o.pipe(.gate_down));
        bind(e, &.{ a.h[0], a.h[if (a.gates > 1) 1 else 0], a.h[if (a.gates > 2) 2 else 0], a.msg });
        bind3(e, 4, a.down_w, a.gates);
        e.setBuffer(a.down.buffer, a.down.offset, 7);
        e.setValue(GateKernelArgs{
            .hidden = a.hidden,
            .rank = a.rank,
            .rows = a.rows,
            .msg_stride = a.msg_stride,
            .slot0 = a.slots[0],
            .slot1 = a.slots[1],
            .slot2 = a.slots[2],
            .out_stride = 0,
            .eps_unweighted = a.eps_unweighted,
            .eps_norm = 0,
        }, 8);
        e.dispatchGroups(mtl.Size.of(a.rows, a.gates, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn gateMix(o: *const Ops, e: mtl.ComputeEncoder, a: GateMixArgs) !void {
        try gateCheck(a.rows, a.hidden, a.rank, a.gates);
        e.setPipeline(o.pipe(.gate_mix));
        bind(e, &.{ a.h[0], a.h[if (a.gates > 1) 1 else 0], a.h[if (a.gates > 2) 2 else 0], a.msg, a.down });
        bind3(e, 5, a.forget, a.gates);
        bind3(e, 8, a.gain, a.gates);
        bind3(e, 11, a.norms, a.gates);
        e.setBuffer(a.out.buffer, a.out.offset, 14);
        e.setValue(GateKernelArgs{
            .hidden = a.hidden,
            .rank = a.rank,
            .rows = a.rows,
            .msg_stride = a.msg_stride,
            .slot0 = a.slots[0],
            .slot1 = a.slots[1],
            .slot2 = a.slots[2],
            .out_stride = a.out_stride,
            .eps_unweighted = 0,
            .eps_norm = a.eps,
        }, 15);
        e.dispatchGroups(mtl.Size.of(a.rows, a.gates, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, views: []const View) void {
    for (views, 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
}

/// Three views from buffer index `first`; the ones past `gates` repeat the first (the kernel never reads them).
fn bind3(e: mtl.ComputeEncoder, first: usize, views: [max_gates]View, gates: u32) void {
    for (0..max_gates) |g| {
        const v = if (g < gates) views[g] else views[0];
        e.setBuffer(v.buffer, v.offset, first + g);
    }
}
