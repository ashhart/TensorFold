//! Mini's core ops on Metal: message attention, gates (loop.metal); caller owns encoder and buffers; rows independent.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const exl3 = @import("../../core/exl3_projection.zig");

pub const View = exl3.View;

pub const max_sources = 9;
pub const max_queries = 4;
pub const max_rank = 128;
/// Gate-down ranks one threadgroup computes (LP_RANK_BLOCK in loop.metal).
pub const rank_block = 8;
pub const max_gates = 3;
pub const max_hidden = 6144;
/// Rows up to which the multi-group kernels run: one group a row leaves the GPU idle, bit-equal to the row kernels.
pub const small_rows = 16;
/// Gate-dot ranks one threadgroup computes (LP_DOT_BLOCK in loop.metal).
pub const dot_block = 2;
/// Mix statistics: sources * (1 + queries) floats a row.
const stat_floats = max_sources * (1 + max_queries);
const stats_bytes = 4096;
/// Gate-y MMA tile (inkling2_loop_gate_y_mma): rows and hidden columns of one simdgroup's output, and its rank step.
const gate_tile = 16;
const gate_step = 8;
/// Gate-down MMA (inkling2_loop_gate_down_mma) chains cover 256 elements of a row, so it needs a multiple of that.
const down_hidden = 256;

comptime {
    std.debug.assert(small_rows * stat_floats * 4 <= stats_bytes);
}

const Kernel = enum { mix, gate_down, gate_mix, mix_stats, mix_apply, gate_y, gate_norm, gate_prep, gate_dot, gate_y_mma, gate_down_mma };
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

/// Attention-residual mix of query q over `nb` bf16 blocks and `partial`; fp32 output, one plane per query.
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

/// Gate g: h[g] (fp32), slot plane slots[g], bf16 down [rank][2 hidden]; output down [gate][rows][rank] bf16.
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

/// Gate outputs: bf16 [rows][hidden] planes `out_stride` apart, normed by norms[g]; forget/gain bf16 [hidden][rank].
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

/// The largest call: its pre-norm gate states (fp32, every gate) live in the scratch past the statistics.
pub const Options = struct { rows: u32 = 64, hidden: u32 = max_hidden };

pub const Ops = struct {
    pipes: [kernel_count]mtl.Pipeline,
    scratch: mtl.Buffer,

    pub fn load(device: mtl.Device, options: Options) !Ops {
        return loadSource(device, options, sources.inkling2_loop_ops);
    }

    /// The same operators from other text (the contract's planted controls edit the source).
    pub fn loadSource(device: mtl.Device, options: Options, text: []const u8) !Ops {
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
            if (o.pipes[i].maxThreads() < launchThreads(@fromBackingInt(@intCast(i))) or o.pipes[i].simdWidth() != 32) return error.LoopLaunchLimit;
        }
        const states = max_gates * @max(options.rows, small_rows) * @as(usize, options.hidden) * 4;
        o.scratch = try device.buffer(stats_bytes + states, mtl.ResourceOptions.private);
        return o;
    }

    pub fn deinit(o: *Ops) void {
        for (o.pipes) |p| p.deinit();
        o.scratch.deinit();
    }

    fn pipe(o: *const Ops, k: Kernel) mtl.Pipeline {
        return o.pipes[@backingInt(k)];
    }

    pub fn mix(o: *const Ops, e: mtl.ComputeEncoder, a: MixArgs) !void {
        if (a.rows == 0 or a.nq == 0 or a.nq > max_queries or a.nb >= max_sources or a.hidden == 0 or a.hidden > max_hidden)
            return error.InvalidLoopOp;
        const args = MixKernelArgs{
            .hidden = a.hidden,
            .blocks = a.nb,
            .block_stride = a.block_stride,
            .partial_f32 = @intFromBool(a.partial_f32),
            .queries = a.nq,
            .out_stride = a.out_stride,
            .partial_pitch = if (a.partial_ld == 0) a.hidden else a.partial_ld,
            .eps_unweighted = a.eps_unweighted,
            .scale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(a.hidden)))),
        };
        if (a.rows <= small_rows) {
            const stats = View{ .buffer = o.scratch };
            e.setPipeline(o.pipe(.mix_stats));
            bind(e, &.{ a.blocks, a.partial, a.queries[0], a.queries[1], a.queries[2], a.queries[3], stats });
            e.setValue(args, 7);
            e.dispatchGroups(mtl.Size.of(a.nb + 1, a.rows, 1), mtl.Size.of(256, 1, 1));
            e.setPipeline(o.pipe(.mix_apply));
            bind(e, &.{ a.blocks, a.partial, stats, a.out });
            e.setValue(args, 4);
            e.dispatchGroups(mtl.Size.of(std.math.divCeil(u32, a.hidden, 256) catch unreachable, a.rows, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        e.setPipeline(o.pipe(.mix));
        bind(e, &.{ a.blocks, a.partial, a.queries[0], a.queries[1], a.queries[2], a.queries[3], a.out });
        e.setValue(args, 7);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn gateCheck(rows: u32, hidden: u32, rank: u32, gates: u32) !void {
        if (rows == 0 or hidden == 0 or hidden > max_hidden or rank == 0 or rank > max_rank or gates == 0 or gates > max_gates)
            return error.InvalidLoopOp;
    }

    pub fn gateDown(o: *const Ops, e: mtl.ComputeEncoder, a: GateDownArgs) !void {
        try gateCheck(a.rows, a.hidden, a.rank, a.gates);
        const args = GateKernelArgs{
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
        };
        const normed = View{ .buffer = o.scratch, .offset = stats_bytes };
        const mma = a.hidden % down_hidden == 0 and stats_bytes + @as(usize, a.gates) * a.rows * a.hidden * 4 <= o.scratch.length() and
            aligned16(a.down_w[0..a.gates]);
        if (a.rows <= small_rows or mma) {
            e.setPipeline(o.pipe(.gate_prep));
            bind(e, &.{ a.h[0], a.h[if (a.gates > 1) 1 else 0], a.h[if (a.gates > 2) 2 else 0], a.msg, normed });
            e.setValue(args, 5);
            e.dispatchGroups(mtl.Size.of(a.rows, a.gates, 1), mtl.Size.of(256, 1, 1));
            if (a.rows > small_rows) {
                e.setPipeline(o.pipe(.gate_down_mma));
                e.setBuffer(normed.buffer, normed.offset, 0);
                bind3(e, 1, a.down_w, a.gates);
                e.setBuffer(a.down.buffer, a.down.offset, 4);
                e.setValue(args, 5);
                e.dispatchGroups(mtl.Size.of(divCeil(a.rank, gate_tile), divCeil(a.rows, gate_tile), a.gates), mtl.Size.of(128, 1, 1));
                return;
            }
            e.setPipeline(o.pipe(.gate_dot));
            e.setBuffer(normed.buffer, normed.offset, 0);
            bind3(e, 1, a.down_w, a.gates);
            e.setBuffer(a.down.buffer, a.down.offset, 4);
            e.setValue(args, 5);
            e.dispatchGroups(mtl.Size.of(a.rows, a.gates, std.math.divCeil(u32, a.rank, dot_block) catch unreachable), mtl.Size.of(256, 1, 1));
            return;
        }
        e.setPipeline(o.pipe(.gate_down));
        bind(e, &.{ a.h[0], a.h[if (a.gates > 1) 1 else 0], a.h[if (a.gates > 2) 2 else 0], a.msg });
        bind3(e, 4, a.down_w, a.gates);
        e.setBuffer(a.down.buffer, a.down.offset, 7);
        e.setValue(args, 8);
        e.dispatchGroups(mtl.Size.of(a.rows, a.gates, std.math.divCeil(u32, a.rank, rank_block) catch unreachable), mtl.Size.of(256, 1, 1));
    }

    pub fn gateMix(o: *const Ops, e: mtl.ComputeEncoder, a: GateMixArgs) !void {
        try gateCheck(a.rows, a.hidden, a.rank, a.gates);
        const args = GateKernelArgs{
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
        };
        const hs = [3]View{ a.h[0], a.h[if (a.gates > 1) 1 else 0], a.h[if (a.gates > 2) 2 else 0] };
        if (a.rows <= small_rows) {
            const y = View{ .buffer = o.scratch, .offset = stats_bytes };
            e.setPipeline(o.pipe(.gate_y));
            bind(e, &.{ hs[0], hs[1], hs[2], a.msg, a.down });
            bind3(e, 5, a.forget, a.gates);
            bind3(e, 8, a.gain, a.gates);
            e.setBuffer(y.buffer, y.offset, 11);
            e.setValue(args, 12);
            e.dispatchGroups(mtl.Size.of(std.math.divCeil(u32, a.hidden, 256) catch unreachable, a.gates, a.rows), mtl.Size.of(256, 1, 1));
            e.setPipeline(o.pipe(.gate_norm));
            e.setBuffer(y.buffer, y.offset, 0);
            bind3(e, 1, a.norms, a.gates);
            e.setBuffer(a.out.buffer, a.out.offset, 4);
            e.setValue(args, 5);
            e.dispatchGroups(mtl.Size.of(a.rows, a.gates, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        const y_bytes = @as(usize, a.gates) * a.rows * a.hidden * 4;
        if (a.rank % gate_step == 0 and stats_bytes + y_bytes <= o.scratch.length()) {
            const y = View{ .buffer = o.scratch, .offset = stats_bytes };
            e.setPipeline(o.pipe(.gate_y_mma));
            bind(e, &.{ hs[0], hs[1], hs[2], a.msg, a.down });
            bind3(e, 5, a.forget, a.gates);
            bind3(e, 8, a.gain, a.gates);
            e.setBuffer(y.buffer, y.offset, 11);
            e.setValue(args, 12);
            e.dispatchGroups(mtl.Size.of(divCeil(a.hidden, gate_tile), divCeil(a.rows, gate_tile), a.gates), mtl.Size.of(32, 1, 1));
            e.setPipeline(o.pipe(.gate_norm));
            e.setBuffer(y.buffer, y.offset, 0);
            bind3(e, 1, a.norms, a.gates);
            e.setBuffer(a.out.buffer, a.out.offset, 4);
            e.setValue(args, 5);
            e.dispatchGroups(mtl.Size.of(a.rows, a.gates, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        e.setPipeline(o.pipe(.gate_mix));
        bind(e, &.{ hs[0], hs[1], hs[2], a.msg, a.down });
        bind3(e, 5, a.forget, a.gates);
        bind3(e, 8, a.gain, a.gates);
        bind3(e, 11, a.norms, a.gates);
        e.setBuffer(a.out.buffer, a.out.offset, 14);
        e.setValue(args, 15);
        e.dispatchGroups(mtl.Size.of(a.rows, a.gates, 1), mtl.Size.of(256, 1, 1));
    }
};

/// Threads a kernel is launched with: the MMA kernels use whole simdgroups of a tile, the rest eight.
fn launchThreads(k: Kernel) u32 {
    return switch (k) {
        .gate_y_mma => 32,
        .gate_down_mma => 128,
        else => 256,
    };
}

fn aligned16(views: []const View) bool {
    for (views) |w| if (w.offset % 16 != 0) return false;
    return true;
}

fn divCeil(a: u32, b: u32) u32 {
    return std.math.divCeil(u32, a, b) catch unreachable;
}

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
