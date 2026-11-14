//! Chain entries: an EXL3 projection's epilogue fused with the block-local work that follows it.
const std = @import("std");
const mtl = @import("metal");
const core = @import("exl3_projection_core.zig");
const pipe = @import("exl3_projection_pipelines.zig");
const swi = @import("exl3_projection_swiglu.zig");
const enc = @import("exl3_projection_encode.zig");
const Pipelines = pipe.Pipelines;
const View = core.View;
const Rows = core.Rows;
const Experts = core.Experts;
const bind = core.bind;

/// The chain kernels' resident pipelines; a pipeline that failed to build leaves its entry null.
pub const Set = struct {
    rotate_out_in: ?mtl.Pipeline = null,
    rms_rotate_in: ?mtl.Pipeline = null,
    rotate_out_rope: ?mtl.Pipeline = null,
    rope_rotate_in: ?mtl.Pipeline = null,
    qkv_epilogue: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device, lib: mtl.Library) Set {
        return .{
            .rotate_out_in = pipeline(device, lib, "tf_exl3_rotate_out_in", 128),
            .rms_rotate_in = pipeline(device, lib, "tf_exl3_rms_rotate_in", 256),
            .rotate_out_rope = pipeline(device, lib, "tf_exl3_rotate_out_rope", 128),
            .rope_rotate_in = pipeline(device, lib, "tf_exl3_rope_rotate_in", 128),
            .qkv_epilogue = pipeline(device, lib, "tf_exl3_qkv_epilogue", 320),
        };
    }

    pub fn deinit(s: *Set) void {
        if (s.rotate_out_in) |pf| pf.deinit();
        if (s.rms_rotate_in) |pf| pf.deinit();
        if (s.rotate_out_rope) |pf| pf.deinit();
        if (s.rope_rotate_in) |pf| pf.deinit();
        if (s.qkv_epilogue) |pf| pf.deinit();
    }
};

fn pipeline(device: mtl.Device, lib: mtl.Library, name: [:0]const u8, threads: usize) ?mtl.Pipeline {
    const pf = mtl.Pipeline.init(device, lib, name, false) catch |err| {
        std.log.err("exl3: chain pipeline {s} failed: {s}", .{ name, @errorName(err) });
        return null;
    };
    if (pf.simdWidth() != 32 or pf.maxThreads() < threads) {
        pf.deinit();
        return null;
    }
    return pf;
}

/// The projection that consumes the fused epilogue's bf16 values: its rotate_in scales (`su`, `su_signs`) and its k.
pub const Next = struct { su: View, su_signs: bool, k: usize };

const NextArgs = extern struct { k: u32, su_signs: u32, reserved0: u32 = 0, reserved1: u32 = 0 };

/// Whether `Fused.next` can run on these pipelines.
pub fn rotateOutInReady(p: *const Pipelines) bool {
    return p.chain.rotate_out_in != null;
}

/// Whether `Fused.rope_in` can run on these pipelines.
pub fn ropeRotateInReady(p: *const Pipelines) bool {
    return p.chain.rope_rotate_in != null;
}

/// A rope applied in place to the call's input before its rotate_in (dsv41_rope; rotary columns a multiple of 32).
pub const RopeIn = struct { factors: View, width: u32, rotary: u32, stride: u32, position: u32, step: u32 = 1, inverse: bool = false };

const RopeInArgs = extern struct { width: u32, rotary: u32, stride: u32, position: u32, step: u32, inverse: u32, reserved0: u32 = 0, reserved1: u32 = 0 };

/// Work fused into an indexed call: `rope_in` before its rotate_in, `next` after its rotate_out.
pub const Fused = struct { rope_in: ?RopeIn = null, next: ?Next = null };

/// `encodeIndexed` with `rope_in` before its rotate_in and `next` fused after its rotate_out (rows bf16, contiguous).
pub fn encodeIndexed(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows, f: Fused) !void {
    const a = try enc.indexedArgs(experts, x, y, scratch, r);
    var plain = r;
    if (f.rope_in) |rin| {
        const kernel = p.chain.rope_rotate_in orelse return error.UnsupportedExl3Chain;
        if (r.prerotated or r.grouped or rin.width % 128 != 0 or rin.rotary == 0 or rin.rotary % 32 != 0 or rin.rotary > 128 or
            rin.stride % rin.width != 0 or experts.k % 128 != 0) return error.InvalidExl3Chain;
        e.setPipeline(kernel);
        bind(e, x, 0);
        bind(e, scratch, 1);
        bind(e, scratch, 2);
        e.setValue(a, 3);
        bind(e, r.input_ids orelse scratch, 4);
        bind(e, r.routes.?.ids, 5);
        bind(e, experts.descriptors, 6);
        bind(e, rin.factors, 7);
        e.setValue(RopeInArgs{ .width = rin.width, .rotary = rin.rotary, .stride = rin.stride, .position = rin.position, .step = rin.step, .inverse = @intFromBool(rin.inverse) }, 8);
        e.dispatchGroups(.{ .width = (@as(usize, a.k) / 128 + 3) / 4, .height = a.rows }, .{ .width = 128 });
        e.barrier();
        plain.prerotated = true;
    }
    const next = f.next orelse {
        swi.stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, plain);
        return;
    };
    const kernel = p.chain.rotate_out_in orelse return error.UnsupportedExl3Chain;
    if (r.output != .bf16 or r.output_ids != null or r.grouped or r.y_stride != experts.n or experts.n % 128 != 0 or
        next.k % 128 != 0 or (r.count * experts.n) % next.k != 0 or experts.n > experts.k or
        next.k > std.math.maxInt(u32)) return error.InvalidExl3Chain;
    plain.no_epilogue = true;
    swi.stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, plain);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try core.rotatedBytes(a.k, a.rows) };
    e.setPipeline(kernel);
    bind(e, sums, 0);
    bind(e, scratch, 1);
    bind(e, scratch, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, r.routes.?.ids, 6);
    bind(e, experts.descriptors, 7);
    bind(e, next.su, 8);
    e.setValue(NextArgs{ .k = @intCast(next.k), .su_signs = @intFromBool(next.su_signs) }, 9);
    bind(e, scratch, 10);
    e.dispatchGroups(.{ .width = (@as(usize, a.n) / 128 + 3) / 4, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// A row-wise RMS norm (dsv41_rms) and the next projection's rotate_in; run that projection `prerotated`.
pub const Rms = struct {
    rows: u32,
    width: u32,
    x_stride: u32,
    y_stride: u32,
    eps: f32,
    weight_f32: bool,
    su_signs: bool,
};

const RmsArgs = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32, su_signs: u32, reserved: u32 = 0 };

/// Threads a row's threadgroup runs: one SIMDgroup a 128-block, and at least the eight of the norm's 256 threads.
fn rmsThreads(width: u32) usize {
    return 32 * @max(8, width / 128);
}

/// The widest row `encodeRmsRotateIn` takes on these pipelines (0 when the kernel is unavailable).
pub fn rmsRotateInWidth(p: *const Pipelines) u32 {
    const kernel = p.chain.rms_rotate_in orelse return 0;
    return @intCast(kernel.maxThreads() / 32 * 128);
}

pub fn encodeRmsRotateIn(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, weight: View, y: View, su: View, scratch: View, d: Rms) !void {
    const kernel = p.chain.rms_rotate_in orelse return error.UnsupportedExl3Chain;
    if (d.rows == 0 or d.width % 128 != 0 or d.width > rmsRotateInWidth(p) or d.x_stride < d.width or d.y_stride < d.width)
        return error.InvalidExl3Chain;
    const x_bytes = (@as(usize, d.rows - 1) * d.x_stride + d.width) * 2;
    const y_bytes = (@as(usize, d.rows - 1) * d.y_stride + d.width) * 2;
    const rotated_bytes = try core.rotatedBytes(d.width, d.rows);
    try core.check(x, x_bytes, 2);
    try core.check(y, y_bytes, 2);
    try core.check(scratch, rotated_bytes, 16);
    try core.disjoint(x, x_bytes, scratch, rotated_bytes);
    try core.disjoint(y, y_bytes, scratch, rotated_bytes);
    e.setPipeline(kernel);
    bind(e, x, 0);
    bind(e, weight, 1);
    bind(e, y, 2);
    e.setValue(RmsArgs{ .rows = d.rows, .width = d.width, .x_stride = d.x_stride, .y_stride = d.y_stride, .eps = d.eps, .weight_f32 = @intFromBool(d.weight_f32), .su_signs = @intFromBool(d.su_signs) }, 3);
    bind(e, su, 4);
    bind(e, scratch, 5);
    e.dispatchGroups(.{ .width = d.rows }, .{ .width = rmsThreads(d.width) });
    e.barrier();
}

/// A rope over each output row's heads: `width` columns a head, the last `rotary` rotated at `position + row * step`.
pub const Rope = struct { width: u32, rotary: u32, position: u32, step: u32 = 1, inverse: bool = false };

const RopeArgs = extern struct { width: u32, rotary: u32, position: u32, step: u32, inverse: u32, reserved0: u32 = 0, reserved1: u32 = 0, reserved2: u32 = 0 };

/// `encode` of a dense matrix with its epilogue fused with the output's RoPE (rotary columns a multiple of 32).
pub fn encodeRotateOutRope(p: *const Pipelines, e: mtl.ComputeEncoder, m: core.Matrix, x: View, y: View, scratch: View, r: Rows, factors: View, rope: Rope) !void {
    const kernel = p.chain.rotate_out_rope orelse return error.UnsupportedExl3Chain;
    if (r.output != .bf16 or r.output_ids != null or r.routes != null or r.y_stride != m.n or rope.width % 128 != 0 or
        m.n % rope.width != 0 or rope.rotary == 0 or rope.rotary % 32 != 0 or rope.rotary > 128 or r.count == 0)
        return error.InvalidExl3Chain;
    const a = try enc.matrixArgs(m, x, y, scratch, r);
    var plain = r;
    plain.no_epilogue = true;
    swi.stages(p, e, a, m.trellis, m.suh, m.svh, m.bias orelse m.svh, scratch, x, y, scratch, plain);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try core.rotatedBytes(a.k, a.rows) };
    e.setPipeline(kernel);
    bind(e, sums, 0);
    bind(e, m.svh, 1);
    bind(e, m.bias orelse m.svh, 2);
    bind(e, y, 3);
    e.setValue(a, 4);
    bind(e, scratch, 5);
    bind(e, scratch, 6);
    bind(e, scratch, 7);
    bind(e, factors, 8);
    e.setValue(RopeArgs{ .width = rope.width, .rotary = rope.rotary, .position = rope.position, .step = rope.step, .inverse = @intFromBool(rope.inverse) }, 9);
    e.dispatchGroups(.{ .width = (@as(usize, a.n) / 128 + 3) / 4, .height = a.rows }, .{ .width = 128 });
    e.barrier();
}

/// What rides on the merged query/key call's epilogue: row 0 the latent's norm and wq_b rotate_in, row 1 the key glue.
pub const Qkv = struct {
    q_weight: View,
    q_weight_f32: bool,
    q_su: View,
    q_su_signs: bool,
    kv_width: u32,
    kv_weight: View,
    factors: View,
    packed_view: View,
    packed_stride: u32,
    position: u32,
    rotary: u32,
    eps: f32,
};

const QkvArgs = extern struct { q_width: u32, q_su_signs: u32, q_weight_f32: u32, kv_width: u32, position: u32, rotary: u32, packed_stride: u32, block: u32, eps: f32, reserved0: u32 = 0, reserved1: u32 = 0, reserved2: u32 = 0 };

/// Whether `encodeQkvEpilogue` can run on these pipelines.
pub fn qkvEpilogueReady(p: *const Pipelines) bool {
    const kernel = p.chain.qkv_epilogue orelse return false;
    return kernel.maxThreads() >= 320;
}

/// The merged two-row indexed call with its epilogue and `f`'s work as one dispatch; run wq_b `prerotated`.
pub fn encodeQkvEpilogue(p: *const Pipelines, e: mtl.ComputeEncoder, experts: Experts, x: View, y: View, scratch: View, r: Rows, f: Qkv) !void {
    const kernel = p.chain.qkv_epilogue orelse return error.UnsupportedExl3Chain;
    if (r.count != 2 or experts.count < 2 or r.output != .bf16 or r.grouped or r.routes == null or experts.n % 128 != 0 or
        experts.n > 1280 or f.kv_width != 512 or f.rotary == 0 or f.rotary % 2 != 0 or f.rotary > f.kv_width or
        f.rotary % 32 != 0 or kernel.maxThreads() < 32 * @max(8, experts.n / 128)) return error.InvalidExl3Chain;
    const a = try enc.indexedArgs(experts, x, y, scratch, r);
    // The rotated latent lands at the scratch base, below the sums this dispatch reads.
    if (@as(usize, experts.n) * 2 > try core.rotatedBytes(a.k, a.rows)) return error.InvalidExl3Chain;
    var plain = r;
    plain.no_epilogue = true;
    swi.stages(p, e, a, scratch, scratch, scratch, scratch, experts.descriptors, x, y, scratch, plain);
    const sums = View{ .buffer = scratch.buffer, .offset = scratch.offset + try core.rotatedBytes(a.k, a.rows) };
    e.setPipeline(kernel);
    bind(e, sums, 0);
    bind(e, y, 1);
    e.setValue(a, 2);
    bind(e, r.output_ids orelse scratch, 3);
    bind(e, r.routes.?.ids, 4);
    bind(e, experts.descriptors, 5);
    bind(e, f.q_weight, 6);
    bind(e, f.q_su, 7);
    bind(e, scratch, 8);
    bind(e, f.kv_weight, 9);
    bind(e, f.factors, 10);
    bind(e, f.packed_view, 11);
    e.setValue(QkvArgs{ .q_width = @intCast(experts.n), .q_su_signs = @intFromBool(f.q_su_signs), .q_weight_f32 = @intFromBool(f.q_weight_f32), .kv_width = f.kv_width, .position = f.position, .rotary = f.rotary, .packed_stride = f.packed_stride, .block = 32, .eps = f.eps }, 12);
    e.dispatchGroups(.{ .width = a.rows }, .{ .width = 32 * @max(8, experts.n / 128) });
    e.barrier();
}
