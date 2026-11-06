//! Trusted plaintext encoders: callers validate nonzero shapes/storage; strides count elements, EXL3 is separate.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("core").frags;
pub const View = @import("view.zig").View;

pub const Pipelines = struct {
    rms: mtl.Pipeline,
    dense: mtl.Pipeline,
    embedding: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    combine: mtl.Pipeline,
    add: mtl.Pipeline,
    stream_mean: mtl.Pipeline,
    copy: mtl.Pipeline,
    markov: mtl.Pipeline,
    markov_pick: mtl.Pipeline,
    /// MLX's bf16 tensor-unit GEMM for the router gate (TF_DSV41_ROUTE_GEMM): bf16 logits, N = 384.
    gate: ?mtl.Pipeline = null,
    /// The same gate GEMM with an fp32 destination: the router's own precision, so the picks and weights are the
    /// shipped path's.
    gate_f32: ?mtl.Pipeline = null,
    gate_f16_f32: ?mtl.Pipeline = null,
    /// F16 -> BF16 for a pack whose gate is F16 (the GEMM's operands share one element type).
    cast_bf16: ?mtl.Pipeline = null,
    /// BF16 -> F16 for the f16 gate GEMM: the pack's gate keeps its 10 mantissa bits and the activation moves.
    cast_f16: ?mtl.Pipeline = null,
    /// The same dense arithmetic with one column a SIMDgroup (bit-identical), for a narrow n.
    dense_col: ?mtl.Pipeline = null,
    /// The dtype-specialized twins of dense_col (same operations and order, loads hoisted).
    dense_col_bf16: ?mtl.Pipeline = null,
    dense_col_f16: ?mtl.Pipeline = null,
    dump_f32: ?mtl.Pipeline = null,
    dump_bf16: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        // The router gate's GEMM: the lane's own prefill block with an fp32 destination, nax.h inlined, in the
        // same library as the ops so one pipeline set covers both.
        const gemm_text = frags.source(device, std.heap.page_allocator, sources.deepseek_v41_gate_gemm) catch null;
        defer if (gemm_text) |g| std.heap.page_allocator.free(g);
        const text = if (gemm_text) |g| try std.mem.concat(std.heap.page_allocator, u8, &.{ sources.deepseek_v41_ops, g }) else sources.deepseek_v41_ops;
        defer if (gemm_text != null) std.heap.page_allocator.free(text);
        const lib = try mtl.Library.fromSource(device, text, options);
        defer lib.deinit();
        const rms = try mtl.Pipeline.init(device, lib, "dsv41_rms", false);
        errdefer rms.deinit();
        const dense = try mtl.Pipeline.init(device, lib, "dsv41_dense", false);
        errdefer dense.deinit();
        const embedding = try mtl.Pipeline.init(device, lib, "dsv41_embedding", false);
        errdefer embedding.deinit();
        const swiglu = try mtl.Pipeline.init(device, lib, "dsv41_swiglu", false);
        errdefer swiglu.deinit();
        const combine = try mtl.Pipeline.init(device, lib, "dsv41_combine", false);
        const add = try mtl.Pipeline.init(device, lib, "dsv41_add", false);
        const stream_mean = try mtl.Pipeline.init(device, lib, "dsv41_stream_mean", false);
        errdefer stream_mean.deinit();
        errdefer add.deinit();
        errdefer combine.deinit();
        const copy = try mtl.Pipeline.init(device, lib, "dsv41_copy", false);
        errdefer copy.deinit();
        const markov = try mtl.Pipeline.init(device, lib, "dsv41_markov", false);
        errdefer markov.deinit();
        const markov_pick = try mtl.Pipeline.init(device, lib, "dsv41_markov_pick", false);
        errdefer markov_pick.deinit();
        const dense_col = mtl.Pipeline.init(device, lib, "dsv41_dense_col1", false) catch null;
        const dense_col_bf16 = mtl.Pipeline.init(device, lib, "dsv41_dense_col1_bf16", false) catch null;
        const dense_col_f16 = mtl.Pipeline.init(device, lib, "dsv41_dense_col1_f16", false) catch null;
        const cast_bf16 = mtl.Pipeline.init(device, lib, "dsv41_f16_to_bf16", false) catch null;
        const cast_f16 = mtl.Pipeline.init(device, lib, "dsv41_bf16_to_f16", false) catch null;
        const dump_f32 = mtl.Pipeline.init(device, lib, "dsv41_dump_f32", false) catch null;
        const dump_bf16 = mtl.Pipeline.init(device, lib, "dsv41_dump_bf16", false) catch null;
        var gate: ?mtl.Pipeline = null;
        var gate_f32: ?mtl.Pipeline = null;
        var gate_f16_f32: ?mtl.Pipeline = null;
        if (gemm_text != null) {
            // N = 384 is 128-aligned, a prompt's rows are not 64-aligned: the bounded variant.
            gate = mtl.Pipeline.init(device, lib, "custom_kernel_tf_gemm_nax_bf16_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", false) catch null;
            // The hand-added fp32-destination twin (gemm_nax.metal): fp32 sums out, so route reads f32 logits.
            gate_f32 = mtl.Pipeline.init(device, lib, "custom_kernel_tf_gemm_nax_bf16_f32out_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_float", false) catch null;
            if (gate_f32 == null) std.log.warn("gate GEMM: fp32-destination entry absent; TF_DSV41_ROUTE_GEMM stays on tf-dense-col", .{});
            gate_f16_f32 = mtl.Pipeline.init(device, lib, "custom_kernel_tf_gemm_nax_f16_f32out_n_t_n_t_n_64_128_256_2_4_float16_t_float16_t_int32_t_float", false) catch null;
            if (gate_f16_f32 == null) std.log.warn("gate GEMM: f16 entry absent; an f16 gate pack stays on tf-dense-col", .{});
        }
        return .{ .rms = rms, .dense = dense, .dense_col = dense_col, .dense_col_bf16 = dense_col_bf16, .dense_col_f16 = dense_col_f16, .embedding = embedding, .swiglu = swiglu, .combine = combine, .add = add, .stream_mean = stream_mean, .copy = copy, .markov = markov, .markov_pick = markov_pick, .gate = gate, .gate_f32 = gate_f32, .gate_f16_f32 = gate_f16_f32, .cast_bf16 = cast_bf16, .cast_f16 = cast_f16, .dump_f32 = dump_f32, .dump_bf16 = dump_bf16 };
    }
    pub fn deinit(p: *Pipelines) void {
        if (p.gate) |g| g.deinit();
        if (p.gate_f32) |g| g.deinit();
        if (p.gate_f16_f32) |g| g.deinit();
        if (p.cast_bf16) |g| g.deinit();
        if (p.cast_f16) |g| g.deinit();
        if (p.dump_f32) |g| g.deinit();
        if (p.dump_bf16) |g| g.deinit();
        if (p.dense_col) |g| g.deinit();
        if (p.dense_col_bf16) |g| g.deinit();
        if (p.dense_col_f16) |g| g.deinit();
        p.rms.deinit();
        p.dense.deinit();
        p.embedding.deinit();
        p.swiglu.deinit();
        p.combine.deinit();
        p.add.deinit();
        p.stream_mean.deinit();
        p.copy.deinit();
        p.markov.deinit();
        p.markov_pick.deinit();
    }
};
fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}
pub const Rms = struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: bool = false };
/// The kernel's `Rms`: the bool above selects the weight's dtype, the kernel reads a uint.
pub const RmsArgs = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32 };
/// BF16 x/y [rows,width], BF16 or FP32 weight [width]; sums, mean and affine product are FP32.
pub fn encodeRms(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, weight: View, y: View, d: Rms) void {
    e.setPipeline(p.rms);
    bind(e, .{ x, weight, y });
    e.setValue(RmsArgs{ .rows = d.rows, .width = d.width, .x_stride = d.x_stride, .y_stride = d.y_stride, .eps = d.eps, .weight_f32 = @intFromBool(d.weight_f32) }, 3);
    e.dispatchGroups(mtl.Size.of(d.rows, 1, 1), mtl.Size.of(256, 1, 1));
}
/// BF16 -> F16 over `n` elements: exact for a bf16 significand inside f16's range, the f16 gate GEMM's bridge.
pub fn encodeBf16ToF16(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, y: View, n: u32) void {
    const pipe = p.cast_f16 orelse return;
    e.setPipeline(pipe);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(y.buffer, y.offset, 1);
    e.setBytes(std.mem.asBytes(&n), 2);
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

/// F16 -> BF16 over `n` elements (round to nearest even), the gate GEMM's operand bridge.
pub fn encodeF16ToBf16(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, y: View, n: u32) void {
    const pipe = p.cast_bf16 orelse return;
    e.setPipeline(pipe);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(y.buffer, y.offset, 1);
    e.setBytes(std.mem.asBytes(&n), 2);
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

pub fn encodeDump(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, y: View, n: u32, bf16_src: bool) void {
    const pipe = if (bf16_src) p.dump_bf16 else p.dump_f32;
    const q = pipe orelse return;
    e.setPipeline(q);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(y.buffer, y.offset, 1);
    e.setBytes(std.mem.asBytes(&n), 2);
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(64, 1, 1));
}

pub const Gate = struct { rows: u32, k: u32, n: u32 = 384 };
/// x[rows, k] bf16 @ gate[n, k] bf16 -> y[rows, n] bf16 on the tensor units (MLX's gemm_nax, fp32 sums). The
/// same dispatch step5/vision.zig uses: P carries the shapes, the strides, the tile counts and the swizzle.
pub fn encodeGateGemm(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Gate) void {
    const pipe = p.gate orelse return;
    const tn = (d.n + 127) / 128;
    const tm = (d.rows + 63) / 64;
    e.setPipeline(pipe);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(w.buffer, w.offset, 1);
    var q: [16]i32 = @splat(0);
    const vals = [_]u32{ d.rows, d.n, d.k, d.k, d.k, d.n, tn, tm, 2, d.k / 256, 0, 0, 0, 0, 0 };
    for (vals, 0..) |v, i| q[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&q), 2);
    e.setBuffer(y.buffer, y.offset, 3);
    e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}
/// x[rows, k] bf16 @ gate[n, k] bf16 -> y[rows, n] f32 on the tensor units: the same block and k order as
/// encodeGateGemm, with an fp32 destination so the router reads the precision tf-dense-col gives it.
pub fn encodeGateGemmF16F32(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Gate) void {
    const pipe = p.gate_f16_f32 orelse return;
    const tn = (d.n + 127) / 128;
    const tm = (d.rows + 63) / 64;
    e.setPipeline(pipe);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(w.buffer, w.offset, 1);
    var q: [16]i32 = @splat(0);
    const vals = [_]u32{ d.rows, d.n, d.k, d.k, d.k, d.n, tn, tm, 2, d.k / 256, 0, 0, 0, 0, 0 };
    for (vals, 0..) |v, k| q[k] = @intCast(v);
    e.setBytes(std.mem.asBytes(&q), 2);
    e.setBuffer(y.buffer, y.offset, 3);
    e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}

pub fn encodeGateGemmF32(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Gate) void {
    const pipe = p.gate_f32 orelse return;
    const tn = (d.n + 127) / 128;
    const tm = (d.rows + 63) / 64;
    e.setPipeline(pipe);
    e.setBuffer(x.buffer, x.offset, 0);
    e.setBuffer(w.buffer, w.offset, 1);
    var q: [16]i32 = @splat(0);
    const vals = [_]u32{ d.rows, d.n, d.k, d.k, d.k, d.n, tn, tm, 2, d.k / 256, 0, 0, 0, 0, 0 };
    for (vals, 0..) |v, i| q[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&q), 2);
    e.setBuffer(y.buffer, y.offset, 3);
    e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}

pub const Dtype = enum(u32) { bf16 = 0, f32 = 1, f16 = 2 };
pub const Dense = struct {
    rows: u32,
    k: u32,
    n: u32,
    groups: u32 = 1,
    x_stride: u32,
    w_stride: u32,
    y_stride: u32,
    x_dtype: Dtype = .bf16,
    w_dtype: Dtype = .bf16,
    y_dtype: Dtype = .bf16,
};

/// The kernel's `Dense` record, field for field (it was a positional payload until this round).
pub const DenseArgs = extern struct { rows: u32, k: u32, n: u32, groups: u32, x_stride: u32, w_stride: u32, y_stride: u32, x_dtype: u32, w_dtype: u32, y_dtype: u32 };

/// x[rows,groups,k], w[groups,n,k], y[rows,groups,n]; packed groups, padded token pitches, FP32 sums, final BF16.
pub fn encodeDense(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Dense) void {
    e.setPipeline(p.dense);
    bind(e, .{ x, w, y });
    e.setValue(DenseArgs{ .rows = d.rows, .k = d.k, .n = d.n, .groups = d.groups, .x_stride = d.x_stride, .w_stride = d.w_stride, .y_stride = d.y_stride, .x_dtype = @backingInt(d.x_dtype), .w_dtype = @backingInt(d.w_dtype), .y_dtype = @backingInt(d.y_dtype) }, 3);
    e.dispatchGroups(mtl.Size.of((d.n + 15) / 16, d.rows, d.groups), mtl.Size.of(128, 1, 1));
}
/// The same arithmetic and the same per-column sums, one column a SIMDgroup (bit-identical). X=bf16, y=bf16 and k %
/// 128 == 0 take the specialized twin (same sums, loads hoisted); every other shape takes the shipped kernel.
pub fn encodeDenseCol(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Dense) bool {
    if (!denseColSpecOff() and d.x_dtype == .bf16 and d.y_dtype == .bf16 and d.k % 128 == 0) {
        const spec = if (d.w_dtype == .bf16) p.dense_col_bf16 else if (d.w_dtype == .f16) p.dense_col_f16 else null;
        if (spec) |pipe| {
            e.setPipeline(pipe);
            bind(e, .{ x, w, y });
            e.setValue(DenseArgs{ .rows = d.rows, .k = d.k, .n = d.n, .groups = d.groups, .x_stride = d.x_stride, .w_stride = d.w_stride, .y_stride = d.y_stride, .x_dtype = @backingInt(d.x_dtype), .w_dtype = @backingInt(d.w_dtype), .y_dtype = @backingInt(d.y_dtype) }, 3);
            e.dispatchGroups(mtl.Size.of((d.n + 3) / 4, d.rows, d.groups), mtl.Size.of(128, 1, 1));
            return true;
        }
    }
    return encodeDenseColGeneric(e, p, x, w, y, d);
}
/// TF_DSV41_DENSE_COL_SPEC=0 is the same-binary control: the shipped one-column kernel.
fn denseColSpecOff() bool {
    const S = struct {
        var cached: ?bool = null;
    };
    if (S.cached) |v| return v;
    var v = false;
    if (std.c.getenv("TF_DSV41_DENSE_COL_SPEC")) |e| v = std.mem.eql(u8, std.mem.span(e), "0");
    S.cached = v;
    return v;
}
pub fn encodeDenseColGeneric(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, w: View, y: View, d: Dense) bool {
    const pipe = p.dense_col orelse return false;
    e.setPipeline(pipe);
    bind(e, .{ x, w, y });
    e.setValue(DenseArgs{ .rows = d.rows, .k = d.k, .n = d.n, .groups = d.groups, .x_stride = d.x_stride, .w_stride = d.w_stride, .y_stride = d.y_stride, .x_dtype = @backingInt(d.x_dtype), .w_dtype = @backingInt(d.w_dtype), .y_dtype = @backingInt(d.y_dtype) }, 3);
    e.dispatchGroups(mtl.Size.of((d.n + 3) / 4, d.rows, d.groups), mtl.Size.of(128, 1, 1));
    return true;
}
pub const Embedding = extern struct { rows: u32, width: u32, table_stride: u32, y_stride: u32, copies: u32 = 4 };
comptime {
    @import("metal/index_mirror.zig").check(Embedding, sources.deepseek_v41_ops, "Embedding");
}
/// ids[rows] u32 must be in vocabulary; table[vocab,width] BF16 -> y[rows,copies,width] BF16.
pub fn encodeEmbedding(e: mtl.ComputeEncoder, p: *const Pipelines, ids: View, table: View, y: View, d: Embedding) void {
    e.setPipeline(p.embedding);
    bind(e, .{ ids, table, y });
    e.setValue(d, 3);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
pub const Activation = struct {
    rows: u32,
    width: u32,
    input_stride: u32,
    output_stride: u32,
    weighted: bool = false,
    input_f32: bool = false,
};

/// The kernel's `Activation` record, field for field (`weighted`/`input_f32` as the kernel's uints).
pub const ActivationArgs = extern struct { rows: u32, width: u32, input_stride: u32, output_stride: u32, weighted: u32, input_f32: u32 };

/// Separate gate/up; apply FP32 weights before BF16 cast/down. Unweighted calls borrow but never read weights.
pub fn encodeSwiGLU(e: mtl.ComputeEncoder, p: *const Pipelines, gate: View, up: View, weights: View, y: View, d: Activation) void {
    e.setPipeline(p.swiglu);
    bind(e, .{ gate, up, weights, y });
    e.setValue(ActivationArgs{ .rows = d.rows, .width = d.width, .input_stride = d.input_stride, .output_stride = d.output_stride, .weighted = @intFromBool(d.weighted), .input_f32 = @intFromBool(d.input_f32) }, 4);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}
pub const Copy = struct { rows: u32, width: u32, x_stride: u32, y_stride: u32 };

/// The kernel's `Copy` record, field for field.
pub const CopyArgs = extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32 };

/// BF16 rows copied verbatim: what the drafter's tap does to a layer's hidden state before the forward reuses it.
pub fn encodeCopy(e: mtl.ComputeEncoder, p: *const Pipelines, x: View, y: View, d: Copy) void {
    e.setPipeline(p.copy);
    bind(e, .{ x, y });
    e.setValue(CopyArgs{ .rows = d.rows, .width = d.width, .x_stride = d.x_stride, .y_stride = d.y_stride }, 2);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}

pub const Combine = struct {
    rows: u32,
    width: u32,
    routed_row_stride: u32,
    slot_stride: u32,
    shared_stride: u32,
    y_stride: u32,
    input_f32: bool = false,
    id_stride: u32 = 6,
};

/// The kernel's `Combine` record, field for field (`input_f32` as the kernel's uint).
pub const CombineArgs = extern struct { rows: u32, width: u32, routed_row_stride: u32, slot_stride: u32, shared_stride: u32, y_stride: u32, input_f32: u32, id_stride: u32 };

/// Sum six weighted down slots by ascending expert ID in FP32, then shared; final BF16, invalid IDs yield NaN.
pub const StreamMean = extern struct { rows: u32, width: u32, stream_stride: u32, out_stride: u32, slot: u32, row_base: u32 = 0, ring: u32 = 128 };
comptime {
    @import("metal/index_mirror.zig").check(StreamMean, sources.deepseek_v41_ops, "StreamMean");
}
/// The mean of the four hyper-connection stream copies, one row into `out` at `slot * rows`.
pub fn encodeStreamMean(e: mtl.ComputeEncoder, p: *const Pipelines, stream: View, out: View, d: StreamMean) void {
    e.setPipeline(p.stream_mean);
    bind(e, .{ stream, out });
    e.setValue(d, 2);
    e.dispatchGroups(mtl.Size.of((d.width + 63) / 64, d.rows, 1), mtl.Size.of(64, 1, 1));
}

pub const Add = extern struct { rows: u32, width: u32, a_stride: u32, b_stride: u32, y_stride: u32 };
/// y = a + b, a row at a time: the drafter's projected hidden state plus a token's embedding.
pub fn encodeAdd(e: mtl.ComputeEncoder, p: *const Pipelines, a: View, b: View, y: View, d: Add) void {
    e.setPipeline(p.add);
    bind(e, .{ a, b, y });
    e.setValue(d, 3);
    e.dispatchGroups(mtl.Size.of((d.width + 63) / 64, d.rows, 1), mtl.Size.of(64, 1, 1));
}

pub fn encodeCombine(e: mtl.ComputeEncoder, p: *const Pipelines, routed: View, shared: View, ids: View, y: View, d: Combine) void {
    e.setPipeline(p.combine);
    bind(e, .{ routed, shared, ids, y });
    e.setValue(CombineArgs{ .rows = d.rows, .width = d.width, .routed_row_stride = d.routed_row_stride, .slot_stride = d.slot_stride, .shared_stride = d.shared_stride, .y_stride = d.y_stride, .input_f32 = @intFromBool(d.input_f32), .id_stride = d.id_stride }, 4);
    e.dispatchThreads(mtl.Size.of(d.width, d.rows, 1), mtl.Size.of(256, 1, 1));
}

pub const Markov = extern struct { vocab: i32, rank: i32, step: i32, pad: i32 = 0 };
/// One Markov step: logits [rows, vocab] f32 row `step`, bf16 embed/head [vocab, rank], ids u32; bests to part.
pub fn encodeMarkov(e: mtl.ComputeEncoder, p: *const Pipelines, logits: View, embed: View, head: View, ids: View, part: View, d: Markov) void {
    e.setPipeline(p.markov);
    bind(e, .{ logits, embed, head, ids, part });
    e.setValue(d, 5);
    const groups: usize = @intCast(@divTrunc(d.vocab + 255, 256));
    e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(256, 1, 1));
}
/// The step's token: the best of part's `groups` bests into ids[step + 1].
pub fn encodeMarkovPick(e: mtl.ComputeEncoder, p: *const Pipelines, part: View, ids: View, groups: u32, step: u32) void {
    e.setPipeline(p.markov_pick);
    bind(e, .{ part, ids });
    e.setValue([2]i32{ @intCast(groups), @intCast(step) }, 2);
    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
}
