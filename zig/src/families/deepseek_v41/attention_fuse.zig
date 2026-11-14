//! The decode attention chain's fused dispatches (one row); each has a kill switch (TF_DSV41_NO_FUSE_*).
const std = @import("std");
const environment = @import("sys").environment;
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const weights = @import("weights.zig");
const config = @import("config.zig");
const common = @import("attention_common.zig");
const View = common.View;
const Workspace = common.Workspace;
const core = common.core;
const tensor = common.tensor;
const kv = @import("kv.zig");

/// The fusion kill switches, read once: a process does not toggle them.
pub const Switches = struct {
    /// wo_a's epilogue and wo_b's rotate_in in one dispatch.
    wo_chain: bool,
    /// The query latent's RMS norm and wq_b's rotate_in in one dispatch.
    qnorm: bool,
    /// wq_b's epilogue and the query's RoPE in one dispatch.
    qrope: bool,
    /// The output's inverse RoPE and wo_a's rotate_in in one dispatch.
    orope: bool,
    /// The indexer's wq_b epilogue and its query RoPE in one dispatch.
    iqrope: bool,
    /// The merged query/key call's epilogue with the query latent's norm, wq_b's rotate_in and the key's window glue.
    qkv_tail: bool,
};

var cached: ?Switches = null;

pub fn switches() Switches {
    if (cached) |s| return s;
    const s: Switches = .{
        .wo_chain = !(environment.read(bool, "TF_DSV41_NO_FUSE_WO_CHAIN", false) catch unreachable),
        .qnorm = !(environment.read(bool, "TF_DSV41_NO_FUSE_QNORM", false) catch unreachable),
        .qrope = !(environment.read(bool, "TF_DSV41_NO_FUSE_QROPE", false) catch unreachable),
        .orope = !(environment.read(bool, "TF_DSV41_NO_FUSE_OROPE", false) catch unreachable),
        .iqrope = !(environment.read(bool, "TF_DSV41_NO_FUSE_IQROPE", false) catch unreachable),
        .qkv_tail = !(environment.read(bool, "TF_DSV41_NO_FUSE_QKV_TAIL", false) catch unreachable),
    };
    cached = s;
    return s;
}

/// The rope's rotation factors for a layer of compress ratio `ratio`: the compressed layers' follow the plain ones.
fn ropeFactors(ap: *const common.Pipelines, ratio: u32) exl3.View {
    const offset: usize = if (ratio == 0) 0 else @as(usize, ap.rotary.capacity) * ap.rotary.rotary * @sizeOf(f32);
    return .{ .buffer = ap.rotary.factors, .offset = offset };
}

/// Whether the indexer's query epilogue and its RoPE of one decode row can run as one dispatch.
pub fn iqRopeReady(ep: *const exl3.Pipelines, ap: *const common.Pipelines, c: config.Config, position: u32) bool {
    return switches().iqrope and ep.chain.rotate_out_rope != null and ap.rotary.rotary == c.rope_dim and
        position < ap.rotary.capacity and c.index_dim % 128 == 0 and c.rope_dim % 32 == 0 and c.rope_dim <= 128;
}

/// The indexer's wq_b over the normalized latent into `iq`, with the query RoPE (heads of `index_dim` columns) fused.
pub fn encodeIqRope(ep: *const exl3.Pipelines, ap: *const common.Pipelines, e: mtl.ComputeEncoder, m: weights.Matrix, qr: View, iq: View, scratch: exl3.View, c: config.Config, ratio: u32, position: u32) !void {
    try exl3.chain.encodeRotateOutRope(ep, e, m, core(qr), core(iq), scratch, .{ .count = 1, .x_stride = c.q_lora, .y_stride = c.index_heads * c.index_dim }, ropeFactors(ap, ratio), .{
        .width = c.index_dim,
        .rotary = c.rope_dim,
        .position = position,
    });
}

/// The eight output groups' indexed wo_a call, optionally with the output rope in and wo_b's rotate_in out.
pub fn encodeWoA(ep: *const exl3.Pipelines, ap: *const common.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, work: *const Workspace, q: View, grouped: View, scratch: exl3.View, c: config.Config, chained: bool, rope: ?struct { ratio: u32, position: u32 }) !void {
    for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
    const q_width = c.heads * c.head_dim;
    const group_input = q_width / c.o_groups;
    var f: exl3.chain.Fused = .{};
    if (rope) |r| f.rope_in = .{ .factors = ropeFactors(ap, r.ratio), .width = c.head_dim, .rotary = c.rope_dim, .stride = q_width, .position = r.position, .inverse = true };
    if (chained) f.next = .{ .su = w.wo_b.suh, .su_signs = w.wo_b.suh_format == .packed_signs, .k = c.o_groups * c.o_lora };
    try exl3.chain.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
        .count = c.o_groups,
        .x_stride = group_input,
        .y_stride = c.o_lora,
        .output = .bf16,
        .output_rows = c.o_groups,
        .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
    }, f);
}

/// Whether the output's inverse RoPE can wait for wo_a's rotate_in (`encodeWoA` with `rope`), one decode row.
pub fn oRopeReady(ep: *const exl3.Pipelines, ap: *const common.Pipelines, w: *const weights.Attention, c: config.Config, position: u32) bool {
    return switches().orope and w.wo_a_descriptors != null and exl3.chain.ropeRotateInReady(ep) and
        ap.rotary.rotary == c.rope_dim and position < ap.rotary.capacity and c.head_dim % 128 == 0 and
        c.rope_dim % 32 == 0 and c.rope_dim <= 128 and (c.heads * c.head_dim / c.o_groups) % c.head_dim == 0 and
        std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null and std.c.getenv("TF_DSV41_DOUBLE_ATTN_WOA") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null;
}

/// wo_b over the rotated input `encodeWoA` left in the scratch base.
pub fn encodeWoB(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, y: View, scratch: exl3.View, c: config.Config) !void {
    try exl3.encode(ep, e, w.wo_b, scratch, core(y), scratch, .{ .count = 1, .x_stride = c.o_groups * c.o_lora, .y_stride = c.hidden, .prerotated = true });
}

/// Whether one decode row's `qr` norm and wq_b's rotate_in can run as one dispatch.
pub fn qNormReady(ep: *const exl3.Pipelines, w: *const weights.Attention, c: config.Config) bool {
    return switches().qnorm and w.wq_b.k == c.q_lora and c.q_lora <= exl3.chain.rmsRotateInWidth(ep) and
        std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") == null;
}

/// `qr = q_norm(qr)` in place, and wq_b's rotated input in the scratch base (`encodeQb` runs wq_b).
pub fn encodeQNorm(ep: *const exl3.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, qr: View, scratch: exl3.View, c: config.Config) !void {
    try exl3.chain.encodeRmsRotateIn(ep, e, core(qr), core(tensor(w.q_norm)), core(qr), w.wq_b.suh, scratch, .{
        .rows = 1,
        .width = c.q_lora,
        .x_stride = c.q_lora,
        .y_stride = c.q_lora,
        .eps = c.eps,
        .weight_f32 = w.q_norm.dtype == .f32,
        .su_signs = w.wq_b.suh_format == .packed_signs,
    });
}

/// Whether wq_b's epilogue and the query's RoPE of one decode row can run as one dispatch.
pub fn qRopeReady(ep: *const exl3.Pipelines, ap: *const common.Pipelines, c: config.Config, position: u32) bool {
    return switches().qrope and ep.chain.rotate_out_rope != null and ap.rotary.rotary == c.rope_dim and
        position < ap.rotary.capacity and c.head_dim % 128 == 0 and c.rope_dim % 32 == 0 and c.rope_dim <= 128 and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null;
}

/// wq_b into `q` (`rotated`: the input is already rotated in the scratch base), the query RoPE optionally fused.
pub fn encodeQb(ep: *const exl3.Pipelines, ap: *const common.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, qr: View, q: View, scratch: exl3.View, c: config.Config, rotated: bool, rope: ?struct { ratio: u32, position: u32 }) !void {
    const rows: exl3.Rows = .{ .count = 1, .x_stride = c.q_lora, .y_stride = c.heads * c.head_dim, .prerotated = rotated };
    const x = if (rotated) scratch else core(qr);
    const r = rope orelse return exl3.encode(ep, e, w.wq_b, x, core(q), scratch, rows);
    try exl3.chain.encodeRotateOutRope(ep, e, w.wq_b, x, core(q), scratch, rows, ropeFactors(ap, r.ratio), .{
        .width = c.head_dim,
        .rotary = c.rope_dim,
        .position = r.position,
    });
}

/// Whether the merged call's epilogue can carry the query norm, wq_b's rotate_in and the key glue (one decode row).
pub fn qkvTailReady(ep: *const exl3.Pipelines, ap: *const common.Pipelines, w: *const weights.Attention, c: config.Config, position: u32) bool {
    const sw = switches();
    return sw.qkv_tail and sw.qnorm and std.c.getenv("TF_DSV41_NO_FUSE_KVGLUE") == null and
        exl3.chain.qkvEpilogueReady(ep) and w.kv_norm.dtype == .bf16 and c.head_dim == 512 and c.q_lora % 128 == 0 and
        c.q_lora <= 1280 and w.wq_b.k == c.q_lora and ap.rotary.rotary == c.rope_dim and position < ap.rotary.capacity and
        c.rope_dim % 32 == 0 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null and std.c.getenv("TF_DSV41_DOUBLE_ATTN_PACK") == null;
}

/// The merged wq_a + wkv call with the tail `qkvTailReady` names fused in; the caller runs wq_b `prerotated`.
pub fn encodeQkvTail(ep: *const exl3.Pipelines, ap: *const common.Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, work: *const Workspace, x: View, fused: View, chunk: View, scratch: exl3.View, c: config.Config, ratio: u32, position: u32) !void {
    try exl3.chain.encodeQkvEpilogue(ep, e, .{ .descriptors = .{ .buffer = w.qkv_descriptors.?, .offset = 0 }, .count = 2, .k = c.hidden, .n = c.q_lora }, core(x), core(fused), scratch, .{
        .count = 2,
        .x_stride = c.hidden,
        .y_stride = c.q_lora,
        .output = .bf16,
        .input_ids = .{ .buffer = work.pair_in, .offset = 0 },
        .input_rows = 1,
        .output_ids = .{ .buffer = work.ids, .offset = 0 },
        .output_rows = 2,
        .routes = .{ .ids = .{ .buffer = work.ids, .offset = 0 }, .repeat = 1 },
    }, .{
        .q_weight = core(tensor(w.q_norm)),
        .q_weight_f32 = w.q_norm.dtype == .f32,
        .q_su = w.wq_b.suh,
        .q_su_signs = w.wq_b.suh_format == .packed_signs,
        .kv_width = c.head_dim,
        .kv_weight = core(tensor(w.kv_norm)),
        .factors = ropeFactors(ap, ratio),
        .packed_view = core(chunk),
        .packed_stride = kv.cache.ring_stride,
        .position = position,
        .rotary = c.rope_dim,
        .eps = c.eps,
    });
}
