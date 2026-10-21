//! Pinned DeepSeek V4.1 attention, resident EXL3 projections and causal packed-cache chunk execution.
const mtl = @import("metal");
const exl3 = @import("../../core/exl3_projection.zig");
const checkpoint = @import("../../core/checkpoint_metal.zig");
const weights = @import("weights.zig");
const config = @import("config.zig");
const ops = @import("ops.zig");
const csa2 = @import("csa2.zig");
const kv = @import("kv.zig");
const index = @import("csa2_index.zig");
const rope = @import("rope.zig");
pub const View = @import("view.zig").View;
pub const State = kv.cache.State;

/// Opt-in bounded intermediate capture; ordinary execution does not install an observer.
pub const DiagnosticStage = enum {
    qr_projection,
    qr_norm,
    q_projection,
    q,
    kv_projection,
    kv_norm,
    kv_rope,
    kv_quantized,
    sparse_o,
    o,
    grouped,
    wo_b,
};
pub const DiagnosticObserver = struct {
    ctx: *anyopaque,
    capture: *const fn (*anyopaque, mtl.ComputeEncoder, DiagnosticStage, View) anyerror!void,

    fn observe(comptime enabled: bool, self: ?DiagnosticObserver, e: mtl.ComputeEncoder, stage: DiagnosticStage, source: View) !void {
        if (enabled) if (self) |observer| try observer.capture(observer.ctx, e, stage, source);
    }
};

pub const Pipelines = struct {
    plain: ops.Pipelines,
    cache: kv.Pipelines,
    select: index.Pipelines,
    rotary: rope.Pipelines,
    sparse: csa2.Pipelines,

    pub fn load(device: mtl.Device, capacity: u32, c: config.Config) !Pipelines {
        var plain = try ops.Pipelines.load(device);
        errdefer plain.deinit();
        var cache = try kv.Pipelines.load(device);
        errdefer cache.deinit();
        var select = try index.Pipelines.load(device);
        errdefer select.deinit();
        var rotary = try rope.Pipelines.load(device, capacity, c.rope_dim, .{ .theta = @floatCast(c.rope.theta), .original = 0 }, .{ .theta = @floatCast(c.rope.compress_theta), .factor = @floatCast(c.rope.factor), .original = @intCast(c.rope.original_max_positions), .beta_fast = @floatCast(c.rope.beta_fast), .beta_slow = @floatCast(c.rope.beta_slow) });
        errdefer rotary.deinit();
        const sparse = try csa2.Pipelines.load(device);
        return .{ .plain = plain, .cache = cache, .select = select, .rotary = rotary, .sparse = sparse };
    }
    pub fn residentBytes(capacity: u32, c: config.Config) usize {
        return rope.Pipelines.residentBytes(capacity, c.rope_dim);
    }
    pub fn registerResidency(p: *const Pipelines, set: mtl.ResidencySet) void {
        p.rotary.registerResidency(set);
    }
    pub fn deinit(p: *Pipelines) void {
        p.plain.deinit();
        p.cache.deinit();
        p.select.deinit();
        p.rotary.deinit();
        p.sparse.deinit();
    }
};

// Row-sized regions never contain dequantized weights or a query-by-context tensor.
const regions = [_]usize{ 1280 * 2, 32768 * 2, 512 * 2, 512 * 4, 512 * 4, 512 * 2, 128 * 2, 4096 * 2, 32 * 2, 8192 * 2 };
const Region = enum(usize) { qr, query, raw_kv, values, gates, latent, index_k, index_q, index_weights, grouped };

pub const Workspace = struct {
    buffer: mtl.Buffer,
    selection: index.Scratch,
    max_rows: u32,
    selection_layer: ?u32 = null,
    selection_position: u32 = 0,
    selection_rows: u32 = 0,
    candidate_position: u32 = 0,
    candidate_rows: u32 = 0,

    pub fn bytes(max_rows: u32) usize {
        var per_row: usize = 0;
        for (regions) |size| per_row += size;
        return @as(usize, max_rows) * per_row + index.Scratch.bytes(max_rows);
    }
    pub fn init(device: mtl.Device, max_rows: u32) !Workspace {
        if (max_rows == 0 or max_rows > 4096) return error.InvalidAttentionWorkspace;
        const buffer = try device.buffer(bytes(max_rows) - index.Scratch.bytes(max_rows), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        const selection = try index.Scratch.init(device, max_rows);
        return .{ .buffer = buffer, .selection = selection, .max_rows = max_rows };
    }
    pub fn deinit(w: *Workspace) void {
        w.buffer.deinit();
        w.selection.deinit();
    }
    pub fn residentBytes(w: *const Workspace) usize {
        return bytes(w.max_rows);
    }
    pub fn registerResidency(w: *const Workspace, set: mtl.ResidencySet) void {
        set.add(w.buffer);
        set.add(w.selection.buffer);
    }
    fn view(w: *const Workspace, region: Region) View {
        var offset: usize = 0;
        for (regions[0..@backingInt(region)]) |size| offset += size * w.max_rows;
        return .{ .buffer = w.buffer, .offset = offset };
    }
};

fn tensor(t: checkpoint.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}
fn core(v: View) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}
fn projection(p: *const exl3.Pipelines, e: mtl.ComputeEncoder, m: weights.AnyMatrix, x: View, y: View, scratch: exl3.View, rows: u32, xs: u32, ys: u32, output: exl3.Output) !void {
    switch (m) {
        .trellis => |t| try exl3.encode(p, e, t, core(x), core(y), scratch, .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output }),
        // A kept-dense repack: the same rotate stages, the middle a plain dense product.
        .dense => |d| try exl3.encodeDense(p, e, .{
            .weight = core(tensor(d.weight)),
            .suh = core(tensor(d.suh)),
            .svh = core(tensor(d.svh)),
            .k = d.k,
            .n = d.n,
        }, core(x), core(y), scratch, .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output }),
    }
}
fn norm(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, weight: checkpoint.Tensor, rows: u32, width: u32, eps: f32) void {
    ops.encodeRms(e, &p.plain, x, tensor(weight), x, .{ .rows = rows, .width = width, .x_stride = width, .y_stride = width, .eps = eps, .weight_f32 = weight.dtype == .f32 });
    e.barrier();
}
fn rotate(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, rows: u32, heads: u32, width: u32, position: u32, ratio: u32, step: u32, inverse: bool, c: config.Config) !void {
    try p.rotary.encode(e, x, .{ .rows = rows, .heads = heads, .width = width, .rotary = c.rope_dim, .stride = heads * width, .position = position, .step = step, .inverse = @intFromBool(inverse), .spectrum = if (ratio == 0) .normal else .compressed });
}
fn precedingIndex(layer: u32) u32 {
    var source: u32 = 2;
    for (kv.cache.index_sources) |id| {
        if (id > layer) break;
        source = id;
    }
    return source;
}

/// BF16 x/y [rows,5120]; layers 0..39 share the chunk-start cursor, appended by the caller after all layers.

// No extra RMS after wq_b or index-query Hadamard outside EXL3.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config) !void {
    return encodeDiagnostic(p, e, w, state, work, x, y, ep, scratch, layer, rows, c, false, null);
}

/// Same encoder and arithmetic as encode, with optional copies at operator boundaries.
pub fn encodeDiagnostic(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config, comptime diagnostic: bool, observer: ?DiagnosticObserver) !void {
    if (layer >= 40 or rows == 0 or rows > work.max_rows or
        state.position > state.plan.capacity or rows > state.plan.capacity - state.position)
        return error.InvalidAttentionChunk;
    const position = state.position;
    const ratio = c.compress_ratios.items()[layer];
    const end = position + rows;
    const q_width = c.heads * c.head_dim;
    const qr = work.view(.qr);
    const q = work.view(.query);
    const raw = work.view(.raw_kv);
    const ring = try state.ring(layer);
    const chunk = work.selection.windowChunk();
    e.barrier();
    try projection(ep, e, .{ .trellis = w.wq_a }, x, qr, scratch, rows, c.hidden, c.q_lora, .bf16);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .qr_projection, qr);
    norm(p, e, qr, w.q_norm, rows, c.q_lora, c.eps);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .qr_norm, qr);
    try projection(ep, e, .{ .trellis = w.wq_b }, qr, q, scratch, rows, c.q_lora, q_width, .bf16);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .q_projection, q);
    try rotate(p, e, q, rows, c.heads, c.head_dim, position, ratio, 1, false, c);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .q, q);
    try projection(ep, e, .{ .trellis = w.wkv }, x, raw, scratch, rows, c.hidden, c.head_dim, .bf16);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .kv_projection, raw);
    norm(p, e, raw, w.kv_norm, rows, c.head_dim, c.eps);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .kv_norm, raw);
    try rotate(p, e, raw, rows, 1, c.head_dim, position, ratio, 1, false, c);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .kv_rope, raw);
    try p.cache.encodePack(e, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .kv_quantized, chunk);

    var main = chunk;
    var compressed_rows: u32 = 0;
    if (ratio != 0) {
        const source = try state.source(layer, ratio);
        main = source.main;
        compressed_rows = end / ratio;
        const complete = compressed_rows - position / ratio;
        const latent = work.view(.latent);
        if (source.owner == layer) {
            const compressor = if (w.compressor) |*value| value else return error.MissingAttentionCompressor;
            const output: exl3.Output = if (ratio == 1) .bf16 else .f32;
            const values = work.view(.values);
            const gates = work.view(.gates);
            try projection(ep, e, compressor.wkv, x, values, scratch, rows, c.hidden, c.head_dim, output);
            if (ratio > 1) {
                const gate = compressor.wgate orelse return error.MissingAttentionCompressorGate;
                try projection(ep, e, gate, x, gates, scratch, rows, c.hidden, c.head_dim, .f32);
            }
            try p.cache.encodePool(e, values, gates, source.partial, tensor(compressor.norm), latent, .{ .rows = rows, .ratio = ratio, .position = position, .stride = c.head_dim, .input_f32 = @intFromBool(ratio > 1), .eps = c.eps });
            if (complete != 0) {
                const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
                const wk = selector.wk orelse return error.MissingAttentionIndexKeyProjection;
                const k_norm = selector.k_norm orelse return error.MissingAttentionIndexKeyNorm;
                const key = work.view(.index_k);
                try projection(ep, e, wk, latent, key, scratch, complete, c.head_dim, c.index_dim, .bf16);
                norm(p, e, key, k_norm, complete, c.index_dim, c.eps);
                try rotate(p, e, key, complete, 1, c.index_dim, position / ratio * ratio, ratio, ratio, false, c);
                try p.cache.encodePack(e, key, source.index, .{ .rows = complete, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .first = position / ratio });
            }
        }
        if (kv.cache.isIndexSource(layer)) {
            const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
            const iq = work.view(.index_q);
            const iw = work.view(.index_weights);
            const iq_width = c.index_heads * c.index_dim;
            if (compressed_rows != 0) {
                try projection(ep, e, selector.wq_b, qr, iq, scratch, rows, c.q_lora, iq_width, .bf16);
                try rotate(p, e, iq, rows, c.index_heads, c.index_dim, position, ratio, 1, false, c);
                try p.cache.encodePack(e, iq, work.selection.queryPacked(), .{ .rows = rows * c.index_heads, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .inplace = 1 });
                ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), iw, .{ .rows = rows, .k = c.hidden, .n = c.index_heads, .x_stride = c.hidden, .w_stride = c.hidden, .y_stride = c.index_heads, .w_dtype = switch (selector.weights_proj.dtype) {
                    .bf16 => .bf16,
                    else => .f16,
                } });
                e.barrier();
            }
            const make_candidates = layer == c.candidate_source;
            const use_candidates = layer > c.candidate_source;
            if (use_candidates and (work.candidate_position != position or work.candidate_rows != rows))
                return error.MissingAttentionCandidates;
            try p.select.encode(e, iq, iw, source.index, work.selection, .{ .rows = rows, .q_stride = iq_width, .weight_stride = c.index_heads, .position = position, .ratio = ratio, .keys = compressed_rows, .make_candidates = @intFromBool(make_candidates), .use_candidates = @intFromBool(use_candidates) });
            work.selection_layer = layer;
            work.selection_position = position;
            work.selection_rows = rows;
            if (make_candidates) {
                work.candidate_position = position;
                work.candidate_rows = rows;
            }
        } else if (work.selection_layer != precedingIndex(layer) or
            work.selection_position != position or work.selection_rows != rows) return error.MissingAttentionSelection;
        // Index K consumes the normalized latent before this in-place RoPE mutates it.
        if (source.owner == layer and complete != 0) {
            try rotate(p, e, latent, complete, 1, c.head_dim, position / ratio * ratio, ratio, ratio, false, c);
            try p.cache.encodePack(e, latent, source.main, .{ .rows = complete, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.main_stride, .block = 16, .kind = .main_fp4, .first = position / ratio });
        }
    }
    try p.sparse.encode(e, q, ring, chunk, main, work.selection.positions(), tensor(w.sink), q, .{ .rows = rows, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = ratio, .compressed_rows = compressed_rows, .topk = @min(c.index_topk, compressed_rows) });
    e.barrier();
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .sparse_o, q);
    try p.cache.publishWindow(e, chunk, ring, rows, position);
    try rotate(p, e, q, rows, c.heads, c.head_dim, position, ratio, 1, true, c);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .o, q);
    const grouped = work.view(.grouped);
    const group_input = q_width / c.o_groups;
    const group_output = c.o_groups * c.o_lora;
    for (w.wo_a, 0..) |matrix, group| {
        try projection(ep, e, .{ .trellis = matrix }, q.shifted(group * group_input * 2), grouped.shifted(group * c.o_lora * 2), scratch, rows, q_width, group_output, .bf16);
    }
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .grouped, grouped);
    try projection(ep, e, .{ .trellis = w.wo_b }, grouped, y, scratch, rows, group_output, c.hidden, .bf16);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, .wo_b, y);
}
