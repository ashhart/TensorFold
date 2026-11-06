// ! Pinned DeepSeek V4.1 attention, resident EXL3 projections and causal packed-cache chunk execution.
const std = @import("std");
const mtl = @import("metal");
const exl3 = @import("core").exl3_projection;
const checkpoint = @import("core").checkpoint_metal;
const weights = @import("weights.zig");
const config = @import("config.zig");
const ops = @import("ops.zig");
const scan_probe = @import("scan_probe.zig");
const csa2 = @import("csa2.zig");
const kv = @import("kv.zig");
const index = @import("csa2_index.zig");
const rope = @import("rope.zig");
pub const View = @import("view.zig").View;
pub const State = kv.cache.State;

/// One segment of a batched decode forward: a slot's state, its first position, its first row in the launch's buffers
/// and its row count. The batch rung is one-row segments (a shared round's lanes), so `n` is 1 today; the type
/// carries it so the core is written once.
pub const Seg = struct { state: *State, p0: u32, r0: u32, n: u32 };

/// Layer 0's wq_b input (the post-q_norm qr, [rows, q_lora] bf16), for the GPTQ calibration Hessian. Null unless
/// TF_DSV41_DUMP_WQB_X is set; rows written are reported in wqb_x_rows.
pub var wqb_x_debug: ?mtl.Buffer = null;
pub var wqb_x_rows: u32 = 0;

/// The attention stage's per-block GPU split (TF_DSV41_ATTN_PROF=1). The block marks (`attnMark`) only count
/// dispatches; this ends the command buffer at each mark, commits, waits and prints that block's GPU window, then
/// starts a fresh buffer. Default-inert: no split, no byte of the path moves.
pub const Prof = struct {
    queue: mtl.Queue,
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
    on: bool,
    total_ms: f64 = 0,
    parts: u32 = 0,

    fn split(self: *Prof, label: []const u8, layer: u32, enc: *mtl.ComputeEncoder) void {
        if (!self.on) return;
        enc.end();
        self.cb.commit();
        self.cb.wait();
        const ms = self.cb.gpuSeconds() * 1e3;
        self.total_ms += ms;
        self.parts += 1;
        std.log.info("attn part {s} l{d}: {d:.3} ms", .{ label, layer, ms });
        self.cb = self.queue.commandBufferUnretained();
        const ne = self.cb.compute(.serial);
        enc.* = ne;
        self.enc = ne;
    }
};

var attn_prof_flag: ?bool = null;
var attn_prof_layers: ?[]const u8 = null;

/// TF_DSV41_ATTN_PROF_LAYERS names the layers to split (default "0": an ordinary layer).
pub fn profLayerOn(layer: u32) bool {
    if (attn_prof_flag == null) attn_prof_flag = std.c.getenv("TF_DSV41_ATTN_PROF") != null;
    if (!attn_prof_flag.?) return false;
    if (attn_prof_layers == null) {
        const raw = std.c.getenv("TF_DSV41_ATTN_PROF_LAYERS");
        attn_prof_layers = if (raw) |r| std.mem.span(r) else "0";
    }
    var it = std.mem.splitScalar(u8, attn_prof_layers.?, ',');
    while (it.next()) |part| {
        const v = std.fmt.parseInt(u32, std.mem.trim(u8, part, " \t"), 10) catch continue;
        if (v == layer) return true;
    }
    return false;
}

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
    capture: *const fn (*anyopaque, mtl.ComputeEncoder, u32, DiagnosticStage, View) anyerror!void,

    fn observe(comptime enabled: bool, self: ?DiagnosticObserver, e: mtl.ComputeEncoder, layer: u32, stage: DiagnosticStage, source: View) !void {
        if (enabled) if (self) |observer| try observer.capture(observer.ctx, e, layer, stage, source);
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
const regions = [_]usize{ 1280 * 2, 32768 * 2, 512 * 2, 512 * 4, 512 * 4, 512 * 2, 128 * 2, 4096 * 2, 32 * 2, 8192 * 2, 2560 * 2 };
const Region = enum(usize) { qr, query, raw_kv, values, gates, latent, index_k, index_q, index_weights, grouped, qkv };

pub const Workspace = struct {
    buffer: mtl.Buffer,
    /// Every visible key decoded once a layer: the window ring, the chunk, then the compressed picks.
    dense_kv: mtl.Buffer,
    /// A decode layer's split key scans: (split, head group, lane) by (max, sum, 64 accumulators).
    partials: mtl.Buffer,
    selection: index.Scratch,
    /// The batch's per-segment selection scratches; segment 0 uses `selection`. Three extras cover a four-lane shared
    /// round.
    sel_extra: [3]?index.Scratch = .{ null, null, null },
    /// Identity map [0..7] for the fused wo_a indexed dispatch (one row per output group), repeated for the multi-row
    /// path.
    ids: mtl.Buffer,
    /// A scratch row for the doubling arms (the workspace rows are live for a prompt chunk, so a repeat there would
    /// change the reply).
    dup: mtl.Buffer,
    /// The batched wq_a+wkv call's input-row map -- both entries read the one decode row (0, 0). The descriptor-id
    /// map is the existing `ids` buffer, whose first two entries are 0, 1.
    pair_in: mtl.Buffer,
    max_rows: u32,
    selection_layer: ?u32 = null,
    selection_position: u32 = 0,
    selection_rows: u32 = 0,
    candidate_position: u32 = 0,
    candidate_rows: u32 = 0,
    /// A merged tail chunk's PREFIX (its rows before the tail mark) is published into this stash ring as well as the
    /// whole chunk into the live one, so the state at the mark can be copied from it.
    publish_limit: u32 = 0,
    publish_stash: ?View = null,

    /// The selection scratch of batch segment `s`; the one-stream path is segment 0.
    pub fn selectionFor(w: *Workspace, s: usize) *index.Scratch {
        if (s == 0) return &w.selection;
        return &w.sel_extra[s - 1].?;
    }
    pub fn bytes(max_rows: u32) usize {
        var per_row: usize = 0;
        for (regions) |size| per_row += size;
        return @as(usize, max_rows) * per_row + index.Scratch.bytes(max_rows) + 32;
    }
    pub fn init(device: mtl.Device, max_rows: u32) !Workspace {
        if (max_rows == 0 or max_rows > 16384) return error.InvalidAttentionWorkspace;
        const buffer = try device.buffer(bytes(max_rows) - index.Scratch.bytes(max_rows), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        const selection = try index.Scratch.init(device, max_rows);
        var sel_extra: [3]?index.Scratch = .{ null, null, null };
        errdefer for (&sel_extra) |*e| if (e.*) |*x| x.deinit();
        for (&sel_extra) |*e| e.* = try index.Scratch.init(device, max_rows);
        const dense_kv = try device.buffer(denseKvBytes(max_rows), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer dense_kv.deinit();
        const partials = try device.buffer(partialBytes(@min(max_rows, 8)), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer partials.deinit();
        const ids = try device.buffer(64 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer ids.deinit();
        const dup = try device.buffer(65536 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer dup.deinit();
        const pair_in = try device.buffer(2 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer pair_in.deinit();
        @memset(pair_in.contents()[0 .. 2 * 4], 0);
        const map: [*]u32 = @ptrCast(@alignCast(ids.contents()));
        for (0..64) |id| map[id] = @intCast(id % 8);
        return .{ .buffer = buffer, .dense_kv = dense_kv, .partials = partials, .selection = selection, .sel_extra = sel_extra, .ids = ids, .dup = dup, .pair_in = pair_in, .max_rows = max_rows };
    }
    pub fn deinit(w: *Workspace) void {
        w.buffer.deinit();
        w.dense_kv.deinit();
        w.partials.deinit();
        w.ids.deinit();
        w.dup.deinit();
        w.pair_in.deinit();
        w.selection.deinit();
        for (&w.sel_extra) |*e| if (e.*) |*x| x.deinit();
    }
    pub fn residentBytes(w: *const Workspace) usize {
        return bytes(w.max_rows) + denseKvBytes(w.max_rows) + 3 * index.Scratch.bytes(w.max_rows);
    }
    pub fn dense(self: *const Workspace) View {
        return .{ .buffer = self.dense_kv, .offset = 0 };
    }
    /// The identity map for the fused wo_a's indexed dispatch (one row per output group), repeated for the multi-row
    /// path. The drafter's own attention takes the same fused form.
    pub fn idMap(self: *const Workspace) View {
        return .{ .buffer = self.ids, .offset = 0 };
    }
    pub fn splitPartials(self: *const Workspace) View {
        return .{ .buffer = self.partials, .offset = 0 };
    }
    pub fn registerResidency(w: *const Workspace, set: mtl.ResidencySet) void {
        set.add(w.buffer);
        set.add(w.dense_kv);
        set.add(w.partials);
        set.add(w.ids);
        set.add(w.pair_in);
        set.add(w.selection.buffer);
        for (w.sel_extra) |e| if (e) |x| set.add(x.buffer);
    }
    /// A region's view into the attention workspace: public so a second model (the drafter) can assemble its own scan
    /// from the same buffers.
    pub fn view(w: *const Workspace, region: Region) View {
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
        .trellis => |t| {
            // The prompt-row int8 twin. The resident prompt path reaches the projection here, not through
            // Model.project; the gate is the call row count, so decode (rows <= 8) keeps the trellis.
            if (t.int8) |side| {
                if (rows >= 1024 and std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
                    try exl3.encodePrefillInt8(p, e, t, side, core(x), core(y), scratch, .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output }, t.int8_direct);
                    return;
                }
            }
            try exl3.encode(p, e, t, core(x), core(y), scratch, .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        },
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

fn normRepeat(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, dup: View, weight: checkpoint.Tensor, rows: u32, width: u32, eps: f32) void {
    ops.encodeRms(e, &p.plain, x, tensor(weight), dup, .{ .rows = rows, .width = width, .x_stride = width, .y_stride = width, .eps = eps, .weight_f32 = weight.dtype == .f32 });
    e.barrier();
}
fn rotate(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, rows: u32, heads: u32, width: u32, position: u32, ratio: u32, step: u32, inverse: bool, c: config.Config) !void {
    try p.rotary.encode(e, x, .{ .rows = rows, .heads = heads, .width = width, .rotary = c.rope_dim, .stride = heads * width, .position = position, .step = step, .inverse = @intFromBool(inverse), .spectrum = if (ratio == 0) .normal else .compressed });
}

fn rotateRepeat(p: *const Pipelines, e: mtl.ComputeEncoder, dup: View, rows: u32, heads: u32, width: u32, position: u32, ratio: u32, step: u32, inverse: bool, c: config.Config) !void {
    try p.rotary.encode(e, dup, .{ .rows = rows, .heads = heads, .width = width, .rotary = c.rope_dim, .stride = heads * width, .position = position, .step = step, .inverse = @intFromBool(inverse), .spectrum = if (ratio == 0) .normal else .compressed });
}
// The attention stage's own block ledger. The doubling arms isolate a dispatch's marginal cost; this counts the
// dispatches each block issues, by reading the engine's own counter at each block boundary (an atomic load: no
// command-buffer split, no perturbation of the stream). It prints a block's count for the layers
// TF_DSV41_ATTN_COUNT_LAYERS names (default "0,2,3": an ordinary layer, an owner layer that also sources the index,
// and a layer that reads a compressed source).
var attn_count_flag: ?bool = null;
var attn_count_layers: ?[]const u8 = null;
var attn_count_base: u64 = 0;
var attn_group_base: u64 = 0;

fn attnCountOn(layer: u32) bool {
    if (attn_count_flag == null) {
        attn_count_flag = std.c.getenv("TF_DSV41_ATTN_COUNT") != null;
        // The engine-s dispatch counter is only incremented when the recorder asks for it.
        if (attn_count_flag.?) mtl.command.count_enabled = true;
    }
    if (!attn_count_flag.?) return false;
    if (attn_count_layers == null) {
        const raw = std.c.getenv("TF_DSV41_ATTN_COUNT_LAYERS");
        attn_count_layers = if (raw) |r| std.mem.span(r) else "0,2,3";
    }
    var it = std.mem.splitScalar(u8, attn_count_layers.?, ',');
    while (it.next()) |part| {
        const v = std.fmt.parseInt(u32, std.mem.trim(u8, part, " \t"), 10) catch continue;
        if (v == layer) return true;
    }
    return false;
}

/// The attention's entry: the ledger's baseline for this layer.
fn attnBegin(layer: u32) void {
    if (!attnCountOn(layer)) return;
    attn_count_base = mtl.command.dispatch_count.load(.monotonic);
    attn_group_base = mtl.command.dispatch_threadgroups.load(.monotonic);
}

/// A block boundary: the dispatches this block issued, and (TF_DSV41_ATTN_PROF) the block's GPU window.
fn attnMark(layer: u32, block: []const u8, prof: ?*Prof, enc: *mtl.ComputeEncoder) void {
    if (prof) |pf| pf.split(block, layer, enc);
    if (!attnCountOn(layer)) return;
    const now = mtl.command.dispatch_count.load(.monotonic);
    const groups = mtl.command.dispatch_threadgroups.load(.monotonic);
    // The threadgroup count beside the dispatch count -- the parallelism ledger.
    std.log.info("attn {d} {s}: {d} dispatches, {d} threadgroups", .{ layer, block, now - attn_count_base, groups - attn_group_base });
    attn_count_base = now;
    attn_group_base = groups;
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
/// A decode layer's split scans: 32 splits at most, 4 head groups of 64 heads, 4 SIMDgroups a group, 32 lanes, 68
/// floats a lane.
fn partialBytes(rows: u32) usize {
    // A split's (max, sum, accumulators) for every row: a drafted window above one row keeps its own parts, so its
    // rows sum in exactly the order a plain step's does (drafted == plain).
    return @as(usize, rows) * 32 * 4 * 4 * 32 * 68 * @sizeOf(f32);
}
/// The window ring, up to `max_rows` chunk rows, and as many compressed rows as the context allows.
fn denseKvBytes(max_rows: u32) usize {
    return (@as(usize, 128) + max_rows + kv.cache.max_context) * 512 * @sizeOf(u16);
}

var split_scan: ?u32 = null;
/// TF_DSV41_ATTN_SPLITS: key-scan slices a decode layer's sparse attention runs (default 8).
fn splitScan() u32 {
    if (split_scan) |v| return v;
    var out: u32 = 8;
    if (std.c.getenv("TF_DSV41_ATTN_SPLITS")) |v| {
        const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch out;
        if (n >= 1 and n <= 32) out = n;
    }
    split_scan = out;
    return out;
}

pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config) !void {
    return encodeDiagnostic(p, e, w, state, work, x, y, ep, scratch, layer, rows, c, false, null, null, &.{});
}

/// The same encode with the per-block GPU split armed (TF_DSV41_ATTN_PROF).
pub fn encodeProfiled(p: *const Pipelines, e: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config, prof: *Prof) !void {
    return encodeDiagnostic(p, e, w, state, work, x, y, ep, scratch, layer, rows, c, false, null, prof, &.{});
}

/// Same encoder and arithmetic as encode, with optional copies at operator boundaries. A decode-sized batch of
/// one-row segments -- the shared round's lanes in one launch. The row-wise projections run once over every segment's
/// rows (the dense EXL3 weights decode once instead of once a lane); every slot-bound step runs once a segment
/// against that segment's own state and selection scratch: the window KV's norm+rope+pack, the compressor's pool, the
/// index key and its pick, the sparse scan, the ring publish and the output rope. The output projections then run
/// once over all rows. Only one-row segments are admitted: this rung amortises the dense attention weights, it is not
/// a prefill batch.
fn encodeMulti(p: *const Pipelines, e_in: mtl.ComputeEncoder, w: *const weights.Attention, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, c: config.Config, segs: []const Seg) !void {
    var e = e_in;
    const nsegs = segs.len;
    if (nsegs < 2 or nsegs > 4) return error.InvalidAttentionChunk;
    const rows: u32 = segs[nsegs - 1].r0 + segs[nsegs - 1].n;
    if (layer >= 40 or rows == 0 or rows > work.max_rows or rows > 8) return error.InvalidAttentionChunk;
    for (segs) |sg| {
        if (sg.n != 1) return error.InvalidAttentionChunk;
        if (sg.state.position != sg.p0) return error.PositionMismatch;
        if (sg.p0 > sg.state.plan.capacity or sg.n > sg.state.plan.capacity - sg.p0) return error.InvalidAttentionChunk;
    }
    const ratio = c.compress_ratios.items()[layer];
    const q_width = c.heads * c.head_dim;
    const qr = work.view(.qr);
    const q = work.view(.query);
    const raw = work.view(.raw_kv);
    const iq_width = c.index_heads * c.index_dim;
    var kv_src = false;
    if (ratio != 0) {
        const src0 = try segs[0].state.source(layer, ratio);
        kv_src = src0.owner == layer;
    }
    const ix_src = ratio != 0 and kv.cache.isIndexSource(layer);
    if (std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("encodeMulti: layer={d} segs={d} rows={d}", .{ layer, nsegs, rows });
    // The row-wise phase: one call over every segment's rows.
    try projection(ep, e, .{ .trellis = w.wq_a }, x, qr, scratch, rows, c.hidden, c.q_lora, .bf16);
    norm(p, e, qr, w.q_norm, rows, c.q_lora, c.eps);
    try projection(ep, e, .{ .trellis = w.wq_b }, qr, q, scratch, rows, c.q_lora, q_width, .bf16);
    try projection(ep, e, .{ .trellis = w.wkv }, x, raw, scratch, rows, c.hidden, c.head_dim, .bf16);
    if (kv_src) {
        const compressor = if (w.compressor) |*value| value else return error.MissingAttentionCompressor;
        const output: exl3.Output = if (ratio == 1) .bf16 else .f32;
        try projection(ep, e, compressor.wkv, x, work.view(.values), scratch, rows, c.hidden, c.head_dim, output);
        if (ratio > 1) {
            const gate = compressor.wgate orelse return error.MissingAttentionCompressorGate;
            try projection(ep, e, gate, x, work.view(.gates), scratch, rows, c.hidden, c.head_dim, .f32);
        }
    }
    if (ix_src) {
        const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
        try projection(ep, e, selector.wq_b, qr, work.view(.index_q), scratch, rows, c.q_lora, iq_width, .bf16);
        const iq_w_args: ops.Dense = .{ .rows = rows, .k = c.hidden, .n = c.index_heads, .x_stride = c.hidden, .w_stride = c.hidden, .y_stride = c.index_heads, .w_dtype = switch (selector.weights_proj.dtype) {
            .bf16 => .bf16,
            else => .f16,
        } };
        const dense_col_env = std.c.getenv("TF_DSV41_DENSE_COL");
        const dense_col = (dense_col_env == null or
            !std.mem.eql(u8, std.mem.span(dense_col_env.?), "0")) and
            ops.encodeDenseCol(e, &p.plain, x, tensor(selector.weights_proj), work.view(.index_weights), iq_w_args);
        if (!dense_col) ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), work.view(.index_weights), iq_w_args);
    }
    e.barrier();
    // The slot-bound phase: one segment at a time, each against its own state and selection scratch.
    var s: usize = 0;
    while (s < nsegs) : (s += 1) {
        const sg = segs[s];
        const st = sg.state;
        const pos = sg.p0;
        const r0 = sg.r0;
        const sel = work.selectionFor(s);
        const qs = q.shifted(@as(usize, r0) * q_width * 2);
        const raws = raw.shifted(@as(usize, r0) * c.head_dim * 2);
        try rotate(p, e, qs, 1, c.heads, c.head_dim, pos, ratio, 1, false, c);
        const chunk = sel.windowChunk();
        if (w.kv_norm.dtype == .bf16 and c.head_dim == 512) {
            const factor_offset: usize = if (ratio == 0) 0 else @as(usize, p.rotary.capacity) * p.rotary.rotary * @sizeOf(f32);
            const factors: View = .{ .buffer = p.rotary.factors, .offset = factor_offset };
            try p.cache.encodeGlue(e, raws, tensor(w.kv_norm), factors, chunk, chunk, .{
                .rows = 1,
                .width = c.head_dim,
                .stride = c.head_dim,
                .packed_stride = kv.cache.ring_stride,
                .block = 32,
                .position = pos,
                .rotary = c.rope_dim,
                .write_y = 0,
                .eps = c.eps,
            });
        } else {
            norm(p, e, raws, w.kv_norm, 1, c.head_dim, c.eps);
            try rotate(p, e, raws, 1, 1, c.head_dim, pos, ratio, 1, false, c);
            try p.cache.encodePack(e, raws, chunk, .{ .rows = 1, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        }
        e.barrier();
        var main: View = chunk;
        var compressed_rows: u32 = 0;
        if (ratio != 0) {
            const source = try st.source(layer, ratio);
            compressed_rows = (pos + 1) / ratio;
            const complete = compressed_rows - pos / ratio;
            if (source.owner == layer) {
                const compressor = if (w.compressor) |*value| value else return error.MissingAttentionCompressor;
                const values_row: usize = if (ratio == 1) c.head_dim * 2 else c.head_dim * 4;
                try p.cache.encodePool(e, work.view(.values).shifted(@as(usize, r0) * values_row), work.view(.gates).shifted(@as(usize, r0) * c.head_dim * 4), source.partial, tensor(compressor.norm), work.view(.latent).shifted(@as(usize, r0) * c.head_dim * 2), source.capture, .{
                    .rows = 1,
                    .ratio = ratio,
                    .position = pos,
                    .stride = c.head_dim,
                    .input_f32 = @intFromBool(ratio > 1),
                    .eps = c.eps,
                    .capture_rows = if (ratio > 1) kv.cache.capture_rows else 0,
                });
                e.barrier();
                if (complete != 0) {
                    const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
                    const wk = selector.wk orelse return error.MissingAttentionIndexKeyProjection;
                    const k_norm = selector.k_norm orelse return error.MissingAttentionIndexKeyNorm;
                    const latent = work.view(.latent).shifted(@as(usize, r0) * c.head_dim * 2);
                    const key = work.view(.index_k).shifted(@as(usize, r0) * c.index_dim * 2);
                    try projection(ep, e, wk, latent, key, scratch, 1, c.head_dim, c.index_dim, .bf16);
                    norm(p, e, key, k_norm, 1, c.index_dim, c.eps);
                    try rotate(p, e, key, 1, 1, c.index_dim, pos / ratio * ratio, ratio, ratio, false, c);
                    try p.cache.encodePack(e, key, source.index, .{ .rows = 1, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .first = pos / ratio });
                    e.barrier();
                }
            }
            if (ix_src) {
                const iq = work.view(.index_q).shifted(@as(usize, r0) * iq_width * 2);
                if (compressed_rows != 0 and compressed_rows > c.index_topk) {
                    try rotate(p, e, iq, 1, c.index_heads, c.index_dim, pos, ratio, 1, false, c);
                    try p.cache.encodePack(e, iq, sel.queryPacked(), .{ .rows = c.index_heads, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .inplace = 1 });
                    e.barrier();
                    const selp: index.Params = .{ .rows = 1, .q_stride = iq_width, .weight_stride = c.index_heads, .position = pos, .ratio = ratio, .keys = compressed_rows, .make_candidates = 0, .use_candidates = 0, .layer = layer, .skip_mask = 0 };
                    try p.select.encode(e, iq, work.view(.index_weights).shifted(@as(usize, r0) * c.index_heads * 2), source.index, sel.*, selp);
                    e.barrier();
                } else {
                    const selp: index.Params = .{ .rows = 1, .q_stride = iq_width, .weight_stride = c.index_heads, .position = pos, .ratio = ratio, .keys = compressed_rows, .layer = layer };
                    p.select.encodeIdentity(e, sel.*, selp);
                }
                sel.sel_layer = layer;
                sel.sel_position = pos;
                sel.sel_rows = 1;
            } else {
                const prev = precedingIndex(layer);
                if (sel.sel_layer == null or sel.sel_layer.? != prev or sel.sel_position != pos or sel.sel_rows != 1) return error.MissingAttentionSelection;
            }
            if (source.owner == layer and complete != 0) {
                const latent = work.view(.latent).shifted(@as(usize, r0) * c.head_dim * 2);
                try rotate(p, e, latent, 1, 1, c.head_dim, pos / ratio * ratio, ratio, ratio, false, c);
                try p.cache.encodePack(e, latent, source.main, .{ .rows = 1, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.main_stride, .block = 16, .kind = .main_fp4, .first = pos / ratio });
                e.barrier();
            }
            main = source.main;
        }
        const ring = try st.ring(layer);
        try p.sparse.encode(e, qs, ring, chunk, main, sel.positions(), tensor(w.sink), qs, work.dense(), work.splitPartials(), .{
            .rows = 1,
            .q_stride = q_width,
            .position = pos,
            .chunk_rows = 1,
            .ratio = ratio,
            .compressed_rows = compressed_rows,
            .topk = @min(c.index_topk, compressed_rows),
            .splits = splitScan(),
        });
        e.barrier();
        try p.cache.publishWindow(e, chunk, ring, 1, pos);
        try rotate(p, e, qs, 1, c.heads, c.head_dim, pos, ratio, 1, true, c);
        e.barrier();
    }
    // The output phase: one call over every segment's rows.
    const grouped = work.view(.grouped);
    const group_input = q_width / c.o_groups;
    const group_output = c.o_groups * c.o_lora;
    if (w.wo_a_descriptors != null and std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null) {
        for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
        try exl3.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
            .count = rows * c.o_groups,
            .x_stride = group_input,
            .y_stride = c.o_lora,
            .output = .bf16,
            .output_rows = rows * c.o_groups,
            .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
        });
    } else {
        var r0: u32 = 0;
        while (r0 < rows) : (r0 += 1) {
            for (w.wo_a, 0..) |matrix, group| {
                try projection(ep, e, .{ .trellis = matrix }, q.shifted(@as(usize, r0) * q_width * 2).shifted(group * group_input * 2), grouped.shifted(@as(usize, r0) * group_output * 2).shifted(group * c.o_lora * 2), scratch, 1, q_width, group_output, .bf16);
            }
        }
    }
    e.barrier();
    try projection(ep, e, .{ .trellis = w.wo_b }, grouped, y, scratch, rows, group_output, c.hidden, .bf16);
}

pub fn encodeDiagnostic(p: *const Pipelines, e_in: mtl.ComputeEncoder, w: *const weights.Attention, state: *State, work: *Workspace, x: View, y: View, ep: *const exl3.Pipelines, scratch: exl3.View, layer: u32, rows: u32, c: config.Config, comptime diagnostic: bool, observer: ?DiagnosticObserver, prof: ?*Prof, segs: []const Seg) !void {
    if (segs.len > 1) return encodeMulti(p, e_in, w, work, x, y, ep, scratch, layer, c, segs);
    var e = e_in;
    if (layer >= 40 or rows == 0 or rows > work.max_rows or
        state.position > state.plan.capacity or rows > state.plan.capacity - state.position)
        return error.InvalidAttentionChunk;
    attnBegin(layer);
    // Start the first block from a fresh buffer so the "pre" window is not charged to `qkv`.
    if (prof) |pf| pf.split("pre", layer, &e);
    const position = state.position;
    const ratio = c.compress_ratios.items()[layer];
    const end = position + rows;
    const q_width = c.heads * c.head_dim;
    var qr = work.view(.qr);
    const q = work.view(.query);
    var raw = work.view(.raw_kv);
    const ring = try state.ring(layer);
    const chunk = work.selection.windowChunk();
    e.barrier();
    const dense_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_DENSE") != null) 2 else 1;
    const dup_view: View = .{ .buffer = work.dup, .offset = 0 };
    // Wq_a (n 1280) and wkv (n 512) read the same input row and the same k, so one indexed call carries both: one
    // rotate_in (each row its own descriptor's suh), one register-only GEMM (each row bounded by its descriptor's own
    // width) and one rotate_out (each row its own svh). The decode row's qr comes back at the region base and its kv
    // at +1280 halves, both contiguous, so every consumer below sees the same values at the same strides for a
    // one-row call. Rows > 1 keep the two separate calls (their layouts are per-row strided); the bodies, the k order
    // and the per-row arithmetic are the shipped ones, so the bits do not move. TF_DSV41_NO_FUSE_QKV=1 restores the
    // two calls for a same-binary A/B.
    const qkv_merged = rows == 1 and w.qkv_descriptors != null and exl3.indexedWidthBody(ep) and
        std.c.getenv("TF_DSV41_NO_FUSE_QKV") == null;
    // THE TRANSFORM FOLD. When the merged wq_a+wkv call leaves its f32 sums in place (`no_epilogue`), one kernel does
    // both output transforms, the q_norm and wq_b's input transform with the three shipped kernels' arithmetic and
    // order, so the chain's three dispatches become one.
    const norm_fold = qkv_merged and ep.norm_transform != null and std.c.getenv("TF_DSV41_FUSE_NORM_TRANSFORM") != null;
    if (qkv_merged) {
        const fused = work.view(.qkv);
        // TF_DSV41_DOUBLE_ATTN_QKV -- the merged call's same-output duplicate. The merged path bypasses the
        // dense_passes loop, so DOUBLE_ATTN_DENSE never covered it.
        const qkv_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_QKV") != null) 2 else 1;
        for (0..qkv_passes) |_| {
            try exl3.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.qkv_descriptors.?, .offset = 0 }, .count = 2, .k = c.hidden, .n = c.q_lora }, core(x), core(fused), scratch, .{
                .count = 2,
                .x_stride = c.hidden,
                .y_stride = c.q_lora,
                .output = .bf16,
                .input_ids = .{ .buffer = work.pair_in, .offset = 0 },
                .input_rows = 1,
                .output_ids = .{ .buffer = work.ids, .offset = 0 },
                .output_rows = 2,
                .routes = .{ .ids = .{ .buffer = work.ids, .offset = 0 }, .repeat = 1 },
                .no_epilogue = norm_fold,
            });
        }
        qr = fused;
        raw = .{ .buffer = fused.buffer, .offset = fused.offset + c.q_lora * 2 };
        if (norm_fold) {
            try exl3.encodeNormTransform(ep, e, .{ .buffer = scratch.buffer, .offset = scratch.offset + exl3.sumsOffset(c.hidden, 2) }, w.wq_a.svh, w.wkv.svh, core(tensor(w.q_norm)), w.wq_b.suh, core(qr), core(raw), .{ .buffer = scratch.buffer, .offset = scratch.offset }, .{
                .n0 = c.q_lora,
                .n1 = c.head_dim,
                .k_next = c.q_lora,
                .su_signs = @intFromBool(w.wq_b.suh_format == .packed_signs),
                .sv_signs0 = @intFromBool(w.wq_a.svh_format == .packed_signs),
                .sv_signs1 = @intFromBool(w.wkv.svh_format == .packed_signs),
                .eps = c.eps,
            });
            e.barrier();
        }
    } else for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wq_a }, x, qr, scratch, rows, c.hidden, c.q_lora, .bf16);
    attnMark(layer, "qkv", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .qr_projection, qr);
    if (!norm_fold) norm(p, e, qr, w.q_norm, rows, c.q_lora, c.eps);
    if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") != null) normRepeat(p, e, qr, dup_view, w.q_norm, rows, c.q_lora, c.eps);
    attnMark(layer, "norm_qr", prof, &e);
    if (wqb_x_debug) |b| {
        if (layer == 0 and !norm_fold and rows >= 64 and rows <= 2048) {
            ops.encodeDump(e, &p.plain, qr, .{ .buffer = b, .offset = 0 }, rows * c.q_lora, true);
            wqb_x_rows = rows;
        }
    }
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .qr_norm, qr);
    if (norm_fold) try exl3.encode(ep, e, w.wq_b, scratch, core(q), scratch, .{ .count = rows, .x_stride = c.q_lora, .y_stride = q_width, .output = .bf16, .prerotated = true }) else for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wq_b }, qr, q, scratch, rows, c.q_lora, q_width, .bf16);
    attnMark(layer, "wq_b", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .q_projection, q);
    try rotate(p, e, q, rows, c.heads, c.head_dim, position, ratio, 1, false, c);
    if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ROPE") != null) try rotateRepeat(p, e, dup_view, rows, c.heads, c.head_dim, position, ratio, 1, false, c);
    attnMark(layer, "q_rope", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .q, q);
    if (!qkv_merged) for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wkv }, x, raw, scratch, rows, c.hidden, c.head_dim, .bf16);
    attnMark(layer, "wkv", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .kv_projection, raw);
    // The window KV's norm + rope + fp8 pack as ONE dispatch, at the pack's own threadgroup geometry (16 a row, 256
    // threads) with the shipped rms reduction order, the rope's source expression and the pack's exact
    // max/scale/code. TF_DSV41_NO_FUSE_KVGLUE=1 is the same-binary control. It is used only where no instrument reads
    // the intermediate rows, and never above 8 rows (the prompt path keeps the three wide kernels).
    const fuse_kv_glue = rows <= 8 and w.kv_norm.dtype == .bf16 and c.head_dim == 512 and !diagnostic and
        std.c.getenv("TF_DSV41_NO_FUSE_KVGLUE") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ROPE") == null and
        std.c.getenv("TF_DSV41_DOUBLE_ATTN_PACK") == null;
    if (fuse_kv_glue) {
        const factor_offset: usize = if (ratio == 0) 0 else @as(usize, p.rotary.capacity) * p.rotary.rotary * @sizeOf(f32);
        const factors: View = .{ .buffer = p.rotary.factors, .offset = factor_offset };
        try p.cache.encodeGlue(e, raw, tensor(w.kv_norm), factors, chunk, chunk, .{
            .rows = rows,
            .width = c.head_dim,
            .stride = c.head_dim,
            .packed_stride = kv.cache.ring_stride,
            .block = 32,
            .position = position,
            .rotary = c.rope_dim,
            .write_y = 0,
            .eps = c.eps,
        });
        attnMark(layer, "kv_glue", prof, &e);
    } else {
        norm(p, e, raw, w.kv_norm, rows, c.head_dim, c.eps);
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_NORM") != null) normRepeat(p, e, raw, dup_view, w.kv_norm, rows, c.head_dim, c.eps);
        attnMark(layer, "norm_kv", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .kv_norm, raw);
        try rotate(p, e, raw, rows, 1, c.head_dim, position, ratio, 1, false, c);
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ROPE") != null) try rotateRepeat(p, e, dup_view, rows, 1, c.head_dim, position, ratio, 1, false, c);
        attnMark(layer, "kv_rope", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .kv_rope, raw);
        try p.cache.encodePack(e, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        // TF_DSV41_DOUBLE_ATTN_PACK -- the window's fp8 cache pack's same-output duplicate (the pack is a pure
        // function of the row, so the second pass writes the same bytes).
        if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_PACK") != null)
            try p.cache.encodePack(e, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        attnMark(layer, "pack", prof, &e);
        if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .kv_quantized, chunk);
    }

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
            // TF_DSV41_DOUBLE_ATTN_COMPRESSOR issues the two compressor EXL3 calls a second time -- the same output,
            // so the reply, the pooling and the routing cannot move -- and isolates the compressor inside the attention
            // stage: the term the record seven doubling arms (the four dense calls, the transforms, the norms, the
            // ropes, the dense-KV and the scan) do not name.
            const compressor_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_COMPRESSOR") != null) 2 else 1;
            for (0..compressor_passes) |_| {
                try projection(ep, e, compressor.wkv, x, values, scratch, rows, c.hidden, c.head_dim, output);
                if (ratio > 1) {
                    const gate = compressor.wgate orelse return error.MissingAttentionCompressorGate;
                    try projection(ep, e, gate, x, gates, scratch, rows, c.hidden, c.head_dim, .f32);
                }
            }
            attnMark(layer, "compressor", prof, &e);
            try p.cache.encodePool(e, values, gates, source.partial, tensor(compressor.norm), latent, source.capture, .{ .rows = rows, .ratio = ratio, .position = position, .stride = c.head_dim, .input_f32 = @intFromBool(ratio > 1), .eps = c.eps, .capture_rows = if (ratio > 1) kv.cache.capture_rows else 0 });
            // TF_DSV41_DOUBLE_ATTN_POOL -- the owner's pool encode's same-output duplicate. The pool kernel writes
            // the latent and the capture and the tail kernel writes the partial, all from the same inputs, so the
            // second pass writes the same bytes.
            if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_POOL") != null)
                try p.cache.encodePool(e, values, gates, source.partial, tensor(compressor.norm), latent, source.capture, .{ .rows = rows, .ratio = ratio, .position = position, .stride = c.head_dim, .input_f32 = @intFromBool(ratio > 1), .eps = c.eps, .capture_rows = if (ratio > 1) kv.cache.capture_rows else 0 });
            attnMark(layer, "pool", prof, &e);
            if (complete != 0) {
                const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
                const wk = selector.wk orelse return error.MissingAttentionIndexKeyProjection;
                const k_norm = selector.k_norm orelse return error.MissingAttentionIndexKeyNorm;
                const key = work.view(.index_k);
                try projection(ep, e, wk, latent, key, scratch, complete, c.head_dim, c.index_dim, .bf16);
                norm(p, e, key, k_norm, complete, c.index_dim, c.eps);
                try rotate(p, e, key, complete, 1, c.index_dim, position / ratio * ratio, ratio, ratio, false, c);
                try p.cache.encodePack(e, key, source.index, .{ .rows = complete, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .first = position / ratio });
                attnMark(layer, "index_k", prof, &e);
            }
        }
        // Every visible key is kept when there are no more compressed keys than the selection keeps: the picks are
        // then the identity in ascending position order, which is what the score/heap/sort path returns for the same
        // keys. So the query projection, its rotation and pack, the weights projection and four selection dispatches
        // buy nothing there, and the block candidate machinery has nothing to filter.
        const few_keys = compressed_rows <= c.index_topk;
        if (kv.cache.isIndexSource(layer)) {
            const selector = if (w.indexer) |*value| value else return error.MissingAttentionIndexer;
            const iq = work.view(.index_q);
            const iw = work.view(.index_weights);
            const iq_width = c.index_heads * c.index_dim;
            if (compressed_rows != 0 and !few_keys) {
                // TF_DSV41_DOUBLE_ATTN_INDEXER -- the same-output duplicate of the index source layers index-query
                // projection.
                const indexer_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_INDEXER") != null) 2 else 1;
                for (0..indexer_passes) |_| try projection(ep, e, selector.wq_b, qr, iq, scratch, rows, c.q_lora, iq_width, .bf16);
                // The index_q block's four dispatches split.
                attnMark(layer, "iq_proj", prof, &e);
                try rotate(p, e, iq, rows, c.index_heads, c.index_dim, position, ratio, 1, false, c);
                attnMark(layer, "iq_rope", prof, &e);
                try p.cache.encodePack(e, iq, work.selection.queryPacked(), .{ .rows = rows * c.index_heads, .width = c.index_dim, .stride = c.index_dim, .packed_stride = kv.cache.index_stride, .block = 32, .kind = .index_fp4, .inplace = 1 });
                attnMark(layer, "iq_pack", prof, &e);
                const iq_w_args: ops.Dense = .{ .rows = rows, .k = c.hidden, .n = c.index_heads, .x_stride = c.hidden, .w_stride = c.hidden, .y_stride = c.index_heads, .w_dtype = switch (selector.weights_proj.dtype) {
                    .bf16 => .bf16,
                    else => .f16,
                } };
                // The one-column form is the same per-column arithmetic (same lane k-stride, same compensated order,
                // same 16..1 butterfly), so it is bit-identical. TF_DSV41_DENSE_COL=0 is the same-binary control.
                const dense_col_env = std.c.getenv("TF_DSV41_DENSE_COL");
                const dense_col = (dense_col_env == null or
                    !std.mem.eql(u8, std.mem.span(dense_col_env.?), "0")) and
                    ops.encodeDenseCol(e, &p.plain, x, tensor(selector.weights_proj), iw, iq_w_args);
                if (!dense_col) ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), iw, iq_w_args);
                // The same-value duplicate of the index-query weights projection.
                if (std.c.getenv("TF_DSV41_DOUBLE_INDEX_W") != null)
                    ops.encodeDense(e, &p.plain, x, tensor(selector.weights_proj), iw, iq_w_args);
                e.barrier();
                attnMark(layer, "index_q", prof, &e);
            }
            // The block filter is a shortcut only when its candidate set is narrower than the keys; with 2,048 blocks
            // of 8 it covers 16,384 slots, so under ~131k tokens it scores more slots than exist and its block heap
            // costs a serial sort. Off there, every key is scored directly once -- the same keys, the same scores,
            // the same strict top-k order, so the picks are identical. The build is a PER-ROW SERIAL SORT, so a
            // decode row pays it alone. Decode (rows <= 8) therefore takes the direct path. "narrower" is not enough
            // -- the filter must HALVE the scored keys. Where it is off the picks are the direct path's own (the
            // exact set), so no arithmetic moves. TF_DSV41_BLOCK_FILTER=1|0 still forces either way. The candidate
            // path pays a per-forward build plus the non-NAX score kernel on the four reader layers, and that does
            // not pay while the set is half the keys. The gate is an eighth: only at >= 131k keys (past this model's
            // 65,536 window) does it stand.
            const filter_env: ?bool = if (std.c.getenv("TF_DSV41_BLOCK_FILTER")) |v|
                std.mem.eql(u8, std.mem.span(v), "1")
            else
                null;
            const candidate_slots: i64 = @as(i64, c.candidate_blocks) * @as(i64, c.candidate_block_size);
            const filter_narrow = candidate_slots * 8 <= @as(i64, @intCast(compressed_rows));
            const block_filter = !few_keys and candidate_slots < @as(i64, @intCast(compressed_rows)) and
                (filter_env orelse (rows > 8 and filter_narrow));
            const make_candidates = block_filter and layer == c.candidate_source;
            const use_candidates = block_filter and layer > c.candidate_source;
            if (use_candidates and (work.candidate_position != position or work.candidate_rows != rows))
                return error.MissingAttentionCandidates;
            const passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_INDEX") != null) 2 else 1;
            // The select block's REMOVAL arms (TF_DSV41_SKIP_SELECT). The FIRST index-source layer always runs the
            // block, so its picks stay valid and the layers below it read a valid set of keys: the delta is the
            // removed dispatch's own cost, not a smaller downstream.
            var first_index: u32 = 0;
            for (c.index_sources.items()) |l| if (c.isIndexSource(l)) {
                first_index = l;
                break;
            };
            const skip_mask: u32 = if (layer == first_index) 0 else blk: {
                const v = std.c.getenv("TF_DSV41_SKIP_SELECT") orelse break :blk 0;
                break :blk std.fmt.parseInt(u32, std.mem.span(v), 0) catch 0;
            };
            const sel: index.Params = .{ .rows = rows, .q_stride = iq_width, .weight_stride = c.index_heads, .position = position, .ratio = ratio, .keys = compressed_rows, .make_candidates = @intFromBool(make_candidates), .use_candidates = @intFromBool(use_candidates), .layer = layer, .skip_mask = skip_mask };
            if (std.c.getenv("TF_DSV41_SKIP_INDEX") == null) {
                if (few_keys) p.select.encodeIdentity(e, work.selection, sel) else for (0..passes) |_| try p.select.encode(e, iq, iw, source.index, work.selection, sel);
            }
            // The same selection a second time, on unchanged inputs -- the arm that says whether the kernel is
            // deterministic at all.
            if (std.c.getenv("TF_DSV41_SELECT_TWICE") != null and scan_probe.active(layer) and !few_keys) {
                e.barrier();
                scan_probe.record(e, &p.plain, work.selection.positions(), rows, 512 * 4 / 2, .pos, layer);
                scan_probe.record(e, &p.plain, work.selection.positions().shifted(@as(usize, rows - 1) * 512 * 4), 1, 512 * 4 / 2, .posl1, layer);
                scan_probe.recordStrided(e, &p.plain, work.selection.scores(), rows, 8192, 8192, .scof1, layer);
                if (ratio != 0 and rows > 1) {
                    var sk1: u32 = 0;
                    while (sk1 < 4 and sk1 < rows) : (sk1 += 1) {
                        const r1 = rows - 1 - @as(u32, @intCast((@as(usize, rows - 1) * sk1) / 3));
                        const vis1: u32 = @intCast((position + r1 + 1) / ratio);
                        if (vis1 != 0) scan_probe.recordStrided(e, &p.plain, work.selection.scores().shifted(@as(usize, r1) * 4096 * 4), 1, 4096, vis1 * 2, .sco1, 400000 + r1);
                    }
                }
                e.barrier();
                try p.select.encode(e, iq, iw, source.index, work.selection, sel);
                e.barrier();
                scan_probe.record(e, &p.plain, work.selection.positions(), rows, 512 * 4 / 2, .pos2, layer);
                scan_probe.record(e, &p.plain, work.selection.positions().shifted(@as(usize, rows - 1) * 512 * 4), 1, 512 * 4 / 2, .posl2, layer);
                scan_probe.recordStrided(e, &p.plain, work.selection.scores(), rows, 8192, 8192, .scof2, layer);
                e.barrier();
            }
            attnMark(layer, "select", prof, &e);
            // (TF_DSV41_DUMP_SELECT=1, needs ATTN_PROF on this layer, whose own commit+wait is what makes the scratch
            // readable): the selection's picks and the heap's worst.
            if (std.c.getenv("TF_DSV41_DUMP_SELECT") != null and prof != null) {
                const v = work.selection.positions();
                const base: [*]const u8 = @ptrCast(v.buffer.contents());
                const pos: [*]const i32 = @ptrCast(@alignCast(base + v.offset));
                var lo: i32 = std.math.maxInt(i32);
                var hi: i32 = -1;
                var over: u32 = 0;
                var live: u32 = 0;
                for (0..512) |i| {
                    const key = pos[i];
                    if (key < 0) continue;
                    live += 1;
                    lo = @min(lo, key);
                    hi = @max(hi, key);
                    if (key >= 4096) over += 1;
                }
                var sum: u32 = 0;
                var xr: u32 = 0;
                for (0..512) |i| {
                    const key = pos[i];
                    sum +%= @as(u32, @bitCast(key));
                    xr ^= @as(u32, @bitCast(key)) *% 2654435761;
                }
                const hv = work.selection.heap();
                const hb: [*]const u8 = @ptrCast(hv.buffer.contents());
                const hf: [*]const f32 = @ptrCast(@alignCast(hb + hv.offset));
                const sv = work.selection.scores();
                const sb: [*]const u8 = @ptrCast(sv.buffer.contents());
                const sf: [*]const f32 = @ptrCast(@alignCast(sb + sv.offset));
                if (std.c.getenv("TF_DSV41_DUMP_SELECT_REF") != null) {
                    // The attention's order: score desc, then position asc
                    const Pick = struct { score: f32, pos: i32 };
                    var ref: [512]Pick = undefined;
                    var n: u32 = 0;
                    const visible = @min(compressed_rows, (position + 1) / ratio);
                    const tlw: usize = 4096;
                    for (0..visible) |k| {
                        const t = k / tlw;
                        const f = sf[(t * rows + 0) * tlw + (k % tlw)];
                        if (!std.math.isFinite(f)) continue;
                        const cand = Pick{ .score = f, .pos = @intCast(k) };
                        const worse = struct {
                            fn lt(a: Pick, b: Pick) bool {
                                return a.score < b.score or (a.score == b.score and a.pos > b.pos);
                            }
                        }.lt;
                        if (n < 512) {
                            ref[n] = cand;
                            n += 1;
                            var at = n - 1;
                            while (at > 0 and worse(ref[at], ref[(at - 1) / 2])) {
                                const tmp = ref[at];
                                ref[at] = ref[(at - 1) / 2];
                                ref[(at - 1) / 2] = tmp;
                                at = (at - 1) / 2;
                            }
                        } else if (worse(ref[0], cand)) {
                            var at: u32 = 0;
                            while (2 * at + 1 < n) {
                                var child = 2 * at + 1;
                                if (child + 1 < n and worse(ref[child + 1], ref[child])) child += 1;
                                if (!worse(ref[child], cand)) break;
                                ref[at] = ref[child];
                                at = child;
                            }
                            ref[at] = cand;
                        }
                    }
                    var refpos: [512]i32 = undefined;
                    for (0..512) |i| refpos[i] = ref[i].pos;
                    // Ascending insertion sort (512 entries, once)
                    for (1..512) |i| {
                        const cur = refpos[i];
                        var j = i;
                        while (j > 0 and refpos[j - 1] > cur) {
                            refpos[j] = refpos[j - 1];
                            j -= 1;
                        }
                        refpos[j] = cur;
                    }
                    var diff: u32 = 0;
                    for (0..512) |i| {
                        if (refpos[i] != pos[i]) diff += 1;
                    }
                    std.log.info("select ref l{d}: visible={d} diffs={d} ref0..7={d},{d},{d},{d},{d},{d},{d},{d} got0..7={d},{d},{d},{d},{d},{d},{d},{d} ref_thr={d} ref_minpos={d}", .{ layer, visible, diff, refpos[0], refpos[1], refpos[2], refpos[3], refpos[4], refpos[5], refpos[6], refpos[7], pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7], ref[0].score, ref[0].pos });
                }
                // The picks of EVERY row, not just row 0's: the small-chunk one-shot selection is a
                // per-row path, and the row-0 array alone cannot see a later row's move.
                var all_hash: u64 = 1469598103934665603;
                {
                    const total: usize = @as(usize, rows) * 512;
                    for (0..total) |i| all_hash = (all_hash ^ @as(u64, @as(u32, @bitCast(pos[i])))) *% 1099511628211;
                }
                var score_hash: u64 = 1469598103934665603;
                var nan: u32 = 0;
                {
                    const vis = @min(compressed_rows, (position + 1) / ratio);
                    for (0..vis) |k| {
                        const f = sf[(k / 4096 * rows + 0) * 4096 + (k % 4096)];
                        if (std.math.isNan(f)) nan += 1;
                        const bits: u32 = @bitCast(f);
                        score_hash = (score_hash ^ @as(u64, bits)) *% 1099511628211;
                    }
                }
                std.log.info("select dump l{d}: cr={d} topk={d} pos={d} rows={d} few={} | live={d} lo={d} hi={d} over={d} | heap0_score={d} sum={d} xr={d} first4={d},{d},{d},{d} scores0to3={d},{d},{d},{d} shash={x} allhash={x} nan={d}", .{ layer, compressed_rows, c.index_topk, position, rows, compressed_rows <= c.index_topk, live, lo, hi, over, hf[0], sum, xr, pos[0], pos[1], pos[2], pos[3], sf[0], sf[1], sf[2], sf[3], score_hash, all_hash, nan });
            }
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
            attnMark(layer, "main_pack", prof, &e);
        }
    }
    if (scan_probe.active(layer)) {
        scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_pre, layer);
        scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_pre2, layer);
        scan_probe.record(e, &p.plain, chunk, rows, 528 / 2, .chunk, layer);
        scan_probe.record(e, &p.plain, tensor(w.sink), 1, 128, .sink, layer);
        if (ratio != 0 and compressed_rows != 0)
            scan_probe.record(e, &p.plain, main, compressed_rows, 288 / 2, .main, layer);
        scan_probe.record(e, &p.plain, work.selection.positions(), rows, 512 * 4 / 2, .pos, layer);
        if (ratio != 0 and rows > 1) {
            var sk: u32 = 0;
            while (sk < 4 and sk < rows) : (sk += 1) {
                const r = rows - 1 - @as(u32, @intCast((@as(usize, rows - 1) * sk) / 3));
                const vis: u32 = @intCast((position + r + 1) / ratio);
                if (vis != 0) scan_probe.recordStrided(e, &p.plain, work.selection.scores().shifted(@as(usize, r) * 4096 * 4), 1, 4096, vis * 2, .sco, 300000 + r);
            }
        }
    }
    try p.sparse.encode(e, q, ring, chunk, main, work.selection.positions(), tensor(w.sink), q, work.dense(), work.splitPartials(), .{ .rows = rows, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = ratio, .compressed_rows = compressed_rows, .topk = @min(c.index_topk, compressed_rows), .splits = if (rows <= 8) splitScan() else 1 });
    if (scan_probe.active(layer)) {
        e.barrier();
        scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_post, layer);
        scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_post2, layer);
        const live: u32 = @min(rows + compressed_rows, 4096);
        scan_probe.recordAt(e, &p.plain, work.dense(), live, 512, .dense, layer, 128, 0);

        // The SAME scan a second time, on the query the first one read (restored from the probe's own copy). Two
        // outputs of one kernel, one process, byte-identical inputs: the arm that says whether the kernel is
        // deterministic at all.
        if (std.c.getenv("TF_DSV41_SCAN_TWICE") != null) {
            if (scan_probe.stage) |stg| {
                ops.encodeCopy(e, &p.plain, .{ .buffer = stg, .offset = 0 }, q, .{ .rows = rows, .width = @intCast(q_width), .x_stride = @intCast(q_width), .y_stride = @intCast(q_width) });
                e.barrier();
                try p.sparse.encode(e, q, ring, chunk, main, work.selection.positions(), tensor(w.sink), q, work.dense(), work.splitPartials(), .{ .rows = rows, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = ratio, .compressed_rows = compressed_rows, .topk = @min(c.index_topk, compressed_rows), .splits = if (rows <= 8) splitScan() else 1 });
                e.barrier();
                scan_probe.record(e, &p.plain, q, rows, @intCast(q_width), .q_post2, layer);
            }
        }
    }
    e.barrier();
    attnMark(layer, "sparse", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .sparse_o, q);
    if (work.publish_stash) |stash| if (work.publish_limit != 0) {
        const prefix = @min(rows, work.publish_limit);
        if (prefix < rows) try p.cache.publishWindow(e, chunk, .{ .buffer = stash.buffer, .offset = stash.offset + @as(usize, layer) * kv.cache.window * kv.cache.ring_stride }, prefix, position);
    };
    try p.cache.publishWindow(e, chunk, ring, rows, position);
    // TF_DSV41_DOUBLE_ATTN_PUBLISH -- the KV ring publish's same-output duplicate (the publish copies the chunk into
    // the ring's slots; the second pass copies the same bytes).
    if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ATTN_PUBLISH") != null)
        try p.cache.publishWindow(e, chunk, ring, rows, position);
    attnMark(layer, "publish", prof, &e);
    try rotate(p, e, q, rows, c.heads, c.head_dim, position, ratio, 1, true, c);
    if (rows == 1 and std.c.getenv("TF_DSV41_DOUBLE_ROPE") != null) try rotateRepeat(p, e, dup_view, rows, c.heads, c.head_dim, position, ratio, 1, true, c);
    attnMark(layer, "o_rope", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .o, q);
    const grouped = work.view(.grouped);
    const group_input = q_width / c.o_groups;
    const group_output = c.o_groups * c.o_lora;
    // The multi-row path takes the same dispatch -- q is row-major, so `rows * o_groups` rows of `group_input` are
    // exactly the (row, group) slices, and `grouped` is the same row-major layout at `o_lora`; the ids repeat 0..7
    // per row. The prefill (rows > 8) keeps the eight separate projections, where one dispatch per group is the right
    // shape.
    if (rows <= 8 and w.wo_a_descriptors != null and std.c.getenv("TF_DSV41_NO_FUSE_WO_A") == null) {
        for (&w.wo_a) |*m| exl3.useMatrix(e, m.*);
        // TF_DSV41_DOUBLE_ATTN_WOA -- the merged wo_a call's same-output duplicate.
        const wo_a_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ATTN_WOA") != null) 2 else 1;
        for (0..wo_a_passes) |_| try exl3.encodeIndexed(ep, e, .{ .descriptors = .{ .buffer = w.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = group_input, .n = c.o_lora }, core(q), core(grouped), scratch, .{
            .count = rows * c.o_groups,
            .x_stride = group_input,
            .y_stride = c.o_lora,
            .output = .bf16,
            .output_rows = rows * c.o_groups,
            .routes = .{ .ids = core(.{ .buffer = work.ids, .offset = 0 }), .repeat = 1 },
        });
        attnMark(layer, "wo_a", prof, &e);
    } else {
        for (w.wo_a, 0..) |matrix, group| {
            for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = matrix }, q.shifted(group * group_input * 2), grouped.shifted(group * c.o_lora * 2), scratch, rows, q_width, group_output, .bf16);
        }
        // The prompt's eight separate wo_a calls had no mark -- the block ledger charged their GEMMs and transforms
        // to `wo_b`. The mark is the same boundary the fused path uses.
        attnMark(layer, "wo_a", prof, &e);
    }
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .grouped, grouped);
    for (0..dense_passes) |_| try projection(ep, e, .{ .trellis = w.wo_b }, grouped, y, scratch, rows, group_output, c.hidden, .bf16);
    attnMark(layer, "wo_b", prof, &e);
    if (diagnostic) try DiagnosticObserver.observe(true, observer, e, layer, .wo_b, y);
    if (prof) |pf| pf.enc = e;
}
