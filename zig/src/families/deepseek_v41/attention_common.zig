//! Pinned DeepSeek V4.1 attention, resident EXL3 projections and causal packed-cache chunk execution.
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

/// One segment of a batched decode forward: a slot's state, its first position.
pub const Seg = struct { state: *State, p0: u32, r0: u32, n: u32 };

/// The attention stage's per-block GPU split (TF_DSV41_ATTN_PROF=1).
pub const Prof = struct {
    queue: mtl.Queue,
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
    on: bool,
    total_ms: f64 = 0,
    parts: u32 = 0,

    pub fn split(self: *Prof, label: []const u8, layer: u32, enc: *mtl.ComputeEncoder) void {
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

pub var attn_prof_flag: ?bool = null;
pub var attn_prof_layers: ?[]const u8 = null;

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

    pub fn observe(comptime enabled: bool, self: ?DiagnosticObserver, e: mtl.ComputeEncoder, layer: u32, stage: DiagnosticStage, source: View) !void {
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
pub const regions = [_]usize{ 1280 * 2, 32768 * 2, 512 * 2, 512 * 4, 512 * 4, 512 * 2, 128 * 2, 4096 * 2, 32 * 2, 8192 * 2, 2560 * 2 };
pub const Region = enum(usize) { qr, query, raw_kv, values, gates, latent, index_k, index_q, index_weights, grouped, qkv };

/// The batched decode window's per-forward tables and per-slot scratch (attention_rows.zig): slot count, table layout.
pub const RowsScratch = struct {
    pub const slots = 8;
    /// The table: positions, then 4 ulongs a slot of addresses (live, 0, selection, 0), then 4 of owner main addresses.
    pub const pos_at = 0;
    pub const addr_at = slots * 4;
    pub const mains_at = addr_at + slots * 4 * 8;
    pub const tab_bytes = mains_at + slots * 4 * 8;
    tab: mtl.Buffer,
    /// One window chunk row (kv.cache.ring_stride bytes) a slot.
    chunk: mtl.Buffer,
    /// A dense key table (csa2.rows_dense rows of 512 bf16) a slot.
    dense: mtl.Buffer,
    /// The table's last upload; a forward re-uploads only a changed table (the GPU may still be reading it).
    uploaded: [tab_bytes]u8 = undefined,
    uploaded_valid: bool = false,

    pub fn denseBytes() usize {
        return slots * csa2.rows_dense * 512 * @sizeOf(u16);
    }
    pub fn init(device: mtl.Device) !RowsScratch {
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const tab = try device.buffer(tab_bytes, opts);
        errdefer tab.deinit();
        const chunk = try device.buffer(slots * kv.cache.ring_stride, opts);
        errdefer chunk.deinit();
        const dense = try device.buffer(denseBytes(), opts);
        return .{ .tab = tab, .chunk = chunk, .dense = dense };
    }
    pub fn deinit(r: *RowsScratch) void {
        r.tab.deinit();
        r.chunk.deinit();
        r.dense.deinit();
    }
    pub fn residentBytes() usize {
        return tab_bytes + slots * kv.cache.ring_stride + denseBytes();
    }
};

pub const Workspace = struct {
    buffer: mtl.Buffer,
    /// Every visible key decoded once a layer: the window ring, the chunk, then the compressed picks.
    dense_kv: mtl.Buffer,
    /// A decode layer's split key scans: (split, head group, lane) by (max, sum, 64 accumulators).
    partials: mtl.Buffer,
    selection: index.Scratch,
    /// The batch's per-segment selection scratches; segment 0 uses `selection`.
    sel_extra: [RowsScratch.slots - 1]?index.Scratch = @splat(null),
    /// Identity map [0..7] for the fused wo_a indexed dispatch (one row per output group).
    ids: mtl.Buffer,
    /// A scratch row for the doubling arms; the live workspace rows would change the reply if reused.
    dup: mtl.Buffer,
    /// The batched wq_a+wkv call's input-row map -- both entries read the one decode row (0, 0).
    pair_in: mtl.Buffer,
    max_rows: u32,
    rows: RowsScratch,
    selection_layer: ?u32 = null,
    selection_position: u32 = 0,
    selection_rows: u32 = 0,
    candidate_position: u32 = 0,
    candidate_rows: u32 = 0,
    /// A merged tail chunk's PREFIX (its rows before the tail mark) is published here as well as the whole chunk.
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
        var sel_extra: [RowsScratch.slots - 1]?index.Scratch = @splat(null);
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
        const rows = try RowsScratch.init(device);
        return .{ .buffer = buffer, .dense_kv = dense_kv, .partials = partials, .selection = selection, .sel_extra = sel_extra, .ids = ids, .dup = dup, .pair_in = pair_in, .max_rows = max_rows, .rows = rows };
    }
    pub fn deinit(w: *Workspace) void {
        w.buffer.deinit();
        w.dense_kv.deinit();
        w.partials.deinit();
        w.ids.deinit();
        w.dup.deinit();
        w.pair_in.deinit();
        w.selection.deinit();
        w.rows.deinit();
        for (&w.sel_extra) |*e| if (e.*) |*x| x.deinit();
    }
    pub fn residentBytes(w: *const Workspace) usize {
        return bytes(w.max_rows) + denseKvBytes(w.max_rows) + w.sel_extra.len * index.Scratch.bytes(w.max_rows) + RowsScratch.residentBytes();
    }
    pub fn dense(self: *const Workspace) View {
        return .{ .buffer = self.dense_kv, .offset = 0 };
    }
    /// The identity map for the fused wo_a's indexed dispatch (one row per output group).
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
        set.add(w.rows.tab);
        set.add(w.rows.chunk);
        set.add(w.rows.dense);
    }
    /// A region's view into the attention workspace, public so the drafter can assemble its own scan.
    pub fn view(w: *const Workspace, region: Region) View {
        var offset: usize = 0;
        for (regions[0..@backingInt(region)]) |size| offset += size * w.max_rows;
        return .{ .buffer = w.buffer, .offset = offset };
    }
};

pub fn tensor(t: checkpoint.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}
pub fn core(v: View) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}
pub fn projection(p: *const exl3.Pipelines, e: mtl.ComputeEncoder, m: weights.AnyMatrix, x: View, y: View, scratch: exl3.View, rows: u32, xs: u32, ys: u32, output: exl3.Output) !void {
    switch (m) {
        .trellis => |t| {
            // The prompt-row int8 twin.
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
pub fn norm(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, weight: checkpoint.Tensor, rows: u32, width: u32, eps: f32) void {
    ops.encodeRms(e, &p.plain, x, tensor(weight), x, .{ .rows = rows, .width = width, .x_stride = width, .y_stride = width, .eps = eps, .weight_f32 = weight.dtype == .f32 });
    e.barrier();
}

pub fn normRepeat(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, dup: View, weight: checkpoint.Tensor, rows: u32, width: u32, eps: f32) void {
    ops.encodeRms(e, &p.plain, x, tensor(weight), dup, .{ .rows = rows, .width = width, .x_stride = width, .y_stride = width, .eps = eps, .weight_f32 = weight.dtype == .f32 });
    e.barrier();
}
pub fn rotate(p: *const Pipelines, e: mtl.ComputeEncoder, x: View, rows: u32, heads: u32, width: u32, position: u32, ratio: u32, step: u32, inverse: bool, c: config.Config) !void {
    try p.rotary.encode(e, x, .{ .rows = rows, .heads = heads, .width = width, .rotary = c.rope_dim, .stride = heads * width, .position = position, .step = step, .inverse = @intFromBool(inverse), .spectrum = if (ratio == 0) .normal else .compressed });
}

pub fn rotateRepeat(p: *const Pipelines, e: mtl.ComputeEncoder, dup: View, rows: u32, heads: u32, width: u32, position: u32, ratio: u32, step: u32, inverse: bool, c: config.Config) !void {
    try p.rotary.encode(e, dup, .{ .rows = rows, .heads = heads, .width = width, .rotary = c.rope_dim, .stride = heads * width, .position = position, .step = step, .inverse = @intFromBool(inverse), .spectrum = if (ratio == 0) .normal else .compressed });
}
// The attention stage's own block ledger.
pub var attn_count_flag: ?bool = null;
pub var attn_count_layers: ?[]const u8 = null;
pub var attn_count_base: u64 = 0;
pub var attn_group_base: u64 = 0;

pub fn attnCountOn(layer: u32) bool {
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
pub fn attnBegin(layer: u32) void {
    if (!attnCountOn(layer)) return;
    attn_count_base = mtl.command.dispatch_count.load(.monotonic);
    attn_group_base = mtl.command.dispatch_threadgroups.load(.monotonic);
}

/// A block boundary: the dispatches this block issued, and (TF_DSV41_ATTN_PROF) the block's GPU window.
pub fn attnMark(layer: u32, block: []const u8, prof: ?*Prof, enc: *mtl.ComputeEncoder) void {
    if (prof) |pf| pf.split(block, layer, enc);
    if (!attnCountOn(layer)) return;
    const now = mtl.command.dispatch_count.load(.monotonic);
    const groups = mtl.command.dispatch_threadgroups.load(.monotonic);
    // The threadgroup count beside the dispatch count -- the parallelism ledger.
    std.log.info("attn {d} {s}: {d} dispatches, {d} threadgroups", .{ layer, block, now - attn_count_base, groups - attn_group_base });
    attn_count_base = now;
    attn_group_base = groups;
}

pub fn precedingIndex(layer: u32) u32 {
    var source: u32 = 2;
    for (kv.cache.index_sources) |id| {
        if (id > layer) break;
        source = id;
    }
    return source;
}

/// BF16 x/y [rows,5120]; layers 0..39 share the chunk-start cursor, appended by the caller after all layers.

// No extra RMS after wq_b or index-query Hadamard outside EXL3.
pub fn partialBytes(rows: u32) usize {
    // A split's (max, sum, accumulators) for every row: a drafted window above one row keeps its own parts.
    return @as(usize, rows) * 32 * 4 * 4 * 32 * 68 * @sizeOf(f32);
}
/// The window ring, up to `max_rows` chunk rows, and as many compressed rows as the context allows.
pub fn denseKvBytes(max_rows: u32) usize {
    return (@as(usize, 128) + max_rows + kv.cache.max_context) * 512 * @sizeOf(u16);
}

pub var split_scan: ?u32 = null;
/// TF_DSV41_ATTN_SPLITS: key-scan slices a decode layer's sparse attention runs (default 8).
pub fn splitScan() u32 {
    if (split_scan) |v| return v;
    var out: u32 = 8;
    if (std.c.getenv("TF_DSV41_ATTN_SPLITS")) |v| {
        const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch out;
        if (n >= 1 and n <= 32) out = n;
    }
    split_scan = out;
    return out;
}
