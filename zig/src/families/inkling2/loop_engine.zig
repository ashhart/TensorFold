//! One-rank, text-only Inkling 2 Mini: the complete 48-visit CUDA forward.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const loop = @import("loop.zig");
const weights_ = @import("loop_weights.zig");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const loop_math = @import("loop_math_cuda.zig");
const round = @import("loop_round.zig");
const dspark = @import("dspark.zig");

pub const prompt_rows: usize = 2048;
pub const window_rows: usize = 16;
/// A shared forward's rows at most: several streams' windows in one pass (loop_round.zig).
pub const round_rows: usize = 128;

pub const Kernels = struct {
    glue: ops.Kernels,
    attn: attention.Kernels,
    moe: modelopt.Kernels,
    loop: loop_math.Kernels,
    sample_mod: cuda.Module,
    draws: cuda.sample.Draws,
    draft: ?dspark.Kernels = null,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var glue = try ops.Kernels.load(d);
        errdefer glue.unload();
        var attn = try attention.Kernels.load(d);
        errdefer attn.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        var moe = try modelopt.Kernels.load(d, sms);
        errdefer moe.unload();
        var lm = try loop_math.Kernels.load(d);
        errdefer lm.unload();
        var sample_mod = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer sample_mod.unload();
        return .{
            .glue = glue,
            .attn = attn,
            .moe = moe,
            .loop = lm,
            .sample_mod = sample_mod,
            .draws = try cuda.sample.Draws.resolve(sample_mod),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.sample_mod.unload();
        if (k.draft) |*draft| draft.unload();
        k.loop.unload();
        k.moe.unload();
        k.attn.unload();
        k.glue.unload();
        k.* = undefined;
    }
};

const Scratch = struct {
    ids: u64,
    pos: u64,
    seg_of: u64,
    segs: u64,
    tiles: u64,
    blocks: [ops.max_blocks]u64 = @splat(0),
    partial_a: u64,
    partial_b: u64,
    prelude: u64,
    bank0: u64,
    bank1: u64,
    raw: u64, // three FP32 AR/gate rows per token
    zeta: u64, // a gate's 64 BF16 bottleneck values per token, between its down and up halves
    messages: u64, // four FP32 message rows per token
    normalized: u64, // three BF16 rows per token
    t: u64,
    q_raw: u64,
    rel_raw: u64,
    sink: u64,
    gate: u64,
    k_raw: u64,
    v_raw: u64,
    k_conv: u64,
    v_conv: u64,
    k_dynamic: u64,
    v_dynamic: u64,
    q_prepared: u64,
    rel_prepared: u64,
    rel_table: u64,
    part_o: u64,
    part_ml: u64,
    attn_heads: u64,
    attn_out: u64,
    route_scores: u64,
    route_pick: u64,
    route_weights: u64,
    moe_out: u64,
    moe_scratch: u64,
    hidden: u64,
    logits: u64,
    taps: u64 = 0, // allocated only when a DSpark drafter attaches
};

/// A stream's state (every visit's K/V rows and conv states, its rewind sites, its window's backup and history).
const Slot = struct { cache: cuda.DeviceBuffer, sites: cuda.DeviceBuffer, backup: cuda.DeviceBuffer, hist: cuda.DeviceBuffer };
pub const max_slots = 16;

/// A slot's allocation sizes: its cache layout, four rewind sites a visit, its window's backup and history.
pub fn slotAllocs(layout: *const loop.CacheLayout, w: *const weights_.Weights) [4]usize {
    const c = w.c.text;
    const kv: usize = @as(usize, c.local.kv) * c.local.dim;
    const visits: usize = w.p.n_visits;
    const cache: usize = @intCast(layout.bytes);
    const backup: usize = @intCast(layout.bytes - layout.at(0).conv_offset);
    return .{ cache, visits * 4 * @sizeOf(ops.RewindSite), backup, visits * (2 * kv + 2 * c.hidden) * window_rows * 4 };
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const weights_.Weights,
    context: usize,
    layout: loop.CacheLayout,
    slots: [max_slots]Slot = undefined,
    n_slots: usize = 0,
    slot: usize = 0, // the slot snapshots and rewinds use; forwards share the scratch
    recorded: [max_slots]bool = @splat(false), // slots whose `backup` and `hist` hold their last window
    s: Scratch = undefined,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn slotBase(m: *const Model) u64 {
        return m.slots[m.slot].cache.ptr;
    }

    /// Allocates slots until `n` exist, each with zeroed conv states and its own rewind sites.
    pub fn addSlots(m: *Model, s: cuda.Stream, n: usize) !void {
        if (n > max_slots) return error.TooManySlots;
        const keep = m.slot;
        defer m.slot = keep;
        const kv: usize = @as(usize, m.w.c.text.local.kv) * m.w.c.text.local.dim;
        var sites: [config.max_visits * 4]ops.RewindSite = undefined;
        while (m.n_slots < n) {
            var slot = try m.newSlot();
            errdefer for ([4]*cuda.DeviceBuffer{ &slot.hist, &slot.backup, &slot.sites, &slot.cache }) |b| b.free();
            m.slots[m.n_slots] = slot;
            m.slot = m.n_slots;
            for (0..m.w.p.n_visits) |v| for (0..4) |site| {
                const c = if (site < 2) kv else m.w.c.text.hidden;
                const hist = m.histOf(@intCast(v), site) - m.slots[m.slot].hist.ptr;
                sites[v * 4 + site] = .{ .state = m.convOf(@intCast(v), site), .hist = @intCast(hist), .backup = @intCast(m.backupOf(@intCast(v), site)), .c = @intCast(c) };
            };
            try m.upload(s, slot.sites.ptr, std.mem.sliceAsBytes(sites[0 .. m.w.p.n_visits * 4]));
            try m.reset(s);
            try s.synchronize();
            m.n_slots += 1;
        }
    }

    /// A slot's four allocations, sized by slotAllocs: its cache, rewind-site table, window backup and history.
    pub fn newSlot(m: *const Model) !Slot {
        const sizes = slotAllocs(&m.layout, m.w);
        var cache = try cuda.DeviceBuffer.alloc(m.d, sizes[0]);
        errdefer cache.free();
        var sites = try cuda.DeviceBuffer.alloc(m.d, sizes[1]);
        errdefer sites.free();
        var backup = try cuda.DeviceBuffer.alloc(m.d, sizes[2]);
        errdefer backup.free();
        const hist = try cuda.DeviceBuffer.alloc(m.d, sizes[3]);
        return .{ .cache = cache, .sites = sites, .backup = backup, .hist = hist };
    }

    fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    pub fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    pub fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    fn zeros(m: *const Model, s: cuda.Stream, ptr: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemsetD8Async(ptr, 0, bytes, s.handle), "cuMemsetD8Async");
    }

    pub fn layerGeometry(m: *const Model, layer: u32) attention.Layer {
        return if (config.isGlobal(layer))
            attention.Layer.global(@intCast(m.context))
        else
            attention.Layer.sliding(m.w.c.text.window, @intCast(@min(prompt_rows, m.context)));
    }

    pub fn cacheOf(m: *const Model, visit: u32) attention.Cache {
        const slot = m.layout.at(visit);
        const base = m.slotBase() + slot.kv_offset;
        return .{ .k = base, .v = base + slot.kv_bytes / 2 };
    }

    pub fn convOf(m: *const Model, visit: u32, site: usize) u64 {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.slotBase() + m.layout.at(visit).conv_offset;
        for (0..site) |i| at += 3 * widths[i] * 4;
        return at;
    }

    fn histBytes(m: *const Model) usize {
        return slotAllocs(&m.layout, m.w)[3];
    }

    pub fn histOf(m: *const Model, visit: u32, site: usize) u64 {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.slots[m.slot].hist.ptr + visit * window_rows * (2 * kv + 2 * h) * 4;
        for (0..site) |i| at += window_rows * widths[i] * 4;
        return at;
    }

    fn backupOf(m: *const Model, visit: u32, site: usize) usize {
        return m.convOf(visit, site) - (m.slotBase() + m.layout.at(0).conv_offset);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const weights_.Weights, context: usize) !Model {
        const layout = try loop.CacheLayout.init(&w.c.text, &w.p, @intCast(context));
        var m: Model = .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .context = context,
            .layout = layout,
        };
        errdefer m.deinitBuffers();
        const c = w.c.text;
        const h: usize = c.hidden;
        const q: usize = @as(usize, c.global.q) * c.global.dim;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const rows = prompt_rows;
        const rel_rows = rows * c.global.q * attention.extent;
        const local = attention.Layer.sliding(c.window, @intCast(@min(prompt_rows, context)));
        const global = attention.Layer.global(@intCast(context));
        const local_slots = attention.slotsFor(local, context);
        const global_slots = attention.slotsFor(global, context);
        const part_slots = @max(rows * c.local.q * local_slots, rows * c.global.q * global_slots);
        m.s = .{
            .ids = try m.alloc(rows * 4),
            .pos = try m.alloc(rows * 4),
            .seg_of = try m.alloc(rows * 4),
            .segs = try m.alloc(max_slots * @sizeOf(attention.Seg)),
            .tiles = try m.alloc(max_slots * 2 * 128 * @sizeOf(attention.Tile)),
            .partial_a = try m.alloc(rows * h * 2),
            .partial_b = try m.alloc(rows * h * 2),
            .prelude = try m.alloc(rows * h * 2),
            .bank0 = try m.alloc(rows * h * 2),
            .bank1 = try m.alloc(rows * h * 2),
            .raw = try m.alloc(3 * rows * h * 4),
            .zeta = try m.alloc(rows * loop_math.rank * 2),
            .messages = try m.alloc(4 * rows * h * 4),
            .normalized = try m.alloc(3 * rows * h * 2),
            .t = try m.alloc(rows * h * 4),
            .q_raw = try m.alloc(rows * q * 2),
            .rel_raw = try m.alloc(rows * c.global.q * c.d_rel * 2),
            .sink = try m.alloc(rows * c.global.q * 2),
            .gate = try m.alloc(rows * q * 2),
            .k_raw = try m.alloc(rows * kv * 2),
            .v_raw = try m.alloc(rows * kv * 2),
            .k_conv = try m.alloc(rows * kv * 2),
            .v_conv = try m.alloc(rows * kv * 2),
            .k_dynamic = try m.alloc(rows * c.global.kv * config.conv_taps * 4),
            .v_dynamic = try m.alloc(rows * c.global.kv * config.conv_taps * 4),
            .q_prepared = try m.alloc(rows * q * 2),
            .rel_prepared = try m.alloc(rows * c.global.q * c.d_rel * 2),
            .rel_table = try m.alloc(rel_rows * 2),
            .part_o = try m.alloc(part_slots * attention.head_dim * 4),
            .part_ml = try m.alloc(part_slots * 8),
            .attn_heads = try m.alloc(rows * q * 2),
            .attn_out = try m.alloc(rows * h * 2),
            .route_scores = try m.alloc(rows * c.experts * 4),
            .route_pick = try m.alloc(rows * c.experts * 4),
            .route_weights = try m.alloc(rows * c.experts * 4),
            .moe_out = try m.alloc(rows * h * 2),
            .moe_scratch = try m.alloc(try modelopt.scratchBytes(rows, c.topk, h, c.expert_inter)),
            .hidden = try m.alloc(rows * h * 2),
            .logits = try m.alloc(round_rows * c.unpadded_vocab * 4),
        };
        for (0..ops.max_blocks) |i| m.s.blocks[i] = try m.alloc(rows * h * 2);
        try m.addSlots(s, 1);
        return m;
    }

    pub fn prepareTaps(m: *Model) !void {
        if (m.s.taps == 0) m.s.taps = try m.alloc(prompt_rows * dspark.taps.len * m.w.c.text.hidden * 2);
    }

    fn deinitBuffers(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        m.deinitBuffers();
        for (m.slots[0..m.n_slots]) |*slot| for ([4]*cuda.DeviceBuffer{ &slot.hist, &slot.backup, &slot.sites, &slot.cache }) |b| b.free();
        m.* = undefined;
    }

    pub fn reset(m: *const Model, s: cuda.Stream) !void {
        try m.zeros(s, m.slotBase() + m.layout.at(0).conv_offset, m.layout.bytes - m.layout.at(0).conv_offset);
    }

    pub fn rewind(m: *const Model, k: *const Kernels, s: cuda.Stream, kept: usize) !void {
        if (kept > window_rows) return error.WindowTooWide;
        if (!m.recorded[m.slot]) return error.RewindOfAnotherSlot;
        const slot = &m.slots[m.slot];
        try ops.rewind(&k.glue, s, slot.sites.ptr, m.w.p.n_visits * 4, m.w.c.text.hidden, slot.backup.ptr, slot.hist.ptr, 0, 0, kept);
    }

    pub fn arBlocks(m: *const Model, blocks: []const u64, partial: u64, query: u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) loop_math.ArArgs {
        var a: loop_math.ArArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
        };
        for (blocks, 0..) |b, i| a.blocks[i] = b;
        return a;
    }

    pub fn mixArgs(m: *const Model, blocks: []const u64, partial: u64, query: u64, norms: [3]u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) ops.MixArgs {
        var a: ops.MixArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
        };
        for (blocks, 0..) |b, i| a.blocks[i] = b;
        a.norm = norms;
        return a;
    }

    /// One prompt chunk or one decode window of the current slot. `record` keeps every convolution input for rollback.
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, tokens: []const u32, p0: usize, head0: usize, heads: usize, record: bool) !void {
        if (tokens.len == 0 or tokens.len > prompt_rows or (!record and heads > window_rows)) return error.BadLoopRows;
        if (record and tokens.len > window_rows) return error.WindowTooWide;
        return round.forward(m, k, s, &.{.{ .slot = m.slot, .p0 = p0, .tokens = tokens }}, head0, heads, record);
    }

    pub const Chain = round.Chain;
    pub const forwardChains = round.forward;

    pub const Saved = struct {
        at: u32,
        visits: u32,
        passes: u32,
        hidden: u32,
        buf: cuda.DeviceBuffer,
    };

    /// Compact bytes for all valid K/V rows and every convolution state of all 48 visits.
    pub fn snapshotBytes(m: *const Model, at: usize) u64 {
        var bytes = m.layout.bytes - m.layout.at(0).conv_offset;
        for (0..m.w.p.n_visits) |v| {
            const layer = m.w.p.at(m.w.p.visits[v].layer);
            const global = layer.kind == .global;
            const count = if (global) at else @min(at, m.w.c.text.window - 1);
            const row = @as(u64, layer.heads.kv) * layer.heads.dim * 2;
            bytes += 2 * count * row;
        }
        return bytes;
    }

    fn copyState(m: *const Model, s: cuda.Stream, saved: *const Saved, out: bool) !void {
        const at: usize = saved.at;
        var off: usize = 0;
        for (0..m.w.p.n_visits) |v| {
            const visit = m.w.p.visits[v];
            const layer = m.w.p.at(visit.layer);
            const geometry = m.layerGeometry(visit.layer);
            const rb = @as(usize, layer.heads.dim) * 2; // the caches are head-major: [kv head][ring][dim]
            const global = layer.kind == .global;
            const count = if (global) at else @min(at, m.w.c.text.window - 1);
            const first = at - count;
            const cache = m.cacheOf(visit.cache_slot);
            for ([2]u64{ cache.k, cache.v }) |k_or_v| for (0..layer.heads.kv) |head| {
                const base = k_or_v + head * geometry.ring * rb;
                var pos = first;
                while (pos < at) {
                    const row = pos % geometry.ring;
                    const n = @min(at - pos, geometry.ring - row);
                    const dev = base + row * rb;
                    const src = if (out) dev else saved.buf.ptr + off;
                    const dst = if (out) saved.buf.ptr + off else dev;
                    try m.copy(s, dst, src, n * rb);
                    off += n * rb;
                    pos += n;
                }
            };
        }
        const conv = m.slotBase() + m.layout.at(0).conv_offset;
        const bytes = m.layout.bytes - m.layout.at(0).conv_offset;
        try m.copy(s, if (out) saved.buf.ptr + off else conv, if (out) conv else saved.buf.ptr + off, bytes);
    }

    pub fn save(m: *const Model, s: cuda.Stream, at: u32) !*Saved {
        if (at == 0 or at > m.context) return error.NotAtMark;
        const gpa = m.gpa;
        const saved = try gpa.create(Saved);
        errdefer gpa.destroy(saved);
        const lp = m.w.c.text.loop.?;
        saved.* = .{ .at = at, .visits = m.w.p.n_visits, .passes = lp.passes, .hidden = m.w.c.text.hidden, .buf = try cuda.DeviceBuffer.alloc(m.d, @intCast(@max(snapshotBytes(m, at), 256))) };
        errdefer saved.buf.free();
        try m.copyState(s, saved, true);
        return saved;
    }

    pub fn restore(m: *const Model, s: cuda.Stream, saved: *const Saved) !void {
        const lp = m.w.c.text.loop.?;
        if (saved.visits != m.w.p.n_visits or saved.passes != lp.passes or saved.hidden != m.w.c.text.hidden)
            return error.StateOfAnotherModel;
        if (saved.at == 0 or saved.at > m.context or saved.buf.len < snapshotBytes(m, saved.at)) return error.BadLoopSnapshot;
        try m.copyState(s, saved, false);
    }

    pub fn drop(m: *const Model, s: cuda.Stream, saved: *Saved) void {
        s.synchronize() catch {};
        saved.buf.free();
        m.gpa.destroy(saved);
    }
};
