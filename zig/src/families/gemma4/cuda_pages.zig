//! Global K/V as core-pool pages; sliding layers retain their existing per-slot rings.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const cfg = @import("cuda_config.zig");

pub const tokens = cuda.window_attention.page_unit;

pub fn perSlot(m: anytype) usize {
    return (m.context + tokens - 1) / tokens;
}

pub fn pageElems(m: anytype) usize {
    return tokens * m.geo.global_kv_heads * cfg.global_dim;
}

pub fn globalLayers(m: anytype) usize {
    var n: usize = 0;
    for (0..m.layers) |i| n += @intFromBool(cfg.global(i));
    return n;
}

pub fn pageBytes(m: anytype) u64 {
    return globalLayers(m) * 2 * pageElems(m) * 2;
}

pub fn layout(m: anytype) kv.Layout {
    var fixed: u64 = perSlot(m) * 4;
    for (0..m.layers) |i| if (!cfg.global(i)) {
        fixed += 2 * m.cacheLen(i) * m.geo.kvHeads(i) * cfg.headDim(i) * 2;
    };
    return .{ .fixed_bytes = fixed, .token_bytes = pageBytes(m) / tokens, .page_tokens = tokens, .snap_bytes = 0 };
}

pub fn view(m: anytype, slot: ?usize) ?cuda.window_attention.Pages {
    if (!m.paged) return null;
    return .{ .table = m.table + (slot orelse 0) * perSlot(m) * 4, .page_elems = pageElems(m), .tokens = tokens, .stride = perSlot(m) };
}

/// One warmup page before attach; otherwise the unchanged linear or sliding cache allocation.
pub fn build(m: anytype, s: cuda.Stream) !void {
    for (0..m.layers) |i| {
        const n = if (m.paged and cfg.global(i)) pageElems(m) else m.slots * m.cacheLen(i) * m.geo.kvHeads(i) * cfg.headDim(i);
        m.kc[i] = try m.zeros(s, n * 2);
        m.vc[i] = try m.zeros(s, n * 2);
    }
    if (m.paged) {
        try tables(m, s);
        if (m.planned == null) m.caps[0] = tokens;
    }
}

fn tables(m: anytype, s: cuda.Stream) !void {
    m.table = try m.alloc(m.slots * perSlot(m) * 4);
    if (m.planned != null) return;
    m.ids = try m.gpa.alloc(u32, m.slots * perSlot(m));
    @memset(m.ids, 0);
    m.caps = try m.gpa.alloc(usize, m.slots);
    @memset(m.caps, 0);
    try m.upload(s, m.table, std.mem.sliceAsBytes(m.ids));
}

fn forget(m: anytype, ptr: u64) void {
    for (m.buffers.items, 0..) |*b, i| if (b.ptr == ptr) {
        b.free();
        _ = m.buffers.swapRemove(i);
        return;
    };
}

/// The native host has priced every slot and page; no forward may run until attach succeeds.
pub fn attach(m: anytype, s: cuda.Stream, extra: usize, count: usize) !void {
    if (!m.paged or m.pages != 0 or count == 0 or extra >= m.graphs.len - 1 or globalLayers(m) == 0) return error.BadAttach;
    try s.synchronize();
    m.dropGraphs();
    for (m.kc, m.vc) |k, v| {
        forget(m, k);
        forget(m, v);
    }
    forget(m, m.table);
    m.gpa.free(m.ids);
    m.gpa.free(m.caps);
    m.ids = &.{};
    m.caps = &.{};
    m.slots = extra + 1;
    m.pages = count;
    const region = count * pageElems(m) * 2;
    const arena = try m.zeros(s, globalLayers(m) * 2 * region);
    var block: usize = 0;
    for (0..m.layers) |i| {
        if (cfg.global(i)) {
            m.kc[i] = arena + block * region;
            m.vc[i] = m.kc[i] + region;
            block += 2;
        } else {
            const bytes = m.slots * m.cacheLen(i) * m.geo.kvHeads(i) * cfg.headDim(i) * 2;
            m.kc[i] = try m.zeros(s, bytes);
            m.vc[i] = try m.zeros(s, bytes);
        }
    }
    try tables(m, s);
    try s.synchronize();
}

pub fn map(m: anytype, s: cuda.Stream, slot: usize, ids: []const u32, from: usize) !void {
    const per = perSlot(m);
    if (!m.paged or slot >= m.slots or from > per or ids.len > per - from) return error.BadChain;
    for (ids) |id| if (id >= m.pages) return error.BadChain;
    const at = slot * per + from;
    @memcpy(m.ids[at..][0..ids.len], ids);
    m.caps[slot] = (from + ids.len) * tokens;
    try m.upload(s, m.table + at * 4, std.mem.sliceAsBytes(m.ids[at..][0..ids.len]));
}
