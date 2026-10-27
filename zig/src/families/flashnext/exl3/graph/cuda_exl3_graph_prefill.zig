//! Prefill graphs of the EXL3 engine: one layer's sub-blocks of a deep chunk captured and replayed.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const hub = @import("cuda_exl3_graph.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");

// ---- prefill: one layer's sub-blocks of a deep chunk.

pub const Prefill = enum { eager, replayed, capturing };
const prefill_binding = @import("../prefill/cuda_exl3_prefill_binding.zig");

pub const PBuffer = struct {
    driver: *const cuda.Driver,
    address: u64,
    bytes: usize,

    pub fn of(buffer: cuda.DeviceBuffer) PBuffer {
        return .{ .driver = buffer.d, .address = buffer.ptr, .bytes = buffer.len };
    }
};

pub const PGdn = struct {
    conv: PBuffer,
    state: PBuffer,

    pub fn of(sequence: anytype, layer: usize) PGdn {
        const index = wt.linearIndex(layer);
        return .{ .conv = PBuffer.of(sequence.conv[index]), .state = PBuffer.of(sequence.state[index]) };
    }
};

pub const PKey = struct {
    layer: usize,
    rows: usize,
    sub: usize,
    /// The chunk's first position (attention layers only; 0 for DeltaNet layers).
    pos0: usize,
    deep: u64,
    base: u64,
    base_cap: usize,
    seq: [5]u64,
    paging: [6]u64,
    wait_count: usize,
    wait_key: u64,
    scores: u64,
    tiles: bool,
    positions: u64 = 0,
    stream: u64 = 0,
    gdn: ?PGdn = null,
};

pub const PEntry = struct { key: PKey, exec: cuda.graph.Exec, used: u64, admission: ?memory.Ticket = null, stream: ?cuda.Stream = null, binding: ?*prefill_binding.Binding = null, binding_admission: ?memory.Ticket = null };

pub fn setPrefillBinding(entry: *PEntry, binding: ?*prefill_binding.Binding, admission: ?memory.Ticket) void {
    if (entry.binding) |old| old.deinit(std.heap.c_allocator);
    if (entry.binding_admission) |ticket| memory.release(ticket) catch unreachable;
    entry.binding = binding;
    entry.binding_admission = admission;
}

pub fn dropPrefill(entry: *PEntry) void {
    if (entry.stream) |stream| stream.synchronize() catch @panic("prefill retirement synchronization failed");
    entry.exec.deinit();
    setPrefillBinding(entry, null, null);
    if (entry.admission) |ticket| memory.release(ticket) catch unreachable;
}
pub const pcache_n = 112;
pub var pentries: [pcache_n]?PEntry = @splat(null);
pub var pseen: [pcache_n * 2]?struct { key: PKey, n: u32 } = @splat(null);
pub var pseen_at: usize = 0;
pub var pcur: ?PKey = null;
pub var pcur_admission: ?memory.Ticket = null;
pub var pcur_binding_admission: ?memory.Ticket = null;
pub var pcur_binding_request: ?prefill_binding.Request = null;

pub fn prefillAdmissionMiss() Prefill {
    hub.prefill_stats.eager += 1;
    hub.prefill_stats.admission_misses += 1;
    const misses = hub.prefill_stats.admission_misses;
    if (misses == 1 or misses % 256 == 0) std.log.info("flashnext exl3 prefill graph budget admission declines: {d}; retained cache preserved, layer runs eagerly", .{misses});
    return .eager;
}

/// TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1 captures prefill layers under TF_FLASHNEXT_EXL3_GRAPH=1 (default off pending
pub fn prefillEnabled() bool {
    if (!hub.enabled() or gcache.broken) return false;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_PREFILL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub fn pshape(k: PKey) PKey {
    var s = k;
    s.pos0 = 0;
    s.paging[5] = 0;
    return s;
}

pub fn prefillOriginCandidate(key: PKey, request: prefill_binding.Request) ?*PEntry {
    for (&pentries) |*maybe| if (maybe.*) |*entry| {
        const binding = entry.binding orelse continue;
        if (!binding.poisoned and std.meta.eql(pshape(entry.key), pshape(key)) and binding.request.compatible(request)) return entry;
    };
    return null;
}

pub fn psighting(k: PKey) u32 {
    const s = pshape(k);
    for (&pseen) |*e| if (e.*) |*x| if (std.meta.eql(x.key, s)) {
        x.n += 1;
        hub.prefill_stats.sighting_hits += 1;
        return x.n;
    };
    hub.prefill_stats.sighting_misses += 1;
    pseen[pseen_at % pseen.len] = .{ .key = s, .n = 1 };
    pseen_at += 1;
    return 1;
}

/// TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER: the sighting of a layer shape that captures it (default 2: a chunk takes
pub fn prefillAfter() u32 {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER") orelse return 2;
    return @max(1, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2);
}

pub fn clearPrefill() void {
    for (&pentries) |*e| if (e.*) |*x| {
        dropPrefill(x);
        e.* = null;
    };
    pseen = @splat(null);
}

/// Before a layer's sub-blocks:
pub fn prefillBegin(p: anytype, linear: bool, layer: usize, rows: usize, sub: usize, pos0: usize, deep: u64, base: anytype, tiles: bool, waits: []const cuda.Event, positions: u64) !Prefill {
    if (hub.enabled() and gcache.broken) return error.GraphCaptureFailed;
    if (p.seq.paged) |pg| try pg.validateSharedLeaseOn(p.stream);
    if (!prefillEnabled()) return .eager;
    const q = p.seq;
    if (!linear) if (q.paged) |pg| {
        try pg.prepareLayer(p.stream, wt.attnIndex(layer), pos0, rows);
        pg.graph_prepared = true;
    };
    errdefer if (q.paged) |pg| {
        pg.graph_prepared = false;
    };
    var wait_key: u64 = 0;
    for (waits) |ev| {
        const handle = if (ev.handle) |h| @intFromPtr(h) else 0;
        wait_key = std.hash.Wyhash.hash(wait_key, std.mem.asBytes(&handle));
    }
    const key: PKey = .{
        .layer = layer,
        .rows = rows,
        .sub = sub,
        .pos0 = if (linear) 0 else pos0,
        .deep = deep,
        .base = base.ids,
        .base_cap = base.cap,
        .seq = .{ if (q.conv.len > 0) q.conv[0].ptr else 0, if (q.kc.len > 0) q.kc[0].ptr else 0, q.tail.ptr, q.cap, if (q.state.len > 0) q.state[0].ptr else 0 },
        .paging = if (!linear) (if (q.paged) |pg| pg.prefillGraphKey() else @splat(0)) else @splat(0),
        .wait_count = waits.len,
        .wait_key = wait_key,
        .scores = q.scores,
        .tiles = tiles,
        .positions = positions,
        .stream = @intFromPtr(p.stream.handle),
        .gdn = if (linear) PGdn.of(q, layer) else null,
    };
    gcache.clock += 1;
    for (&pentries) |*e| if (e.*) |*x| if (std.meta.eql(x.key, key)) {
        if (x.binding) |binding| if (binding.poisoned) continue;
        x.used = gcache.clock;
        hub.prefill_stats.replays += 1;
        if (x.binding != null) hub.prefill_stats.paged_replays += 1;
        try x.exec.launchOn(p.stream);
        return .replayed;
    };
    const request = prefill_binding.requestFor(p, linear, layer, rows, sub, pos0, positions);
    if (request) |target| if (prefillOriginCandidate(key, target)) |entry| {
        try p.stream.synchronize();
        try entry.binding.?.apply(prefill_binding.Backend{ .exec = entry.exec }, target);
        entry.key = key;
        entry.used = gcache.clock;
        hub.prefill_stats.rebinds += 1;
        hub.prefill_stats.replays += 1;
        hub.prefill_stats.paged_replays += 1;
        try entry.exec.launchOn(p.stream);
        return .replayed;
    };
    if (psighting(key) < prefillAfter()) {
        hub.prefill_stats.eager += 1;
        return .eager;
    }
    var free_slot = false;
    for (pentries) |entry| free_slot = free_slot or entry == null;
    if (!free_slot) return prefillAdmissionMiss();
    _ = try memory.startupReservation();
    const admission = memory.admitWithQuery(.prefill, 0, p.stream, gbudget.freeMemory) catch |err| switch (err) {
        error.GraphBudgetBusy, error.GraphBudgetExceeded, error.GraphMemoryHeadroom, error.GraphEvictionRequired => return prefillAdmissionMiss(),
        else => return err,
    };
    errdefer memory.release(admission) catch unreachable;
    const binding_admission: ?memory.Ticket = if (request != null) memory.admitWithQuery(.prefill, 0, p.stream, gbudget.freeMemory) catch null else null;
    errdefer if (binding_admission) |ticket| memory.release(ticket) catch unreachable;
    try p.stream.synchronize();
    try cuda.graph.beginCapture(p.stream, .thread_local);
    hub.capturing = true;
    pcur = key;
    pcur_admission = admission;
    pcur_binding_request = request;
    pcur_binding_admission = binding_admission;
    return .capturing;
}

/// After a layer's sub-blocks ran under `.capturing`: the graph is instantiated (or a cached one of the same shape
pub fn prefillEnd(p: anytype, how: Prefill) !void {
    defer if (p.seq.paged) |pg| {
        pg.graph_prepared = false;
    };
    if (how != .capturing) return;
    const t0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    hub.capturing = false;
    const key = pcur.?;
    pcur = null;
    const admission = pcur_admission.?;
    pcur_admission = null;
    const request = pcur_binding_request;
    pcur_binding_request = null;
    const binding_admission = pcur_binding_admission;
    pcur_binding_admission = null;
    var binding_admitted = true;
    defer if (binding_admitted) if (binding_admission) |ticket| memory.release(ticket) catch unreachable;
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    var graph = try cuda.graph.endCapture(p.stream);
    const binding: ?*prefill_binding.Binding = if (binding_admission != null and request != null) prefill_binding.Binding.create(std.heap.c_allocator, graph, request.?) catch null else null;
    var kept = false;
    defer if (!kept) {
        if (binding) |owned| owned.deinit(std.heap.c_allocator) else graph.deinit();
    };
    if (binding != null) try memory.installed(binding_admission.?);
    const at: *PEntry = blk: {
        for (&pentries) |*e| if (e.*) |*x| if (binding == null and std.meta.eql(pshape(x.key), pshape(key))) {
            if ((try x.exec.update(graph)) == .success) {
                try gbudget.uploadResident(&x.exec, p.stream);
                x.key = key;
                x.used = gcache.clock;
                setPrefillBinding(x, binding, if (binding != null) binding_admission else null);
                kept = binding != null;
                binding_admitted = binding == null;
                hub.prefill_stats.updates += 1;
                break :blk x;
            }
        };
        var best: usize = 0;
        var best_used: u64 = std.math.maxInt(u64);
        var same_shape: ?usize = null;
        for (pentries, 0..) |entry, index| if (entry) |cached| {
            if (std.meta.eql(pshape(cached.key), pshape(key))) {
                same_shape = index;
                break;
            }
        };
        for (&pentries, 0..) |*e, i| {
            if (e.* == null) {
                best = i;
                break;
            }
            if (e.*.?.used < best_used) {
                best_used = e.*.?.used;
                best = i;
            }
        }
        if (same_shape) |index| best = index;
        if (pentries[best]) |*old| {
            dropPrefill(old);
            pentries[best] = null;
        }
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try gbudget.uploadResident(&exec, p.stream);
        try memory.installed(admission);
        pentries[best] = .{ .key = key, .exec = exec, .used = gcache.clock, .admission = admission, .stream = p.stream };
        setPrefillBinding(&pentries[best].?, binding, if (binding != null) binding_admission else null);
        kept = binding != null;
        binding_admitted = binding == null;
        admitted = false;
        break :blk &pentries[best].?;
    };
    hub.prefill_stats.captures += 1;
    hub.prefill_stats.capture_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - t0);
    try at.exec.launchOn(p.stream);
}

/// A failed capture: closed and dropped; prefill graphs stay off for the process (the chunk's error propagates).
pub fn prefillAbort(p: anytype, how: Prefill) void {
    if (p.seq.paged) |pg| pg.graph_prepared = false;
    if (how != .capturing) return;
    hub.capturing = false;
    pcur = null;
    pcur_binding_request = null;
    defer if (pcur_binding_admission) |ticket| {
        memory.release(ticket) catch unreachable;
        pcur_binding_admission = null;
    };
    defer if (pcur_admission) |ticket| {
        memory.release(ticket) catch unreachable;
        pcur_admission = null;
    };
    if (cuda.graph.endCapture(p.stream)) |g| {
        var d = g;
        d.deinit();
    } else |_| {}
    std.log.warn("flashnext exl3 graph: a prefill layer could not be captured; prefill and decode graphs are off", .{});
    gcache.broken = true;
}
