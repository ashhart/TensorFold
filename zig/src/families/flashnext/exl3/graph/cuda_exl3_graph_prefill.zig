//! Prefill graphs of the EXL3 engine: one layer's sub-blocks of a deep chunk captured and replayed.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const hub = @import("cuda_exl3_graph.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gfall = @import("cuda_exl3_graph_fallback.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");

// ---- prefill: one layer's sub-blocks of a deep chunk.

pub const Prefill = enum { eager, replayed, capturing };

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
    owner: u64 = 0,
    layer: usize,
    rows: usize,
    sub: usize,
    /// The chunk's first position (attention layers only; 0 for DeltaNet layers).
    pos0: usize,
    deep: u64,
    base: u64,
    base_cap: usize,
    seq: [5]u64,
    pages: [5]u64 = @splat(0),
    tier: [5]u64 = @splat(0),
    wait_count: usize,
    wait_key: u64,
    scores: u64,
    tiles: bool,
    positions: u64 = 0,
    stream: u64 = 0,
    gdn: ?PGdn = null,
};

pub const PEntry = struct { key: PKey, exec: cuda.graph.Exec, used: u64, admission: ?memory.Ticket = null, stream: ?cuda.Stream = null };

pub fn dropPrefill(entry: *PEntry) void {
    if (entry.stream) |stream| stream.synchronize() catch @panic("prefill retirement synchronization failed");
    entry.exec.deinit();
    if (entry.admission) |ticket| memory.release(ticket) catch unreachable;
}
pub const pcache_n = 112;
pub var pentries: [pcache_n]?PEntry = @splat(null);
pub var pseen: [pcache_n * 2]?struct { key: PKey, n: u32 } = @splat(null);
pub var pseen_at: usize = 0;
pub var pcur: ?PKey = null;
pub var pcur_admission: ?memory.Ticket = null;

pub fn prefillAdmissionMiss() Prefill {
    hub.prefill_stats.eager += 1;
    hub.prefill_stats.admission_misses += 1;
    const misses = hub.prefill_stats.admission_misses;
    if (misses == 1 or misses % 256 == 0) std.log.info("flashnext exl3 prefill graph budget admission declines: {d}; retained cache preserved, layer runs eagerly", .{misses});
    return .eager;
}

/// Prefill layers are captured whenever graphs are on (TF_FLASHNEXT_EXL3_GRAPH=1).
pub fn prefillEnabled() bool {
    return hub.enabled();
}

pub fn pshape(k: PKey) PKey {
    var s = k;
    s.pos0 = 0;
    return s;
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

/// The sighting of a layer shape that captures it (a chunk takes every layer's shape twice).
pub fn prefillAfter() u32 {
    return gcache.after();
}

pub fn clearPrefill() void {
    for (&pentries) |*e| if (e.*) |*x| {
        dropPrefill(x);
        e.* = null;
    };
    pseen = @splat(null);
}

pub fn clearRuntime(owner: u64) void {
    for (&pentries) |*entry| if (entry.*) |*held| if (held.key.owner == owner) {
        dropPrefill(held);
        entry.* = null;
    };
    for (&pseen) |*entry| if (entry.*) |held| if (held.key.owner == owner) {
        entry.* = null;
    };
}

/// Before a layer's sub-blocks:
pub fn prefillBegin(p: anytype, linear: bool, layer: usize, rows: usize, sub: usize, pos0: usize, deep: u64, base: anytype, tiles: bool, waits: []const cuda.Event, positions: u64) !Prefill {
    if (!prefillEnabled()) return .eager;
    if (!linear) if (p.seq.sel) |selection| if (!selection.graph_staged) return .eager;
    const q = p.seq;
    var wait_key: u64 = 0;
    for (waits) |ev| {
        const handle = if (ev.handle) |h| @intFromPtr(h) else 0;
        wait_key = std.hash.Wyhash.hash(wait_key, std.mem.asBytes(&handle));
    }
    const key: PKey = .{
        .owner = if (p.m.prefill) |runtime| @intFromPtr(runtime) else 0,
        .layer = layer,
        .rows = rows,
        .sub = sub,
        .pos0 = if (linear) 0 else pos0,
        .deep = deep,
        .base = base.ids,
        .base_cap = base.cap,
        .seq = .{ if (q.conv.len > 0) q.conv[0].ptr else 0, if (q.kc.len > 0) q.kc[0].ptr else 0, q.tail.ptr, q.cap, if (q.state.len > 0) q.state[0].ptr else 0 },
        .pages = @import("cuda_exl3_graph_key.zig").Key.pageOf(q.pg),
        .tier = if (q.sel) |selection| @import("cuda_exl3_graph_key.zig").Key.tierOf(selection, pos0, rows) else @splat(0),
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
        x.used = gcache.clock;
        hub.prefill_stats.replays += 1;
        try x.exec.launchOn(p.stream);
        return .replayed;
    };
    if (psighting(key) < prefillAfter() or gfall.blocked(gfall.hashOf(2, pshape(key)), gcache.clock)) {
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
    try p.stream.synchronize();
    try cuda.graph.beginCapture(p.stream, .thread_local);
    hub.capturing = true;
    pcur = key;
    pcur_admission = admission;
    return .capturing;
}

/// After a layer's sub-blocks ran under `.capturing`: true once the layer ran from its graph, false: run it eagerly.
pub fn prefillEnd(p: anytype, how: Prefill) !bool {
    if (how != .capturing) return true;
    const t0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    hub.capturing = false;
    const key = pcur.?;
    pcur = null;
    const admission = pcur_admission.?;
    pcur_admission = null;
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    var graph = cuda.graph.endCapture(p.stream) catch return unusable(key);
    defer graph.deinit();
    if (gfall.take(&gfall.inject_capture_failures)) return unusable(key);
    const at: *PEntry = blk: {
        for (&pentries) |*e| if (e.*) |*x| if (std.meta.eql(pshape(x.key), pshape(key))) {
            if ((try x.exec.update(graph)) == .success) {
                try gbudget.uploadResident(&x.exec, p.stream);
                x.key = key;
                x.used = gcache.clock;
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
        var exec = (if (gfall.take(&gfall.inject_instantiate_failures)) error.CudaFailed else graph.instantiate()) catch return unusable(key);
        errdefer exec.deinit();
        try gbudget.uploadResident(&exec, p.stream);
        try memory.installed(admission);
        pentries[best] = .{ .key = key, .exec = exec, .used = gcache.clock, .admission = admission, .stream = p.stream };
        admitted = false;
        break :blk &pentries[best].?;
    };
    hub.prefill_stats.captures += 1;
    hub.prefill_stats.capture_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - t0);
    try at.exec.launchOn(p.stream);
    return true;
}

fn unusable(key: PKey) bool {
    gfall.record(gfall.hashOf(2, pshape(key)), gcache.clock);
    hub.prefill_stats.eager += 1;
    return false;
}

/// Runs one layer's sub-blocks (`blocks.run(mode)`); a layer that cannot be captured runs eagerly, its shape backs off.
pub fn runLayer(p: anytype, how: Prefill, blocks: anytype) !void {
    var mode = how;
    while (mode != .replayed) {
        blocks.run(mode) catch |err| {
            if (mode != .capturing) return err;
            prefillAbort(p, mode);
            mode = .eager;
            continue;
        };
        if (try prefillEnd(p, mode)) return;
        mode = .eager;
    }
}

/// A capture that failed while its sub-blocks were enqueued: closed and dropped, the shape backs off.
pub fn prefillAbort(p: anytype, how: Prefill) void {
    if (how != .capturing) return;
    hub.capturing = false;
    if (pcur) |key| {
        gfall.record(gfall.hashOf(2, pshape(key)), gcache.clock);
        hub.prefill_stats.eager += 1;
    }
    pcur = null;
    defer if (pcur_admission) |ticket| {
        memory.release(ticket) catch unreachable;
        pcur_admission = null;
    };
    if (cuda.graph.endCapture(p.stream)) |g| {
        var d = g;
        d.deinit();
    } else |_| {}
    std.log.warn("flashnext exl3 graph: a prefill layer could not be captured; it runs eagerly and is tried again later", .{});
}
