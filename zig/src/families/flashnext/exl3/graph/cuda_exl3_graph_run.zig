//! One rank's EXL3 decode step: replay the graph cached under its key, or capture it after enough sightings.

const std = @import("std");
const cuda = @import("cuda");
const wire = @import("../../cuda_rank_wire.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const hub = @import("cuda_exl3_graph.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");
const grebind = @import("cuda_exl3_graph_rebind.zig");
const glog = @import("cuda_exl3_graph_log.zig");
const gfall = @import("cuda_exl3_graph_fallback.zig");
const gtogether = @import("cuda_exl3_graph_together.zig");
const Key = gkey.Key;
const Entry = gcache.Entry;
const Pending = gpending.Pending;

/// The key's capture or instantiation failed: it backs off, and this round runs eagerly (nothing else is affected).
fn fellBack(key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    gfall.record(gfall.hashOf(1, key), gcache.clock);
    hub.stats.eager_dplan += 1;
    glog.note(key, .eager);
    pass.prepared = true;
    return pass.chunk(ids, pos0, null);
}

pub fn run(key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    gcache.clock += 1;
    if (pass.m.collective) |c| return gtogether.runTogether(c, key, pass, ids, pos0);
    if (gcache.find(key)) |e| {
        e.used = gcache.clock;
        if (!gcache.settle(e)) {
            if (e.pending == null) {
                // the instantiation failed: forget the key, back it off, run this round eagerly
                gcache.forget(e);
                return fellBack(key, pass, ids, pos0);
            }
            hub.stats.eager_dplan += 1;
            glog.note(key, .eager);
            pass.prepared = true;
            return pass.chunk(ids, pos0, null);
        }
        hub.stats.replays += 1;
        glog.note(key, .replay);
        return e.exec.?.launchOn(pass.stream);
    }
    if (try grebind.tryRebind(key, pass)) return;
    if (gcache.sighting(key) < gcache.afterKey(key) or gfall.blocked(gfall.hashOf(1, key), gcache.clock)) {
        hub.stats.eager_dplan += 1;
        glog.note(key, .eager);
        pass.prepared = true;
        return pass.chunk(ids, pos0, null);
    }
    const t0 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    try pass.stream.synchronize();
    const admission = gbudget.reserveMemory(pass.stream, .decode, 0) catch |err| {
        if (gfall.admissionDenied(err)) return fellBack(key, pass, ids, pos0);
        return err;
    };
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    var graph = gfall.captureOnce(pass, ids, pos0) orelse return fellBack(key, pass, ids, pos0);
    var kept = false;
    defer if (!kept) graph.deinit();
    const t1 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    var how: []const u8 = "instantiated";
    const tp = pass.m.collective != null;
    const at: *Entry = blk: {
        // a same-shape graph is updated in place only when it is stale (its key not used for a while, e.g. a finished
        var free_slot = false;
        for (&gcache.entries) |*e| free_slot = free_slot or e.* == null;
        if (grebind.capturedBinding(key, pass, graph) == null) {
            for (&gcache.entries) |*e| if (e.*) |*x| if (!x.poisoned and x.exec != null and x.pending == null and gkey.sameShape(x.key, key) and (!free_slot or gcache.clock - x.used > gcache.stale_after)) {
                // every earlier launch finished (each step ends on the host), so the old parameters are free to go
                const r = try x.exec.?.update(graph);
                if (r == .success) {
                    if (x.binding) |*binding| binding.invalidate();
                    x.binding = null;
                    try gbudget.uploadResident(&x.exec.?, pass.stream);
                    x.key = key;
                    x.used = gcache.clock;
                    if (x.template) |*old| old.deinit();
                    x.template = null;
                    if (tp) {
                        x.template = graph;
                        kept = true;
                    }
                    hub.stats.updates += 1;
                    glog.note(key, .updated);
                    how = "updated";
                    break :blk x;
                }
            };
        }
        if (gpending.asyncInstantiate()) if (gpending.ctxApi()) |api| {
            var ctx: ?*anyopaque = null;
            if (api.get(&ctx) == 0) {
                const sl = gcache.slot();
                const pd = try std.heap.c_allocator.create(Pending);
                const binding = grebind.capturedBinding(key, pass, graph);
                pd.* = .{ .graph = graph, .ctx = ctx, .keep_template = tp or binding != null, .started_ns = wire.now() };
                kept = true;
                pd.thread = std.Thread.spawn(.{}, Pending.main, .{pd}) catch |err| {
                    kept = false;
                    std.heap.c_allocator.destroy(pd);
                    return err;
                };
                sl.* = .{ .key = key, .used = gcache.clock, .pending = pd, .admission = admission, .stream = pass.stream, .binding = binding };
                admitted = false;
                glog.note(key, .instantiated);
                hub.stats.captures += 1;
                // this step runs eagerly while the graph instantiates
                hub.capturing = false;
                pass.prepared = true;
                return pass.chunk(ids, pos0, null);
            }
        };
        const sl = gcache.slot();
        var exec = (if (gfall.take(&gfall.inject_instantiate_failures)) error.CudaFailed else graph.instantiate()) catch return fellBack(key, pass, ids, pos0);
        errdefer exec.deinit();
        try gbudget.uploadResident(&exec, pass.stream);
        try memory.installed(admission);
        sl.* = .{ .key = key, .exec = exec, .used = gcache.clock, .admission = admission, .stream = pass.stream };
        sl.*.?.binding = grebind.capturedBinding(key, pass, graph);
        if (sl.*.?.binding != null) {
            sl.*.?.template = graph;
            kept = true;
        }
        admitted = false;
        glog.note(key, .instantiated);
        break :blk &sl.*.?;
    };
    hub.stats.captures += 1;
    const t2 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    hub.stats.capture_ns += @intCast(t2 - t0);
    std.log.info("flashnext exl3 graph: captured a {s} of {d} rows in {d:.1} ms + {s} in {d:.1} ms ({d} captures, {d} updates, {d} replays, {d} eager, {d} declined steps)", .{ if (key.kind == 1) "decode round" else "window", key.rows, @as(f64, @floatFromInt(t1 - t0)) / 1e6, how, @as(f64, @floatFromInt(t2 - t1)) / 1e6, hub.stats.captures, hub.stats.updates, hub.stats.replays, hub.stats.eager_dplan, hub.stats.declined });
    try at.exec.?.launchOn(pass.stream);
}
