//! A tensor-parallel EXL3 decode step: the ranks agree on replay, capture, update and retirement together.

const std = @import("std");
const cuda = @import("cuda");
const collective = @import("../../cuda_collective.zig");
const wire = @import("../../cuda_rank_wire.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const hub = @import("cuda_exl3_graph.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");
const grebind = @import("cuda_exl3_graph_rebind.zig");
const glog = @import("cuda_exl3_graph_log.zig");
const Key = gkey.Key;
const Entry = gcache.Entry;

pub fn graphVote(entry: ?*Entry, timeout_ms: u64) collective.GraphVote {
    const x = entry orelse return .{ .state = .capture };
    if (gcache.settle(x)) return .{ .state = .ready, .cohort = x.cohort };
    if (x.pending) |pd| {
        const elapsed = wire.now() -| pd.started_ns;
        return .{ .state = if (elapsed >= timeout_ms *| std.time.ns_per_ms) .timed_out else .pending, .cohort = x.cohort };
    }
    return .{ .state = .failed, .cohort = x.cohort };
}

pub fn runTogether(c: *collective.Collective, key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    errdefer c.failGraph();
    const signature = gkey.graphSignature(key, pos0);
    var entry = gcache.find(key);
    if (entry) |x| x.used = gcache.clock;
    const vote = if (entry != null) graphVote(entry, c.graphTimeoutMs()) else collective.GraphVote{
        .state = if (grebind.stateRequest(key, pass)) |request|
            if (grebind.rebindCandidate(key, request, true) != null) .capture else if (gcache.sighting(key) < gcache.after(key)) .eager else .capture
        else if (gcache.sighting(key) < gcache.after(key)) .eager else .capture,
    };
    const decision = try c.graphAgree(signature, vote);
    switch (decision.choice) {
        .eager => {
            hub.stats.eager_dplan += 1;
            glog.note(key, .eager);
            pass.prepared = true;
            return pass.chunk(ids, pos0, null);
        },
        .replay => {
            hub.stats.replays += 1;
            glog.note(key, .replay);
            return entry.?.exec.?.launchOn(pass.stream);
        },
        .capture => {},
    }
    if (try grebind.tryRebindTogether(c, signature, key, pass)) return;
    const started = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    try pass.stream.synchronize();
    const admission = try gbudget.reserveTogether(c, pass.stream, signature, decision.cohort);
    entry = gcache.find(key);
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    try cuda.graph.beginCapture(pass.stream, .thread_local);
    pass.prepared = true;
    hub.capturing = true;
    pass.chunk(ids, pos0, null) catch |err| {
        hub.capturing = false;
        if (cuda.graph.endCapture(pass.stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    hub.capturing = false;
    var graph = try cuda.graph.endCapture(pass.stream);
    var kept = false;
    defer if (!kept) graph.deinit();
    var updated = false;
    var candidate: ?*Entry = null;
    if (gcache.tpUpdate() and grebind.capturedBinding(key, pass, graph) == null) for (&gcache.entries) |*maybe| if (maybe.*) |*current| {
        if (!current.poisoned and current.cohort != 0 and current.exec != null and current.pending == null and gkey.sameShape(current.key, key) and
            (std.meta.eql(current.key, key) or gcache.clock - current.used > gcache.stale_after))
        {
            if (candidate == null or current.used < candidate.?.used) candidate = current;
        }
    };
    const update_agreement = try c.graphAgree(signature ^ 0x757064617465, if (candidate) |current|
        .{ .state = .ready, .cohort = current.cohort }
    else
        .{ .state = .capture });
    if (update_agreement.choice == .replay) {
        const current = candidate.?;
        const result = try current.exec.?.update(graph);
        if (result == .success) {
            if (current.binding) |*binding| binding.invalidate();
            current.binding = null;
            if (current.template) |*old| old.deinit();
            current.template = graph;
            kept = true;
            updated = true;
        }
        const result_agreement = try c.graphAgree(signature ^ 0x75706461746564, .{ .state = if (updated) .ready else .capture, .cohort = decision.cohort });
        if (result_agreement.choice == .replay) {
            if (current.admission) |old| try memory.release(old);
            current.admission = admission;
            admitted = false;
            try gbudget.uploadResident(&current.exec.?, pass.stream);
            try memory.installed(admission);
            current.key = key;
            current.used = gcache.clock;
            current.cohort = decision.cohort;
            current.stream = pass.stream;
            entry = current;
        } else {
            if (updated) {
                current.template = null;
                kept = false;
            }
            gcache.drop(current);
            for (&gcache.entries) |*slot_entry| if (slot_entry.*) |*cached| if (cached == current) {
                slot_entry.* = null;
            };
            entry = gcache.find(key);
            updated = false;
            try gbudget.retirementBarrier(c, signature ^ 0x72656275696c64, decision.cohort);
        }
    }
    if (!updated) {
        const target = if (entry) |old| blk: {
            gcache.drop(old);
            for (&gcache.entries) |*slot_entry| if (slot_entry.*) |*x| if (x == old) {
                slot_entry.* = null;
                break :blk slot_entry;
            };
            unreachable;
        } else blk: {
            for (&gcache.entries) |*slot_entry| if (slot_entry.* == null) break :blk slot_entry;
            return error.GraphBudgetBusy;
        };
        if (try gpending.startInstantiate(graph, true)) |pending| {
            target.* = .{ .key = key, .used = gcache.clock, .pending = pending, .cohort = decision.cohort, .admission = admission, .stream = pass.stream, .binding = grebind.capturedBinding(key, pass, graph) };
        } else {
            var exec = try graph.instantiate();
            errdefer exec.deinit();
            try gbudget.uploadResident(&exec, pass.stream);
            try memory.installed(admission);
            target.* = .{ .key = key, .exec = exec, .used = gcache.clock, .template = graph, .cohort = decision.cohort, .admission = admission, .stream = pass.stream, .binding = grebind.capturedBinding(key, pass, graph) };
        }
        admitted = false;
        kept = true;
        entry = &target.*.?;
    }
    hub.stats.captures += 1;
    if (updated) hub.stats.updates += 1;
    glog.note(key, if (updated) .updated else .instantiated);
    hub.stats.capture_ns += @intCast(std.Io.Clock.awake.now(pass.io).toNanoseconds() - started);
    const installed = try c.graphAgree(signature ^ 0x696e7374616c6c, graphVote(entry, c.graphTimeoutMs()));
    switch (installed.choice) {
        .replay => try entry.?.exec.?.launchOn(pass.stream),
        .eager => {
            hub.stats.eager_dplan += 1;
            pass.prepared = true;
            try pass.chunk(ids, pos0, null);
        },
        .capture => return error.GraphRendezvousMismatch,
    }
}
