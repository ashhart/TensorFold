//! Rebinding a cached window graph to another sequence's DeltaNet state (TF_FLASHNEXT_EXL3_GRAPH_STATE_REBIND=1).

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const collective = @import("../../cuda_collective.zig");
const state_binding = @import("../kv/exl3_state_binding.zig");
const state_driver = @import("../kv/cuda_exl3_state_binding.zig");
const hub = @import("cuda_exl3_graph.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const glog = @import("cuda_exl3_graph_log.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const Key = gkey.Key;
const Entry = gcache.Entry;

pub fn stateRebindEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_STATE_REBIND") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn stateRequest(key: Key, pass: anytype) ?state_binding.Request {
    if (!stateRebindEnabled()) return null;
    return windowStateRequest(key, pass);
}

pub fn windowStateRequest(key: Key, pass: anytype) ?state_binding.Request {
    const Pass = @typeInfo(@TypeOf(pass)).pointer.child;
    if (comptime !@hasField(Pass, "lanes")) return null;
    if (key.kind != 2 or !state_binding.supportsWindow(key.rows) or pass.lanes.len != 0) return null;
    const gdn = @import("../../cuda_gdn.zig");
    var request: state_binding.Request = .{
        .width = key.rows,
        .chain = @intFromPtr(pass.k.chain.handle),
        .chain_to = @intFromPtr(pass.k.chain_to.handle),
        .geometry = .{ .grid = .{ gdn.nv, gdn.dv / 16, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = pass.s.gq, .gk = pass.s.gk, .gv = pass.s.gv, .ggb = pass.s.ggb, .rows = @intCast(key.rows), .ys = pass.s.ys },
    };
    for (pass.m.layers, 0..) |layer, index| {
        if (!layer.linear) continue;
        const linear = wt.linearIndex(index);
        if (request.count == state_binding.max_layers or linear >= pass.seq.state.len) return null;
        request.layers[request.count] = .{
            .layer = index,
            .state_in = pass.seq.state[linear].ptr,
            .state_out = if (pass.hooks.keep) |keep| keep.layers[index].state_out else 0,
        };
        if (request.layers[request.count].state_out == 0) return null;
        request.count += 1;
    }
    request.validate() catch return null;
    return request;
}

pub fn capturedBinding(key: Key, pass: anytype, graph: cuda.graph.Graph) ?state_binding.Binding {
    const request = stateRequest(key, pass) orelse return null;
    if (state_driver.nodeType(graph.d) == null) return null;
    return state_binding.Binding.init(@intFromPtr(graph.handle), request) catch null;
}

pub fn rebindCandidate(key: Key, request: state_binding.Request, tp: bool) ?*Entry {
    var candidate: ?*Entry = null;
    for (&gcache.entries) |*maybe| if (maybe.*) |*entry| {
        if (entry.poisoned or entry.pending != null or entry.exec == null or entry.template == null or (tp and entry.cohort == 0)) continue;
        const binding = entry.binding orelse continue;
        if (binding.poisoned or !std.meta.eql(state_binding.normalized(entry.key), state_binding.normalized(key)) or !binding.request.compatible(request)) continue;
        if (candidate == null or entry.cohort < candidate.?.cohort or (entry.cohort == candidate.?.cohort and entry.used > candidate.?.used)) candidate = entry;
    };
    return candidate;
}

pub fn prepareBinding(entry: *Entry) !?state_driver.Backend {
    const backend = state_driver.Backend.init(entry.template.?, entry.exec.?) orelse return null;
    entry.binding.?.load(backend) catch |err| switch (err) {
        error.UnsupportedBinding => {
            entry.binding.?.invalidate();
            entry.binding = null;
            return null;
        },
        else => return err,
    };
    return backend;
}

pub fn tryRebind(key: Key, pass: anytype) !bool {
    const request = stateRequest(key, pass) orelse return false;
    const entry = rebindCandidate(key, request, false) orelse {
        hub.stats.rebind_declined += 1;
        return false;
    };
    const prepared = prepareBinding(entry) catch |err| {
        hub.stats.rebind_failures += 1;
        return err;
    };
    const backend = prepared orelse {
        hub.stats.rebind_declined += 1;
        return false;
    };
    hub.stats.rebind_attempts += 1;
    entry.binding.?.apply(backend, request) catch |err| {
        entry.poisoned = true;
        hub.stats.rebind_failures += 1;
        return err;
    };
    entry.key = key;
    entry.used = gcache.clock;
    hub.stats.rebinds += 1;
    hub.stats.replays += 1;
    glog.note(key, .replay);
    try entry.exec.?.launchOn(pass.stream);
    return true;
}

pub fn tryRebindTogether(c: *collective.Collective, signature: u64, key: Key, pass: anytype) !bool {
    if (!stateRebindEnabled() or key.kind != 2 or !state_binding.supportsWindow(key.rows)) return false;
    const request = stateRequest(key, pass);
    const candidate = if (request) |target| rebindCandidate(key, target, true) else null;
    var backend: ?state_driver.Backend = null;
    var failed = false;
    if (candidate) |entry| backend = prepareBinding(entry) catch blk: {
        hub.stats.rebind_failures += 1;
        failed = true;
        break :blk null;
    };
    const selected = try c.graphAgree(signature ^ 0x737461746573656c, .{
        .state = if (failed) .failed else if (backend != null) .ready else .capture,
        .cohort = if (candidate) |entry| entry.cohort else 0,
    });
    if (selected.choice != .replay) {
        hub.stats.rebind_declined += 1;
        return false;
    }
    const entry = candidate.?;
    hub.stats.rebind_attempts += 1;
    entry.binding.?.applyTogether(backend.?, request.?, c, signature ^ 0x7374617465646f6e, selected.cohort) catch |err| {
        entry.poisoned = true;
        hub.stats.rebind_failures += 1;
        return err;
    };
    entry.key = key;
    entry.used = gcache.clock;
    hub.stats.rebinds += 1;
    hub.stats.replays += 1;
    glog.note(key, .replay);
    try entry.exec.?.launchOn(pass.stream);
    return true;
}
