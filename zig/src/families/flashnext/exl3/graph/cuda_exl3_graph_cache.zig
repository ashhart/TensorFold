//! The cache of captured EXL3 decode graphs: entries, LRU slots, sightings and the retirement of a sequence's graphs.

const std = @import("std");
const cuda = @import("cuda");
const profile_module = @import("exl3_graph_profile.zig");
const state_binding = @import("../kv/exl3_state_binding.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const dplan = @import("../dplan/cuda_exl3_dplan.zig");
const hub = @import("cuda_exl3_graph.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");
const glog = @import("cuda_exl3_graph_log.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const Key = gkey.Key;
const Pending = gpending.Pending;

/// `template`: the graph an update took this exec's parameters from, kept while the exec uses them.
pub const Entry = struct { key: Key, exec: ?cuda.graph.Exec = null, used: u64, template: ?cuda.graph.Graph = null, pending: ?*Pending = null, cohort: u64 = 0, admission: ?memory.Ticket = null, stream: ?cuda.Stream = null, binding: ?state_binding.Binding = null, poisoned: bool = false };

/// A pending entry whose instantiation finished becomes a usable graph (or is dropped if it failed); true if usable.
pub fn settle(x: *Entry) bool {
    if (x.poisoned) return false;
    const pd = x.pending orelse return x.exec != null;
    if (!pd.done.load(.acquire)) return false;
    pd.thread.join();
    x.pending = null;
    defer std.heap.c_allocator.destroy(pd);
    if (pd.exec) |e| {
        var exec = e;
        if (x.stream) |stream| gbudget.uploadResident(&exec, stream) catch {
            exec.deinit();
            pd.graph.deinit();
            return false;
        };
        x.exec = exec;
        if (pd.keep_template) x.template = pd.graph else pd.graph.deinit();
        if (x.admission) |ticket| memory.installed(ticket) catch unreachable;
        return true;
    }
    pd.graph.deinit();
    return false;
}

pub fn drop(x: *Entry) void {
    if (x.binding) |*binding| binding.invalidate();
    if (x.stream) |stream| stream.synchronize() catch @panic("graph retirement synchronization failed");
    defer if (x.admission) |ticket| {
        memory.release(ticket) catch unreachable;
        x.admission = null;
    };
    if (x.pending) |pd| {
        pd.thread.join();
        if (pd.exec) |*e| e.deinit();
        pd.graph.deinit();
        std.heap.c_allocator.destroy(pd);
        return;
    }
    if (x.exec) |*e| e.deinit();
    if (x.template) |*g| g.deinit();
}

/// Steps after which a graph nobody replayed counts as stale (another stream's, or a finished one's).
pub const stale_after: u64 = 512;

/// Graphs kept (LRU): a round and up to `max_rows` window sizes per stream, at two seams.
pub const cache_n = 128;
pub var entries: [cache_n]?Entry = @splat(null);
pub var clock: u64 = 0;
/// A capture that failed once turns graphs off for the process (the steps keep the device plan).
pub var broken: bool = false;

pub fn clear() void {
    gprefill.clearPrefill();
    for (&entries) |*e| if (e.*) |*x| {
        drop(x);
        e.* = null;
    };
    seen = @splat(null);
}

pub fn references(key: Key, tail: u64) bool {
    if (tail == 0) return false;
    for (key.seq) |seq| if (seq[2] == tail) return true;
    return false;
}

pub fn releaseSeqImpl(seq: anytype, keep_prefill: bool) void {
    const tail = seq.tail.ptr;
    if (tail == 0) return;
    for (&entries) |*entry| if (entry.*) |*x| if (references(x.key, tail)) {
        drop(x);
        entry.* = null;
    };
    for (&seen) |*entry| if (entry.*) |x| if (references(x.key, tail)) {
        entry.* = null;
    };
    for (&glog.last_key) |*kinds| for (kinds) |*key| if (key.*) |k| if (references(k, tail)) {
        key.* = null;
    };
    if (!keep_prefill) {
        for (&gprefill.pentries) |*entry| if (entry.*) |*x| if (x.key.seq[2] == tail) {
            gprefill.dropPrefill(x);
            entry.* = null;
        };
        for (&gprefill.pseen) |*entry| if (entry.*) |*x| if (x.key.seq[2] == tail) {
            entry.* = null;
        };
    }
}

pub fn releaseSeq(seq: anytype) void {
    releaseSeqImpl(seq, false);
}

pub fn releaseDecodeSeq(seq: anytype) void {
    releaseSeqImpl(seq, true);
}

/// The engine's deinit: graphs and the device planner go.
pub fn releaseFor(owner: *const anyopaque) void {
    clear();
    dplan.releaseFor(owner);
}

pub fn find(key: Key) ?*Entry {
    for (&entries) |*e| if (e.*) |*x| if (std.meta.eql(x.key, key)) return x;
    return null;
}

pub fn slot() *?Entry {
    var best: usize = 0;
    var best_used: u64 = std.math.maxInt(u64);
    for (&entries, 0..) |*e, i| {
        if (e.* == null) return e;
        if (e.*.?.used < best_used) {
            best_used = e.*.?.used;
            best = i;
        }
    }
    if (hub.profileEnabled()) hub.profile.evict(profile_module.Sample.fromKey(entries[best].?.key));
    drop(&entries[best].?);
    entries[best] = null;
    return &entries[best];
}

/// Keys seen without a graph, and how often (a key is captured at its `after`-th sighting; one-off shapes, e.g. a
pub const Seen = struct { key: Key, n: u32, used: u64 };
pub var seen: [256]?Seen = @splat(null);

pub fn sighting(key: Key) u32 {
    var oldest: usize = 0;
    var oldest_used: u64 = std.math.maxInt(u64);
    for (&seen, 0..) |*e, i| {
        if (e.*) |*x| {
            if (std.meta.eql(x.key, key)) {
                x.n += 1;
                x.used = clock;
                return x.n;
            }
            if (x.used < oldest_used) {
                oldest_used = x.used;
                oldest = i;
            }
        } else {
            oldest = i;
            oldest_used = 0;
        }
    }
    seen[oldest] = .{ .key = key, .n = 1, .used = clock };
    return 1;
}

/// TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE=0: tp graphs are always instantiated; default: updated in place like tp=1's.
pub fn tpUpdate() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_FLASHNEXT_EXL3_GRAPH_DECODE_AFTER: the sighting of a key that captures it, for rounds and windows alike.
pub fn after(key: Key) u32 {
    if (std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_DECODE_AFTER")) |v| return @max(1, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2);
    return if (key.kind == 2) 12 else 2;
}
