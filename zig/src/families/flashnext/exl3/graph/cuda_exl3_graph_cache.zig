//! The cache of captured EXL3 decode graphs: entries, LRU slots, sightings and the retirement of a sequence's graphs.

const std = @import("std");
const cuda = @import("cuda");
const state_binding = @import("../kv/exl3_state_binding.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
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

fn resourceSeq(seq: *const [5]u64, tail: u64, state: u64) bool {
    return (tail != 0 and seq[2] == tail) or (state != 0 and seq[4] == state);
}

fn resourceKey(key: *const Key, tail: u64, state: u64) bool {
    if (state != 0 and key.keep_state == state) return true;
    for (&key.seq) |*seq| if (resourceSeq(seq, tail, state)) return true;
    return false;
}

fn draftKey(key: *const Key, draft: anytype) bool {
    return (draft.logits.ptr != 0 and key.output == draft.logits.ptr) or
        (draft.streams.ptr != 0 and key.streams == draft.streams.ptr) or
        resourceKey(key, draft.kc.ptr, draft.spare[0].ptr);
}

fn onDriver(held: *const Entry, driver: *const cuda.Driver) bool {
    if (held.stream) |stream| return stream.d == driver;
    if (held.exec) |exec| return exec.d == driver;
    if (held.template) |template| return template.d == driver;
    if (held.pending) |pending| return pending.graph.d == driver;
    return false;
}

/// Drafts exchange whole state sets with sequences; retire every graph that still borrows their allocations.
pub fn releaseDraft(draft: anytype) void {
    for (&entries) |*entry| if (entry.*) |*held| {
        if (!draftKey(&held.key, draft) or !onDriver(held, draft.dlogits.d)) continue;
        drop(held);
        entry.* = null;
    };
    for (&gprefill.pentries) |*entry| if (entry.*) |*held| {
        if (held.exec.d != draft.dlogits.d or !resourceSeq(&held.key.seq, draft.kc.ptr, draft.spare[0].ptr)) continue;
        gprefill.dropPrefill(held);
        entry.* = null;
    };
}

pub fn releaseSeqImpl(seq: anytype, keep_prefill: bool) void {
    const tail = seq.tail.ptr;
    if (tail == 0) return;
    const state = if (seq.state.len > 0) seq.state[0].ptr else 0;
    for (&entries) |*entry| if (entry.*) |*x| {
        if (!resourceKey(&x.key, tail, state) or !onDriver(x, seq.tail.d)) continue;
        drop(x);
        entry.* = null;
    };
    for (&seen) |*entry| if (entry.*) |x| if (references(x.key, tail)) {
        entry.* = null;
    };
    if (!keep_prefill) {
        for (&gprefill.pentries) |*entry| if (entry.*) |*x| if (x.exec.d == seq.tail.d and resourceSeq(&x.key.seq, tail, state)) {
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

/// The engine's deinit: the graphs go (the engine releases its own planner state).
pub fn releaseFor(owner: *@import("../../cuda_prompt.zig").Engine) void {
    const id = @intFromPtr(owner);
    for (&entries) |*entry| if (entry.*) |*held| if (held.key.owner == id) {
        drop(held);
        entry.* = null;
    };
    for (&seen) |*entry| if (entry.*) |held| if (held.key.owner == id) {
        entry.* = null;
    };
    @import("cuda_exl3_graph_fallback.zig").clearFor(id);
    gprefill.clearRuntime(@intFromPtr(&owner.exl3_prefill));
}

/// Drops `x` and empties its slot.
pub fn forget(x: *Entry) void {
    for (&entries) |*en| if (en.*) |*y| if (y == x) {
        drop(y);
        en.* = null;
        return;
    };
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

/// The sighting of a key that captures it: a one-off shape never pays for a capture.
pub const capture_after: u32 = 2;
/// A test's override of `capture_after` (1 captures at once); not a user switch.
pub var capture_after_override: ?u32 = null;

pub fn after() u32 {
    return capture_after_override orelse capture_after;
}

/// A window (kind 2) is keyed by rows, state set and seams, so each is seen a few times and a capture costs more.
pub const window_capture_after: u32 = 12;

pub fn afterKey(key: Key) u32 {
    return capture_after_override orelse if (key.kind == 2) window_capture_after else capture_after;
}
