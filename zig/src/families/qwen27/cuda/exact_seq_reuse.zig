//! Private target-only allocation reuse after proven full-N cache restore; reset and draft lifecycle are unchanged.
const std = @import("std");
const st = @import("state.zig");
const dr = @import("draft.zig");
const lanes = @import("lanes");
const Cuda = @import("lanes.zig").Cuda;
const draft_reuse = @import("exact_draft_reuse.zig");
pub const policy = @import("seq_pool.zig");
pub const Proofs = @import("restore_proof.zig").Proofs;
pub const Storage = policy.Pool(st.Seq);
pub const reset_skip = @import("reset_skip.zig");
pub const Opened = struct { seq: *st.Seq, deferred_reset: bool = false, pooled: bool = false };

fn destroy(x: *Cuda, seq: *st.Seq) void {
    seq.deinit();
    x.gpa.destroy(seq);
}

pub fn clear(x: *Cuda) void {
    while (x.seq_pool.evict()) |seq| destroy(x, seq);
}

pub fn fail(x: *Cuda) void {
    draft_reuse.fail(x);
    if (!x.seq_pool.stats.enabled) return;
    x.seq_pool.fail();
    clear(x);
}

pub fn shutdown(x: *Cuda) void {
    if (x.reset_skip.enabled) std.debug.print("PR608_EXACT_CACHE_RESET_SKIP {f}\n", .{std.json.fmt(x.reset_skip, .{})});
    if (x.seq_pool.stats.enabled) std.debug.print("PR608_EXACT_CACHE_SEQ_POOL {f}\n", .{std.json.fmt(x.seq_pool.stats, .{})});
    clear(x);
}

/// A new stream's state, sized for its prompt and reply.
pub fn open(x: *Cuda, s: *lanes.Stream) !*st.Seq {
    errdefer fail(x);
    try x.settle();
    const ids = s.prompt();
    if (ids.len == 0 or x.seqs.contains(s)) return error.InvalidQwenPrompt;
    const capacity = @min(x.e.context, ids.len + s.max_new + st.max_rows);
    if (ids.len >= capacity) return error.PromptTooLong;
    try draft_reuse.beforeOpen(x, s, capacity);
    return obtain(x, s, capacity);
}

/// A peek permits acquisition only; a separate successful lookup records restore provenance.
pub fn obtain(x: *Cuda, s: *lanes.Stream, capacity: usize) !*st.Seq {
    return (try obtainState(x, s, capacity, false)).seq;
}

fn obtainState(x: *Cuda, s: *lanes.Stream, capacity: usize, defer_reset: bool) !Opened {
    errdefer fail(x);
    if (x.seq_pool.active()) {
        if (x.seqs.count() >= x.seq_pool.limit) return error.SeqReuseOwnershipLimit;
        if (@import("exact_eligible.zig").text(x.e.kv8, s) and s.reuse.hook == null and
            x.exact_cache.canRestore(s, x.drafter != null and s.drafts) and capacity <= x.e.context)
        {
            if (x.seq_pool.take(capacity)) |seq| {
                errdefer destroy(x, seq);
                if (seq.capacity != capacity) return error.SeqReuseCapacityMismatch;
                if (defer_reset) {
                    seq.pos = 0;
                    seq.rope_delta = 0;
                    seq.media = null;
                } else try seq.reset(x.e.ops(), x.e.w.g);
                try x.seqs.put(x.gpa, s, seq);
                if (defer_reset) x.reset_skip.deferred_pool += 1;
                return .{ .seq = seq, .deferred_reset = defer_reset, .pooled = true };
            }
        }
        // Every fresh creation drains retained ownership, including a prompt-key miss with room.
        x.seq_pool.beforeCreate(x, destroy);
        if (!x.seq_pool.hasRoom(x.seqs.count())) return error.SeqReuseOwnershipLimit;
    }
    const seq = try x.gpa.create(st.Seq);
    errdefer x.gpa.destroy(seq);
    x.seq_pool.stats.create_attempts += 1;
    seq.* = (if (defer_reset) x.e.sequenceForRestore(capacity) else x.e.sequence(capacity)) catch |err| return if (err == error.OutOfMemory) err else error.OutOfDeviceMemory;
    x.seq_pool.stats.driver_creations += 1;
    errdefer seq.deinit();
    try x.seqs.put(x.gpa, s, seq);
    if (defer_reset) x.reset_skip.deferred_fresh += 1;
    return .{ .seq = seq, .deferred_reset = defer_reset };
}

/// Private admission only; public Cuda.open always initializes the target.
pub fn openPrefill(x: *Cuda, s: *lanes.Stream) !Opened {
    if (!x.reset_skip.enabled) return .{ .seq = try x.open(s) };
    errdefer fail(x);
    try x.settle();
    const ids = s.prompt();
    if (ids.len == 0 or x.seqs.contains(s)) return error.InvalidQwenPrompt;
    const capacity = @min(x.e.context, ids.len + s.max_new + st.max_rows);
    if (ids.len >= capacity) return error.PromptTooLong;
    try draft_reuse.beforeOpen(x, s, capacity);
    const hint = capacity <= reset_skip.max_capacity and @import("exact_eligible.zig").text(x.e.kv8, s) and s.reuse.hook == null and
        x.exact_cache.canRestore(s, x.drafter != null and s.drafts);
    return obtainState(x, s, capacity, hint);
}

const Resolve = struct {
    x: *Cuda,
    stream: *lanes.Stream,
    seq: *st.Seq,
    ctx: ?*dr.Context,
    pub fn cancelled(a: Resolve) bool {
        return a.stream.isCancelled();
    }
    pub fn lookup(a: Resolve) !bool {
        return @import("exact_seq_reuse.zig").lookup(a.x, a.stream, a.seq, a.ctx);
    }
    pub fn reset(a: Resolve) !void {
        try a.seq.reset(a.x.e.ops(), a.x.e.w.g);
    }
};

/// No target call may precede resolution; restore errors abort, never fall back from a partial copy.
pub fn lookupPrefill(x: *Cuda, s: *lanes.Stream, opened: *Opened, ctx: ?*dr.Context) !bool {
    errdefer fail(x);
    if (!opened.deferred_reset) return lookup(x, s, opened.seq, ctx);
    const action: Resolve = .{ .x = x, .stream = s, .seq = opened.seq, .ctx = ctx };
    return reset_skip.resolve(action, &opened.deferred_reset, opened.pooled, &x.reset_skip, reset_skip.resetWork(x.e.w.g));
}

pub fn lookup(x: *Cuda, s: *lanes.Stream, seq: *st.Seq, ctx: ?*dr.Context) !bool {
    errdefer fail(x);
    if (!@import("exact_eligible.zig").text(x.e.kv8, s)) return false;
    const hit = try x.exact_cache.lookup(s, seq, ctx, x.e.ops(), x.e.w.g);
    if (hit and (x.seq_pool.active() or x.exact_cache.logits_enabled)) {
        if (x.seqs.get(s) != seq or seq.pos != s.prompt().len or s.cached != s.prompt().len)
            return error.InvalidSeqRestoreProof;
        try x.restored.mark(s, seq, s.prompt().len);
    }
    if (hit) try draft_reuse.initialized(x, s);
    return hit;
}

fn retainOrDestroy(x: *Cuda, s: *lanes.Stream, seq: *st.Seq, restored: bool, settled: bool, cache_ok: bool) void {
    const bytes = st.Seq.bytes(x.e.w.g, seq.capacity, seq.kv8);
    const safe = @import("exact_eligible.zig").text(x.e.kv8, s) and !seq.kv8 and cache_ok and !x.exact_cache.failed and seq.capacity <= x.e.context and
        seq.pos >= s.prompt_len and seq.pos <= seq.capacity;
    if (safe and policy.completed(s, settled, restored, cache_ok, x.exact_cache.failed) and x.seq_pool.canRetain(x.seqs.count(), seq.capacity, bytes)) {
        x.e.s.synchronize() catch |err| {
            std.log.err("synchronizing before exact-cache Seq retention: {s}", .{@errorName(err)});
            fail(x);
            x.seq_pool.stats.discarded += 1;
            destroy(x, seq);
            return;
        };
        if (policy.completed(s, settled, restored, cache_ok, x.exact_cache.failed) and !x.exact_cache.failed and
            x.seq_pool.retain(seq, x.seqs.count(), seq.capacity, bytes)) return;
    }
    x.seq_pool.stats.discarded += 1;
    destroy(x, seq);
}

/// Release retains only detached, synchronized ownership after actual settle and cache-release success.
pub fn release(ptr: *anyopaque, s: *lanes.Stream) void {
    const x: *Cuda = @ptrCast(@alignCast(ptr));
    const settled_ok = blk: {
        x.settle() catch |err| {
            std.log.err("committing the last round before a release: {s}", .{@errorName(err)});
            break :blk false;
        };
        break :blk true;
    };
    const cache_ok = blk: {
        x.exact_cache.release(s, settled_ok, x.e.ops()) catch |err| {
            std.log.err("exact cache release: {s}", .{@errorName(err)});
            fail(x);
            break :blk false;
        };
        break :blk true;
    };
    const draft_seq: ?*const anyopaque = if (x.seqs.get(s)) |seq| seq else null;
    const proof = x.restored.remove(s);
    if (x.seqs.fetchRemove(s)) |entry| {
        const restored = if (proof) |p| p.seq == @as(*const anyopaque, entry.value) and p.tokens == s.prompt_len else false;
        retainOrDestroy(x, s, entry.value, restored, settled_ok, cache_ok);
    }
    draft_reuse.release(x, s, draft_seq, settled_ok, cache_ok);
}

test {
    _ = policy;
    _ = @import("restore_proof.zig");
}
