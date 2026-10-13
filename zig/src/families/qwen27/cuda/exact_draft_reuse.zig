//! Private cold-capable Context storage recycling; uncertain in-flight ownership is quarantined, never reused.
const std = @import("std");
const Cuda = @import("lanes.zig").Cuda;
const lanes = @import("lanes");
const drafts = @import("drafts.zig");
const dr = @import("draft.zig");
const seq_reuse = @import("exact_seq_reuse.zig");
pub const policy = @import("draft_pool.zig");
pub const Storage = policy.Pool(drafts.Stream);

fn destroy(x: *Cuda, ds: *drafts.Stream) void {
    ds.deinit();
    x.gpa.destroy(ds);
}

fn clearIdle(x: *Cuda) void {
    while (x.draft_pool.evict()) |ds| destroy(x, ds);
}

/// Only already-fenced idle storage can be freed without another fence.
pub fn fail(x: *Cuda) void {
    if (!x.draft_pool.stats.enabled) return;
    x.draft_pool.fail();
    clearIdle(x);
}

fn fresh(s: *lanes.Stream) bool {
    return s.chunks.len == 0 and !s.reuse_failed and s.reuse.saved == null and s.reuse.at == 0 and
        s.reuse.marks.len == 0 and s.reuse.hook == null;
}

fn eligible(x: *Cuda, s: *lanes.Stream, capacity: usize) bool {
    return @import("exact_eligible.zig").text(x.e.kv8, s) and policy.eligible(s.drafts, x.drafter != null, s.prompt().len, capacity, fresh(s), s.isCancelled());
}

fn reap(x: *Cuda) !void {
    if (x.draft_pool.stats.quarantined_slots == 0) return;
    try x.e.s.synchronize();
    while (x.draft_pool.reap(true)) |ds| destroy(x, ds);
}

/// Misses within the fixed small class may take storage; unsupported admissions drain before target allocation.
pub fn beforeOpen(x: *Cuda, s: *lanes.Stream, capacity: usize) !void {
    if (!x.draft_pool.stats.enabled) return;
    try reap(x);
    if (x.seqs.count() >= x.draft_pool.limit) return error.DraftPoolOwnershipLimit;
    if (!x.draft_pool.active() or !eligible(x, s, capacity)) clearIdle(x);
}

pub fn obtain(x: *Cuda, s: *lanes.Stream) !*drafts.Stream {
    errdefer seq_reuse.fail(x);
    if (x.drafting.get(s)) |ds| return ds;
    var ds: *drafts.Stream = undefined;
    if (x.draft_pool.stats.enabled) {
        try reap(x);
        const seq = x.seqs.get(s) orelse return error.UnknownQwenStream;
        if (x.draft_pool.active() and eligible(x, s, seq.capacity)) {
            if (x.draft_pool.take()) |saved| {
                ds = saved;
                errdefer destroy(x, ds);
                ds.resetStorage();
                try x.drafting.put(x.gpa, s, ds);
                return ds;
            }
        } else clearIdle(x);
        if (!x.draft_pool.hasRoom(x.drafting.count())) return error.DraftPoolOwnershipLimit;
    }
    ds = try x.gpa.create(drafts.Stream);
    errdefer x.gpa.destroy(ds);
    const ctx = dr.Context.init(x.e.ctx.d) catch |err| return if (err == error.OutOfMemory) err else error.OutOfDeviceMemory;
    if (x.draft_pool.stats.enabled) x.draft_pool.stats.driver_creations += 1;
    ds.* = drafts.Stream.init(x.gpa, ctx);
    errdefer ds.deinit();
    try x.drafting.put(x.gpa, s, ds);
    return ds;
}

/// The tag authorizes recycling empty storage later, not reuse of prior request state now.
pub fn initialized(x: *Cuda, s: *lanes.Stream) !void {
    if (!x.draft_pool.active()) return;
    errdefer fail(x);
    const seq = x.seqs.get(s) orelse return error.UnknownQwenStream;
    if (!eligible(x, s, seq.capacity)) return;
    const ds = x.drafting.get(s) orelse return error.UnknownDraftStream;
    const buf = ds.ctx.arena.buf orelse return error.InvalidDraftInitialization;
    if (seq.pos != s.prompt_len or ds.ctx.len != s.prompt_len or ds.ctx.end != s.prompt_len)
        return error.InvalidDraftInitialization;
    try x.draft_initialized.mark(.{ .stream = s, .seq = seq, .wrapper = ds, .arena = buf.ptr, .tokens = s.prompt_len, .capacity = seq.capacity });
    x.draft_pool.stats.initialized += 1;
}

fn detach(x: *Cuda, s: *lanes.Stream) ?*drafts.Stream {
    if (x.drafting.fetchRemove(s)) |entry| return entry.value;
    return null;
}

/// Error ownership never enters the reusable pool, even when a later fence succeeds.
pub fn drop(x: *Cuda, s: *lanes.Stream) void {
    _ = x.draft_initialized.remove(s);
    const ds = detach(x, s) orelse return;
    if (!x.draft_pool.stats.enabled) return destroy(x, ds);
    fail(x);
    x.e.s.synchronize() catch |err| {
        std.log.err("quarantining failed draft Context: {s}", .{@errorName(err)});
        x.draft_pool.park(ds, x.drafting.count());
        return;
    };
    x.draft_pool.stats.discarded += 1;
    destroy(x, ds);
}

/// Target release has already happened; pointer comparison does not dereference that former Seq owner.
pub fn release(x: *Cuda, s: *lanes.Stream, seq: ?*const anyopaque, settled: bool, cache_ok: bool) void {
    const proof = x.draft_initialized.remove(s);
    const ds = detach(x, s) orelse return;
    if (!x.draft_pool.stats.enabled) return destroy(x, ds);
    const normal = s.finished and (s.reason == .stop or s.reason == .length);
    const valid = if (proof) |p| p.seq == seq and p.wrapper == @as(*const anyopaque, ds) and
        p.arena == (if (ds.ctx.arena.buf) |buf| buf.ptr else 0) and p.tokens == s.prompt_len and
        eligible(x, s, p.capacity) else false;
    const clean = policy.completed(valid, normal, !s.isCancelled(), settled, cache_ok, x.exact_cache.failed);
    if (!settled or !cache_ok or x.exact_cache.failed) fail(x);
    x.e.s.synchronize() catch |err| {
        std.log.err("quarantining released draft Context: {s}", .{@errorName(err)});
        fail(x);
        x.draft_pool.park(ds, x.drafting.count());
        return;
    };
    if (clean and !s.isCancelled() and x.draft_pool.active()) {
        ds.resetStorage();
        if (x.draft_pool.retain(ds, x.drafting.count())) return;
    }
    x.draft_pool.stats.discarded += 1;
    destroy(x, ds);
}

fn shutdownFence(stream: anytype) !void {
    stream.synchronize() catch |first_err| {
        std.log.warn("draft teardown fence, retrying once: {s}", .{@errorName(first_err)});
        try stream.synchronize();
    };
}

/// One bounded retry precedes a fatal caller exit; no uncertain Context is freed on either failed fence.
pub fn shutdown(x: *Cuda) !void {
    if (!x.draft_pool.stats.enabled) return;
    if (x.draft_pool.stats.quarantined_slots > 0 or x.drafting.count() > 0) {
        try shutdownFence(&x.e.s);
    }
    while (x.draft_pool.reap(true)) |ds| destroy(x, ds);
    clearIdle(x);
    std.debug.print("PR608_EXACT_CACHE_DRAFT_POOL {f}\n", .{std.json.fmt(x.draft_pool.stats, .{})});
}

test "shutdown fence has one bounded retry and propagates persistent failure" {
    const Mock = struct {
        calls: usize = 0,
        failures: usize,
        pub fn synchronize(x: *@This()) !void {
            x.calls += 1;
            if (x.calls <= x.failures) return error.MockFenceFailed;
        }
    };
    var good: Mock = .{ .failures = 0 };
    try shutdownFence(&good);
    try std.testing.expectEqual(@as(usize, 1), good.calls);
    var recovered: Mock = .{ .failures = 1 };
    try shutdownFence(&recovered);
    try std.testing.expectEqual(@as(usize, 2), recovered.calls);
    var failed: Mock = .{ .failures = 3 };
    try std.testing.expectError(error.MockFenceFailed, shutdownFence(&failed));
    try std.testing.expectEqual(@as(usize, 2), failed.calls);
}
