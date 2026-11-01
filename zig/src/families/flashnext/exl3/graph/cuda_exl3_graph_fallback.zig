//! A failed capture costs its round an eager run: the key backs off and retries a bounded number of times.

const std = @import("std");
const cuda = @import("cuda");
const hub = @import("cuda_exl3_graph.zig");

/// A graph is optional when its memory admission is refused; other driver and collective errors still propagate.
pub fn admissionDenied(err: anyerror) bool {
    return switch (err) {
        error.GraphBudgetBusy, error.GraphBudgetExceeded, error.GraphMemoryHeadroom, error.GraphEvictionRequired => true,
        else => false,
    };
}

/// Failures a key tolerates; the one after the last stays eager until the cache is cleared.
pub const max_failures: u8 = 3;
/// Rounds the first failure waits before the key is captured again; each further failure doubles the wait.
pub const first_backoff: u64 = 32;
const slots = 64;

pub const Failure = struct { hash: u64, failures: u8, retry_at: u64 };
var table: [slots]?Failure = @splat(null);
var next_slot: usize = 0;

pub fn hashOf(salt: u64, key: anytype) u64 {
    var h = std.hash.Wyhash.init(salt);
    std.hash.autoHashStrat(&h, key, .Deep);
    return h.final();
}

/// True while `hash` waits out a backoff (`now`: the graph cache's clock) or has used its retries.
pub fn blocked(hash: u64, now: u64) bool {
    for (table) |slot| if (slot) |f| if (f.hash == hash) return f.failures > max_failures or now < f.retry_at;
    return false;
}

/// A capture or instantiation of `hash` failed: its next attempt waits `first_backoff << failures` rounds.
pub fn record(hash: u64, now: u64) void {
    hub.stats.capture_failures += 1;
    for (&table) |*slot| if (slot.*) |*f| if (f.hash == hash) {
        f.failures +|= 1;
        f.retry_at = now +| (first_backoff << @intCast(@min(f.failures, 8)));
        return;
    };
    for (&table) |*slot| if (slot.* == null) {
        slot.* = .{ .hash = hash, .failures = 1, .retry_at = now +| (first_backoff << 1) };
        return;
    };
    table[next_slot % slots] = .{ .hash = hash, .failures = 1, .retry_at = now +| (first_backoff << 1) };
    next_slot += 1;
}

pub fn failures(hash: u64) u8 {
    for (table) |slot| if (slot) |f| if (f.hash == hash) return f.failures;
    return 0;
}

pub fn clear() void {
    table = @splat(null);
    next_slot = 0;
}

/// A test's way to make the next captures fail after they ran (the graph is then discarded as a failed capture).
pub var inject_capture_failures: u32 = 0;
/// A test's way to make the next instantiations fail.
pub var inject_instantiate_failures: u32 = 0;

pub fn take(counter: *u32) bool {
    if (counter.* == 0) return false;
    counter.* -= 1;
    return true;
}

/// Captures `pass.chunk(ids, pos0, null)` once; null when it could not (closed and dropped, nothing ran).
pub fn captureOnce(pass: anytype, ids: []const u32, pos0: usize) ?cuda.graph.Graph {
    cuda.graph.beginCapture(pass.stream, .thread_local) catch return null;
    pass.prepared = true;
    hub.capturing = true;
    const ran = pass.chunk(ids, pos0, null);
    hub.capturing = false;
    ran catch {
        if (cuda.graph.endCapture(pass.stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return null;
    };
    var graph = cuda.graph.endCapture(pass.stream) catch return null;
    if (take(&inject_capture_failures)) {
        graph.deinit();
        return null;
    }
    return graph;
}

/// One rank's capture attempt (a graph or null), agreed through `c.graphAgree`: kept only if both ranks hold one.
pub fn agreedCapture(c: anytype, signature: u64, cohort: u64, attempt: anytype) !@TypeOf(attempt.run()) {
    var mine = attempt.run();
    const decision = c.graphAgree(signature ^ capture_tag, .{ .state = if (mine != null) .ready else .eager, .cohort = cohort }) catch |err| {
        if (mine) |*g| g.deinit();
        return err;
    };
    if (decision.choice == .replay) return mine;
    if (mine) |*g| g.deinit();
    return null;
}

const capture_tag: u64 = 0x63617074757265;

test "exl3: a failed key backs off, retries a bounded number of times and then stays eager" {
    clear();
    defer clear();
    const h: u64 = 0x1234;
    try std.testing.expect(!blocked(h, 0));
    var now: u64 = 100;
    var tries: u32 = 0;
    // a key that always fails: every attempt waits longer, and the attempts stop
    while (tries < 20) : (tries += 1) {
        while (blocked(h, now) and now < 100_000) now += 1;
        if (blocked(h, now)) break;
        record(h, now);
    }
    try std.testing.expectEqual(@as(u8, max_failures + 1), failures(h));
    try std.testing.expect(blocked(h, now + 1_000_000));
    // other keys are untouched
    try std.testing.expect(!blocked(h +% 1, now));
    clear();
    try std.testing.expect(!blocked(h, now));
}

test "exl3: retry waits double with each failure" {
    clear();
    defer clear();
    record(7, 0);
    try std.testing.expect(blocked(7, first_backoff * 2 - 1));
    try std.testing.expect(!blocked(7, first_backoff * 2));
    record(7, first_backoff * 2);
    try std.testing.expect(blocked(7, first_backoff * 2 + first_backoff * 4 - 1));
    try std.testing.expect(!blocked(7, first_backoff * 2 + first_backoff * 4));
}
