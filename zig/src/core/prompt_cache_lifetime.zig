//! Cache invalidation is immediate; storage and peer drops wait for a host boundary and the last restore borrower.
const std = @import("std");
const pc = @import("prompt_cache.zig");

/// Called inside backend operations: no destruction, allocation, or peer command is allowed here.
pub fn expired(s: *pc.Store, saved: pc.Saved) void {
    for (s.entries.items) |e| if (e.saved == saved and !e.expired) {
        e.expired = true;
        s.counts.evicted += 1;
        return;
    };
}

/// Reclaim invalid entries between backend operations, keeping every outstanding lookup alive.
pub fn collect(s: *pc.Store) void {
    var i: usize = 0;
    while (i < s.entries.items.len) {
        const e = s.entries.items[i];
        if (e.expired and e.borrowers == 0) s.remove(i) else i += 1;
    }
}

/// Release a lookup without reporting a restore, including cancellation and admission failure.
pub fn release(s: *pc.Store, e: *pc.Entry) void {
    if (e.borrowers > 0) e.borrowers -= 1;
    collect(s);
}

/// A completed restore updates reuse history; a refusal expires the exact borrowed state.
pub fn resumedWithReason(s: *pc.Store, e: *pc.Entry, prompt: []const u32, failure: ?anyerror) void {
    if (std.mem.indexOfScalar(*pc.Entry, s.entries.items, e) == null) return;
    defer s.release(e);
    if (failure) |err| {
        @import("prompt_cache_restore.zig").report(s.restore_log, .{ .kept = e.at, .prompt = prompt.len, .expired = e.expired, .err = err });
        s.counts.failed += 1;
        e.expired = true;
        return;
    }
    s.clock += 1;
    e.used = e.seen.hit(s.clock);
    @import("prompt_cache_decay.zig").resumed(s, e);
    const branch = @import("prompt_cache_policy.zig").continuedLater(s, e) and !std.mem.eql(u32, prompt, e.last);
    e.branched = e.branched or branch; // an identical re-send adds no branch
    s.counts.hits += 1;
    const last = s.gpa.dupe(u32, prompt) catch return;
    if (e.own.len == 0 or e.last.ptr != e.own.ptr) s.gpa.free(e.last); // the first `last` is the tokens' copy
    e.last = last;
}
