//! Cache invalidation is immediate; storage and peer drops wait for a host boundary and the last restore borrower.
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
