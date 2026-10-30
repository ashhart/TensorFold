//! Memory pressure's shedding: kept states leave, least valuable first, until the store holds no more than a watermark.
const pc = @import("prompt_cache.zig");

/// One unborrowed state leaves (spilled first under --spill-gib) while the store holds past `mark`; spare past it too.
pub fn shed(s: *pc.Store, mark: u64) bool {
    s.collectExpired();
    var went = false;
    if (s.held > mark) if (s.victimBut(null, &.{}, 0)) |i| {
        s.remove(i);
        s.counts.evicted += 1;
        went = true;
    };
    const room = mark -| s.held;
    if (s.family.vtable.trim) |f| if (s.spare() > room) f(s.family.ptr, room);
    return went;
}
