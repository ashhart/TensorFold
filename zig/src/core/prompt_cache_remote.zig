//! #1594 step 2b's kept remote states: a state built on another machine's prefix, kept apart from the engine's own.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

/// The engine's own copy `e` just kept hides each remote entry it serves alike: same at, tokens, spans and images.
pub fn retireTwins(s: *pc.Store, e: *const pc.Entry) void {
    for (s.entries.items) |o| if (o != e and o.remote and !o.expired and twins(o, e)) {
        o.expired = true; // reclaimed at the next safe host boundary, like any expired entry (a borrower keeps it)
    };
}

/// Whether every lookup `o` serves, `e` serves too: the keys keepBorn and index.find compare, entry against entry.
fn twins(o: *const pc.Entry, e: *const pc.Entry) bool {
    if (o.at != e.at or !std.mem.eql(u32, o.tokens, e.tokens) or !modes.equal(o.decode_spans, e.decode_spans, e.at)) return false;
    if (o.image_frame != e.image_frame or o.images.len != e.images.len) return false;
    for (o.images, e.images) |x, y| if (!std.meta.eql(x, y)) return false;
    return true;
}
