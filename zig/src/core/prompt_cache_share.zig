//! Compatible snapshot parents and accounting for KV pages held by more than one kept state.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");

pub fn parent(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) ?pc.Saved {
    var best: ?*pc.Entry = null;
    for (s.entries.items) |e| {
        if (e.expired or e.at > at or e.tokens.len > prompt.len or
            !std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) or
            !modes.equal(e.decode_spans, spans, e.at) or !images.matches(s, e)) continue;
        if (best == null or e.at > best.?.at) best = e;
    }
    return if (best) |e| e.saved else null;
}

pub fn bytes(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) u64 {
    const f = s.family.vtable.incremental orelse return s.family.vtable.bytes(s.family.ptr, at);
    return f(s.family.ptr, parent(s, prompt, at, spans), at);
}

pub fn save(s: *pc.Store, prompt: []const u32, owner: ?*anyopaque, at: u32, spans: []const modes.Span) !pc.Saved {
    const f = s.family.vtable.save_from orelse return s.family.vtable.save(s.family.ptr, owner, at);
    return f(s.family.ptr, owner, at, parent(s, prompt, at, spans));
}

/// A page's last sibling may have gone: exclusive charges move to its surviving state.
pub fn recount(s: *pc.Store) void {
    const f = s.family.vtable.shared_bytes orelse return;
    s.held = f(s.family.ptr);
    for (s.entries.items) |e| {
        e.bytes = s.family.vtable.charged.?(s.family.ptr, e.saved);
        s.held += e.bytes;
    }
}
