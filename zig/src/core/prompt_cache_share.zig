//! Compatible snapshot parents and accounting for KV pages held by more than one kept state.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");

pub fn parent(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) ?pc.Saved {
    return parentFor(s, null, prompt, at, spans);
}

pub fn parentFor(s: *const pc.Store, owner: ?*anyopaque, prompt: []const u32, at: u32, spans: []const modes.Span) ?pc.Saved {
    const eligible = s.family.vtable.can_share orelse return null;
    var best: ?*pc.Entry = null;
    for (s.entries.items) |e| {
        if (e.expired or e.at > at or e.tokens.len > prompt.len or
            !std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) or
            !modes.equal(e.decode_spans, spans, e.at) or !images.matches(s, e) or
            !eligible(s.family.ptr, owner, e.saved, at)) continue;
        if (best == null or e.at > best.?.at) best = e;
    }
    return if (best) |e| e.saved else null;
}

/// Planning can anticipate a matching restore; a keep recomputes demand using its actual live owner.
pub fn bytes(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) u64 {
    const f = s.family.vtable.incremental orelse return s.family.vtable.bytes(s.family.ptr, at);
    var need = f(s.family.ptr, null, at);
    if (s.family.vtable.can_share == null) return need;
    for (s.entries.items) |e| {
        if (e.expired or e.at > at or e.tokens.len > prompt.len or
            !std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) or
            !modes.equal(e.decode_spans, spans, e.at) or !images.matches(s, e)) continue;
        need = @min(need, f(s.family.ptr, e.saved, at));
    }
    return need;
}

pub fn liveBytes(s: *const pc.Store, prompt: []const u32, owner: ?*anyopaque, at: u32, spans: []const modes.Span) u64 {
    const f = s.family.vtable.incremental orelse return s.family.vtable.bytes(s.family.ptr, at);
    return f(s.family.ptr, parentFor(s, owner, prompt, at, spans), at);
}

pub fn save(s: *pc.Store, prompt: []const u32, owner: ?*anyopaque, at: u32, spans: []const modes.Span) !pc.Saved {
    const f = s.family.vtable.save_from orelse return s.family.vtable.save(s.family.ptr, owner, at);
    return f(s.family.ptr, owner, at, parentFor(s, owner, prompt, at, spans));
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

pub const Freed = struct { bytes: u64 = 0, slots: usize = 0 };

/// Count exclusive storage plus shared pages whose last holders are all removable by this victim order.
pub fn freeSet(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, own_only: bool) Freed {
    const C = struct {
        s: *const pc.Store,
        skip: ?*pc.Entry,
        prompt: []const u32,
        own_only: bool,
        fn get(ptr: *const anyopaque, i: usize) ?pc.Saved {
            const c: *const @This() = @ptrCast(@alignCast(ptr));
            const e = c.s.entries.items[i];
            if (e.borrowers > 0 or e == c.skip) return null;
            if (c.s.reservation_guard) |g| if (g.protects(c.s, e)) return null;
            if (c.own_only and !@import("prompt_cache_policy.zig").own(e, c.skip, c.prompt)) return null;
            return e.saved;
        }
    };
    const c = C{ .s = s, .skip = skip, .prompt = prompt, .own_only = own_only };
    const set = pc.SavedSet{ .ctx = &c, .len = s.entries.items.len, .get = C.get };
    var out: Freed = .{};
    for (s.entries.items, 0..) |e, i| if (set.get(set.ctx, i) != null) {
        out.bytes += e.bytes;
        out.slots += 1;
    };
    if (s.family.vtable.shared_freed) |f| out.bytes += f(s.family.ptr, set);
    return out;
}
