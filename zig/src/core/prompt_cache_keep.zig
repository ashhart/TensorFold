//! Keep a prompt snapshot within the unpromised budget and protect queued resumptions.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Entry = pc.Entry;
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const learned = @import("prompt_learn.zig");
const index = @import("prompt_cache_index.zig");
const policy = @import("prompt_cache_policy.zig");
const others = @import("prompt_cache_others.zig");
const rows = @import("rows_store.zig");
const note = pc.note;
const sharedKey = pc.Store.sharedKey;

pub fn keepBorn(s: *Store, prompt: []const u32, born: u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
    s.collectExpired();
    const n = @as(usize, at) + s.rules.lookahead;
    if (at == 0 or n > prompt.len or s.hold) return false;
    s.clock += 1;
    const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
    for (s.entries.items) |e| if (!e.expired and e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e)) {
        e.used = e.seen.touch(s.clock); // the same state again: no copy
        e.shared = e.shared or shared;
        e.rewind = e.rewind or (at == s.active_rewind_at and sharedKey(e.tokens) == s.active_rewind_key);
        if (shared) learned.learn(s, e, starts);
        return true;
    };
    const image_ids = images.prefix(s.gpa, s.image_query orelse &.{}, @intCast(n)) catch |err| return s.fail(at, err);
    var owns_images = true;
    defer if (owns_images) s.gpa.free(image_ids);
    const decoded = modes.prefix(s.gpa, spans, at) catch |err| return s.fail(at, err);
    var owns_decoded = true;
    defer if (owns_decoded) s.gpa.free(decoded);
    const bytes = s.family.vtable.bytes(s.family.ptr, at);
    if (bytes > s.budget or s.most == 0) {
        s.counts.refused += 1;
        note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget{s}", .{ at, bytes >> 20, s.budget >> 20, if (bytes > s.budget) s.over else "" });
        return false;
    }
    const promised = if (s.reservation_guard) |g| g.promised else @import("prompt_cache_reserve.zig").Room{};
    const limit = s.budget -| promised.bytes;
    const slots = s.most -| promised.slots;
    if (s.reservation_guard != null) {
        const can = @import("prompt_cache_reserve.zig").evictable(s, null, prompt);
        // with no outside promise a freed or spare buffer may take the state for less than `bytes`
        const reuse = promised.bytes == 0 and s.family.vtable.reuses != null;
        if (bytes > limit or slots == 0 or (!reuse and s.held -| can.bytes + bytes > limit) or
            s.entries.items.len - can.slots >= slots) return s.fail(at, error.CachePromised);
    }
    // Use fitting spare storage, otherwise make room for new storage.
    while (s.entries.items.len > 0) {
        if (s.family.vtable.reuses) |f| if (s.entries.items.len < slots and (promised.bytes == 0 or s.held + bytes <= limit) and f(s.family.ptr, at)) break;
        const free_ = s.spare();
        if (s.held + free_ + bytes <= limit and s.entries.items.len < slots) break;
        if (free_ > 0) {
            if (s.family.vtable.trim) |f| f(s.family.ptr, limit -| s.held -| bytes); // trim unusable spare first
            if (s.spare() < free_) continue;
        }
        s.remove(others.victim(s, null, prompt, born, bytes, 1) orelse return if (s.keep_others) others.refuse(s, at, bytes) else s.fail(at, error.CacheBorrowed));
        s.counts.evicted += 1;
    }
    if (bytes > limit or slots == 0) return s.fail(at, error.CachePromised);
    if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
    const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
    const last = s.gpa.dupe(u32, prompt) catch { // one copy: the state's tokens are its prefix
        s.gpa.destroy(e);
        return s.fail(at, error.OutOfMemory);
    };
    const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
        s.gpa.free(last);
        s.gpa.destroy(e);
        return s.fail(at, err);
    };
    const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
    e.* = .{ .tokens = last[0..n], .images = image_ids, .image_frame = s.image_frame, .decode_spans = decoded, .at = at, .saved = saved, .bytes = charged, .born = born, .used = s.clock, .seen = .of(s, last[0..n], decoded, starts, at), .last = last, .shared = shared, .rewind = at == s.active_rewind_at and sharedKey(last[0..n]) == s.active_rewind_key, .intermediate = @import("prompt_intermediate.zig").is(s.rules, prompt.len, at), .own = last };
    owns_decoded = false;
    owns_images = false;
    index.add(s, e) catch {
        s.free(e);
        return s.fail(at, error.OutOfMemory);
    };
    policy.link(s, e);
    s.held += charged;
    s.counts.kept += 1;
    if (shared) learned.learn(s, e, starts);
    if (s.disk) |r| rows.write.queued(r, e, starts);
    if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only within the budget's room
    return true;
}
