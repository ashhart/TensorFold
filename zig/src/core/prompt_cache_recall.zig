//! A learned state read back from disk as a shared prompt-cache entry (the store's lookup, in its own file for size).
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const learned = @import("prompt_learn.zig");
const policy = @import("prompt_cache_policy.zig");
const index = @import("prompt_cache_index.zig");
const Store = pc.Store;
const Entry = pc.Entry;

/// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
pub fn recall(s: *Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*Entry) ?*Entry {
    const im = s.imprint orelse return have;
    const read = s.family.vtable.read orelse return have;
    const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
    if (s.image_frame != 0 or !images.diskSafe(m.tokens) or !images.equal(&.{}, s.image_query orelse &.{}, @intCast(m.tokens.len))) return have;
    if (!modes.equal(&.{}, spans, m.at)) return have; // the request decodes rows the learned prompt pass prefilled
    const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
    while (bytes <= s.budget and (s.held + s.spare() + bytes > s.budget or s.entries.items.len >= s.most)) {
        s.remove(pc.others.victim(s, have, prompt, born, bytes, 1) orelse return have);
        s.counts.evicted += 1;
    }
    if (bytes > s.budget) return have;
    const e = s.gpa.create(Entry) catch return have;
    e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = born, .used = s.clock, .seen = .of(s, m.tokens, &.{}, m.starts, m.at), .last = &.{}, .shared = true };
    e.tokens = s.gpa.dupe(u32, m.tokens) catch return drop(s, e, have, false);
    e.last = s.gpa.dupe(u32, prompt) catch return drop(s, e, have, false);
    const key = m.key;
    e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
        pc.note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
        learned.unlearn(s, im, key);
        return drop(s, e, have, false);
    };
    im.touch(key);
    index.add(s, e) catch return drop(s, e, have, true);
    policy.link(s, e);
    s.held += bytes;
    if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
    return e;
}

/// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
    if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
    s.gpa.free(e.tokens);
    s.gpa.free(e.last);
    s.gpa.destroy(e);
    return have;
}
