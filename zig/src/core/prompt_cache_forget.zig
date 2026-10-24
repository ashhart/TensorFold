//! Span erasure of retained and learned prompt copies, with bounded outcome diagnostics.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const learned = @import("prompt_learn.zig");

pub fn holdsSpan(tokens: []const u32, span: []const u32) bool {
    if (span.len == 0) return false;
    if (std.mem.indexOf(u32, tokens, span) != null) return true;
    var n = @min(tokens.len, span.len - 1);
    while (n >= 2) : (n -= 1) {
        if (std.mem.eql(u32, tokens[tokens.len - n ..], span[0..n])) return true;
    }
    return false;
}

pub fn weakSpanTail(tokens: []const u32, span: []const u32) bool {
    return tokens.len > 0 and span.len > 0 and tokens[tokens.len - 1] == span[0];
}

/// What a forget of a token span did: the kept and learned states it removed, and what it could not reach.
pub const Forgotten = struct {
    folders: @import("forget_folders.zig").Folders = .{},
    states: u32 = 0, // kept states whose tokens held the span: dropped, their tokens zeroed
    scrubbed: u32 = 0, // kept states that did not, but whose last prompt did: that prompt replaced by their own tokens
    learned: u32 = 0, // learned states on disk (--learn) whose tokens held it: files and index records removed
    held: u32 = 0, // kept states a running prompt pass is restoring: left in place, a later forget takes them
    failed: u32 = 0, // learned states whose files or index record could not be removed, or a folder sync that failed
    in_flight: u32 = 0, // requests queued or running whose prompt holds the span (the engine counts them)
    conversations: u32 = 0, // saved conversation states removed from disk
    conversation_disk: bool = false, // the current saved-conversation identity was searched
    disk: bool = false, // the learned states' index (--learn) was searched: false when this store keeps none on disk

    /// Nothing the engine keeps holds the span any more, and no request carrying it is still running.
    pub fn complete(f: Forgotten) bool {
        return f.held == 0 and f.failed == 0 and f.in_flight == 0;
    }
};

/// Erase matching copies only when their owner can release them, and report every unresolved holder.
pub fn forgetSpan(s: *pc.Store, span: []const u32, held: []const *const pc.Entry) Forgotten {
    var out: Forgotten = .{};
    if (span.len == 0) return out;
    if (s.family.vtable.forget_peers_clear) |peers_clear| {
        if (!peers_clear(s.family.ptr)) out.held += 1;
    }
    if (s.disk) |r| @import("rows_forget.zig").forget(r, span, &out);
    var i = s.entries.items.len;
    while (i > 0) {
        i -= 1;
        const e = s.entries.items[i];
        if (!holdsSpan(e.tokens, span)) {
            if (weakSpanTail(e.tokens, span)) {
                out.held += 1;
                continue;
            }
            if (!holdsSpan(e.last, span) and !weakSpanTail(e.last, span)) continue;
            const own = s.gpa.dupe(u32, e.tokens) catch @constCast(@as([]const u32, &.{})); // out of memory: an empty last prompt only loses a mark
            std.crypto.secureZero(u32, e.last);
            s.gpa.free(e.last);
            e.last = own;
            out.scrubbed += 1;
            continue;
        }
        if (e.borrowers > 0 or std.mem.indexOfScalar(*const pc.Entry, held, e) != null) {
            e.expired = true;
            out.held += 1;
            continue;
        }
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, pc.Store.sharedKey(e.tokens))) |k| _ = s.shared_keys.orderedRemove(k);
        std.crypto.secureZero(u32, e.tokens);
        std.crypto.secureZero(u32, e.last);
        s.remove(i);
        out.states += 1;
    }
    if (s.imprint) |im| {
        learned.forget(s, im, span, &out);
        learned.forgetOthers(s, im, span, &out);
        out.disk = true;
    }
    return out;
}
