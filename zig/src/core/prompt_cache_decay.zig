//! Optional bounded resume credit for the RAM snapshot store; default eviction remains main's order.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const policy = @import("prompt_cache_policy.zig");
pub const Policy = enum { main, decayed_resume };
pub const Credit = struct { key: u64 = 0, n: usize = 0, at: u32 = 0, resumes: u2 = 0 };

pub fn fromText(raw: ?[]const u8) Policy {
    return if (std.mem.eql(u8, raw orelse "", "decayed_resume")) .decayed_resume else .main;
}

pub fn fromEnv() Policy {
    return fromText(@import("profile").get(.TF_CACHE_EVICT));
}

pub fn resumed(s: *pc.Store, e: *pc.Entry) void {
    if (s.eviction != .decayed_resume) return;
    e.resume_credit = @min(@as(u3, e.resume_credit) + 1, 2);
    s.resume_carry = if (e.shared) .{} else .{ .key = pc.Store.sharedKey(e.tokens), .n = e.tokens.len, .at = e.at, .resumes = e.resume_credit };
}

pub fn kept(s: *pc.Store, e: *pc.Entry, prompt: []const u32) void {
    if (s.eviction != .decayed_resume or e.shared) return;
    const carry = s.resume_carry;
    if (carry.resumes > 0 and e.at >= carry.at and prompt.len >= carry.n and
        pc.Store.sharedKey(prompt[0..carry.n]) == carry.key)
        e.resume_credit = @max(e.resume_credit, carry.resumes);
}

fn rewind(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32) bool {
    return !e.expired and e.at == s.active_rewind_at and e.tokens.len <= prompt.len and
        std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
}

fn age(s: *const pc.Store, e: *const pc.Entry) u64 {
    return (s.clock -| e.used) -| (@as(u64, 8) * e.resume_credit);
}

pub fn victim(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32, own_only: bool, native: usize) usize {
    if (policy.redundant(s, s.entries.items[native], prompt, born)) return native;
    var best = native;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |guard| if (guard.protects(s, e)) continue;
        if (own_only and !policy.own(e, skip, prompt)) continue;
        const old = s.entries.items[best];
        const er = rewind(s, e, prompt);
        const br = rewind(s, old, prompt);
        if (er != br) {
            if (!er) best = i;
        } else if (age(s, e) > age(s, old) or (age(s, e) == age(s, old) and e.used < old.used)) {
            best = i;
        }
    }
    return best;
}
