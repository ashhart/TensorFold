//! TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS=1: a new state that fits only by evicting another conversation's is not kept.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const policy = @import("prompt_cache_policy.zig");

pub const name = "TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS";

/// Exactly `1` turns the rule on; unset, empty and `0` leave it off (the startup table refuses any other text).
pub fn wanted(raw: ?[]const u8) bool {
    return std.mem.eql(u8, raw orelse "", "1");
}

/// The switch as the profile registry resolves it.
pub fn fromEnv() bool {
    return wanted(@import("profile").get(.TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS));
}

/// Whether `bytes` more in `new` more states fit in what other conversations' states leave (spare storage counts free).
pub fn fits(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, bytes: u64, new: usize) bool {
    var freed: u64 = 0;
    var n: usize = 0;
    for (s.entries.items) |e| if (policy.own(e, skip, prompt)) {
        freed += e.bytes;
        n += 1;
    };
    return (s.held -| freed) + bytes <= s.budget and s.entries.items.len - n + new <= s.most;
}

/// The state to free for a new one: victimBut's when off; when on, only an own state, and none unless the new fits.
pub fn victim(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32, bytes: u64, new: usize) ?usize {
    if (!s.keep_others) return policy.victimBut(s, skip, prompt, born);
    if (!fits(s, skip, prompt, bytes, new)) return null;
    return policy.victimIn(s, skip, prompt, born, true);
}

/// A state the rule keeps out, counted as refused, with one line naming its size and what no kept state holds.
pub fn refuse(s: *pc.Store, at: u32, bytes: u64) bool {
    s.counts.refused += 1;
    pc.note("kept nothing at {d} tokens: {d} MiB does not fit beside {d} kept states; {d} MiB of the {d} MiB budget free ({s}=1)", .{ at, bytes >> 20, s.entries.items.len, (s.budget -| s.held) >> 20, s.budget >> 20, name });
    return false;
}
