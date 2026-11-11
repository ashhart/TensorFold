//! A prompt in pieces between other slots' rounds (iface.Target begin / piece / finish) over the whole-prompt path.
const std = @import("std");
const host = @import("dsv41");
const target_ = @import("target.zig");
const prompt = @import("prompt.zig");

const iface = host.iface;
const lanes = host.lanes;
const Target = target_.Target;
const Slot = target_.Slot;

/// The slot cleared or a kept state restored, as `prefill` starts; rows from `at` follow in pieces, then `finish`.
pub fn begin(ptr: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
    const t = Target.self(ptr);
    const st = try t.startPrompt(slot, ids);
    const s = st.s;
    if (st.hit) |e| if (e.ids.len == ids.len) {
        s.hit.clearRetainingCapacity();
        try s.hit.appendSlice(t.a, e.logits.?);
        s.hit_on = true;
    };
    // the profile (TF_DSV41_PROMPT_PROF) counts every prompt's encoder chunks from here to this one's `finish`
    t.pr.prof = .{};
    s.began = prompt.now();
    s.filling = true;
    return .{ .at = s.resumed };
}

/// Prompt rows [start, end) through the encoder (its chunks from the begun position); the cone waits for `finish`.
pub fn piece(ptr: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void {
    const t = Target.self(ptr);
    const s = try begun(t, slot, ids);
    errdefer abandon(t, s);
    // a mid-prompt state is never kept: a kept entry holds a whole prompt's state (the decoder cone run)
    if (start == end and save and start == s.tail.end) return;
    if (start != s.tail.end or end <= start or end >= ids.len) return error.NotAChain;
    t.pr.encode(t.promptEnv(), &s.kv, &s.tail, s.hist.items, @intCast(end)) catch |e| return t.fail(e);
    t.stats.pieces += 1;
}

/// The rows the pieces left, the decoder's cone once over the prompt's end, then `prefill`'s choice and keep.
pub fn finish(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
    const t = Target.self(ptr);
    const s = try begun(t, slot, ids);
    errdefer abandon(t, s);
    const n: u32 = @intCast(ids.len);
    if (s.hit_on) {
        s.filling = false;
        return t.endPrompt(s, s.hit.items, smp, draw);
    }
    const env = t.promptEnv();
    t.pr.encode(env, &s.kv, &s.tail, s.hist.items, n) catch |e| return t.fail(e);
    t.pr.finish(env, &s.kv, &s.tail) catch |e| return t.fail(e);
    t.pr.report(n - s.resumed, prompt.now() - s.began);
    s.filling = false;
    return t.endPrompt(s, null, smp, draw);
}

/// The slot's begun prompt, refused unless `ids` are the ones it began with.
fn begun(t: *Target, slot: u32, ids: []const u32) !*Slot {
    if (slot >= t.slots.len) return error.NoSlot;
    const s = &t.slots[slot];
    if (!s.filling) return error.NotBegun;
    if (ids.len != s.hist.items.len) return error.NotAChain;
    return s;
}

/// A begun prompt ends unfinished (released, cancelled or failed): nothing of it is kept, the slot is free again.
pub fn abandon(t: *Target, s: *Slot) void {
    t.pr.forget(&s.tail);
    s.filling = false;
    s.hit_on = false;
    s.whole = false;
    t.stats.abandoned += 1;
}
