//! A provisional removal changes only a new neuron's input row; rejection restores that row's exact bits.
const std = @import("std");
pub const Action = enum { none, next, keep, restore };
pub const State = struct {
    at: usize = 0,
    pending: bool = false,
    used: usize = 0,
    up: std.ArrayList(f32) = .empty,
    pub fn reset(s: *State) void {
        s.at = 0;
        s.pending = false;
    }
    pub fn deinit(s: *State, a: std.mem.Allocator) void {
        s.up.deinit(a);
    }
};

/// `true` means a candidate is staged; no disk write or rounding is allowed until it is resolved.
pub fn act(p: anytype, t: anytype, host: anytype, action: Action) !bool {
    if (!p.locate.enabled) return false;
    const s = &p.shrink;
    if (action == .none) return false;
    if (action == .next) {
        if (s.pending) return error.ShrinkPending;
        if (s.at >= p.taken.items.len) return false;
        const x = p.taken.items[s.at];
        const u = p.neuron(p.up.items, x.slot, x.neuron, p.hidden);
        try s.up.resize(p.gpa, u.len);
        @memcpy(s.up.items, u);
        s.used = p.used.items[x.slot];
        s.pending = true;
        @memset(u, 0);
        if (x.neuron + 1 == s.used) p.used.items[x.slot] = x.neuron;
        try p.upload(t, host, x.slot);
        return true;
    }
    if (!s.pending) return error.NoShrinkPending;
    const x = p.taken.items[s.at];
    if (action == .restore) {
        @memcpy(p.neuron(p.up.items, x.slot, x.neuron, p.hidden), s.up.items);
        p.used.items[x.slot] = s.used;
        try p.upload(t, host, x.slot);
        s.at += 1;
    } else {
        // The zero input is already live and checked. Keep the down row's quantization group untouched.
        _ = p.taken.orderedRemove(s.at);
        var i = p.learned_before;
        while (i < p.learned.items.len) : (i += 1) {
            const e = p.learned.items[i];
            if (e.slot == x.slot and e.neuron == x.neuron) {
                _ = p.learned.orderedRemove(i);
                break;
            }
        }
        std.log.info("slide shrink: kept without layer {d}, neuron {d}", .{ p.moe.items[x.slot], x.neuron });
    }
    s.pending = false;
    return false;
}
