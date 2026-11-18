//! Opt-in reuse of silent interior neurons left by checked deletion; live or pending-retirement owners stay put.
const std = @import("std");

pub fn next(p: anytype, slot: usize) ?usize {
    if (p.reuse_deleted) finding: for (0..p.used.items[slot]) |n| {
        if (!std.mem.allEqual(f32, p.neuron(p.up.items, slot, n, p.hidden), 0)) continue;
        for (p.learned.items) |e| if (e.slot == slot and e.neuron == n) continue :finding;
        for (p.retired.items) |e| if (e[0] == slot and e[1] == n) continue :finding;
        return n;
    };
    return if (p.used.items[slot] < @TypeOf(p.*).neurons) p.used.items[slot] else null;
}
