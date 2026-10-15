//! The pocket's neurons as the shards hold them, read once, at the layers the family's layout learns at.
const std = @import("std");
const neuron_math = @import("neuron.zig");

/// The pocket's neurons as the shards hold them, read once, at the layers the family's layout learns at.
pub fn load(p: anytype, host: anytype) !void {
    const P = @TypeOf(p.*);
    if (p.moe.items.len > 0) return;
    const lay = host.layout();
    if (lay.layers > 64) return error.TooManyLayers; // a step's nudges mark their layers in a u64
    p.hidden = lay.hidden;
    p.layers = lay.layers;
    for (0..lay.layers) |l| if (lay.learn.isSet(l)) try p.moe.append(p.gpa, l);
    const n = P.neurons * lay.hidden;
    try p.up.appendNTimes(p.gpa, 0, p.moe.items.len * n);
    try p.down.appendNTimes(p.gpa, 0, p.moe.items.len * n);
    try p.used.appendNTimes(p.gpa, 0, p.moe.items.len);
    if (P.swiglu) {
        try p.gate.appendNTimes(p.gpa, 0, p.moe.items.len * n);
        try p.scratch.appendNTimes(p.gpa, 0, n);
    }
    for (p.moe.items, 0..) |l, j| {
        const up = p.up.items[j * n ..][0..n];
        const down = p.down.items[j * n ..][0..n];
        if (P.swiglu) {
            p.used.items[j] = host.pocketReadSwiglu(l, p.gate.items[j * n ..][0..n], p.scratch.items, down);
            neuron_math.fromUp(P.of.neuron, p.scratch.items, up);
        } else p.used.items[j] = host.pocketRead(l, up, down);
    }
}
