//! Bytes of a stream's state and of the prompt cache: pages of keys and values, and the linear layers' state.

const config = @import("../../../weights/config.zig");
const pages = @import("../forward/pages.zig");

/// Bytes of one page of the pool: the keys and values of 64 positions in every attention layer.
pub fn pageBytes(s: config.Spec, act_bytes: usize) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    return full * 2 * s.kv_heads * s.head_dim * act_bytes * pages.tokens;
}

/// One stream's (or one snapshot's) linear state: the conv window and the recurrent state of every linear layer.
pub fn linearBytes(s: config.Spec) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    const conv = (s.keyWidth() * 2 + s.valueWidth()) * (s.conv - 1) * 4;
    const recurrent = s.value_heads * s.value_dim * s.key_dim * 4;
    return (s.n_layers - full) * (conv + recurrent);
}

/// Bytes one stream's caches hold at `tokens` positions: attention keys and values plus the linear layers' state.
pub fn stateBytes(s: config.Spec, act_bytes: usize, tokens: usize) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    return full * 2 * s.kv_heads * s.head_dim * act_bytes * tokens + linearBytes(s);
}

/// What `budget` bytes of prompt cache hold: at most `slots` snapshots, up to half of it, and pages for the rest.
pub const Cache = struct { pages: usize, snaps: usize };

pub fn cache(s: config.Spec, act_bytes: usize, budget: usize, slots: usize) Cache {
    const snap = @max(linearBytes(s), 1);
    const snaps = @min(slots, budget / 2 / snap);
    return .{ .pages = (budget - snaps * snap) / @max(pageBytes(s, act_bytes), 1), .snaps = snaps };
}
