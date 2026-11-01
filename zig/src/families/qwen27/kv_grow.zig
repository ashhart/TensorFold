//! Grow-as-used K/V (opt-in): a slot holds the rows its stream reached, not the whole context.
const std = @import("std");

/// One blit of a K/V buffer's head into a wider buffer, in bytes.
pub const Move = struct { from: usize, to: usize, len: usize };

/// TF_Q27_KV_GROW: the rows a slot starts with and grows by (a multiple of 16); unset or 0 keeps every slot full.
pub fn step(text: ?[]const u8) !?u32 {
    const value = text orelse return null;
    const n = std.fmt.parseInt(u32, value, 10) catch return error.BadKvGrowStep;
    if (n == 0) return null;
    if (n % 16 != 0) return error.BadKvGrowStep;
    return n;
}

/// A slot's first rows: one step, at most the context.
pub fn first(s: u32, context: u32) u32 {
    return @min(s, context);
}

/// Rows for `need`: the need rounded up to the step, or a quarter more than `held`, at most the context.
pub fn rows(held: u32, need: u32, s: u32, context: u32) u32 {
    if (need <= held) return held;
    const rounded = (@as(u64, need) + s - 1) / s * s;
    const quarter = @as(u64, held) + held / 4;
    return @intCast(@min(@max(rounded, quarter), context));
}

/// Head h of a [heads][old][dim] bf16 buffer to its place in a [heads][new][dim] one.
pub fn moves(out: []Move, heads: usize, dim: usize, old: u32, new: u32) []Move {
    std.debug.assert(out.len >= heads and new >= old);
    const row = dim * @sizeOf(u16);
    for (out[0..heads], 0..) |*m, h| m.* = .{ .from = h * old * row, .to = h * new * row, .len = @as(usize, old) * row };
    return out[0..heads];
}

/// A slot's K and V bytes across `layers` attention layers at `r` rows of `kv` bf16 values.
pub fn slotBytes(layers: usize, kv: usize, r: u32) usize {
    return layers * 2 * @as(usize, r) * kv * @sizeOf(u16);
}
