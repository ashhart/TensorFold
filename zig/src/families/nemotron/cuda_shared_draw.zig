//! A shared window's batched draw: its host rules and positions, when every stream shares a mode.
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("cuda_forward.zig");

/// One stream of a shared window's draw: its sampling, position and rows.
pub const SharedDraw = struct { sampling: ?@import("lanes").Sampling, pos: usize, rows: usize };

/// A shared window's batched draw when every stream shares a mode; null when the modes are mixed.
pub fn batch(parts: []const SharedDraw, pinned: cuda.HostBuffer, rules: u64, meta: u64, tokens: u64, pin_rules: usize, pin_meta: usize) ?forward.Batch {
    var sampled: usize = 0;
    var total: usize = 0;
    for (parts) |p| {
        sampled += @intFromBool(cuda.sample.ruleFor(p.sampling) != null);
        total += p.rows;
    }
    if (sampled != 0 and sampled != parts.len) return null;
    var b: forward.Batch = .{ .rules = rules, .meta = meta, .tokens = tokens, .host_rules = &.{}, .host_meta = &.{}, .sampled = sampled != 0 };
    if (sampled == 0) return b;
    const rb = std.mem.sliceAsBytes(pinned.slice(u32)[pin_rules..][0 .. total * 8]);
    const hm = pinned.slice(i32)[pin_meta..][0..total];
    var at: usize = 0;
    for (parts) |p| {
        const r = cuda.sample.ruleFor(p.sampling).?;
        for (0..p.rows) |r_| {
            @memcpy(rb[(at + r_) * 32 ..][0..32], std.mem.asBytes(&r));
            hm[at + r_] = @intCast(p.pos + r_);
        }
        at += p.rows;
    }
    b.host_rules = rb;
    b.host_meta = hm;
    return b;
}

/// A sampled batch's rules and positions to the device, with the window's other pinned words (before `copied`).
pub fn upload(b: forward.Batch, ops: anytype) !void {
    if (!b.sampled) return;
    try ops.upload(b.rules, b.host_rules);
    try ops.upload(b.meta, std.mem.sliceAsBytes(b.host_meta));
}
