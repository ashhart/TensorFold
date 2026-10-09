//! A completed draft head yields bounded candidate rows and retains all selector inputs beyond workspace reuse.
const std = @import("std");
const op = @import("operators.zig");
const Config = @import("config.zig").Config;
const Frame = @import("runtime_frame.zig").Frame;
fn value(r: @import("runtime_frame.zig").Ref, at: usize) f32 {
    const raw = std.mem.readInt(u16, r.buf.contents()[r.off + at * 2 ..][0..2], .little);
    return @bitCast(@as(u32, raw) << 16);
}
pub fn read(a: std.mem.Allocator, frame: *const Frame, topk: op.View, hidden: op.View, c: Config, head: op.Head) !op.Lattice {
    if (!frame.completed) return error.DraftLatticeShape;
    try topk.check(head.logits.rows, @sizeOf(@import("core").bf16_topk.Row), .u8);
    try hidden.check(head.logits.rows, c.rank, .bf16);
    const picked = try frame.ref(topk);
    const projected = try frame.ref(hidden);
    const mapping = if (head.token_ids) |ids| try frame.ref(ids) else null;
    const count = @as(usize, head.logits.rows) * c.topk;
    var out = op.Lattice{ .gpa = a, .depth = head.logits.rows, .topk = c.topk, .rank = c.rank, .candidates = try a.alloc(u32, count), .unary = undefined, .projected = undefined };
    errdefer a.free(out.candidates);
    out.unary = try a.alloc(f32, count);
    errdefer a.free(out.unary);
    out.projected = try a.alloc(f32, @as(usize, head.logits.rows) * c.rank);
    errdefer a.free(out.projected);
    for (0..head.logits.rows) |row| {
        const raw: *const @import("core").bf16_topk.Row = @ptrCast(@alignCast(picked.buf.contents() + picked.off + row * topk.stride));
        const entries = try @import("core").bf16_topk.decode(raw, .{ .rows = head.logits.rows, .vocab = head.logits.width, .k = c.topk, .stride = head.logits.stride });
        for (entries, 0..) |entry, column| {
            out.candidates[row * c.topk + column] = if (mapping) |ids| std.mem.readInt(u32, ids.buf.contents()[ids.off + @as(usize, entry.token) * 4 ..][0..4], .little) else entry.token;
            out.unary[row * c.topk + column] = entry.score;
        }
        for (0..c.rank) |r| out.projected[row * c.rank + r] = value(projected, row * hidden.stride + r);
    }
    try out.check(c);
    return out;
}
