//! The draft vocabulary borrows exact rows of the checkpoint's dense or affine language head.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const projection = @import("cuda_projection.zig");

pub fn draft(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, ids: []const u32) !wt.Q4 {
    const tensor = try store.tensor("language_model.lm_head.weight");
    if (tensor.dtype == .bf16) {
        if (!tensor.is(.bf16, &.{ wt.vocab, wt.dims })) return error.UnexpectedTensor;
        const rows = try a.alloc(u8, ids.len * wt.dims * 2);
        defer a.free(rows);
        for (ids, 0..) |id, i| {
            if (id >= wt.vocab) return error.InvalidToken;
            @memcpy(rows[i * wt.dims * 2 ..][0 .. wt.dims * 2], tensor.bytes[id * wt.dims * 2 ..][0 .. wt.dims * 2]);
        }
        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ ids.len, wt.dims, 1, 1, 1 }, .bytes = rows });
    }
    var buf: [120]u8 = undefined;
    const head = try wt.join3(a, store, &buf, "language_model.lm_head", "");
    defer a.free(head.w);
    defer a.free(head.s);
    defer a.free(head.b);
    const wrow = wt.dims / 2;
    const grow = wt.dims / 32 * 2;
    if (head.w.len != wt.vocab * wrow or head.s.len != wt.vocab * grow) return error.UnexpectedTensor;
    const w = try a.alloc(u8, ids.len * wrow);
    defer a.free(w);
    const s = try a.alloc(u8, ids.len * grow);
    defer a.free(s);
    const b = try a.alloc(u8, ids.len * grow);
    defer a.free(b);
    for (ids, 0..) |id, i| {
        if (id >= wt.vocab) return error.InvalidToken;
        @memcpy(w[i * wrow ..][0..wrow], head.w[id * wrow ..][0..wrow]);
        @memcpy(s[i * grow ..][0..grow], head.s[id * grow ..][0..grow]);
        @memcpy(b[i * grow ..][0..grow], head.b[id * grow ..][0..grow]);
    }
    return wt.uploadQ4(a, d, w, s, b, ids.len, wt.dims / 8);
}
