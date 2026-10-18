//! The loader owns each tail allocation until all books, scratch and tables are ready.
const std = @import("std");
pub const Sizes = struct { book: usize, ctx: usize, cand: usize, hidden: usize };

pub fn acquire(g: std.mem.Allocator, dr: anytype, ctx: anytype, comptime Ops: type, n: Sizes) !void {
    dr.pred = try g.alloc(u16, n.book);
    errdefer g.free(dr.pred);
    try Ops.readBook(ctx, 0, dr.pred);
    dr.succ = try g.alloc(u16, n.book);
    errdefer g.free(dr.succ);
    try Ops.readBook(ctx, 1, dr.succ);
    try Ops.scratch(ctx, dr);
    errdefer Ops.freeScratch(ctx, dr);
    dr.h_cpos = try g.alloc(i64, n.ctx);
    errdefer g.free(dr.h_cpos);
    dr.h_crings = try g.alloc(u64, n.ctx);
    errdefer g.free(dr.h_crings);
    dr.cand = try g.alloc(i32, n.cand);
    errdefer g.free(dr.cand);
    dr.gh = try g.alloc(f32, n.hidden);
}
