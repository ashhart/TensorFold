//! Lane startup steps whose failures release the partially opened lane.
const cuda = @import("cuda");

pub fn open(l: anytype, ctx: *const cuda.Context) !void {
    errdefer l.gpa.destroy(l);
    l.target = .{ .e = l.e, .caches = &l.caches, .seen = &l.seen, .device = try ctx.attribute(.integrated) != 0 };
    try @import("cuda_prefill_trace.zig").open(l);
}
