//! QK row dispatches for plain Qwen27 matrices.
const lane = @import("core").lane_projection;

pub fn encode(qks: anytype, e: anytype, p: anytype, x: anytype, xs: anytype, y: anytype, rows: u32, sk: u32, own: anytype) !void {
    const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
    const sb = lane.Ref{ .buf = p.pairs.buffer, .off = p.pairs.offset };
    var first: u32 = 0;
    while (first < rows) : (first += lane.reg_rows) {
        const take = @min(rows - first, lane.reg_rows);
        const xr = lane.Ref{ .buf = x.buffer, .off = x.offset + @as(usize, first) * p.k * 2 };
        const yr = lane.Ref{ .buf = y.buffer, .off = y.offset + @as(usize, first) * p.n * 2 };
        const sums: lane.Sums = if (xs) |s| .{ .ref = .{ .buf = s.ref.buffer, .off = s.ref.offset + @as(usize, first) * 4 }, .stride = s.stride } else try own.?.value_ptr.encodeOwnSums(e, xr, take);
        try qks.encode(e, p.n, p.k, sk, xr, w, sb, sums, yr, take);
        if (xs == null) e.barrier();
    }
}
