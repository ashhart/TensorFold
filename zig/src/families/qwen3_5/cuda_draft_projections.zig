//! Grouped draft dispatch retains the established projection and shared-sum ordering.
const cuda = @import("cuda");
const precision = @import("cuda_draft_precision.zig");

pub fn projectMany(d: anytype, ls: []const cuda.qlinear.Affine4, x: u64, outs: []const u64, rows: usize, sums: bool, segmented: bool) !void {
    const e = d.e;
    if (ls.len == 0 or ls.len != outs.len) return error.UnexpectedTensor;
    const prompt = precision.prompt(d.projection_role, rows, precision.dflash_projection_prompt_cutoff);
    const overridden = if (d.w.bf16) |*p| p.containsAny(ls) else false;
    if (overridden or (!e.affine.pdl and !segmented) or (rows > 16 and (!e.affine.pdl or prompt)) or ls.len > cuda.qlinear.Affine4Kernels.segs_max or ls.len == 1) {
        for (ls, outs, 0..) |l, out, i| try d.projSums(l, x, out, rows, sums and (overridden or i == 0));
        return;
    }
    if (sums) try e.run(e.fns.sums, .{ rows, (ls[0].k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, d.xs, @as(c_int, @intCast(ls[0].k)), @as(u64, 0) });
    var many: [cuda.qlinear.Affine4Kernels.segs_max]cuda.qlinear.Affine4Kernels.Out = undefined;
    for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l, .out = out };
    try e.affine.verifyManyRows(e.stream, x, d.xs, many[0..ls.len], rows);
}
