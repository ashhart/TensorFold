//! Split prefill's plain residual (full GLM-5.3): tf_glm_residual_add_own's bytes against tf_glm_residual_add's.
const std = @import("std");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const Rig = @import("quant_ref_tests.zig").RigT;

/// Each rank of 2 and 3: the gathered partials whole, then NaN in the rank's own slot with its partial read in place.
pub fn run(r: *Rig, set: glm.glue.Set) !void {
    var cases: usize = 0;
    for ([_]usize{ 2, 3 }) |world| for ([_]usize{ 1, 7, 70, 1024 }) |rows| for ([_]usize{ 6144, 2048 }) |d| for (0..world) |me| {
        const n = rows * d;
        const x = try r.bfs(n, 1.0);
        const g = try r.a.alloc(f32, world * n);
        for (g) |*v| v.* = (r.rng.float(f32) * 2 - 1) * 0.5;
        const own = try r.a.dupe(f32, g[me * n ..][0..n]);
        const holes = try r.a.dupe(f32, g);
        @memset(holes[me * n ..][0..n], std.math.nan(f32)); // a read of the own slot would show in every row
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const want = try r.zeros(n * 2);
        const got = try r.zeros(n * 2);
        try glm.glue.residualAdd(set, r.s, dx, want, try r.dev(std.mem.sliceAsBytes(g)), rows, d, world);
        const dh = try r.dev(std.mem.sliceAsBytes(holes));
        try glm.glue.residualAddOwn(set, r.s, dx, got, dh, rows, d, world, try r.dev(std.mem.sliceAsBytes(own)), me);
        var name: [128]u8 = undefined;
        const what = try std.fmt.bufPrint(&name, "residual_add_own: world {d}, rank {d}, {d} rows of {d}", .{ world, me, rows, d });
        try check.sameBytes(what, try r.back(got, n * 2), try r.back(want, n * 2));
        cases += 1;
    };
    check.pass("residual_add_own equals residual_add byte for byte: {d} cases (worlds 2 and 3, every rank, 1 to 1,024 rows)", .{cases});
}
