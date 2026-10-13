//! tf-glm-rows-speed's candidates (GLM_SPEED_CANDIDATES): glm_rows_speed.metal's launch shapes, bits then time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const speed = @import("glm_rows_speed.zig");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const Ref = affine.Ref;
const Affine = affine.Affine;
const Pool = speed.Pool;
const Io = speed.Io;
const Bench = speed.Bench;
const bind = speed.bind;
const rf = speed.rf;
const rows = speed.rows;
const moeTwo = speed.moeTwo;
const a = speed.a;
const reps = speed.reps;
const N = speed.N;
const D = speed.D;
const E = speed.E;
const TOPK = speed.TOPK;

/// glm_rows.metal with glm_rows_speed.metal's candidate instances.
pub fn candidateLib(device: mtl.Device) !mtl.Library {
    const src = try std.mem.concat(a, u8, &.{ sources.core_row_projection, wide.glue_math, sources.glm_rows, @embedFile("glm_rows_speed.metal") });
    defer a.free(src);
    return mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
}

fn cand(e: mtl.ComputeEncoder, i: usize, pipe: mtl.Pipeline, rps: u32, sg: u32, pool: Pool, io: Io) void {
    const q = pool.at(i);
    e.setPipeline(pipe);
    bind(e, 0, .{ rf(io.x), q.w, q.s, q.b });
    e.setValue(wide.RowDims{ .k = @intCast(q.k), .n = @intCast(q.n), .group = 64, .rows = 1, .ldx = @intCast(q.k), .ldo = @intCast(q.n) }, 4);
    bind(e, 5, .{rf(io.y)});
    e.dispatchGroups(mtl.Size.of((q.n + rps * sg - 1) / (rps * sg), 1, 1), mtl.Size.of(32 * sg, 1, 1));
}

/// glm_rows_speed.metal's launch shapes on `pool` at 8 bits: each one's bits == glm_rows's, then its time.
pub fn candidates(b: *const Bench, lib: mtl.Library, p: *const wide.Pipes, pool: Pool, name: []const u8) !void {
    const q = pool.at(0);
    const base = try b.time(rows, .{ p, pool, b.io });
    try b.run(rows, .{ p, pool, b.io });
    const want = try a.dupe(u16, b.io.y.slice(u16, q.n));
    defer a.free(want);
    var best: f64 = base;
    var best_name: [40]u8 = undefined;
    var best_len: usize = 0;
    for ([_][2]u32{ .{ 4, 2 }, .{ 8, 2 }, .{ 4, 4 }, .{ 8, 4 }, .{ 2, 4 }, .{ 2, 8 }, .{ 4, 8 } }) |rs| for ([_]u32{ 0, q.k }) |kk| {
        var nb: [40]u8 = undefined;
        const fname = try std.fmt.bufPrintSentinel(&nb, "x_q8_r{d}_s{d}_k{d}", .{ rs[0], rs[1], kk }, 0);
        const pipe = try mtl.Pipeline.init(b.device, lib, fname, false);
        defer pipe.deinit();
        @memset(b.io.y.slice(u16, q.n), 0);
        try b.run(cand, .{ pipe, rs[0], rs[1], pool, b.io });
        if (!std.mem.eql(u16, b.io.y.slice(u16, q.n), want)) {
            std.debug.print("  {s} {s}: BITS DIFFER\n", .{ name, fname });
            return error.BitsDiffer;
        }
        const us = try b.time(cand, .{ pipe, rs[0], rs[1], pool, b.io });
        if (us < best) {
            best = us;
            @memcpy(best_name[0..fname.len], fname);
            best_len = fname.len;
        }
        std.debug.print("  {s:<16} {s:<24} {d:8.1} us ({d:5.3}x glm_rows)\n", .{ name, fname, us, base / us });
    };
    std.debug.print("  {s:<16} glm_rows {d:.1} us, best {s} {d:.1} us ({d:.0} GB/s)\n", .{ name, base, best_name[0..best_len], best, @as(f64, @floatFromInt(pool.bytes)) / best / 1e3 });
}

/// moeTwo through candidate pipelines (glu, down) at `rps` columns a simdgroup.
fn moeCand(e: mtl.ComputeEncoder, i: usize, pipes: [2]mtl.Pipeline, rps: [2]u32, gu: [2]Pool, down: Pool, sh: [2]Pool, io: Io) void {
    const z = rf(io.z);
    const ids = rf(io.ids).at((i % reps) * TOPK * 4);
    const steps = [2]struct { w: Affine, up: Affine, sh: Affine, x: Ref, out: Ref, out2: Ref, repeat: u32 }{
        .{ .w = gu[0].at(i), .up = gu[1].at(i), .sh = sh[0].at(i), .x = rf(io.x), .out = z, .out2 = z.at(TOPK * N * 2), .repeat = TOPK },
        .{ .w = down.at(i), .up = down.at(i), .sh = sh[1].at(i), .x = z, .out = rf(io.y), .out2 = rf(io.y).at(TOPK * D * 2), .repeat = 1 },
    };
    for (steps, pipes, rps, 0..) |st, pipe, r, j| {
        e.setPipeline(pipe);
        bind(e, 0, .{ st.x, st.w.w, st.w.s, st.w.b });
        e.setValue(wide.PickDims{ .k = @intCast(st.w.k), .n = @intCast(st.w.n), .group = 64, .slots = TOPK, .repeat = @intCast(st.repeat), .experts = E, .ldo = @intCast(st.w.n), .lim = 10.0 }, 4);
        bind(e, 5, .{ st.out, ids, st.up.w, st.up.s, st.up.b, if (j == 0) rf(io.x) else z.at(TOPK * N * 2), st.sh.w, st.sh.s, st.sh.b, st.out2 });
        e.dispatchGroups(mtl.Size.of((st.w.n + 2 * r - 1) / (2 * r), TOPK + 1, 1), mtl.Size.of(64, 1, 1));
    }
}

/// glm_moe_rows candidates (glm_rows_speed.metal) for oQ2's MoE: bits == wide.moe's, then the time a layer.
pub fn moeCandidates(b: *const Bench, lib: mtl.Library, p: *const wide.Pipes, gu: [2]Pool, down: Pool, sh: [2]Pool) !void {
    const base = try b.time(moeTwo, .{ p, gu, down, sh, b.io });
    try b.run(moeTwo, .{ p, gu, down, sh, b.io });
    const want = try a.dupe(u16, b.io.y.slice(u16, (TOPK + 1) * D));
    defer a.free(want);
    for ([_][2][:0]const u8{ .{ "x_moe_glu_r8_k0", "x_moe_r8_k0" }, .{ "x_moe_glu_r2_k0", "x_moe_r2_k0" }, .{ "x_moe_glu_r4_k4096", "x_moe_r4_k2048" }, .{ "x_moe_glu_r8_k4096", "x_moe_r8_k2048" } }) |names| {
        var pipes: [2]mtl.Pipeline = undefined;
        for (&pipes, names) |*x, n| x.* = try mtl.Pipeline.init(b.device, lib, n, false);
        defer for (pipes) |x| x.deinit();
        const rps = [2]u32{ names[0][11] - '0', names[1][7] - '0' };
        @memset(b.io.y.slice(u16, (TOPK + 1) * D), 0);
        try b.run(moeCand, .{ pipes, rps, gu, down, sh, b.io });
        if (!std.mem.eql(u16, b.io.y.slice(u16, (TOPK + 1) * D), want)) return error.BitsDiffer;
        const us = try b.time(moeCand, .{ pipes, rps, gu, down, sh, b.io });
        std.debug.print("  moe q2+q8 {s} / {s}: {d:8.1} us ({d:5.3}x wide.moe {d:.1} us)\n", .{ names[0], names[1], us, base / us, base });
    }
}
