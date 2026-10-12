//! glm_rows_check's MoE launches: the routed and shared experts' two-launch path (wide.moe) == its four launches.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const rc = @import("glm_rows_check.zig");

fn moeLaunch(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Moe, io: wide.MoeIo, n_rows: u32) void {
    wide.moe(p, e, m, 4, 8, 0.75, io, n_rows);
}

/// wide.moe (routed at q2..q8, shared at 8, two launches) == byIds GLU/down + glu/dense bit for bit; id 5 of 4: zeros.
pub fn check(g: *const rc.Gpu) !usize {
    const n = 256;
    const k = 512;
    const n_rows = 2;
    const slots = 8 * n_rows;
    var pairs: usize = 0;
    const sh = [2]rc.Mat{ try rc.Mat.init(g.device, 2 * n, k, 8, 64, 7300), try rc.Mat.init(g.device, k, n, 8, 64, 7301) };
    defer for (sh) |m| m.deinit();
    for (affine.row_bits) |bits| {
        const ms = [3]rc.Mat{ try rc.Mat.init(g.device, 4 * n, k, bits, 64, 7400 + @as(u64, bits)), try rc.Mat.init(g.device, 4 * n, k, bits, 64, 7500 + @as(u64, bits)), try rc.Mat.init(g.device, 4 * k, n, bits, 64, 7600 + @as(u64, bits)) };
        defer for (ms) |m| m.deinit();
        var q: [3]affine.Affine = undefined;
        for (&q, ms, [_]u32{ n, n, k }) |*x, m, r| {
            x.* = m.q;
            x.n = r;
        }
        const bs = [_]mtl.Buffer{ try rc.buf(g.device, slots * 4), try rc.buf(g.device, 2 * slots * n * 2), try rc.buf(g.device, 2 * n_rows * n * 2), try rc.buf(g.device, 2 * slots * k * 2), try rc.buf(g.device, 2 * n_rows * k * 2) };
        defer for (bs) |b| b.deinit();
        const picks = [slots]i32{ 2, 0, 3, 1, 1, 2, 0, 3, 3, 3, 0, 5, 1, 2, 2, 0 };
        @memcpy(bs[0].slice(i32, slots), &picks);
        const m: wide.Moe = .{ .gate = q[0], .up = q[1], .down = q[2], .sh_gate_up = sh[0].q, .sh_down = sh[1].q };
        if (!wide.moeOne(m) and bits != 4) return error.NotOneLaunch; // 4/64 routed keeps the generated gathers
        const io: wide.MoeIo = .{ .x = rc.ref(ms[0].x()), .ids = rc.ref(bs[0]), .act = rc.ref(bs[1]), .acts = rc.ref(bs[2]), .ye = rc.ref(bs[3]), .ys = rc.ref(bs[4]) };
        try g.run(moeLaunch, .{ &g.p, m, io, @as(u32, n_rows) });
        const half = [5]usize{ 0, slots * n, n_rows * n, slots * k, n_rows * k }; // the reference in each buffer's second half
        try g.run(rc.byIds, .{ &g.p, wide.Picks{ .w = q[0], .up = q[1], .experts = 4, .repeat = 8, .lim = 0.75 }, io.x, io.ids, io.act.at(half[1] * 2), @as(u32, slots) });
        try g.run(rc.gluLaunch, .{ &g.p, sh[0].q, @as(u32, n), io.x, io.acts.at(half[2] * 2), @as(u32, n_rows) });
        try g.run(rc.byIds, .{ &g.p, wide.Picks{ .w = q[2], .experts = 4, .repeat = 1 }, io.act, io.ids, io.ye.at(half[3] * 2), @as(u32, slots) });
        try g.run(rc.dense, .{ &g.p, sh[1].q, wide.Io{ .in = io.acts, .ldx = n, .out = io.ys.at(half[4] * 2), .ldo = k, .rows = n_rows } });
        for (bs[1..], half[1..]) |b, h| {
            const v = b.slice(u16, 2 * h);
            try rc.eql(v[0..h], v[h..]);
            pairs += h;
        }
    }
    std.debug.print("routed experts at q2..q8 and the 8-bit shared expert in two launches == byIds, glu and dense: {d} bit-equal values\n", .{pairs});
    return pairs;
}
