//! glm_rows_check's MoE launches: the routed and shared experts' two-launch path (wide.moe) == its four launches.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const rc = @import("glm_rows_check.zig");

fn moeLaunch(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Moe, io: wide.MoeIo, n_rows: u32) void {
    wide.moe(p, e, m, 4, 8, 0.75, io, n_rows, null);
}

fn moeLaunchBy(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Moe, experts: u32, io: wide.MoeIo, n_rows: u32, u: ?wide.Unique) void {
    wide.moe(p, e, m, experts, 8, 0.75, io, n_rows, u);
}

/// wide.moe (routed at q2..q8, shared at 8, two launches) == byIds GLU/down + glu/dense bit for bit; id 5 of 4: zeros.
pub fn check(g: *const rc.Gpu) !usize {
    const n = 256;
    const k = 512;
    const n_rows = 2;
    const slots = 8 * n_rows;
    var pairs: usize = 0;
    const sh = [2]rc.Mat{
        try rc.Mat.init(g.device, 2 * n, k, 8, 64, 7300),
        try rc.Mat.init(g.device, k, n, 8, 64, 7301),
    };
    defer for (sh) |m| m.deinit();
    for (affine.row_bits) |bits| {
        const ms = [3]rc.Mat{
            try rc.Mat.init(g.device, 4 * n, k, bits, 64, 7400 + @as(u64, bits)),
            try rc.Mat.init(g.device, 4 * n, k, bits, 64, 7500 + @as(u64, bits)),
            try rc.Mat.init(g.device, 4 * k, n, bits, 64, 7600 + @as(u64, bits)),
        };
        defer for (ms) |m| m.deinit();
        var q: [3]affine.Affine = undefined;
        for (&q, ms, [_]u32{ n, n, k }) |*x, m, r| {
            x.* = m.q;
            x.n = r;
        }
        const bs = [_]mtl.Buffer{
            try rc.buf(g.device, slots * 4),
            try rc.buf(g.device, 2 * slots * n * 2),
            try rc.buf(g.device, 2 * n_rows * n * 2),
            try rc.buf(g.device, 2 * slots * k * 2),
            try rc.buf(g.device, 2 * n_rows * k * 2),
        };
        defer for (bs) |b| b.deinit();
        const picks = [slots]i32{ 2, 0, 3, 1, 1, 2, 0, 3, 3, 3, 0, 5, 1, 2, 2, 0 };
        @memcpy(bs[0].slice(i32, slots), &picks);
        const m: wide.Moe = .{ .gate = q[0], .up = q[1], .down = q[2], .sh_gate_up = sh[0].q, .sh_down = sh[1].q };
        if (!wide.moeOne(m) and bits != 4) return error.NotOneLaunch; // 4/64 routed keeps the generated gathers
        const io: wide.MoeIo = .{
            .x = rc.ref(ms[0].x()),
            .ids = rc.ref(bs[0]),
            .act = rc.ref(bs[1]),
            .acts = rc.ref(bs[2]),
            .ye = rc.ref(bs[3]),
            .ys = rc.ref(bs[4]),
        };
        try g.run(moeLaunch, .{ &g.p, m, io, @as(u32, n_rows) });
        const half = [5]usize{
            0,
            slots * n,
            n_rows * n,
            slots * k,
            n_rows * k,
        }; // the reference in each buffer's second half
        try g.run(rc.byIds, .{
            &g.p,
            wide.Picks{ .w = q[0], .up = q[1], .experts = 4, .repeat = 8, .lim = 0.75 },
            io.x,
            io.ids,
            io.act.at(half[1] * 2),
            @as(u32, slots),
        });
        try g.run(rc.gluLaunch, .{ &g.p, sh[0].q, @as(u32, n), io.x, io.acts.at(half[2] * 2), @as(u32, n_rows) });
        try g.run(rc.byIds, .{
            &g.p,
            wide.Picks{ .w = q[2], .experts = 4, .repeat = 1 },
            io.act,
            io.ids,
            io.ye.at(half[3] * 2),
            @as(u32, slots),
        });
        try g.run(rc.dense, .{
            &g.p,
            sh[1].q,
            wide.Io{ .in = io.acts, .ldx = n, .out = io.ys.at(half[4] * 2), .ldo = k, .rows = n_rows },
        });
        for (bs[1..], half[1..]) |b, h| {
            const v = b.slice(u16, 2 * h);
            try rc.eql(v[0..h], v[h..]);
            pairs += h;
        }
    }
    std.debug.print(
        "routed experts at q2..q8 and the 8-bit shared expert in two launches == byIds, glu and dense: " ++
            "{d} bit-equal values\n",
        .{pairs},
    );
    return pairs;
}

/// wide.moe by unique expert == by pick, bit for bit: 2, 4 and 16 rows of 8 distinct picks of 16 experts, q8, q4, q2.
pub fn checkUnique(g: *const rc.Gpu) !usize {
    const n = 256;
    const k = 512;
    const E = 16;
    const top = 8;
    var pairs: usize = 0;
    const sh = [2]rc.Mat{
        try rc.Mat.init(g.device, 2 * n, k, 8, 64, 7310),
        try rc.Mat.init(g.device, k, n, 8, 64, 7311),
    };
    defer for (sh) |mm| mm.deinit();
    var prng = std.Random.DefaultPrng.init(7320);
    const r = prng.random();
    for ([_]u8{ 8, 4, 2 }) |bits| {
        const ms = [3]rc.Mat{
            try rc.Mat.init(g.device, E * n, k, bits, 64, 7410 + @as(u64, bits)),
            try rc.Mat.init(g.device, E * n, k, bits, 64, 7510 + @as(u64, bits)),
            try rc.Mat.init(g.device, E * k, n, bits, 64, 7610 + @as(u64, bits)),
        };
        defer for (ms) |mm| mm.deinit();
        var q: [3]affine.Affine = undefined;
        for (&q, ms, [_]u32{ n, n, k }) |*x, mm, rr| {
            x.* = mm.q;
            x.n = rr;
        }
        const m: wide.Moe = .{ .gate = q[0], .up = q[1], .down = q[2], .sh_gate_up = sh[0].q, .sh_down = sh[1].q };
        for ([_]u32{ 2, 4, 16 }) |n_rows| {
            const slots = top * n_rows;
            const bs = [_]mtl.Buffer{
                try rc.buf(g.device, slots * 4), // picks
                try rc.buf(g.device, 2 * slots * n * 2), // act: by expert, then by pick
                try rc.buf(g.device, 2 * n_rows * n * 2), // acts
                try rc.buf(g.device, 2 * slots * k * 2), // ye
                try rc.buf(g.device, 2 * n_rows * k * 2), // ys
                try rc.buf(g.device, slots * 4), // unique ids
                try rc.buf(g.device, slots * wide.max_members * 4), // their picks
                try rc.buf(g.device, 4), // their count
            };
            defer for (bs) |b| b.deinit();
            const picks = bs[0].slice(i32, slots);
            for (0..n_rows) |row| { // a row's 8 distinct experts, as the route picks them
                var order: [E]i32 = undefined;
                for (&order, 0..) |*o, i| o.* = @intCast(i);
                r.shuffle(i32, &order);
                @memcpy(picks[row * top ..][0..top], order[0..top]);
            }
            const uids = bs[5].slice(i32, slots);
            const umem = bs[6].slice(i32, slots * wide.max_members);
            @memset(umem, -1);
            var count: usize = 0;
            for (0..E) |ex| { // moe_route's order: experts ascending, each one's picks ascending
                var j: usize = 0;
                for (picks, 0..) |pk, pi| if (pk == @as(i32, @intCast(ex))) {
                    umem[count * wide.max_members + j] = @intCast(pi);
                    j += 1;
                };
                if (j == 0) continue;
                uids[count] = @intCast(ex);
                count += 1;
            }
            bs[7].slice(i32, 1)[0] = @intCast(count);
            const half = [5]usize{ 0, slots * n, n_rows * n, slots * k, n_rows * k };
            const io: wide.MoeIo = .{ .x = rc.ref(ms[0].x()), .ids = rc.ref(bs[0]), .act = rc.ref(bs[1]), .acts = rc.ref(bs[2]), .ye = rc.ref(bs[3]), .ys = rc.ref(bs[4]) };
            const by_pick: wide.MoeIo = .{ .x = io.x, .ids = io.ids, .act = io.act.at(half[1] * 2), .acts = io.acts.at(half[2] * 2), .ye = io.ye.at(half[3] * 2), .ys = io.ys.at(half[4] * 2) };
            const u: wide.Unique = .{ .ids = rc.ref(bs[5]), .members = rc.ref(bs[6]), .count = rc.ref(bs[7]) };
            try g.run(moeLaunchBy, .{ &g.p, m, @as(u32, E), io, n_rows, u });
            try g.run(moeLaunchBy, .{ &g.p, m, @as(u32, E), by_pick, n_rows, null });
            for (bs[1..5], half[1..]) |b, h| {
                const v = b.slice(u16, 2 * h);
                try rc.eql(v[0..h], v[h..]);
                pairs += h;
            }
        }
    }
    std.debug.print("routed experts by unique expert == by pick at q8, q4 and q2, 2, 4 and 16 rows: {d} bit-equal values\n", .{pairs});
    return pairs;
}
