//! glm_rows_check's FP8 routed experts (E4M3, 128 x 128 f32 scales): launches agree bit for bit, near float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const rc = @import("glm_rows_check.zig");
const Ref = affine.Ref;

/// E4M3 code `c` as a value; 0x7F and 0xFF (NaN) are never drawn.
fn e4m3(c: u8) f64 {
    const e: i32 = (c >> 3) & 15;
    const m: f64 = @floatFromInt(c & 7);
    const v = if (e == 0) m / 8.0 * std.math.pow(f64, 2, -6) else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
    return if (c & 0x80 != 0) -v else v;
}

fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// A stack [E, n, k] of random codes (no NaN) and block scales in [2^-9, 2^-5].
const Stack = struct {
    codes: mtl.Buffer,
    scales: mtl.Buffer,
    q: affine.Affine,

    fn init(device: mtl.Device, E: usize, n: usize, k: usize, r: std.Random) !Stack {
        const codes = try rc.buf(device, E * n * k);
        errdefer codes.deinit();
        const sb = affine.fp8Scales(n, k).?;
        const scales = try rc.buf(device, E * sb);
        for (codes.slice(u8, E * n * k)) |*c| {
            c.* = r.int(u8);
            if (c.* & 0x7F == 0x7F) c.* ^= 1;
        }
        for (scales.slice(f32, E * sb / 4)) |*s| s.* = std.math.pow(f32, 2, -9 + 4 * r.float(f32));
        const w = rc.ref(codes);
        return .{ .codes = codes, .scales = scales, .q = .{ .w = w, .s = rc.ref(scales), .b = w, .n = @intCast(n), .k = @intCast(k), .bits = 8, .group = affine.fp8_block, .fp8 = true } };
    }

    fn deinit(s: Stack) void {
        s.codes.deinit();
        s.scales.deinit();
    }

    fn value(s: Stack, e: usize, row: usize, col: usize) f64 {
        const n: usize = s.q.n;
        const k: usize = s.q.k;
        const code = s.codes.slice(u8, (e + 1) * n * k)[(e * n + row) * k + col];
        const sb = affine.fp8Scales(n, k).? / 4;
        const scale = s.scales.slice(f32, (e + 1) * sb)[e * sb + (row / 128) * (k / 128) + col / 128];
        return e4m3(code) * scale;
    }
};

fn moeLaunch(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Moe, experts: u32, io: wide.MoeIo, rows: u32, u: ?wide.Unique) void {
    wide.moe(p, e, m, experts, 8, 0.75, io, rows, u);
}

fn sortedLaunch(e: mtl.ComputeEncoder, p: *const wide.Pipes, m: wide.Picks, x: Ref, offsets: Ref, order: Ref, out: Ref, ldo: u32, pairs: u32) void {
    wide.bySorted(p, e, m, x, offsets, order, out, ldo, pairs);
}

fn bfRows(device: mtl.Device, count: usize, r: std.Random) !mtl.Buffer {
    const b = try rc.buf(device, count * 2);
    for (b.slice(u16, count)) |*v| v.* = rc.toBf(r.float(f32) * 2 - 1);
    return b;
}

pub fn check(g: *const rc.Gpu) !usize {
    const n = 256; // gate/up rows, down's input
    const k = 512; // gate/up's input, down's rows
    const E = 16;
    const top = 8;
    var prng = std.Random.DefaultPrng.init(7330);
    const r = prng.random();
    const st = [3]Stack{ try Stack.init(g.device, E, n, k, r), try Stack.init(g.device, E, n, k, r), try Stack.init(g.device, E, k, n, r) };
    defer for (st) |s| s.deinit();
    const sh = [2]rc.Mat{ try rc.Mat.init(g.device, 2 * n, k, 8, 64, 7340), try rc.Mat.init(g.device, k, n, 8, 64, 7341) };
    defer for (sh) |mm| mm.deinit();
    const m: wide.Moe = .{ .gate = st[0].q, .up = st[1].q, .down = st[2].q, .sh_gate_up = sh[0].q, .sh_down = sh[1].q };
    if (!wide.moeOne(m)) return error.NotOneLaunch;
    var pairs: usize = 0;
    var worst: f64 = 0;
    for ([_]u32{ 1, 2, 4, 16 }) |rows| {
        const slots = top * rows;
        const x = try bfRows(g.device, rows * k, r);
        defer x.deinit();
        const bs = [_]mtl.Buffer{
            try rc.buf(g.device, slots * 4), // picks
            try rc.buf(g.device, 3 * slots * n * 2), // act: by expert, by pick, by ids
            try rc.buf(g.device, 2 * rows * n * 2), // acts
            try rc.buf(g.device, 3 * slots * k * 2), // ye: by expert, by pick, by ids
            try rc.buf(g.device, 2 * rows * k * 2), // ys
            try rc.buf(g.device, slots * 4), // unique ids
            try rc.buf(g.device, slots * wide.max_members * 4), // their picks
            try rc.buf(g.device, 4), // their count
            try rc.buf(g.device, 2 * slots * n * 2), // sorted gate/up out, then the act rows in sorted order
            try rc.buf(g.device, slots * k * 2), // sorted down out
            try rc.buf(g.device, E * 4), // offsets
            try rc.buf(g.device, slots * 4), // order
        };
        defer for (bs) |b| b.deinit();
        const picks = bs[0].slice(i32, slots);
        for (0..rows) |row| {
            var order: [E]i32 = undefined;
            for (&order, 0..) |*o, i| o.* = @intCast(i);
            r.shuffle(i32, &order);
            @memcpy(picks[row * top ..][0..top], order[0..top]);
        }
        const uids = bs[5].slice(i32, slots);
        const umem = bs[6].slice(i32, slots * wide.max_members);
        @memset(umem, -1);
        const offsets = bs[10].slice(i32, E);
        const order = bs[11].slice(i32, slots);
        var count: usize = 0;
        var at: usize = 0;
        for (0..E) |ex| { // experts ascending, each one's picks ascending: moe_route's order and the prompt's sort
            offsets[ex] = @intCast(at);
            var j: usize = 0;
            for (picks, 0..) |pk, pi| if (pk == @as(i32, @intCast(ex))) {
                umem[count * wide.max_members + j] = @intCast(pi);
                order[at] = @intCast(pi);
                j += 1;
                at += 1;
            };
            if (j == 0) continue;
            uids[count] = @intCast(ex);
            count += 1;
        }
        bs[7].slice(i32, 1)[0] = @intCast(count);
        const half = [5]usize{ 0, slots * n, rows * n, slots * k, rows * k };
        const io: wide.MoeIo = .{ .x = rc.ref(x), .ids = rc.ref(bs[0]), .act = rc.ref(bs[1]), .acts = rc.ref(bs[2]), .ye = rc.ref(bs[3]), .ys = rc.ref(bs[4]) };
        const by_pick: wide.MoeIo = .{ .x = io.x, .ids = io.ids, .act = io.act.at(half[1] * 2), .acts = io.acts.at(half[2] * 2), .ye = io.ye.at(half[3] * 2), .ys = io.ys.at(half[4] * 2) };
        const u: wide.Unique = .{ .ids = rc.ref(bs[5]), .members = rc.ref(bs[6]), .count = rc.ref(bs[7]) };
        if (rows >= 2) try g.run(moeLaunch, .{ &g.p, m, @as(u32, E), io, rows, u });
        try g.run(moeLaunch, .{ &g.p, m, @as(u32, E), by_pick, rows, null });
        if (rows >= 2) { // by unique expert == by pick
            for (bs[1..5], half[1..]) |b, h| {
                const v = b.slice(u16, 2 * h);
                try rc.eql(v[0..h], v[h..]);
                pairs += h;
            }
        }
        // by pick == byIds through the same experts (gate/up SwiGLU'd, then down from the by-pick act)
        const ids_act = io.act.at(2 * half[1] * 2);
        const ids_ye = io.ye.at(2 * half[3] * 2);
        try g.run(rc.byIds, .{ &g.p, wide.Picks{ .w = st[0].q, .up = st[1].q, .experts = E, .repeat = top, .lim = 0.75 }, io.x, io.ids, ids_act, @as(u32, slots) });
        try g.run(rc.byIds, .{ &g.p, wide.Picks{ .w = st[2].q, .experts = E, .repeat = 1 }, by_pick.act, io.ids, ids_ye, @as(u32, slots) });
        const act = bs[1].slice(u16, 3 * half[1]);
        const ye = bs[3].slice(u16, 3 * half[3]);
        try rc.eql(act[2 * half[1] ..], act[half[1] .. 2 * half[1]]);
        try rc.eql(ye[2 * half[3] ..], ye[half[3] .. 2 * half[3]]);
        pairs += half[1] + half[3];
        // sorted pairs == byIds: gate/up to pair i from x row order[i] / top, down from act row i to order[i]
        try g.run(sortedLaunch, .{ &g.p, wide.Picks{ .w = st[0].q, .up = st[1].q, .experts = E, .repeat = top, .lim = 0.75 }, io.x, rc.ref(bs[10]), rc.ref(bs[11]), rc.ref(bs[8]), @as(u32, n), @as(u32, slots) });
        const sorted = bs[8].slice(u16, 2 * slots * n);
        for (0..slots) |i| {
            const o: usize = @intCast(order[i]);
            try rc.eql(sorted[i * n ..][0..n], act[half[1] + o * n ..][0..n]);
            @memcpy(sorted[slots * n + i * n ..][0..n], act[half[1] + o * n ..][0..n]); // by-pick act rows, sorted
        }
        try g.run(sortedLaunch, .{ &g.p, wide.Picks{ .w = st[2].q, .experts = E, .repeat = 1 }, rc.ref(bs[8]).at(slots * n * 2), rc.ref(bs[10]), rc.ref(bs[11]), rc.ref(bs[9]), @as(u32, k), @as(u32, slots) });
        try rc.eql(bs[9].slice(u16, slots * k), ye[half[3] .. 2 * half[3]]);
        pairs += 2 * slots * n / 2 + slots * k;
        // byIds down against float64 of the decoded codes: within the bf16 output's rounding and fp32 sums
        for (0..slots) |s| {
            const e: usize = @intCast(picks[s]);
            for (0..k) |row| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (0..n) |c| {
                    const t = fromBf(act[half[1] + s * n + c]) * st[2].value(e, row, c);
                    want += t;
                    mag += @abs(t);
                }
                const got = fromBf(ye[2 * half[3] + s * k + row]);
                const bound = @abs(want) * std.math.pow(f64, 2, -8) + mag * 1e-6 + 1e-30;
                worst = @max(worst, @abs(got - want) / bound);
                if (@abs(got - want) > bound) {
                    std.debug.print("fp8 down slot {d} row {d}: {d} vs float64 {d}\n", .{ s, row, got, want });
                    return error.OutOfBound;
                }
            }
        }
    }
    std.debug.print("FP8 routed experts by unique expert, by pick, by ids and sorted pairs at 1, 2, 4 and 16 rows: " ++
        "{d} bit-equal values; byIds within bf16 of float64 (worst err/bound {d:.4})\n", .{ pairs, worst });
    return pairs;
}
