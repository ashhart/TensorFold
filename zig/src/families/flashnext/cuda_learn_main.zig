//! `zig build flashnext-learn`: Flash Next's trainer on the GPU against the serving forward and finite differences.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const prompt = flash.prompt;
const batch = prompt.batch;
const train = flash.train;
const Trainer = train.Trainer(flash.Tri);
const dims = 2560;
const layers_n = 48;
const slots = 11;

fn fetch(d: *const cuda.Driver, gpa: std.mem.Allocator, ptr: u64, n: usize) ![]u8 {
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    try d.check(d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n), "cuMemcpyDtoH");
    return out;
}

/// The serving forward over `ids` against the trainer's keys and loss passes: MoE inputs, gates and scores, by bytes.
fn identity(t: *Trainer, e: *prompt.Engine, tri: flash.Tri, gpa: std.mem.Allocator, ids: []const u32, what: []const u8) !bool {
    const d = e.driver;
    const n = ids.len - 1;
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpa);
    }
    var keep: batch.Keep = .{};
    for (&keep.layers) |*k| {
        try bufs.append(gpa, try cuda.DeviceBuffer.alloc(d, n * dims * 2));
        k.moe_x = bufs.items[bufs.items.len - 1].ptr;
        try bufs.append(gpa, try cuda.DeviceBuffer.alloc(d, n * slots * 4));
        k.moe_wts = bufs.items[bufs.items.len - 1].ptr;
    }
    try bufs.append(gpa, try cuda.DeviceBuffer.alloc(d, n * prompt.vocabulary * 2));
    const logits = bufs.items[bufs.items.len - 1].ptr;
    var seq = try e.newSeq(n + 8);
    defer seq.deinit(gpa);
    try e.forward(flash.Tri, tri, &seq, ids[0..n], .{ .logits = .all, .logits_out = logits, .keep = &keep });
    try e.stream.synchronize();
    const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };
    const sq: train.Seq = .{ .ids = ids, .start = n - 4, .decide = n - 5 };
    var r: [1]train.Result = undefined;
    try t.pass(&.{sq}, .keys, turn, &r);
    var inputs_ok = true;
    var gates_ok = true;
    for (0..layers_n) |l| {
        const x = try fetch(d, gpa, keep.layers[l].moe_x, n * dims * 2);
        defer gpa.free(x);
        const w = try fetch(d, gpa, keep.layers[l].moe_wts, n * slots * 4);
        defer gpa.free(w);
        for (0..n) |row| {
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(t.normed(l, row)), x[row * dims * 2 ..][0 .. dims * 2])) inputs_ok = false;
            const g: f32 = @bitCast(std.mem.readInt(u32, w[(row * slots + slots - 1) * 4 ..][0..4], .little));
            if (@as(u32, @bitCast(g)) != @as(u32, @bitCast(t.gateAt(l, row)))) gates_ok = false;
        }
    }
    try t.scores(sq);
    const want = try fetch(d, gpa, logits, n * prompt.vocabulary * 2);
    defer gpa.free(want);
    const got = try fetch(d, gpa, t.b.logits, n * prompt.vocabulary * 2);
    defer gpa.free(got);
    const rows_ok = std.mem.eql(u8, want, got);
    const ok = inputs_ok and gates_ok and rows_ok;
    std.debug.print("{s} {s}: the trainer's pass against the serving forward over {d} rows: MoE inputs {s}, shared gates {s} at all 48 layers, every row's scores {s}\n", .{ if (ok) "PASS" else "FAIL", what, n, if (inputs_ok) "byte-equal" else "DIFFER", if (gates_ok) "byte-equal" else "DIFFER", if (rows_ok) "byte-equal" else "DIFFER" });
    return ok;
}

/// `n` numbers from the comma list in env `name`, else `default`.
fn envList(comptime n: usize, name: [:0]const u8, default: [n]usize) [n]usize {
    const v = std.c.getenv(name) orelse return default;
    var out = default;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
    for (&out) |*o| o.* = std.fmt.parseInt(usize, it.next() orelse break, 10) catch return default;
    return out;
}

/// Rows before the last keep their scores byte for byte when the last row's token changes (`n` rows, odd or even).
fn causal(e: *prompt.Engine, tri: flash.Tri, gpa: std.mem.Allocator, ids: []const u32) !bool {
    const d = e.driver;
    const n = ids.len;
    var buf = try cuda.DeviceBuffer.alloc(d, 2 * n * prompt.vocabulary * 2);
    defer buf.free();
    const other = try gpa.dupe(u32, ids);
    defer gpa.free(other);
    other[n - 1] = if (ids[n - 1] == 11) 13 else 11;
    for ([_][]const u32{ ids, other }, 0..) |x, i| {
        var seq = try e.newSeq(n + 8);
        defer seq.deinit(gpa);
        try e.forward(flash.Tri, tri, &seq, x, .{ .logits = .all, .logits_out = buf.ptr + i * n * prompt.vocabulary * 2 });
        try e.stream.synchronize();
    }
    const a = try fetch(d, gpa, buf.ptr, (n - 1) * prompt.vocabulary * 2);
    defer gpa.free(a);
    const b = try fetch(d, gpa, buf.ptr + n * prompt.vocabulary * 2, (n - 1) * prompt.vocabulary * 2);
    defer gpa.free(b);
    const ok = std.mem.eql(u8, a, b);
    std.debug.print("{s} causal over {d} rows: the last token changed, the {d} rows before it keep their scores {s}\n", .{ if (ok) "PASS" else "FAIL", n, n - 1, if (ok) "byte for byte" else "NOT byte for byte" });
    return ok;
}

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}

fn dot(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += @as(f64, x) * y;
    return s;
}

fn norm16(xs: []const u16) f64 {
    var s: f64 = 0;
    for (xs) |b| {
        const v: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
        s += v * v;
    }
    return @sqrt(s);
}

/// The loss's slope as sequence `sq`'s key-k nudge at layer l moves along `dir`: least squares through +-h .. +-6h.
fn slope(t: *Trainer, sq: *const train.Seq, k: usize, l: usize, dir: []const f32, h: f64) !f64 {
    const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };
    var num: f64 = 0;
    var den: f64 = 0;
    for (1..7) |m| for ([_]f64{ 1, -1 }) |sgn| {
        const x = sgn * @as(f64, @floatFromInt(m)) * h;
        for (t.deltaAt(0, k, l), dir) |*o, u| o.* = @floatCast(x * u);
        var r: [1]train.Result = undefined;
        try t.pass(&.{sq.*}, .loss, turn, &r);
        if (std.c.getenv("FN_TRACE") != null) std.debug.print("    h {d:.4} loss {d:.5}\n", .{ x, r[0].loss });
        num += x * r[0].loss;
        den += x * x;
    };
    @memset(t.deltaAt(0, k, l), 0);
    return num / den;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: flashnext-learn <model-dir> <ids-file>\n", .{});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
    // the first 256 tokens a sequence; the windows of (2) slide over the rest
    const all = ids.items;
    const n = @min(all.len, 256);
    if (n < 16) return 2;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var glue = try flash.Glue.load(&driver);
    defer glue.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const tri: flash.Tri = .{ .own = &glue, .s = stream };
    const l0 = now(io);
    var e = try prompt.Engine.init(gpa, io, &driver, &stream, args[1]);
    defer e.deinit();
    _ = try e.reserve(train.pack_rows);
    const l1 = now(io);
    const t = try Trainer.init(gpa, io, e, tri);
    defer t.deinit();
    const l2 = now(io);
    const use = cuda.usage(false);
    std.debug.print("engine loaded in {d:.1} s, trainer in {d:.1} s; device bytes held {d:.2} GiB\n", .{ l1 - l0, l2 - l1, @as(f64, @floatFromInt(use.device)) / (1 << 30) });
    var failures: usize = 0;
    const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };

    // (0) the trainer's forward is the serving one, untaught and with a pocket neuron at layer 31
    const seq_ids = all[0..n];
    // FN_ONLY=grad runs the finite differences alone
    const only_grad = std.c.getenv("FN_ONLY") != null;
    for ([_]usize{ 255, 254 }) |len| {
        if (!try causal(e, tri, gpa, seq_ids[0..@min(n, len)])) failures += 1;
    }
    if (!only_grad and !try identity(t, e, tri, gpa, seq_ids[0..@min(n, 120)], "untaught")) failures += 1;
    if (!only_grad) {
        // the 64 neurons section (3) uses, the first live
        const g = try gpa.alloc(f32, 64 * dims);
        defer gpa.free(g);
        const u = try gpa.alloc(f32, 64 * dims);
        defer gpa.free(u);
        const o = try gpa.alloc(f32, 64 * dims);
        defer gpa.free(o);
        @memset(g, 0);
        @memset(u, 0);
        @memset(o, 0);
        var prng0 = std.Random.DefaultPrng.init(31);
        for (g[0..dims], u[0..dims], o[0..dims]) |*a, *b, *c| {
            a.* = prng0.random().floatNorm(f32) * 0.05;
            b.* = prng0.random().floatNorm(f32) * 0.05;
            c.* = prng0.random().floatNorm(f32) * 0.02;
        }
        try e.setPocket(31, g, u, o, 1);
        if (!try identity(t, e, tri, gpa, seq_ids[0..@min(n, 120)], "a pocket neuron at layer 31")) failures += 1;
        @memset(g, 0);
        @memset(u, 0);
        @memset(o, 0);
        try e.setPocket(31, g, u, o, 0);
    }

    // the sequence: the prompt with its last tokens as the answer (FN_ANSWER of them, 8 by default)
    const answer: usize = if (std.c.getenv("FN_ANSWER")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 8 else 8;
    const start = n - @min(answer, n / 2);

    // (1) packed sequences: each one's loss as alone (a pass's rows are each sequence's own)
    if (!only_grad) {
        var seqs: [6]train.Seq = undefined;
        var alone: [6]f32 = undefined;
        for (&seqs, 0..) |*s, i| {
            const len = @min(n, 120) - 3 * i; // odd and even lengths, so some sequences sit after a gap row
            s.* = .{ .ids = seq_ids[0..len], .start = len - 6, .decide = len - 7 };
            var one: [1]train.Result = undefined;
            try t.pass(seqs[i .. i + 1], .loss, turn, &one);
            alone[i] = one[0].loss;
        }
        var got: [6]train.Result = undefined;
        const p0 = now(io);
        try t.pass(&seqs, .loss, turn, &got);
        const p1 = now(io);
        var worst: f32 = 0;
        for (got, alone) |g, a| worst = @max(worst, @abs(g.loss - a));
        const ok = worst <= 1e-4;
        std.debug.print("{s} six sequences side by side ({d} rows in {d:.0} ms): each loss as alone within {e} (alone: {d:.4} {d:.4} {d:.4} ...)\n", .{ if (ok) "PASS" else "FAIL", t.rows, (p1 - p0) * 1e3, worst, alone[0], alone[1], alone[2] });
        if (!ok) failures += 1;
    }

    // (2) push against finite differences in FN_WINDOWS (4) windows of the prompt, each at four layers (FN_LAYERS)
    const layers = envList(4, "FN_LAYERS", .{ train.keep_low, 23, 31, layers_n - 1 });
    const windows = @min(envList(1, "FN_WINDOWS", .{4})[0], all.len - n + 1);
    if (windows == 0) return error.EmptyGradientSweep;
    for (layers) |layer| if (layer < train.keep_low or layer >= layers_n) return error.UnsupportedGradientLayer;
    // steps of 0.4% and 3% of the residual's size: the rounding decides below the first, the loss's curvature above
    const fracs = [_]f64{ 0.004, 0.03 };
    var errs: [2][2]std.ArrayList(f64) = .{ .{ .empty, .empty }, .{ .empty, .empty } }; // [direct, cross][step]
    defer for (&errs) |*a| for (a) |*b| b.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(7);
    for (0..windows) |w| for ([_]usize{ 0, start - 1, start - 5 }) |cross_row| {
        // the last row (its nudge reaches only its own loss: the direct path) and a row before the answer (cross-row)
        const rows = [_]usize{ n - 2, cross_row };
        var sq: train.Seq = .{ .ids = all[w..][0..n], .start = start, .decide = start - 1 };
        var mask: u64 = 0;
        for (layers) |l| mask |= @as(u64, 1) << @intCast(l);
        sq.nudge = .{ .layers = .{ mask, mask, 0 }, .rows = .{ rows[0], rows[1], 0 } };
        for (layers) |l| for (0..2) |k| @memset(t.deltaAt(0, k, l), 0);
        var out: [1]train.Result = undefined;
        const q0 = now(io);
        try t.pass(&.{sq}, .target, turn, &out);
        const q1 = now(io);
        std.debug.print("window {d}: target pass over {d} rows, loss {d:.5}, back to layer {d} in {d:.0} ms\n", .{ w, n - 1, out[0].loss, layers[0], (q1 - q0) * 1e3 });
        const push = try gpa.alloc(f32, layers.len * 2 * dims);
        defer gpa.free(push);
        for (layers, 0..) |l, li| for (0..2) |k| @memcpy(push[(li * 2 + k) * dims ..][0..dims], t.pushAt(0, k, l));
        for (layers, 0..) |l, li| for (0..2) |k| {
            const p = push[(li * 2 + k) * dims ..][0..dims];
            const pn = @sqrt(dot(p, p));
            if (!std.math.isFinite(pn) or pn == 0) {
                failures += 1;
                continue;
            }
            const rn = norm16(t.residual(l + 1, rows[k]));
            var dir: [dims]f32 = undefined;
            for (0..2) |which| {
                // along the push, then half along it and half along a random direction
                var r: [dims]f32 = undefined;
                var ss: f64 = 0;
                for (&r) |*x| {
                    x.* = prng.random().floatNorm(f32);
                    ss += @as(f64, x.*) * x.*;
                }
                for (&dir, p, r) |*x, v, q| x.* = @floatCast(if (which == 0) v / pn else (v / pn + q / @sqrt(ss)) / std.math.sqrt2);
                const want = dot(p, &dir);
                for (fracs, 0..) |frac, fi| {
                    const fd = try slope(t, &sq, k, l, &dir, frac * @max(rn, 1.0));
                    const rel = @abs(fd - want) / @max(@abs(want), 1e-30);
                    try errs[k][fi].append(gpa, rel);
                    std.debug.print("  layer {d} row {d} ({s}, {s}, step {d:.1}% of |residual| {d:.1}): push . u {e:.4}, finite differences' slope {e:.4}, relative error {d:.4}\n", .{ l, rows[k], if (k == 0) "direct" else "cross-row", if (which == 0) "along push" else "half random", frac * 100, rn, want, fd, rel });
                }
            }
        };
    };
    var stat: [2][2][2]f64 = undefined; // [path][step][median, worst]
    for (&errs, &stat) |*a, *sa| for (a, sa) |*b, *sb| {
        if (b.items.len == 0) return error.EmptyGradientSweep;
        for (b.items) |value| if (!std.math.isFinite(value)) return error.NonFiniteGradient;
        std.mem.sort(f64, b.items, {}, std.sort.asc(f64));
        sb.* = .{ b.items[b.items.len / 2], b.items[b.items.len - 1] };
    };
    // Both paths and steps now gate every sample, including first and pre-answer rows.
    const grad_ok = flash.gradient_gate.accepts(stat);
    std.debug.print("{s} push against finite differences, {d} windows, {d} checks a path and step: direct path median relative error {d:.4} (worst {d:.4}) at 0.4% steps, {d:.4} ({d:.4}) at 3%; cross-row paths {d:.4} ({d:.4}) at 0.4%, {d:.4} ({d:.4}) at 3%\n", .{ if (grad_ok) "PASS" else "FAIL", windows, errs[0][0].items.len, stat[0][0][0], stat[0][0][1], stat[0][1][0], stat[0][1][1], stat[1][0][0], stat[1][0][1], stat[1][1][0], stat[1][1][1] });
    if (!grad_ok) failures += 1;
    var sq: train.Seq = .{ .ids = seq_ids, .start = start, .decide = start - 1 };
    const rows = [_]usize{ n - 2, start - 5 };

    // (3) a pocket neuron at layer 31: its output's gradient (gate act push) against the loss moving with it
    if (!only_grad) {
        const l: usize = 31;
        const r = rows[0];
        var keys: [1]train.Result = undefined;
        sq.nudge = .{};
        try t.pass(&.{sq}, .keys, turn, &keys);
        // input weights that read 2 on this row and little elsewhere: the row less the sequence's mean row, scaled
        var z: [dims]f64 = @splat(0);
        const rows_n = sq.ids.len - 1;
        for (0..rows_n) |j| {
            if (j == r) continue;
            for (t.normed(l, j), 0..) |b, i| z[i] -= @as(f64, @as(f32, @bitCast(@as(u32, b) << 16))) / @as(f64, @floatFromInt(rows_n - 1));
        }
        const x = t.normed(l, r);
        var zx: f64 = 0;
        for (x, 0..) |b, i| {
            z[i] += @as(f32, @bitCast(@as(u32, b) << 16));
            zx += z[i] * @as(f32, @bitCast(@as(u32, b) << 16));
        }
        var u: [dims]f32 = undefined;
        for (&u, z) |*o, v| o.* = @floatCast(2 * v / zx);
        var most: f64 = -1e30;
        for (0..rows_n) |j| {
            if (j == r) continue;
            var d: f64 = 0;
            for (t.normed(l, j), u) |b, w| d += @as(f64, @as(f32, @bitCast(@as(u32, b) << 16))) * w;
            most = @max(most, d);
        }
        std.debug.print("  pocket neuron: u x = 2 on row {d}, at most {d:.3} on the sequence's other rows\n", .{ r, most });
        const neurons = 64;
        const gate = try gpa.alloc(f32, neurons * dims);
        defer gpa.free(gate);
        const upr = try gpa.alloc(f32, neurons * dims);
        defer gpa.free(upr);
        const down = try gpa.alloc(f32, neurons * dims);
        defer gpa.free(down);
        @memset(gate, 0);
        @memset(upr, 0);
        @memset(down, 0);
        for (0..dims) |i| {
            gate[i] = 16 * u[i];
            upr[i] = u[i] / 16;
        }
        try e.setPocket(l, gate, upr, down, 1);
        sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(l), 0, 0 }, .rows = .{ r, 0, 0 } };
        @memset(t.deltaAt(0, 0, l), 0);
        var got: [1]train.Result = undefined;
        try t.pass(&.{sq}, .target, turn, &got);
        const xr = t.normed(l, r);
        var zg: f64 = 0;
        var zu: f64 = 0;
        for (xr, 0..) |b, i| {
            const v: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
            zg += gate[i] * v;
            zu += upr[i] * v;
        }
        const act = zg / (1 + @exp(-zg)) * zu;
        const g = t.gateAt(l, r);
        const pg = t.pushAt(0, 0, l);
        // the neuron's output along the push, by 1e-2 of the residual's size, and back
        var pn: f64 = 0;
        for (pg) |v| pn += @as(f64, v) * v;
        pn = @sqrt(pn);
        const resid = t.residual(l + 1, r);
        var rn: f64 = 0;
        for (resid) |b| {
            const v: f64 = @as(f32, @bitCast(@as(u32, b) << 16));
            rn += v * v;
        }
        rn = @sqrt(rn);
        sq.nudge = .{};
        var fds: [fracs.len]f64 = undefined;
        for (fracs, &fds) |frac, *fd_out| {
            const step = frac * rn / @max(g * act, 1e-6);
            var num: f64 = 0;
            var den: f64 = 0;
            for (1..7) |m| for ([_]f64{ 1, -1 }) |sgn| {
                const xstep = sgn * @as(f64, @floatFromInt(m)) * step;
                for (down[0..dims], pg) |*o, v| o.* = @floatCast(xstep * v / pn);
                try e.setPocket(l, gate, upr, down, 1);
                var rr: [1]train.Result = undefined;
                try t.pass(&.{sq}, .loss, turn, &rr);
                num += xstep * rr[0].loss;
                den += xstep * xstep;
            };
            fd_out.* = num / den;
        }
        const fd = fds[1];
        {
            // the forward's change at that row after layer l, against g act v (as the learner's verifyForward reads it)
            for (down[0..dims], pg) |*o, v| o.* = @floatCast(rn * v / pn);
            try e.setPocket(l, gate, upr, down, 1);
            var rr: [1]train.Result = undefined;
            try t.pass(&.{sq}, .target, turn, &rr);
            var with: [dims]f32 = undefined;
            for (&with, t.residual(l + 1, r)) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
            @memset(down, 0);
            try e.setPocket(l, gate, upr, down, 1);
            try t.pass(&.{sq}, .target, turn, &rr);
            var dd: f64 = 0;
            var dw: f64 = 0;
            for (&with, t.residual(l + 1, r), pg) |w, b, v| {
                const delta = w - @as(f32, @bitCast(@as(u32, b) << 16));
                dd += delta * delta;
                dw += delta * v / pn;
            }
            const moved = g * act * rn;
            const ok = @abs(@sqrt(dd) - moved) <= 0.02 * moved and @abs(dw - moved) <= 0.02 * moved;
            std.debug.print("{s} a pocket neuron at layer {d} in the forward: the residual moves {d:.3} (along v {d:.3}); gate act |v| = {d:.3}\n", .{ if (ok) "PASS" else "FAIL", l, @sqrt(dd), dw, moved });
            if (!ok) failures += 1;
        }
        // reported: the neuron fires a little on the sequence's other rows too, which its row's gradient leaves out
        const want = g * act * pn;
        std.debug.print("  its output's gradient at its row (gate {d:.3}, activation {d:.3}): gate act |push| {e:.4}; finite difference of the loss as it moves {e:.4} at 3% steps, {e:.4} at 0.4%\n", .{ g, act, want, fd, fds[0] });
        @memset(down, 0);
        @memset(gate, 0);
        @memset(upr, 0);
        try e.setPocket(l, gate, upr, down, 0);
    }

    // (4) a full pass's time: sixteen sequences, keys then loss then target down to layer 12
    if (!only_grad) {
        var seqs: [16]train.Seq = undefined;
        var rowsum: usize = 0;
        for (&seqs, 0..) |*s, i| {
            const len = @min(n, 40 + 3 * i);
            rowsum += len - 1;
            s.* = .{ .ids = seq_ids[0..len], .start = len - 6, .decide = len - 7, .nudge = .{ .layers = .{ @as(u64, 1) << 12 | @as(u64, 1) << 30, 0, 0 }, .rows = .{ len - 7, 0, 0 } } };
            @memset(t.deltaAt(i, 0, 12), 0);
            @memset(t.deltaAt(i, 0, 30), 0);
        }
        var got: [16]train.Result = undefined;
        for ([_]train.Mode{ .keys, .loss, .target }) |m| {
            const a = now(io);
            try t.pass(&seqs, m, turn, &got);
            const b = now(io);
            std.debug.print("pass {s}: 16 sequences, {d} rows, {d:.0} ms\n", .{ @tagName(m), rowsum, (b - a) * 1e3 });
        }
    }
    const peak = cuda.usage(false);
    std.debug.print("device bytes held at the end {d:.2} GiB, peak {d:.2} GiB\n", .{ @as(f64, @floatFromInt(peak.device)) / (1 << 30), @as(f64, @floatFromInt(peak.peak)) / (1 << 30) });
    std.debug.print("{d} failures\n", .{failures});
    return if (failures == 0) 0 else 1;
}
