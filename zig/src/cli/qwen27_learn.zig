//! tf-qwen27-learn MODEL KERNELS IDS_JSON: the 27B's trainer on the GPU (packing, push vs differences, a pocket).
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen27");
const qwen27_taps = @import("qwen27_taps.zig");
const gates = qwen27_taps.gates;
const scores = qwen27_taps.scores;

const train = qwen.train;
const Trainer = train.Trainer;
const D = 5120;
const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}

fn dot(a: []const f32, b: []const f32) f64 {
    var s: f64 = 0;
    for (a, b) |x, y| s += @as(f64, x) * y;
    return s;
}

pub fn bf(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

fn norm16(xs: []const u16) f64 {
    var s: f64 = 0;
    for (xs) |b| s += bf(b) * bf(b);
    return @sqrt(s);
}

/// The loss's slope along `dir` at sequence sq's key-k nudge of layer l: a least-squares line through +-h .. +-6h.
fn slope(t: *Trainer, sq: *const train.Seq, k: usize, l: usize, dir: []const f32, h: f64) !f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (1..7) |m| for ([_]f64{ 1, -1 }) |sgn| {
        const x = sgn * @as(f64, @floatFromInt(m)) * h;
        for (t.deltaAt(0, k, l), dir) |*o, u| o.* = @floatCast(x * u);
        var r: [1]train.Result = undefined;
        try t.pass(&.{sq.*}, .loss, turn, &r);
        num += x * r[0].loss;
        den += x * x;
    };
    @memset(t.deltaAt(0, k, l), 0);
    return num / den;
}

fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    const arr = switch (doc.value) {
        .array => |a| a,
        .object => |o| (o.get("code") orelse return error.BadIds).array,
        else => return error.BadIds,
    };
    const out = try gpa.alloc(u32, arr.items.len);
    for (out, arr.items) |*o, v| o.* = @intCast(v.integer);
    return out;
}

fn greedyVsPass(gpa: std.mem.Allocator, io: std.Io, e: *qwen.Engine, t: *Trainer, all: []const u32, failures: *usize) !void {
    const V = qwen.config.vocab;
    const steps = 48;
    const prompts = 12;
    var seq = try e.sequence(4096);
    defer seq.deinit();
    const dec = try gpa.alloc(u16, steps * V);
    defer gpa.free(dec);
    const pre = try gpa.alloc(u16, V);
    defer gpa.free(pre);
    var picks_dev = try cuda.DeviceBuffer.alloc(e.ctx.d, 1024 * 4);
    defer picks_dev.free();
    var rows_equal: usize = 0;
    var rows_all: usize = 0;
    var picks_equal: usize = 0;
    var most: f64 = 0;
    const a0 = now(io);
    for (0..prompts) |k| {
        const p = all[k * 300 ..][0 .. 20 + 7 * k];
        // greedy serial decode: the prompt's pass, then one-row rounds, each round's logits kept
        try e.prefill(&seq, p);
        var f = e.forward();
        try f.head(&seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        var toks: std.ArrayList(u32) = .empty;
        defer toks.deinit(gpa);
        try toks.append(gpa, pick[0]);
        try e.ops().download(std.mem.sliceAsBytes(dec[0..V]), e.scratch.logits);
        try e.s.synchronize();
        // the first pick's logits are the head over the prompt's last row (dense, 1 row)
        while (toks.items.len < steps) {
            const last = toks.items[toks.items.len - 1];
            const part = [_]qwen.Part{.{ .seq = &seq, .tokens = &.{last} }};
            try f.round(&part);
            try f.picks(1, &pick);
            try e.ops().download(std.mem.sliceAsBytes(dec[toks.items.len * V ..][0..V]), e.scratch.logits);
            try e.s.synchronize();
            try f.commit(&part, &.{&.{0}});
            try toks.append(gpa, pick[0]);
        }
        // one batched prompt pass over prompt + decoded tokens, every row's logits (the trainer's keep)
        const ids = try std.mem.concat(gpa, u32, &.{ p, toks.items });
        defer gpa.free(ids);
        const n = ids.len - 1;
        var keep = t.keep;
        keep.layers = t.light;
        var g = e.forward();
        g.keep = &keep;
        try t.states[0].reset(e.ops(), e.w.g);
        try g.chunks(&.{.{ .ids = ids[0..n], .seq = &t.states[0], .last = false }});
        try e.ops().argmax(t.logits, V, V, picks_dev.ptr, n);
        var got: [1024]u32 = undefined;
        try e.ops().download(std.mem.sliceAsBytes(got[0..n]), picks_dev.ptr);
        try e.s.synchronize();
        var agree: usize = 0;
        var same_rows: usize = 0;
        for (0..steps) |j| {
            const row = p.len - 1 + j;
            agree += @intFromBool(got[row] == toks.items[j]);
            const d = e.ctx.d;
            try d.check(d.api.cuMemcpyDtoH_v2(pre.ptr, t.logits + row * V * 2, V * 2), "cuMemcpyDtoH");
            var eq = true;
            for (pre, dec[j * V ..][0..V]) |x, y| {
                if (x != y) eq = false;
                most = @max(most, @abs(bf(x) - bf(y)));
            }
            same_rows += @intFromBool(eq);
        }
        rows_all += steps;
        rows_equal += same_rows;
        picks_equal += agree;
        std.debug.print("  prompt {d} ({d} tokens): {d}/{d} pass argmaxes equal the greedy tokens, {d}/{d} rows' logits bit-equal\n", .{ k, p.len, agree, steps, same_rows, steps });
    }
    // the same decoded tokens teacher-forced through decode rounds.
    for ([_]bool{ false, true }) |multi| {
        try e.fit(prompts);
        var seqs: [prompts]qwen.state.Seq = undefined;
        for (&seqs) |*q| q.* = try e.sequence(4096);
        defer for (&seqs) |*q| q.deinit();
        var toks_all: [prompts][]u32 = undefined;
        var bit_rows: usize = 0;
        var pick_rows: usize = 0;
        var rows_n: usize = 0;
        var f = e.forward();
        const decs = try gpa.alloc(u16, prompts * steps * V);
        defer gpa.free(decs);
        // serial decode again per prompt (its logits per row), then the windows
        for (0..prompts) |k| {
            const p = all[k * 300 ..][0 .. 20 + 7 * k];
            try e.prefill(&seqs[k], p);
            try f.head(&seqs[k]);
            var pick: [1]u32 = undefined;
            try f.picks(1, &pick);
            toks_all[k] = try gpa.alloc(u32, steps);
            toks_all[k][0] = pick[0];
            try e.ops().download(std.mem.sliceAsBytes(decs[(k * steps) * V ..][0..V]), e.scratch.logits);
            try e.s.synchronize();
            for (1..steps) |j| {
                const part = [_]qwen.Part{.{ .seq = &seqs[k], .tokens = toks_all[k][j - 1 .. j] }};
                try f.round(&part);
                try f.picks(1, &pick);
                try e.ops().download(std.mem.sliceAsBytes(decs[(k * steps + j) * V ..][0..V]), e.scratch.logits);
                try e.s.synchronize();
                try f.commit(&part, &.{&.{0}});
                toks_all[k][j] = pick[0];
            }
            try e.prefill(&seqs[k], p);
        }
        defer for (toks_all) |x| gpa.free(x);
        const win = 16;
        var at: usize = 0; // window start: tokens at..at+win feed rows predicting at+1..
        const got = try gpa.alloc(u16, prompts * win * V);
        defer gpa.free(got);
        while (at + 1 < steps) : (at += win) {
            const w = @min(win, steps - 1 - at);
            var parts: [prompts]qwen.Part = undefined;
            var paths: [prompts][]const u32 = undefined;
            const path = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
            const groups: usize = if (multi) 1 else prompts;
            for (0..groups) |gi| {
                const members: usize = if (multi) prompts else 1;
                for (0..members) |mi| {
                    const k = if (multi) mi else gi;
                    parts[mi] = .{ .seq = &seqs[k], .tokens = toks_all[k][at..][0..w] };
                    paths[mi] = path[0..w];
                }
                try f.round(parts[0..members]);
                try e.ops().download(std.mem.sliceAsBytes(got[0 .. members * w * V]), e.scratch.logits);
                try e.s.synchronize();
                try f.commit(parts[0..members], paths[0..members]);
                for (0..members) |mi| {
                    const k = if (multi) mi else gi;
                    for (0..w) |r| {
                        const row = got[(mi * w + r) * V ..][0..V];
                        const want = decs[(k * steps + at + 1 + r) * V ..][0..V];
                        bit_rows += @intFromBool(std.mem.eql(u16, row, want));
                        var best: usize = 0;
                        for (row, 0..) |x, i| if (bf(x) > bf(row[best])) {
                            best = i;
                        };
                        pick_rows += @intFromBool(best == toks_all[k][at + 1 + r]);
                        rows_n += 1;
                    }
                }
            }
        }
        std.debug.print("{s} decode rounds teacher-forced ({s}, 16-row windows): {d}/{d} rows' logits bit-equal to serial decode, {d}/{d} picks\n", .{ if (bit_rows == rows_n) "PASS" else "FAIL", if (multi) "every prompt's stream in one round" else "one stream a round", bit_rows, rows_n, pick_rows, rows_n });
        if (bit_rows != rows_n) failures.* += 1;
    }
    // prompts prefilled side by side (Forward.chunks, as the greedy scoring and serving's prefill_batch run them).
    {
        var alone = try e.sequence(4096);
        defer alone.deinit();
        var side: [prompts]qwen.state.Seq = undefined;
        for (&side) |*q| q.* = try e.sequence(4096);
        defer for (&side) |*q| q.deinit();
        var pieces: [prompts]qwen.Forward.Piece = undefined;
        for (0..prompts) |k| {
            try side[k].reset(e.ops(), e.w.g);
            pieces[k] = .{ .ids = all[k * 300 ..][0 .. 20 + 7 * k], .seq = &side[k], .last = true };
        }
        var f = e.forward();
        try f.chunks(&pieces);
        var equal: usize = 0;
        const a1 = try gpa.alloc(u16, V);
        defer gpa.free(a1);
        const b1 = try gpa.alloc(u16, V);
        defer gpa.free(b1);
        for (0..prompts) |k| {
            try e.prefill(&alone, pieces[k].ids);
            for ([_]*qwen.state.Seq{ &alone, &side[k] }, [_][]u16{ a1, b1 }) |q, out| {
                const part = [_]qwen.Part{.{ .seq = q, .tokens = &.{ 13, 220 } }};
                try f.round(&part);
                try e.ops().download(std.mem.sliceAsBytes(out), e.scratch.logits + V * 2);
                try e.s.synchronize();
                try f.commit(&part, &.{&.{ 0, 1 }});
            }
            var ha: [D]u16 = undefined;
            var hb: [D]u16 = undefined;
            try e.ops().download(std.mem.sliceAsBytes(&ha), alone.head_in);
            try e.ops().download(std.mem.sliceAsBytes(&hb), side[k].head_in);
            try e.s.synchronize();
            equal += @intFromBool(std.mem.eql(u16, &ha, &hb) and std.mem.eql(u16, a1, b1));
        }
        std.debug.print("{s} prompts prefilled side by side: {d}/{d} with the head's input and a round's logits after them bit-equal to alone\n", .{ if (equal == prompts) "PASS" else "FAIL", equal, prompts });
        if (equal != prompts) failures.* += 1;
    }
    // the trainer's greedy scoring against serial decode: each decoded reply unchanged, then one token changed
    {
        var replies: [prompts][]u32 = undefined;
        var ids: [2 * prompts][]u32 = undefined;
        var sq: [2 * prompts]train.Seq = undefined;
        for (0..prompts) |k| {
            const p = all[k * 300 ..][0 .. 20 + 7 * k];
            var out: std.ArrayList(u32) = .empty;
            try e.generate(&seq, p, 40, &out);
            replies[k] = try out.toOwnedSlice(gpa);
            ids[k] = try std.mem.concat(gpa, u32, &.{ p, replies[k] });
            ids[prompts + k] = try gpa.dupe(u32, ids[k]);
            const at = p.len + (k * 7) % replies[k].len; // a token changed: the reply is no longer greedy from there
            ids[prompts + k][at] = (ids[prompts + k][at] + 1) % 1000;
            sq[k] = .{ .ids = ids[k], .start = p.len, .decide = p.len - 1 };
            sq[prompts + k] = .{ .ids = ids[prompts + k], .start = p.len, .decide = p.len - 1 };
        }
        defer for (replies) |r| gpa.free(r);
        defer for (ids) |x| gpa.free(x);
        var same: [2 * prompts]bool = undefined;
        const g0 = now(io);
        try t.greedy(sq[0..prompts], same[0..prompts]);
        try t.greedy(sq[prompts..], same[prompts..]);
        const g1 = now(io);
        var right: usize = 0;
        for (same, 0..) |x, i| right += @intFromBool(x == (i < prompts));
        std.debug.print("{s} the trainer's greedy scoring: {d}/{d} verdicts right (decoded replies unchanged, then one token changed), {d:.0} ms\n", .{ if (right == same.len) "PASS" else "FAIL", right, same.len, (g1 - g0) * 1e3 });
        if (right != same.len) failures.* += 1;
    }
    // a note, not a check: a prompt pass over the reply is not decoding's arithmetic (why the scoring runs rounds)
    std.debug.print("NOTE greedy decode vs one prompt pass over the reply: {d}/{d} picks equal, {d}/{d} rows' logits bit-equal (most apart {d:.4}), {d:.1} s\n", .{ picks_equal, rows_all, rows_equal, rows_all, most, now(io) - a0 });
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: tf-qwen27-learn MODEL KERNELS IDS_JSON\n", .{});
        return 2;
    }
    const all = try readIds(gpa, io, args[3]);
    defer gpa.free(all);
    const n = @min(all.len, 256);
    if (n < 24) return 2;
    const seq_ids = all[0..n];
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const l0 = now(io);
    const e = try qwen.Engine.init(gpa, io, &ctx, args[1], args[2], .{ .context = 4096 });
    defer e.deinit();
    const l1 = now(io);
    if (std.c.getenv("Q27_GATE")) |list| return gates(gpa, io, e, std.mem.span(list));
    if (std.c.getenv("Q27_SCORE")) |list| return scores(gpa, io, e, std.mem.span(list));
    const t = try Trainer.init(gpa, io, e);
    defer t.deinit();
    const l2 = now(io);
    std.debug.print("engine loaded in {d:.1} s, trainer in {d:.1} s\n", .{ l1 - l0, l2 - l1 });
    var failures: usize = 0;
    // (0) a prompt in a conversation read through the engine's pass (tap.zig) against the trainer's pass.
    if (std.c.getenv("Q27_TAP") != null) {
        const sq: train.Seq = .{ .ids = seq_ids, .start = n - 4, .decide = n - 5 };
        var keys: [1]train.Result = undefined;
        try t.pass(&.{sq}, .keys, turn, &keys);
        const tl = [_]usize{ 0, 13, 40, 62 };
        const rows_n = n - 1;
        const want = try gpa.alloc(u32, rows_n);
        defer gpa.free(want);
        for (want, 0..) |*w, i| w.* = @intCast(i);
        const got = try gpa.alloc(u16, rows_n * tl.len * D);
        defer gpa.free(got);
        for ([_]usize{ 0, n / 2, n / 2 }, 0..) |context, round| {
            const from: usize = if (round == 2) context else 0; // the third read finds the conversation kept
            const a = now(io);
            try t.tapRows(seq_ids[0..rows_n], context, want[from..], &tl, got[0 .. (rows_n - from) * tl.len * D]);
            const b = now(io);
            var differ: usize = 0;
            for (want[from..], 0..) |r, i| for (tl, 0..) |l, j| {
                for (t.normed(l, r), got[(i * tl.len + j) * D ..][0..D]) |p, q| differ += @intFromBool(p != q);
            };
            const ok = differ == 0;
            std.debug.print("{s} taps after a {d}-row conversation ({d} rows from {d}, {d:.0} ms): {d} values differ from the pass's\n", .{ if (ok) "PASS" else "FAIL", context, rows_n - from, from, (b - a) * 1e3, differ });
            if (!ok) failures += 1;
        }
        std.debug.print("taps: {d} conversations run, {d} found kept\n", .{ t.tap.?.runs, t.tap.?.reused });
    }
    // Q27_GREEDY: greedy decode (serial rounds) against one batched prompt pass over the prompt and the decoded.
    if (std.c.getenv("Q27_GREEDY") != null) try greedyVsPass(gpa, io, e, t, all, &failures);
    // Q27_TIME: only the full pass's time (4)
    const timing = std.c.getenv("Q27_TIME") != null;
    const answer: usize = if (std.c.getenv("Q27_ANSWER")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 8 else 8;
    const start = n - @min(answer, n / 2);

    // (1) packed sequences: each one's loss as alone
    if (!timing) {
        var seqs: [6]train.Seq = undefined;
        var alone: [6]f32 = undefined;
        for (&seqs, 0..) |*s, i| {
            const len = @min(n, 120) - 3 * i;
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

    if (!timing) {
        // (2) push against finite differences at four layers: two rows of the sequence, keys 0 and 1
        var layers = [_]usize{ 13, 26, 39, 52 };
        if (std.c.getenv("Q27_LAYERS")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            for (&layers) |*l| l.* = std.fmt.parseInt(usize, it.next() orelse break, 10) catch l.*;
        }
        const rows = [_]usize{ start - 1, start - 5 };
        var sq: train.Seq = .{ .ids = seq_ids, .start = start, .decide = start - 1 };
        var mask: u64 = 0;
        for (layers) |l| mask |= @as(u64, 1) << @intCast(l);
        sq.nudge = .{ .layers = .{ mask, mask, 0 }, .rows = .{ rows[0], rows[1], 0 } };
        for (layers) |l| for (0..2) |k| @memset(t.deltaAt(0, k, l), 0);
        var out: [1]train.Result = undefined;
        const q0 = now(io);
        try t.pass(&.{sq}, .target, turn, &out);
        const q1 = now(io);
        std.debug.print("target pass: {d} rows, loss {d:.5}, back to layer {d} in {d:.0} ms\n", .{ n - 1, out[0].loss, layers[0], (q1 - q0) * 1e3 });
        if (std.c.getenv("Q27_DEBUG") != null) for (layers) |l| for (0..2) |k| {
            const p = t.pushAt(0, k, l);
            var nan: usize = 0;
            for (p) |v| if (std.math.isNan(v)) {
                nan += 1;
            };
            std.debug.print("  debug layer {d} key {d} row {d}: |push| {e:.4}, {d} NaN\n", .{ l, k, rows[k], @sqrt(dot(p, p)), nan });
        };
        const push = try gpa.alloc(f32, layers.len * 2 * D);
        defer gpa.free(push);
        const resid_n = try gpa.alloc(f64, layers.len * 2);
        defer gpa.free(resid_n);
        for (layers, 0..) |l, li| for (0..2) |k| {
            @memcpy(push[(li * 2 + k) * D ..][0..D], t.pushAt(0, k, l));
            resid_n[li * 2 + k] = norm16(t.residual(l + 1, rows[k]));
        };
        var prng = std.Random.DefaultPrng.init(7);
        const step_frac: f64 = if (std.c.getenv("Q27_STEP")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.004 else 0.004;
        var worst_rel: f64 = 0;
        var lines: usize = 0;
        const dir = try gpa.alloc(f32, D);
        defer gpa.free(dir);
        const rnd = try gpa.alloc(f32, D);
        defer gpa.free(rnd);
        for (layers, 0..) |l, li| for (0..2) |k| {
            const p = push[(li * 2 + k) * D ..][0..D];
            const pn = @sqrt(dot(p, p));
            const rn = resid_n[li * 2 + k];
            for (0..2) |which| {
                var s: f64 = 0;
                for (rnd) |*x| {
                    x.* = prng.random().floatNorm(f32);
                    s += @as(f64, x.*) * x.*;
                }
                for (dir, p, rnd) |*x, v, q| x.* = @floatCast(if (which == 0) v / pn else (v / pn + q / @sqrt(s)) / std.math.sqrt2);
                const want = dot(p, dir);
                // the step moves the loss by about `dl` nats at its first point: small enough to stay linear
                const h = if (std.c.getenv("Q27_DL")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.02) / @max(@abs(want), 1e-12) else step_frac * @max(rn, 1.0);
                const fd = try slope(t, &sq, k, l, dir, h);
                const rel = @abs(fd - want) / @max(@abs(want), 1e-30);
                worst_rel = @max(worst_rel, rel);
                std.debug.print("  layer {d} row {d} ({s}, |residual| {d:.1}, step {d:.3}): push . u {e:.4}, finite differences' slope {e:.4}, relative error {d:.4}\n", .{ l, rows[k], if (which == 0) "along push" else "half random", rn, h, want, fd, rel });
                lines += 1;
            }
        };
        const grad_ok = worst_rel < 0.1;
        std.debug.print("{s} push against finite differences: worst relative error {d:.4} ({d} checks)\n", .{ if (grad_ok) "PASS" else "FAIL", worst_rel, lines });
        if (!grad_ok) failures += 1;

        // (3) a pocket neuron at layer 39: its output's gradient (act push) against the loss moving with it
        {
            const l: usize = 39;
            const r = rows[0];
            var keys: [1]train.Result = undefined;
            sq.nudge = .{};
            try t.pass(&.{sq}, .keys, turn, &keys);
            // input weights that read 2 on this row and little elsewhere: the row less the sequence's mean row, scaled
            const z = try gpa.alloc(f64, D);
            defer gpa.free(z);
            @memset(z, 0);
            const rows_n = sq.ids.len - 1;
            for (0..rows_n) |j| {
                if (j == r) continue;
                for (t.normed(l, j), 0..) |b, i| z[i] -= bf(b) / @as(f64, @floatFromInt(rows_n - 1));
            }
            var zx: f64 = 0;
            for (t.normed(l, r), 0..) |b, i| {
                z[i] += bf(b);
                zx += z[i] * bf(b);
            }
            const u = try gpa.alloc(f32, D);
            defer gpa.free(u);
            for (u, z) |*o, v| o.* = @floatCast(2 * v / zx);
            var most: f64 = -1e30;
            for (0..rows_n) |j| {
                if (j == r) continue;
                var d: f64 = 0;
                for (t.normed(l, j), u) |b, w| d += bf(b) * w;
                most = @max(most, d);
            }
            std.debug.print("  pocket neuron: u x = 2 on row {d}, at most {d:.3} on the sequence's other rows\n", .{ r, most });
            const neurons = 64;
            const rows3 = try gpa.alloc(f32, 3 * neurons * D);
            defer gpa.free(rows3);
            @memset(rows3, 0);
            const gate = rows3[0 .. neurons * D];
            const upr = rows3[neurons * D ..][0 .. neurons * D];
            const down = rows3[2 * neurons * D ..][0 .. neurons * D];
            for (0..D) |i| {
                gate[i] = 16 * u[i];
                upr[i] = u[i] / 16;
            }
            try e.pockets.set(e.s, l, gate, upr, down, 1);
            sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(l), 0, 0 }, .rows = .{ r, 0, 0 } };
            @memset(t.deltaAt(0, 0, l), 0);
            var got: [1]train.Result = undefined;
            try t.pass(&.{sq}, .target, turn, &got);
            var zg: f64 = 0;
            var zu: f64 = 0;
            for (t.normed(l, r), 0..) |b, i| {
                zg += gate[i] * bf(b);
                zu += upr[i] * bf(b);
            }
            const act = zg / (1 + @exp(-zg)) * zu;
            const pg = try gpa.dupe(f32, t.pushAt(0, 0, l));
            defer gpa.free(pg);
            const pn = @sqrt(dot(pg, pg));
            const rn = norm16(t.residual(l + 1, r));
            const step = step_frac * rn / @max(act, 1e-6);
            sq.nudge = .{};
            var num: f64 = 0;
            var den: f64 = 0;
            for (1..7) |m| for ([_]f64{ 1, -1 }) |sgn| {
                const xs = sgn * @as(f64, @floatFromInt(m)) * step;
                for (down[0..D], pg) |*o, v| o.* = @floatCast(xs * v / pn);
                try e.pockets.set(e.s, l, gate, upr, down, 1);
                var rr: [1]train.Result = undefined;
                try t.pass(&.{sq}, .loss, turn, &rr);
                num += xs * rr[0].loss;
                den += xs * xs;
            };
            const fd = num / den;
            const want = act * pn;
            const rel = @abs(fd - want) / @max(@abs(want), 1e-30);
            const ok = rel < 0.1;
            std.debug.print("{s} a pocket neuron at layer {d} (activation {d:.3}): act |push| {e:.4}, finite difference of its output {e:.4}, relative error {d:.4}\n", .{ if (ok) "PASS" else "FAIL", l, act, want, fd, rel });
            if (!ok) failures += 1;
            @memset(rows3, 0);
            try e.pockets.set(e.s, l, gate, upr, down, 0);
        }
    }

    // (4) a full pass's time: sixteen sequences, keys then loss then target down to layer 13
    {
        // Q27_LESSON: a lesson's values pass instead: 8 answers of 78 rows, 22 of them the reply.
        const lesson = std.c.getenv("Q27_LESSON") != null;
        const count: usize = if (lesson) 8 else 16;
        var seqs: [16]train.Seq = undefined;
        var rowsum: usize = 0;
        var mask: u64 = @as(u64, 1) << 13 | @as(u64, 1) << 40;
        if (lesson) {
            mask = 0;
            for ([_]usize{ 19, 20, 21, 23, 24, 25, 26, 27, 28, 36 }) |l| mask |= @as(u64, 1) << @intCast(l);
        }
        for (seqs[0..count], 0..) |*s, i| {
            const len = if (lesson) @min(n, 79) else @min(n, 40 + 3 * i);
            const reply: usize = if (lesson) 22 else 6;
            rowsum += len - 1;
            s.* = .{ .ids = seq_ids[i * 3 ..][0..len], .start = len - reply, .decide = len - reply - 1, .nudge = .{ .layers = .{ mask, 0, 0 }, .rows = .{ len - reply - 1, 0, 0 } } };
            if (!lesson) s.ids = seq_ids[0..len];
            for (0..e.w.g.layers) |l| @memset(t.deltaAt(i, 0, l), 0);
        }
        var got: [16]train.Result = undefined;
        for ([_]train.Mode{ .keys, .loss, .target, .target, .target }) |m| {
            const a = now(io);
            try t.pass(seqs[0..count], m, turn, got[0..count]);
            const b = now(io);
            var h = std.hash.Wyhash.init(0);
            for (got[0..count]) |r| h.update(std.mem.asBytes(&r.loss));
            if (m == .target) for (0..count) |si| for (0..3) |k| for (0..e.w.g.layers) |l| h.update(std.mem.sliceAsBytes(t.pushAt(si, k, l)));
            std.debug.print("pass {s}: {d} sequences, {d} rows, {d:.0} ms (losses and pushes {x:0>16})\n", .{ @tagName(m), count, rowsum, (b - a) * 1e3, h.final() });
        }
    }
    std.debug.print("{d} failures\n", .{failures});
    return if (failures == 0) 0 else 1;
}
