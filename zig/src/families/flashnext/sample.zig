//! A sampled reply's draws on the GPU: tf_sample_full, or for top_k 1 to KMAX the same token in two passes
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");

/// tf_topk_slices' geometry (nemotron_sample.metal): slices a row, threads a slice, values a thread, its widest top_k.
pub const PG = 32;
const PT = 512;
const EMAX = 16;
pub const KMAX = 64;

/// The keyed draw's kernels and the two-pass draw's scratch (each row's slices' top_k keys and ids).
pub const Draws = struct {
    full: mtl.Pipeline, // tf_sample_full: any rule, ~700 us a call over 248,320 ids
    full_ids: mtl.Pipeline, // tf_sample_full_ids
    slices: mtl.Pipeline, // tf_topk_slices, then tf_topk_race[_ids]: top_k 1 to KMAX, ~29 us a call
    race: mtl.Pipeline,
    race_ids: mtl.Pipeline,
    ck: mtl.Buffer,
    ci: mtl.Buffer,

    /// The kernels from nemotron_sample.metal's library `lib`, scratch for `rows` rows.
    pub fn init(dev: mtl.Device, lib: mtl.Library, rows: usize) !Draws {
        var d: Draws = undefined;
        d.full = try mtl.Pipeline.init(dev, lib, "tf_sample_full", false);
        errdefer d.full.deinit();
        d.full_ids = try mtl.Pipeline.init(dev, lib, "tf_sample_full_ids", false);
        errdefer d.full_ids.deinit();
        d.slices = try mtl.Pipeline.init(dev, lib, "tf_topk_slices", false);
        errdefer d.slices.deinit();
        d.race = try mtl.Pipeline.init(dev, lib, "tf_topk_race", false);
        errdefer d.race.deinit();
        d.race_ids = try mtl.Pipeline.init(dev, lib, "tf_topk_race_ids", false);
        errdefer d.race_ids.deinit();
        d.ck = try dev.buffer(rows * PG * KMAX * 4, fz.opts);
        errdefer d.ck.deinit();
        d.ci = try dev.buffer(rows * PG * KMAX * 4, fz.opts);
        return d;
    }

    pub fn deinit(d: Draws) void {
        for ([_]mtl.Pipeline{ d.full, d.full_ids, d.slices, d.race, d.race_ids }) |p| p.deinit();
        d.ck.deinit();
        d.ci.deinit();
    }
};

/// fz_positions: a window's row i draws at the token after it, the cache length (ar[1]) + 1 + i.
pub fn positions(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, ar: fz.Buf, pos: fz.Buf, rows: usize) void {
    enc.setPipeline(pipe);
    enc.setBuffer(pos.b, pos.off, 0);
    enc.setBuffer(ar.b, ar.off, 1);
    const n: u32 = @intCast(rows);
    enc.setBytes(std.mem.asBytes(&n), 2);
    enc.dispatchThreads(mtl.Size.of(rows, 1, 1), mtl.Size.of(rows, 1, 1));
}

/// Where the rows' draw positions come from: the host (as bytes) or a buffer the GPU wrote.
const At = union(enum) { host: []const u32, gpu: fz.Buf };

/// Rule i's keyed draw over row i's `n` columns (ids from `ids`) to out[i]; two passes when every top_k is 1 to KMAX
fn keyedDraws(enc: mtl.ComputeEncoder, d: *const Draws, logits: fz.Buf, ids: ?fz.Buf, n: usize, rules: []const lanes.gpu_rule.Rule, at: At, out: fz.Buf, two_ok: bool) void {
    const rows = rules.len;
    var seeds: [2 * fz.MAXR]u32 = @splat(0);
    var cfgs: [4 * fz.MAXR]f32 = @splat(0);
    var caps: [fz.MAXR]u32 = @splat(0);
    var two = two_ok and n <= PG * PT * EMAX;
    for (rules, 0..) |rule, i| {
        seeds[2 * i ..][0..2].* = rule.seedHalves();
        cfgs[4 * i ..][0..4].* = rule.cfg();
        caps[i] = rule.top_k;
        if (rule.top_k == 0 or rule.top_k > KMAX or rule.top_k >= n) two = false;
    }
    const v: u32 = @intCast(n);
    enc.setBuffer(logits.b, logits.off, 0);
    enc.setBytes(std.mem.sliceAsBytes(seeds[0 .. 2 * rows]), 1);
    switch (at) {
        .host => |p| enc.setBytes(std.mem.sliceAsBytes(p), 2),
        .gpu => |b| enc.setBuffer(b.b, b.off, 2),
    }
    enc.setBytes(std.mem.sliceAsBytes(cfgs[0 .. 4 * rows]), 3);
    enc.setBytes(std.mem.sliceAsBytes(caps[0..rows]), 4);
    enc.setBuffer(out.b, out.off, 5);
    const map = ids orelse out; // the id map tf_sample_full<false> never reads
    enc.setBuffer(map.b, map.off, 6);
    enc.setBytes(std.mem.asBytes(&v), 7);
    if (two) {
        enc.setBuffer(d.ck, 0, 8);
        enc.setBuffer(d.ci, 0, 9);
        enc.setPipeline(d.slices);
        enc.dispatchGroups(mtl.Size.of(PG, rows, 1), mtl.Size.of(PT, 1, 1));
        enc.barrier(); // whatever the encoder's dispatch type
        enc.setPipeline(if (ids != null) d.race_ids else d.race);
    } else enc.setPipeline(if (ids != null) d.full_ids else d.full);
    enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// `rows` bf16 rows of `logits` (`vocab` wide) drawn at the positions in `pos` under `s`: row i's token to picks[i].
pub fn draw(
    enc: mtl.ComputeEncoder,
    d: *const Draws,
    logits: fz.Buf,
    pos: fz.Buf,
    picks: fz.Buf,
    rows: usize,
    vocab: usize,
    s: lanes.Sampling,
) void {
    var each: [fz.MAXR]lanes.Sampling = undefined;
    for (each[0..rows]) |*x| x.* = s;
    drawRows(enc, d, logits, pos, picks, each[0..rows], vocab);
}

/// One row per entry of `ss`, each under its own window's rule: one draw for several windows.
pub fn drawRows(enc: mtl.ComputeEncoder, d: *const Draws, logits: fz.Buf, pos: fz.Buf, picks: fz.Buf, ss: []const lanes.Sampling, vocab: usize) void {
    var rules: [fz.MAXR]lanes.gpu_rule.Rule = undefined;
    for (ss, rules[0..ss.len]) |si, *x| x.* = lanes.gpu_rule.rule(si);
    keyedDraws(enc, d, logits, null, vocab, rules[0..ss.len], .{ .gpu = pos }, picks, true);
}

/// A greedy row among keyed ones: its top token alone races.
const top_one: lanes.Sampling = .{ .seed = 0, .temperature = 1.0, .top_k = 1, .top_p = 1.0 };

/// FZ_KEYED_DRAFTS=0: a sampled reply or stream drafts by argmax, as before.
pub fn keyed() bool {
    const v = std.c.getenv("FZ_KEYED_DRAFTS") orelse return true;
    return v[0] != '0';
}

/// A sampled reply's head draft in GPU-side rounds: its keyed rule, and where the GPU holds the draft's position.
pub const HeadDraw = struct { s: lanes.Sampling, pos: fz.Buf };

/// One bf16 row of `n` token-id columns drawn under `h`'s rule at the position the GPU holds at `h.pos`.
pub fn drawIdAt(enc: mtl.ComputeEncoder, d: *const Draws, logits: fz.Buf, ids: fz.Buf, n: usize, h: HeadDraw, out: fz.Buf) void {
    const rules = [1]lanes.gpu_rule.Rule{lanes.gpu_rule.rule(h.s)};
    keyedDraws(enc, d, logits, ids, n, &rules, .{ .gpu = h.pos }, out, true);
}

/// Bf16 rows of `n` token-id columns: row i's draft at lands[i] by its keyed rule (null: its top token).
pub fn drawIds(enc: mtl.ComputeEncoder, d: *const Draws, logits: fz.Buf, ids: fz.Buf, n: usize, ss: []const ?lanes.Sampling, lands: []const u64, out: fz.Buf) void {
    var rules: [fz.MAXR]lanes.gpu_rule.Rule = undefined;
    var at: [fz.MAXR]u32 = undefined;
    for (ss, lands, 0..) |si, p, i| {
        rules[i] = lanes.gpu_rule.rule(si orelse top_one);
        at[i] = @intCast(p);
    }
    keyedDraws(enc, d, logits, ids, n, rules[0..ss.len], .{ .host = at[0..ss.len] }, out, true);
}

test "a window's draws on the GPU are the shared rule's host draws at the cache length + 1 + row" {
    const ks = @import("kernel_sources");
    const verdict = @import("verdict.zig");
    const gpa = std.testing.allocator;
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const vlib = try mtl.Library.fromSource(dev, verdict.source, mtl.CompileOptions.mlx());
    defer vlib.deinit();
    const slib = try mtl.Library.fromSource(dev, ks.nemotron_sample, mtl.CompileOptions.mlx());
    defer slib.deinit();
    const pos_pipe = try mtl.Pipeline.init(dev, vlib, "fz_positions", false);
    defer pos_pipe.deinit();
    const d = try Draws.init(dev, slib, fz.MAXR);
    defer d.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const vocab = 6000;
    const rows = 5;
    const logits = try dev.buffer(fz.MAXR * vocab * 2, fz.opts);
    defer logits.deinit();
    const ar = try dev.buffer(4 * fz.AR_WORDS, fz.opts);
    defer ar.deinit();
    const pos = try dev.buffer(4 * fz.MAXR, fz.opts);
    defer pos.deinit();
    const picks = try dev.buffer(4 * fz.MAXR, fz.opts);
    defer picks.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const row = try gpa.alloc(f32, vocab);
    defer gpa.free(row);
    const words = logits.slice(u16, fz.MAXR * vocab);
    for (words) |*w| { // spread logits with a few strong tokens, rounded to bf16 as the head writes them
        const x: f32 = rng.floatNorm(f32) * 2.0 + @as(f32, if (rng.uintLessThan(u32, 200) == 0) 8.0 else 0.0);
        const b: u32 = @bitCast(x);
        w.* = @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
    }
    const cases = [_]lanes.Sampling{
        .{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 2, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 3, .temperature = 0.6, .top_k = 0, .top_p = 0.9, .min_p = 0.05 },
        .{ .seed = 1 << 63 | 4, .temperature = 1.3, .top_k = 2000, .top_p = 0.8 },
    };
    for (cases) |s| for ([_]i32{ 17, 40_000 }) |t| {
        ar.slice(i32, fz.AR_WORDS)[1] = t;
        const cb = q.commandBuffer();
        const enc = cb.compute(.serial);
        positions(enc, pos_pipe, .{ .b = ar }, .{ .b = pos }, rows);
        draw(enc, &d, .{ .b = logits }, .{ .b = pos }, .{ .b = picks }, rows, vocab, s);
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        for (0..rows) |i| {
            for (row, words[i * vocab ..][0..vocab]) |*x, w| x.* = @bitCast(@as(u32, w) << 16);
            const at: u32 = @intCast(t + 1 + @as(i32, @intCast(i)));
            try std.testing.expectEqual(at, pos.slice(u32, fz.MAXR)[i]);
            try std.testing.expectEqual(try lanes.gpu_full.sample(gpa, row, s, at, null), picks.slice(u32, fz.MAXR)[i]);
        }
    };
}

test "a sampled stream's draft over the draft head's ids is the shared rule's host draw keyed by those ids" {
    const ks = @import("kernel_sources");
    const gpa = std.testing.allocator;
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const slib = try mtl.Library.fromSource(dev, ks.nemotron_sample, mtl.CompileOptions.mlx());
    defer slib.deinit();
    const d = try Draws.init(dev, slib, fz.MAXR);
    defer d.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const n = 3000;
    const logits = try dev.buffer(2 * n * 2, fz.opts);
    defer logits.deinit();
    const ids = try dev.buffer(n * 4, fz.opts);
    defer ids.deinit();
    const out = try dev.buffer(8, fz.opts);
    defer out.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    for (ids.slice(u32, n), 0..) |*v, i| v.* = @intCast(3 * i + 7); // a cut head's sorted ids
    const words = logits.slice(u16, 2 * n);
    const row = try gpa.alloc(f32, n);
    defer gpa.free(row);
    for (words[0..n], row) |*w, *x| {
        const v: f32 = rng.floatNorm(f32) * 2.0 + @as(f32, if (rng.uintLessThan(u32, 100) == 0) 7.0 else 0.0);
        const b: u32 = @bitCast(v);
        w.* = @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
        x.* = @bitCast(@as(u32, w.*) << 16);
    }
    const cases = [_]lanes.Sampling{
        .{ .seed = 9, .temperature = 1.0, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 1 << 63 | 2, .temperature = 0.6, .top_k = 0, .top_p = 0.9, .min_p = 0.05 },
    };
    for (cases) |s| for ([_]u64{ 3, 70_001 }) |at| {
        const cb = q.commandBuffer();
        const enc = cb.compute(.serial);
        @memcpy(words[n..], words[0..n]); // row 1: the same scores, greedy
        drawIds(enc, &d, .{ .b = logits }, .{ .b = ids }, n, &.{ s, null }, &.{ at, at }, .{ .b = out });
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const want = try lanes.gpu_full.sample(gpa, row, s, @intCast(at), ids.slice(u32, n));
        try std.testing.expectEqual(want, out.slice(u32, 2)[0]);
        const greedy = std.mem.indexOfScalar(u32, ids.slice(u32, n), out.slice(u32, 2)[1]).?;
        try std.testing.expectEqual(std.mem.max(f32, row), row[greedy]); // a top score (ties may go either way)
        // a reply's draft at a position the GPU holds, read at an offset as a round's chain step reads it
        const at_gpu = try dev.buffer(16, fz.opts);
        defer at_gpu.deinit();
        at_gpu.slice(u32, 4)[2] = @intCast(at);
        const cb2 = q.commandBuffer();
        const enc2 = cb2.compute(.serial);
        drawIdAt(enc2, &d, .{ .b = logits }, .{ .b = ids }, n, .{ .s = s, .pos = .{ .b = at_gpu, .off = 8 } }, .{ .b = out, .off = 4 });
        enc2.end();
        cb2.commit();
        cb2.wait();
        try std.testing.expect(cb2.failure() == null);
        try std.testing.expectEqual(want, out.slice(u32, 2)[1]);
    };
}

test "the two-pass top_k draw is tf_sample_full's token on every row" {
    const ks = @import("kernel_sources");
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const slib = try mtl.Library.fromSource(dev, ks.nemotron_sample, mtl.CompileOptions.mlx());
    defer slib.deinit();
    const d = try Draws.init(dev, slib, fz.MAXR);
    defer d.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const rows = 4;
    const calls = 24; // draws a rule and shape: rows of each, at their own positions
    const ids_n = 79_616; // the draft head's ids
    const ids = try dev.buffer(ids_n * 4, fz.opts);
    defer ids.deinit();
    for (ids.slice(u32, ids_n), 0..) |*v, i| v.* = @intCast(3 * i + 7);
    const want = try dev.buffer(calls * rows * 4, fz.opts);
    defer want.deinit();
    const got = try dev.buffer(calls * rows * 4, fz.opts);
    defer got.deinit();
    var prng = std.Random.DefaultPrng.init(23);
    const rng = prng.random();
    const rules = [_]lanes.Sampling{
        .{ .seed = 1, .temperature = 1.0, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 2, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 3, .temperature = 0.6, .top_k = 50, .top_p = 0.9, .min_p = 0.05 },
        .{ .seed = 4, .temperature = 1.3, .top_k = 1, .top_p = 1.0 },
        .{ .seed = 5, .temperature = 0.3, .top_k = 5, .top_p = 0.5, .min_p = 0.1 },
        .{ .seed = 1 << 63 | 6, .temperature = 2.0, .top_k = 64, .top_p = 1.0 },
        .{ .seed = 7, .temperature = 1.0, .top_k = 2, .top_p = 0.6 },
        .{ .seed = 8, .temperature = 1.0, .top_k = 20, .top_p = 0.9999999 },
        .{ .seed = 9, .temperature = 0.8, .top_k = 40, .top_p = 0.95 },
        .{ .seed = 10, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, // past the two-pass draw: both tf_sample_full
    };
    for ([_]usize{ fz.VOCAB, ids_n, 3000 }) |n| {
        const logits = try dev.buffer(rows * n * 2, fz.opts);
        defer logits.deinit();
        const words = logits.slice(u16, rows * n);
        // shapes: a tail with strong tokens; coarse levels (ties); flat; one token far above; a grammar's mask
        for (0..5) |shape| {
            for (words, 0..) |*w, j| {
                const u = rng.float(f32);
                const x: f32 = switch (shape) {
                    0 => if (rng.uintLessThan(u32, 5000) == 0) 6 + 4 * u else -4 + 2.5 * -@log(@max(u, 1e-9)),
                    1 => @as(f32, @floatFromInt(rng.uintLessThan(u32, 12))) * 0.5 - 2,
                    2 => u * 0.2,
                    3 => if (j % n == (j / n) * 7919 % n) 30 else rng.floatNorm(f32),
                    else => if (rng.uintLessThan(u32, 1000) == 0) rng.floatNorm(f32) * 3 else -std.math.inf(f32),
                };
                w.* = @truncate(@as(u32, @bitCast(x)) >> 16);
            }
            for (rules) |s| {
                if (s.top_k >= n) continue;
                for ([_]bool{ false, true }) |two| {
                    const cb = q.commandBuffer();
                    const enc = cb.compute(.serial);
                    for (0..calls) |k| {
                        var rs: [rows]lanes.gpu_rule.Rule = undefined;
                        var at: [rows]u32 = undefined;
                        for (0..rows) |i| {
                            var si = s;
                            si.seed +%= i;
                            rs[i] = lanes.gpu_rule.rule(si);
                            at[i] = @intCast(17 + 977 * k + i);
                        }
                        const map: ?fz.Buf = if (n == ids_n) .{ .b = ids } else null;
                        keyedDraws(enc, &d, .{ .b = logits }, map, n, &rs, .{ .host = &at }, .{ .b = if (two) got else want, .off = k * rows * 4 }, two);
                    }
                    enc.end();
                    cb.commit();
                    cb.wait();
                    try std.testing.expect(cb.failure() == null);
                }
                try std.testing.expectEqualSlices(u32, want.slice(u32, calls * rows), got.slice(u32, calls * rows));
            }
        }
    }
}
