//! A sampled reply's draws on the GPU: tf_sample_full under the shared rule (lanes.gpu_rule), as Nemotron draws.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");

/// fz_positions: a window's row i draws at the token after it, the cache length (ar[1]) + 1 + i.
pub fn positions(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, ar: fz.Buf, pos: fz.Buf, rows: usize) void {
    enc.setPipeline(pipe);
    enc.setBuffer(pos.b, pos.off, 0);
    enc.setBuffer(ar.b, ar.off, 1);
    const n: u32 = @intCast(rows);
    enc.setBytes(std.mem.asBytes(&n), 2);
    enc.dispatchThreads(mtl.Size.of(rows, 1, 1), mtl.Size.of(rows, 1, 1));
}

/// tf_sample_full on `rows` bf16 rows of `logits` (`vocab` wide) at the positions in `pos`: row i's token to picks[i].
pub fn draw(
    enc: mtl.ComputeEncoder,
    pipe: mtl.Pipeline,
    logits: fz.Buf,
    pos: fz.Buf,
    picks: fz.Buf,
    rows: usize,
    vocab: usize,
    s: lanes.Sampling,
) void {
    var each: [fz.MAXR]lanes.Sampling = undefined;
    for (each[0..rows]) |*x| x.* = s;
    drawRows(enc, pipe, logits, pos, picks, each[0..rows], vocab);
}

/// tf_sample_full on one row per entry of `ss`, each row under its own window's rule: one dispatch for several windows.
pub fn drawRows(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, logits: fz.Buf, pos: fz.Buf, picks: fz.Buf, ss: []const lanes.Sampling, vocab: usize) void {
    const rows = ss.len;
    var seeds: [2 * fz.MAXR]u32 = @splat(0);
    var cfgs: [4 * fz.MAXR]f32 = @splat(0);
    var caps: [fz.MAXR]u32 = @splat(0);
    for (ss, 0..) |si, i| {
        const rule = lanes.gpu_rule.rule(si);
        seeds[2 * i ..][0..2].* = rule.seedHalves();
        cfgs[4 * i ..][0..4].* = rule.cfg();
        caps[i] = rule.top_k;
    }
    enc.setPipeline(pipe);
    enc.setBuffer(logits.b, logits.off, 0);
    enc.setBytes(std.mem.sliceAsBytes(&seeds), 1);
    enc.setBuffer(pos.b, pos.off, 2);
    enc.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
    enc.setBytes(std.mem.sliceAsBytes(&caps), 4);
    enc.setBuffer(picks.b, picks.off, 5);
    enc.setBuffer(picks.b, picks.off, 6); // the id map tf_sample_full<false> never reads
    const v: u32 = @intCast(vocab);
    enc.setBytes(std.mem.asBytes(&v), 7);
    enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// tf_sample_full_ids on one bf16 row over `n` columns that are token ids `ids`: the draft at `position`, under the target's keyed rule.
pub fn drawIds(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, logits: fz.Buf, ids: fz.Buf, n: usize, position: u64, s: lanes.Sampling, out: fz.Buf) void {
    const rule = lanes.gpu_rule.rule(s);
    const seeds = rule.seedHalves();
    const cfgs = rule.cfg();
    const cap: u32 = rule.top_k;
    const at: u32 = @intCast(position);
    const v: u32 = @intCast(n);
    enc.setPipeline(pipe);
    enc.setBuffer(logits.b, logits.off, 0);
    enc.setBytes(std.mem.sliceAsBytes(&seeds), 1);
    enc.setBytes(std.mem.asBytes(&at), 2);
    enc.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
    enc.setBytes(std.mem.asBytes(&cap), 4);
    enc.setBuffer(out.b, out.off, 5);
    enc.setBuffer(ids.b, ids.off, 6);
    enc.setBytes(std.mem.asBytes(&v), 7);
    enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
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
    const draw_pipe = try mtl.Pipeline.init(dev, slib, "tf_sample_full", false);
    defer draw_pipe.deinit();
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
        draw(enc, draw_pipe, .{ .b = logits }, .{ .b = pos }, .{ .b = picks }, rows, vocab, s);
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
    const pipe = try mtl.Pipeline.init(dev, slib, "tf_sample_full_ids", false);
    defer pipe.deinit();
    const q = try dev.queue();
    defer q.deinit();
    const n = 3000;
    const logits = try dev.buffer(n * 2, fz.opts);
    defer logits.deinit();
    const ids = try dev.buffer(n * 4, fz.opts);
    defer ids.deinit();
    const out = try dev.buffer(4, fz.opts);
    defer out.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    for (ids.slice(u32, n), 0..) |*v, i| v.* = @intCast(3 * i + 7); // a cut head's sorted ids
    const words = logits.slice(u16, n);
    const row = try gpa.alloc(f32, n);
    defer gpa.free(row);
    for (words, row) |*w, *x| {
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
        drawIds(enc, pipe, .{ .b = logits }, .{ .b = ids }, n, at, s, .{ .b = out });
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const want = try lanes.gpu_full.sample(gpa, row, s, @intCast(at), ids.slice(u32, n));
        try std.testing.expectEqual(want, out.slice(u32, 1)[0]);
    };
}
