//! The prompt chunk's tensor-unit sparse attention against the row kernel and an fp64 host, over dense, sparse and empty key lists.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

const H = 64;
const RANK = 512;
const WIDTH = 2051; // the row kernel's list width (index_topk + kpool - 1)
const CAP = 3000;
const ROWS = 5;

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "sparse attention on the tensor units: the row kernel's values within bf16 rounding" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const gen = for (sources.glm.all) |k| {
        if (std.mem.eql(u8, k.key, "sparse_attention")) break k;
    } else return error.MissingKernel;
    const lib_row = try mtl.Library.fromSource(device, gen.source, mtl.CompileOptions.mlx());
    defer lib_row.deinit();
    const row_pipe = try mtl.Pipeline.init(device, lib_row, gen.functions[0], false);
    defer row_pipe.deinit();
    const nax_src = try frags.source(device, a, sources.glm_sparse_nax);
    defer a.free(nax_src);
    const lib_nax = try mtl.Library.fromSource(device, nax_src, mtl.CompileOptions.mlx());
    defer lib_nax.deinit();
    const nax_pipe = try mtl.Pipeline.init(device, lib_nax, "glm_sparse_nax", false);
    defer nax_pipe.deinit();

    const opts = mtl.ResourceOptions.shared;
    const bufs = [_]mtl.Buffer{ try device.buffer(ROWS * H * RANK * 2, opts), try device.buffer(CAP * RANK * 2, opts), try device.buffer(ROWS * WIDTH * 4, opts), try device.buffer(ROWS * H * RANK * 2, opts), try device.buffer(ROWS * H * RANK * 2, opts) };
    defer for (bufs) |b| b.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const q = bufs[0].slice(u16, ROWS * H * RANK);
    for (q, 0..) |*v, i| v.* = bf16((rnd.float(f32) * 4 - 2) * (if (i / (H * RANK) == 4) @as(f32, 6) else 1)); // row 4: peaked
    const keys = bufs[1].slice(u16, CAP * RANK);
    for (keys) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    const lists = bufs[2].slice(i32, ROWS * WIDTH);
    @memset(lists, -1);
    for (0..1500) |j| lists[j] = @intCast(j); // row 0: a dense row's list, then nothing
    var perm: [CAP]i32 = undefined; // rows 1 and 4: 2,048 distinct keys, then entries that are no key
    for (&perm, 0..) |*p, i| p.* = @intCast(i);
    rnd.shuffle(i32, &perm);
    for (0..2048) |j| {
        lists[WIDTH + j] = perm[j];
        lists[4 * WIDTH + j] = perm[CAP - 1 - j];
    }
    lists[WIDTH + 2048] = CAP + 5; // past the keys written
    for (0..10) |j| lists[2 * WIDTH + 100 + 7 * j] = @intCast(37 * j); // row 2: ten keys among gaps; row 3: none

    const scale: f32 = 1.0 / 16.0;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(row_pipe);
    for (bufs[0..3], 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setValue(scale, 3);
    enc.setValue(@as(i32, CAP), 4);
    enc.setBuffer(bufs[3], 0, 5);
    enc.dispatchThreads(mtl.Size.of(1024, ROWS * H, 1), mtl.Size.of(1024, 1, 1));
    enc.setPipeline(nax_pipe);
    for (bufs[0..3], 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setValue(scale, 3);
    enc.setValue([4]i32{ WIDTH, CAP, 0, 0 }, 4);
    enc.setBuffer(bufs[4], 0, 5);
    enc.dispatchGroups(mtl.Size.of(H / 16, ROWS, 1), mtl.Size.of(128, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);

    const row_out = bufs[3].slice(u16, ROWS * H * RANK);
    const nax_out = bufs[4].slice(u16, ROWS * H * RANK);
    const scores = try a.alloc(f64, WIDTH);
    defer a.free(scores);
    var same: usize = 0;
    var worst: f64 = 0; // the largest difference from the row kernel's output, in bf16 steps of the larger value
    const outs = try a.alloc(f64, RANK);
    defer a.free(outs);
    for (0..ROWS) |r| for (0..H) |h| {
        // the host's fp64 softmax over the valid listed keys
        const qr = q[(r * H + h) * RANK ..][0..RANK];
        var top: f64 = -std.math.inf(f64);
        for (lists[r * WIDTH ..][0..WIDTH], scores) |k, *sc| {
            sc.* = -std.math.inf(f64);
            if (k < 0 or k >= CAP) continue;
            var dot: f64 = 0;
            for (qr, keys[@as(usize, @intCast(k)) * RANK ..][0..RANK]) |x, y| dot += @as(f64, f32of(x)) * f32of(y);
            sc.* = dot * scale;
            top = @max(top, sc.*);
        }
        var total: f64 = 0;
        for (scores) |*sc| {
            sc.* = if (sc.* == -std.math.inf(f64)) 0 else @exp(sc.* - top);
            total += sc.*;
        }
        var big: f64 = 0;
        for (0..RANK) |d| {
            var o: f64 = 0;
            if (total > 0) for (lists[r * WIDTH ..][0..WIDTH], scores) |k, p| {
                if (p != 0) o += p * f32of(keys[@as(usize, @intCast(k)) * RANK + d]);
            };
            outs[d] = if (total > 0) o / total else 0;
            big = @max(big, @abs(outs[d]));
        }
        for (0..RANK) |d| {
            const i = (r * H + h) * RANK + d;
            const got: f64 = f32of(nax_out[i]);
            const theirs: f64 = f32of(row_out[i]);
            // a bf16 step of the value, but no finer than fp32 sums of the head's values can tell (2^-16 of its largest)
            const floor = big / 65536.0;
            try std.testing.expect(@abs(got - outs[d]) <= @max(@abs(outs[d]) / 128.0, floor));
            if (nax_out[i] == row_out[i]) same += 1 else worst = @max(worst, @abs(got - theirs) / @max(@max(@abs(got), @abs(theirs)) / 128.0, floor));
        }
    };
    const share = @as(f64, @floatFromInt(same)) / @as(f64, @floatFromInt(ROWS * H * RANK));
    std.debug.print("sparse attention on the tensor units: {d:.2}% of outputs equal the row kernel's, at most {d:.2} bf16 steps apart\n", .{ share * 100, worst });
    try std.testing.expect(share > 0.99 and worst <= 2);
}

test "the MLA absorb on the tensor units: the row kernel's values within bf16 rounding" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const NOPE = 256;
    const LATENT = 512;
    const PER_HEAD = 512;
    const M = 70; // a partial tile
    const QS = H * NOPE + 64; // the q projection's row pitch (the nope parts first)
    const glue = try mtl.Library.fromSource(device, sources.glm_glue, mtl.CompileOptions.mlx());
    defer glue.deinit();
    const row_pipe = try mtl.Pipeline.init(device, glue, "glm_absorb", false);
    defer row_pipe.deinit();
    const nax_src = try frags.source(device, a, sources.glm_absorb_nax);
    defer a.free(nax_src);
    const lib_nax = try mtl.Library.fromSource(device, nax_src, mtl.CompileOptions.mlx());
    defer lib_nax.deinit();
    const nax_pipe = try mtl.Pipeline.init(device, lib_nax, "glm_absorb_nax", false);
    defer nax_pipe.deinit();
    const opts = mtl.ResourceOptions.shared;
    const WR = H * PER_HEAD;
    const bufs = [_]mtl.Buffer{ try device.buffer(WR * LATENT / 2, opts), try device.buffer(WR * (LATENT / 64) * 2, opts), try device.buffer(WR * (LATENT / 64) * 2, opts), try device.buffer(M * QS * 2, opts), try device.buffer(M * H * LATENT * 2, opts), try device.buffer(M * H * LATENT * 2, opts) };
    defer for (bufs) |b| b.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const wb = bufs[0].slice(u8, WR * LATENT / 2);
    rnd.bytes(wb);
    const sc = bufs[1].slice(u16, WR * (LATENT / 64));
    const bi = bufs[2].slice(u16, WR * (LATENT / 64));
    for (sc, bi) |*s, *b| {
        s.* = bf16((rnd.float(f32) + 0.5) / 32);
        b.* = bf16(-(rnd.float(f32) + 0.5) * 0.25);
    }
    const qp = bufs[3].slice(u16, M * QS);
    for (qp) |*v| v.* = bf16(rnd.float(f32) * 4 - 2);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(row_pipe);
    for (bufs[0..5], 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setValue(@as(u32, QS), 5);
    enc.dispatchGroups(mtl.Size.of(1, LATENT / 64, M * H), mtl.Size.of(64, 1, 1));
    enc.setPipeline(nax_pipe);
    for (bufs[0..4], 0..) |b, i| enc.setBuffer(b, 0, i);
    enc.setBuffer(bufs[5], 0, 4);
    enc.setValue([2]i32{ M, QS }, 5);
    enc.dispatchGroups(mtl.Size.of(LATENT / 64, (M + 63) / 64, H), mtl.Size.of(128, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
    const row_out = bufs[4].slice(u16, M * H * LATENT);
    const nax_out = bufs[5].slice(u16, M * H * LATENT);
    var same: usize = 0;
    var worst: f64 = 0;
    for (0..M) |r| for (0..H) |h| {
        var big: f64 = 0;
        var outs: [LATENT]f64 = undefined;
        for (0..LATENT) |j| {
            var o: f64 = 0;
            for (0..NOPE) |i| {
                const wrow = h * PER_HEAD + i;
                const qv: u32 = (wb[wrow * (LATENT / 2) + j / 2] >> @intCast(4 * (j % 2))) & 15;
                const g = wrow * (LATENT / 64) + j / 64;
                const w: f64 = @as(f32, f32of(sc[g]) * @as(f32, @floatFromInt(qv)) + f32of(bi[g])); // the fp32 weight
                o += w * f32of(qp[r * QS + h * NOPE + i]);
            }
            outs[j] = o;
            big = @max(big, @abs(o));
        }
        for (0..LATENT) |j| {
            const i = (r * H + h) * LATENT + j;
            const got: f64 = f32of(nax_out[i]);
            const theirs: f64 = f32of(row_out[i]);
            const floor = big / 65536.0;
            try std.testing.expect(@abs(got - outs[j]) <= @max(@abs(outs[j]) / 128.0, floor));
            if (nax_out[i] == row_out[i]) same += 1 else worst = @max(worst, @abs(got - theirs) / @max(@max(@abs(got), @abs(theirs)) / 128.0, floor));
        }
    };
    const share = @as(f64, @floatFromInt(same)) / @as(f64, @floatFromInt(M * H * LATENT));
    std.debug.print("absorb on the tensor units: {d:.2}% of outputs equal the row kernel's, at most {d:.2} bf16 steps apart\n", .{ share * 100, worst });
    try std.testing.expect(share > 0.99 and worst <= 2);
}
