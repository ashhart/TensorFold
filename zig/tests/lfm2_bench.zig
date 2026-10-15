//! LFM2 prompt-chunk and one-row decode timing on the real weights (no server).
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm2;
const st = q.state;
const c = q.config;

fn now() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return error.ExpectedModelPromptChunkDecode;
    const length = try std.fmt.parseInt(usize, args[2], 10);
    const chunk = try std.fmt.parseInt(usize, args[3], 10);
    const steps = try std.fmt.parseInt(usize, args[4], 10);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    try prompts(m);
    try grouped(m);
    try promptBudget(m);
    try attnLong(m, 30720, 2048);
    try decodeBudget(m, 64);
    try decodeBudget(m, 32768);
    var big = try st.Scratch.init(init.gpa, m.device, &m.config.attention, chunk);
    defer big.deinit();
    var one = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 1);
    defer one.deinit();
    var cache = try st.Cache.init(init.gpa, m.device, &m.config.attention, length + steps + 1);
    defer cache.deinit();
    const ids = try init.gpa.alloc(u32, length + steps);
    defer init.gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % 120000);
    const t0 = now();
    const g0 = q.forward.gpu_seconds;
    var at: usize = 0;
    while (at < length) {
        const rows = @min(chunk, length - at);
        try q.forward.run(m, &big, &.{.{ .cache = &cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        at += rows;
    }
    const t1 = now();
    const g1 = q.forward.gpu_seconds;
    for (0..steps) |i| try q.forward.run(m, &one, &.{.{ .cache = &cache, .rows = 1 }}, ids[length + i ..][0..1], false, .all);
    const t2 = now();
    const pf: f64 = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const dc: f64 = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    const tps = @as(f64, @floatFromInt(length)) / pf;
    const dps = @as(f64, @floatFromInt(steps)) / dc;
    const line = "prompt {d} in {d}-row chunks: {d:.2} s, {d:.0} tok/s (GPU {d:.2} s); decode {d}: {d:.1} tok/s\n";
    std.debug.print(line, .{ length, chunk, pf, tps, g1 - g0, steps, dps });
    try windows(m);
}

const tile = @import("tensorfold").tile_qmm;

/// GPU TFLOPS of the prompt tiles on 512 rows of a dense 6144x2048 and a 2048x2048 matrix.
fn prompts(m: *q.Model) !void {
    const dev = m.device;
    const rows = 512;
    const in = try dev.buffer((rows + 32) * 11776 * 2, mtl.ResourceOptions.shared);
    defer in.deinit();
    for (in.slice(u16, rows * 2048), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5)) >> 16);
    const out = try dev.buffer(rows * 11776 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const xs = try dev.buffer(rows * 184 * 4, mtl.ResourceOptions.shared);
    defer xs.deinit();
    const b = m.weights.blocks[0];
    for ([_]q.weights.Linear{ b.mixer.conv.in, b.mixer.conv.out, b.ffn.dense.gate }) |w| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..20) |_| try tile.encode(e, &m.kernels.tiles, q.forward.matrix(w), in, 0, xs, 0, out, 0, rows);
        e.end();
        cb.commit();
        cb.wait();
        const t = cb.gpuSeconds() / 20.0;
        std.debug.print("tiles {d}x{d}, {d} rows: {d:.0} us, {d:.2} TFLOPS\n", .{ w.n, w.k, rows, t * 1e6, 2.0 * @as(f64, @floatFromInt(rows * w.n * w.k)) / t / 1e12 });
    }
}

/// Each window width's GPU us of the twin calls at their shapes, by the route forward.zig takes.
fn windows(m: *q.Model) !void {
    const dev = m.device;
    var sc = try st.Scratch.init(m.gpa, dev, &m.config.attention, st.batch_rows);
    defer sc.deinit();
    const slots = st.batch_rows * c.topk;
    var prng = std.Random.DefaultPrng.init(11);
    const picks = sc.picks.slice(u32, slots);
    for (0..st.batch_rows) |r| {
        var k: usize = 0;
        while (k < c.topk) {
            const e = prng.random().uintLessThan(u32, 32);
            if (std.mem.indexOfScalar(u32, picks[r * c.topk ..][0..k], e) != null) continue;
            picks[r * c.topk + k] = e;
            k += 1;
        }
    }
    for (sc.x.slice(u16, st.batch_rows * c.hidden), 0..) |*v, i| v.* = bf16(@floatFromInt(i % 89), 89);
    for (sc.act.slice(u16, slots * c.moe_intermediate), 0..) |*v, i| v.* = bf16(@floatFromInt(i % 97), 97);
    const tq = q.forward.matrix;
    const t = &m.kernels.tiles;
    const attn = m.weights.blocks[2].mixer.attention;
    const conv = m.weights.blocks[0].mixer.conv;
    const moe = m.weights.blocks[10].ffn.moe;
    const names = [_][]const u8{ "q 2048x2048", "conv in 6144x2048", "head", "experts glu", "experts down" };
    std.debug.print("window twin us by width:", .{});
    for (names) |n| std.debug.print(" {s};", .{n});
    std.debug.print("\n", .{});
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |w| {
        std.debug.print("{d:2} rows:", .{w});
        const groups: ?mtl.Buffer = if (w >= tile.group_rows) sc.groups else null;
        const n = w * c.topk;
        inline for (0..names.len) |which| {
            var best: f64 = 1e9;
            for (0..3) |_| {
                const cb = m.queue.commandBuffer();
                const e = cb.compute(.serial);
                for (0..10) |_| switch (which) {
                    0 => try tile.encodeTwin(e, t, tq(attn.q), sc.x, 0, sc.q, 0, w),
                    1 => try tile.encodeTwin(e, t, tq(conv.in), sc.x, 0, sc.bcx, 0, w),
                    2 => try tile.encodeTwin(e, t, tq(m.weights.embedding), sc.x, 0, sc.logits, 0, w),
                    3 => {
                        if (groups != null) try tile.encodeTwinGroup(e, t, sc.picks, n, sc.groups);
                        const gate = tq(moe.gate.one);
                        try tile.encodeTwinGlu(e, t, gate, tq(moe.up.one), sc.x, sc.act, sc.picks, groups, n, c.topk);
                    },
                    else => {
                        if (groups != null) try tile.encodeTwinGroup(e, t, sc.picks, n, sc.groups);
                        try tile.encodeTwinIndexed(e, t, tq(moe.down.one), sc.act, sc.experts, sc.picks, groups, n, 1);
                    },
                };
                e.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| std.debug.print("windows: {s}\n", .{msg});
                best = @min(best, cb.gpuSeconds() * 1e5);
            }
            std.debug.print(" {d:6.0}", .{best});
        }
        std.debug.print("\n", .{});
    }
}

/// A bf16 in [-0.5, 0.5) from i / n.
fn bf16(i: f32, n: f32) u16 {
    return @truncate(@as(u32, @bitCast(i / n - 0.5)) >> 16);
}

/// GPU TFLOPS of the grouped tiles on 8,192 rows, 128 an expert, over layer 10's gate experts.
fn grouped(m: *q.Model) !void {
    const dev = m.device;
    const rows = 8192;
    const x = m.weights.blocks[10].ffn.moe;
    const in = try dev.buffer((rows + 32) * 2048 * 2, mtl.ResourceOptions.shared);
    defer in.deinit();
    for (in.slice(u16, rows * 2048), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5)) >> 16);
    const out = try dev.buffer(rows * 2048 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const xs = try dev.buffer(rows * 32 * 4, mtl.ResourceOptions.shared);
    defer xs.deinit();
    const starts = try dev.buffer(65 * 4, mtl.ResourceOptions.shared);
    defer starts.deinit();
    for (starts.slice(u32, 65), 0..) |*v, i| v.* = @intCast(i * 128);
    const work = try dev.buffer((rows / 32 + 66) * 4, mtl.ResourceOptions.shared);
    defer work.deinit();
    const ident = try dev.buffer(rows * 4, mtl.ResourceOptions.shared);
    defer ident.deinit();
    for (ident.slice(u32, rows), 0..) |*v, i| v.* = @intCast(i);
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    tile.encodeWork(e, &m.kernels.tiles, starts, 64, work);
    for (0..10) |_| try tile.encodeGrouped(e, &m.kernels.tiles, q.forward.matrix(x.gate.one), 64, in, xs, ident, out, rows, starts, work);
    e.end();
    cb.commit();
    cb.wait();
    const t = cb.gpuSeconds() / 10.0;
    std.debug.print("grouped tiles 1536x2048 x64 experts, {d} rows: {d:.0} us, {d:.2} TFLOPS\n", .{ rows, t * 1e6, 2.0 * @as(f64, @floatFromInt(rows * 1536 * 2048)) / t / 1e12 });
}

/// A 2,048-row prompt chunk's GPU budget: each kernel at its real shape (picks spread), times its launches.
fn promptBudget(m: *q.Model) !void {
    const rows = 2048;
    const slots = rows * 4;
    const dev = m.device;
    var sc = try st.Scratch.init(m.gpa, dev, &m.config.attention, rows);
    defer sc.deinit();
    var cache = try st.Cache.init(m.gpa, dev, &m.config.attention, rows + 64);
    defer cache.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    for (sc.picks.slice(u32, slots), 0..) |*v, i| v.* = @intCast((i % 4) * 16 + prng.random().uintLessThan(u32, 16));
    for (sc.x.slice(u16, rows * 2048), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 89)) / 89.0 - 0.5)) >> 16);
    const tq = q.forward.matrix;
    const t = &m.kernels.tiles;
    const conv = m.weights.blocks[0].mixer.conv;
    const attn = m.weights.blocks[2].mixer.attention;
    const moe = m.weights.blocks[10].ffn.moe;
    const dense = m.weights.blocks[0].ffn.dense;
    const Item = struct { name: []const u8, per: f64 };
    const items = [_]Item{
        .{ .name = "sums+tiles conv in", .per = 30 },       .{ .name = "sums+tiles conv out", .per = 30 },
        .{ .name = "sums+tiles attn q,k,v,o", .per = 10 },  .{ .name = "sums+tiles dense x3", .per = 2 },
        .{ .name = "route", .per = 38 },                    .{ .name = "sort+work", .per = 38 },
        .{ .name = "sums+grouped glu gate,up", .per = 38 }, .{ .name = "sums+grouped down", .per = 38 },
        .{ .name = "combine sorted", .per = 38 },           .{ .name = "norm", .per = 81 },
        .{ .name = "conv kernel", .per = 30 },              .{ .name = "attn mma 2k", .per = 10 },
    };
    var total: f64 = 0;
    inline for (items, 0..) |it, which| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        const reps = 3;
        for (0..reps) |_| switch (which) {
            0 => {
                tile.encodeSums(e, t, 2048, sc.x, 0, sc.sums, rows);
                try tile.encode(e, t, tq(conv.in), sc.x, 0, sc.sums, 0, sc.bcx, 0, rows);
            },
            1 => {
                tile.encodeSums(e, t, 2048, sc.y, 0, sc.sums, rows);
                try tile.encode(e, t, tq(conv.out), sc.y, 0, sc.sums, 0, sc.r, 0, rows);
            },
            2 => for ([_]q.weights.Linear{ attn.q, attn.k, attn.v, attn.out }) |w| {
                tile.encodeSums(e, t, 2048, sc.x, 0, sc.sums, rows);
                try tile.encode(e, t, tq(w), sc.x, 0, sc.sums, 0, sc.q, 0, rows);
            },
            3 => for ([_]q.weights.Linear{ dense.gate, dense.up, dense.down }) |w| {
                tile.encodeSums(e, t, w.k, sc.act, 0, sc.sums, rows);
                try tile.encode(e, t, tq(w), sc.act, 0, sc.sums, 0, sc.gate, 0, rows);
            },
            4 => {
                e.setPipeline(m.kernels.get("lfm2_route_logits"));
                e.setBuffer(sc.x, 0, 0);
                e.setBuffer(moe.router.buffer, moe.router.offset, 1);
                e.setBuffer(sc.route, 0, 2);
                e.dispatchGroups(mtl.Size.of(8, rows, 1), mtl.Size.of(256, 1, 1));
                e.setPipeline(m.kernels.get("lfm2_route_select"));
                e.setBuffer(sc.route, 0, 0);
                e.setBuffer(moe.bias.buffer, moe.bias.offset, 1);
                e.setValue(@as(f32, 1), 2);
                e.setBuffer(sc.weights, 0, 4);
                e.setBuffer(cache.logits, 0, 3); // keep the bench's spread picks
                e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
            },
            5 => {
                e.setPipeline(m.kernels.get("lfm2_sort"));
                e.setBuffer(sc.picks, 0, 0);
                e.setValue(@as(u32, slots), 1);
                e.setBuffer(sc.starts, 0, 2);
                e.setBuffer(sc.order, 0, 3);
                e.setBuffer(sc.pos, 0, 4);
                e.setBuffer(sc.tokens, 0, 5);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(64 * 16, 1, 1));
                tile.encodeWork(e, t, sc.starts, 64, sc.work);
            },
            6 => {
                tile.encodeSums(e, t, 2048, sc.x, 0, sc.sums, rows);
                try tile.encodeGroupedGlu(e, t, tq(moe.gate.one), tq(moe.up.one), 64, sc.x, sc.sums, sc.tokens, sc.act, slots, sc.starts, sc.work);
            },
            7 => {
                tile.encodeSums(e, t, 1536, sc.act, 0, sc.sums, slots);
                try tile.encodeGrouped(e, t, tq(moe.down.one), 64, sc.act, sc.sums, sc.ident, sc.experts, slots, sc.starts, sc.work);
            },
            8 => {
                e.setPipeline(m.kernels.get("lfm2_combine_sorted"));
                e.setBuffer(sc.experts, 0, 0);
                e.setBuffer(sc.weights, 0, 1);
                e.setBuffer(sc.r, 0, 2);
                e.setBuffer(sc.pos, 0, 3);
                e.dispatchThreads(mtl.Size.of(2048, rows, 1), mtl.Size.of(256, 1, 1));
            },
            9 => {
                e.setPipeline(m.kernels.get("lfm2_norm"));
                e.setBuffer(sc.h, 0, 0);
                e.setBuffer(sc.r, 0, 1);
                e.setBuffer(conv.weight.buffer, conv.weight.offset, 2);
                e.setValue(@as(f32, 1e-5), 3);
                e.setValue(@as(u32, 1), 4);
                e.setBuffer(sc.x, 0, 5);
                e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(512, 1, 1));
            },
            10 => {
                e.setPipeline(m.kernels.get("lfm2_conv"));
                e.setBuffer(sc.bcx, 0, 0);
                e.setBuffer(cache.blocks[0].conv, 0, 1);
                e.setBuffer(conv.weight.buffer, conv.weight.offset, 2);
                e.setBuffer(sc.y, 0, 3);
                e.setBuffer(sc.snapshots[0].?, 0, 4);
                e.setValue(@as(u32, rows - 1), 5);
                e.dispatchThreads(mtl.Size.of(2048, rows, 1), mtl.Size.of(256, 1, 1));
            },
            else => {
                const kv = cache.blocks[2].attention;
                e.setPipeline(m.kernels.get("lfm2_attn_mma_r2"));
                e.setBuffer(sc.queries, 0, 0);
                e.setBuffer(kv.keys, 0, 1);
                e.setBuffer(kv.values, 0, 2);
                e.setBytes(std.mem.asBytes(&[4]u32{ 0, @intCast(cache.capacity), @bitCast(@as(f32, 0.125)), 1 }), 3);
                e.setBuffer(sc.pm, 0, 4);
                e.setBuffer(sc.pl, 0, 5);
                e.setBuffer(sc.po, 0, 6);
                e.setValue(@as(u32, rows), 7);
                e.dispatchGroups(mtl.Size.of(rows / st.attn_rows, 8, 1), mtl.Size.of(128, 1, 1));
                e.setPipeline(m.kernels.get("lfm2_attn_merge"));
                e.setBuffer(sc.pm, 0, 0);
                e.setBuffer(sc.pl, 0, 1);
                e.setBuffer(sc.po, 0, 2);
                e.setBytes(std.mem.asBytes(&[4]u32{ 0, @intCast(cache.capacity), @bitCast(@as(f32, 0.125)), 1 }), 3);
                e.setBuffer(sc.y, 0, 4);
                e.dispatchGroups(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
            },
        };
        e.end();
        cb.commit();
        cb.wait();
        const ms = cb.gpuSeconds() * 1e3 / reps;
        total += ms * it.per;
        std.debug.print("{s}: {d:.2} ms x {d} = {d:.0} ms\n", .{ it.name, ms, it.per, ms * it.per });
    }
    std.debug.print("prompt chunk budget: {d:.0} ms for {d} rows ({d:.0} tok/s)\n", .{ total, rows, rows / total * 1e3 });
}

/// GPU time of one layer's prompt attention (mma and merge) for `rows` rows over `len` cached keys, best of 7.
fn attnLong(m: *q.Model, len: usize, rows: usize) !void {
    const dev = m.device;
    var sc = try st.Scratch.init(m.gpa, dev, &m.config.attention, rows);
    defer sc.deinit();
    var cache = try st.Cache.init(m.gpa, dev, &m.config.attention, len + rows + 64);
    defer cache.deinit();
    const kv = cache.blocks[2].attention;
    const kv_count = c.kv_heads * cache.capacity * c.head_dim;
    for (kv.keys.slice(u16, kv_count), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 89)) / 89.0 - 0.5)) >> 16);
    @memcpy(kv.values.contents()[0 .. kv_count * 2], kv.keys.contents()[0 .. kv_count * 2]);
    @memcpy(sc.queries.contents()[0 .. rows * c.hidden * 2], kv.keys.contents()[0 .. rows * c.hidden * 2]);
    const args = [4]u32{ @intCast(len), @intCast(cache.capacity), @bitCast(@as(f32, 0.125)), 1 };
    const used = (len + rows + st.chunk_keys - 1) / st.chunk_keys;
    var best: f64 = 1e9;
    for (0..7) |_| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(m.kernels.get("lfm2_attn_mma_r2"));
        e.setBuffer(sc.queries, 0, 0);
        e.setBuffer(kv.keys, 0, 1);
        e.setBuffer(kv.values, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(sc.pm, 0, 4);
        e.setBuffer(sc.pl, 0, 5);
        e.setBuffer(sc.po, 0, 6);
        e.setValue(@as(u32, @intCast(rows)), 7);
        e.dispatchGroups(mtl.Size.of((rows + st.attn_rows - 1) / st.attn_rows, c.kv_heads, used), mtl.Size.of(128, 1, 1));
        e.setPipeline(m.kernels.get("lfm2_attn_merge"));
        e.setBuffer(sc.pm, 0, 0);
        e.setBuffer(sc.pl, 0, 1);
        e.setBuffer(sc.po, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(sc.y, 0, 4);
        e.dispatchGroups(mtl.Size.of(c.query_heads, rows, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("attn microbench: {s}\n", .{msg});
        best = @min(best, cb.gpuSeconds() * 1e3);
    }
    const flops = 4.0 * @as(f64, @floatFromInt(c.query_heads * c.head_dim * rows)) * (@as(f64, @floatFromInt(len)) + @as(f64, @floatFromInt(rows)) / 2.0);
    std.debug.print("attn mma+merge, {d} rows at {d} keys: best of 7 {d:.2} ms, {d:.2} TFLOPS\n", .{ rows, len, best, flops / best / 1e9 });
}

fn linearBytes(w: q.weights.Linear) usize {
    return w.n * w.k * w.bits / 8 + 2 * w.n * (w.k / 64) * 2;
}

/// One decode step's GPU budget at `len` context: each kernel class at its one-row shape over every layer of its kind.
fn decodeBudget(m: *q.Model, len: usize) !void {
    const dev = m.device;
    var sc = try st.Scratch.init(m.gpa, dev, &m.config.attention, 1);
    defer sc.deinit();
    var cache = try st.Cache.init(m.gpa, dev, &m.config.attention, len + 64);
    defer cache.deinit();
    cache.len = len;
    // Rotating through the layers, no call finds its weights in cache; each class reports the bytes it must read.
    const ids = try dev.buffer(64, mtl.ResourceOptions.shared);
    defer ids.deinit();
    for (ids.slice(u32, 4), 0..) |*v, i| v.* = @intCast(i * 13 + 1);
    const tq = q.forward.matrix;
    const t = &m.kernels.tiles;
    const scale: f32 = 1.0 / @sqrt(@as(f32, c.head_dim));
    const names = [_][]const u8{ "norm x43", "conv in twin", "conv kernel", "conv out twin", "attn q,k,v,o twins", "qk+attn+merge", "dense gate,up,swiglu,down", "route", "experts glu", "experts down", "combine+norm", "head twin (6-bit)" };
    var total: f64 = 0;
    inline for (names, 0..) |name, which| {
        var bytes: usize = 0;
        var launches_n: usize = 0;
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (m.weights.blocks, 0..) |block, layer| {
            switch (which) {
                0 => {
                    for (0..@as(usize, if (block.ffn == .dense) 2 else 1) + @intFromBool(layer == 0)) |_| {
                        e.setPipeline(m.kernels.get("lfm2_norm"));
                        e.setBuffer(sc.h, 0, 0);
                        e.setBuffer(sc.r, 0, 1);
                        e.setBuffer(block.operator_norm.buffer, block.operator_norm.offset, 2);
                        e.setValue(@as(f32, 1e-5), 3);
                        e.setValue(@as(u32, 1), 4);
                        e.setBuffer(sc.x, 0, 5);
                        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
                        launches_n += 1;
                    }
                },
                1, 2, 3 => if (block.mixer == .conv) {
                    const cv = block.mixer.conv;
                    if (which == 1) {
                        try tile.encodeTwin(e, t, tq(cv.in), sc.x, 0, sc.bcx, 0, 1);
                        bytes += linearBytes(cv.in);
                    } else if (which == 2) {
                        e.setPipeline(m.kernels.get("lfm2_conv"));
                        e.setBuffer(sc.bcx, 0, 0);
                        e.setBuffer(cache.blocks[layer].conv, 0, 1);
                        e.setBuffer(cv.weight.buffer, cv.weight.offset, 2);
                        e.setBuffer(sc.y, 0, 3);
                        e.setBuffer(sc.snapshots[layer].?, 0, 4);
                        e.setValue(@as(u32, 0), 5);
                        e.dispatchThreads(mtl.Size.of(2048, 1, 1), mtl.Size.of(256, 1, 1));
                    } else {
                        try tile.encodeTwin(e, t, tq(cv.out), sc.y, 0, sc.r, 0, 1);
                        bytes += linearBytes(cv.out);
                    }
                    launches_n += 1;
                },
                4, 5 => if (block.mixer == .attention) {
                    const a = block.mixer.attention;
                    if (which == 4) {
                        for ([_]q.weights.Linear{ a.q, a.k, a.v, a.out }) |w| {
                            try tile.encodeTwin(e, t, tq(w), sc.x, 0, sc.q, 0, 1);
                            bytes += linearBytes(w);
                            launches_n += 1;
                        }
                    } else {
                        const kv = cache.blocks[layer].attention;
                        e.setPipeline(m.kernels.get("lfm2_qk"));
                        e.setBuffer(sc.q, 0, 0);
                        e.setBuffer(sc.k, 0, 1);
                        e.setBuffer(sc.v, 0, 2);
                        e.setBuffer(a.q_norm.buffer, a.q_norm.offset, 3);
                        e.setBuffer(a.k_norm.buffer, a.k_norm.offset, 4);
                        e.setBytes(std.mem.asBytes(&[4]u32{ @intCast(len), @intCast(cache.capacity), @bitCast(@as(f32, 1e-5)), @bitCast(@as(f32, 20.0)) }), 5);
                        e.setBuffer(sc.queries, 0, 6);
                        e.setBuffer(kv.keys, 0, 7);
                        e.setBuffer(kv.values, 0, 8);
                        e.dispatchGroups(mtl.Size.of(c.query_heads + c.kv_heads, 1, 1), mtl.Size.of(32, 1, 1));
                        const used = (len + 1 + st.chunk_keys - 1) / st.chunk_keys * st.subs;
                        e.setPipeline(m.kernels.get("lfm2_attn_mma"));
                        e.setBuffer(sc.queries, 0, 0);
                        e.setBuffer(kv.keys, 0, 1);
                        e.setBuffer(kv.values, 0, 2);
                        e.setBytes(std.mem.asBytes(&[4]u32{ @intCast(len), @intCast(cache.capacity), @bitCast(scale), st.subs }), 3);
                        e.setBuffer(sc.pm, 0, 4);
                        e.setBuffer(sc.pl, 0, 5);
                        e.setBuffer(sc.po, 0, 6);
                        e.setValue(@as(u32, 1), 7);
                        e.dispatchGroups(mtl.Size.of(1, c.kv_heads, used), mtl.Size.of(128, 1, 1));
                        e.setPipeline(m.kernels.get("lfm2_attn_merge"));
                        e.setBuffer(sc.pm, 0, 0);
                        e.setBuffer(sc.pl, 0, 1);
                        e.setBuffer(sc.po, 0, 2);
                        e.setBytes(std.mem.asBytes(&[4]u32{ @intCast(len), @intCast(cache.capacity), @bitCast(scale), st.subs }), 3);
                        e.setBuffer(sc.y, 0, 4);
                        e.dispatchGroups(mtl.Size.of(c.query_heads, 1, 1), mtl.Size.of(32, 1, 1));
                        bytes += 2 * (len + 1) * c.kv_heads * c.head_dim * 2;
                        launches_n += 3;
                    }
                },
                6 => if (block.ffn == .dense) {
                    const d = block.ffn.dense;
                    try tile.encodeTwin(e, t, tq(d.gate), sc.x, 0, sc.gate, 0, 1);
                    try tile.encodeTwin(e, t, tq(d.up), sc.x, 0, sc.up, 0, 1);
                    e.setPipeline(m.kernels.get("lfm2_swiglu"));
                    e.setBuffer(sc.gate, 0, 0);
                    e.setBuffer(sc.up, 0, 1);
                    e.setBuffer(sc.act, 0, 2);
                    e.setValue(@as(u32, c.intermediate), 3);
                    e.dispatchThreads(mtl.Size.of(c.intermediate / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    try tile.encodeTwin(e, t, tq(d.down), sc.act, 0, sc.r, 0, 1);
                    bytes += linearBytes(d.gate) + linearBytes(d.up) + linearBytes(d.down);
                    launches_n += 4;
                },
                7, 8, 9, 10 => if (block.ffn == .moe) {
                    const x = block.ffn.moe;
                    if (which == 7) {
                        e.setPipeline(m.kernels.get("lfm2_route_logits"));
                        e.setBuffer(sc.x, 0, 0);
                        e.setBuffer(x.router.buffer, x.router.offset, 1);
                        e.setBuffer(sc.route, 0, 2);
                        e.dispatchGroups(mtl.Size.of(c.experts / 8, 1, 1), mtl.Size.of(256, 1, 1));
                        e.setPipeline(m.kernels.get("lfm2_route_select"));
                        e.setBuffer(sc.route, 0, 0);
                        e.setBuffer(x.bias.buffer, x.bias.offset, 1);
                        e.setValue(@as(f32, 1), 2);
                        e.setBuffer(sc.picks, 0, 3);
                        e.setBuffer(sc.weights, 0, 4);
                        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
                        bytes += c.experts * c.hidden * 4;
                        launches_n += 2;
                    } else if (which == 8) {
                        try tile.encodeTwinGlu(e, t, tq(x.gate.one), tq(x.up.one), sc.x, sc.act, ids, null, 4, 4);
                        bytes += 4 * (linearBytes(x.gate.one) + linearBytes(x.up.one));
                        launches_n += 1;
                    } else if (which == 9) {
                        try tile.encodeTwinIndexed(e, t, tq(x.down.one), sc.act, sc.experts, ids, null, 4, 1);
                        bytes += 4 * linearBytes(x.down.one);
                        launches_n += 1;
                    } else {
                        e.setPipeline(m.kernels.get("lfm2_combine_norm"));
                        e.setBuffer(sc.experts, 0, 0);
                        e.setBuffer(sc.weights, 0, 1);
                        e.setBuffer(sc.h, 0, 2);
                        e.setBuffer(block.operator_norm.buffer, block.operator_norm.offset, 3);
                        e.setValue(@as(f32, 1e-5), 4);
                        e.setBuffer(sc.x, 0, 5);
                        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
                        launches_n += 1;
                    }
                },
                else => if (layer == 0) {
                    try tile.encodeTwin(e, t, tq(m.weights.embedding), sc.x, 0, sc.logits, 0, 1);
                    bytes += linearBytes(m.weights.embedding);
                    launches_n += 1;
                },
            }
        }
        e.end();
        cb.commit();
        cb.wait();
        const ms = cb.gpuSeconds() * 1e3;
        total += ms;
        const mb = @as(f64, @floatFromInt(bytes)) / 1e6;
        std.debug.print("{s}: {d:.3} ms, {d} launches, {d:.0} MB, {d:.0} GB/s\n", .{ name, ms, launches_n, mb, mb / ms });
    }
    std.debug.print("decode step budget at {d} context: {d:.2} ms ({d:.0} tok/s)\n", .{ len, total, 1e3 / total });
}
