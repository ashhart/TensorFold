//! tf-glm-mm-groups [ROWS...]: prompt experts and dense products in column groups against affine_mm's: bits, time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const affine_mm = tf.affine_mm;
const groups = tf.glm.mm_groups;

const E = 288;
const topk = 8;
const D = 4096;
const I = 2048;

const B = struct { buf: mtl.Buffer, off: usize = 0 };
const Mat = struct { w: B, s: B, b: B, n: u32, k: u32 };

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn matrix(device: mtl.Device, r: std.Random, rows: usize, n: u32, k: usize) !Mat {
    const opts = mtl.ResourceOptions.shared;
    const m: Mat = .{ .w = .{ .buf = try device.buffer(rows * k / 2, opts) }, .s = .{ .buf = try device.buffer(rows * k / 64 * 2, opts) }, .b = .{ .buf = try device.buffer(rows * k / 64 * 2, opts) }, .n = n, .k = @intCast(k) };
    r.bytes(m.w.buf.slice(u8, rows * k / 2));
    for (m.s.buf.slice(u16, rows * k / 64), m.b.buf.slice(u16, rows * k / 64)) |*s, *b| {
        s.* = toBf((r.float(f32) + 0.5) / 256);
        b.* = toBf(-(r.float(f32) + 0.5) * 0.03);
    }
    return m;
}

/// Each row's top-8 distinct experts from skewed weights (load-balanced routers still give experts 0.3-3x the mean).
fn route(r: std.Random, rows: usize, picks: []i32) void {
    var w: [E]f64 = undefined;
    for (&w, 0..) |*v, e| v.* = 0.35 + 2.5 * std.math.pow(f64, @as(f64, @floatFromInt((e * 7919) % E)) / E, 3);
    for (0..rows) |i| {
        var taken: [E]bool = @splat(false);
        for (0..topk) |s| {
            var total: f64 = 0;
            for (w, taken) |v, t| total += if (t) 0 else v;
            var at = r.float(f64) * total;
            const pick = for (w, 0..) |v, e| {
                if (taken[e]) continue;
                at -= v;
                if (at <= 0) break e;
            } else E - 1;
            taken[pick] = true;
            picks[i * topk + s] = @intCast(pick);
        }
    }
}

const Io = struct { x: B, sums: B, gate: Mat, up: Mat, down: Mat, offs: B, order: B, act: [2]B, yp: [2]B, rows: u32, n: u32 };

/// Today's: core/affine_mm.zig's gathers at height `bm` (0: their own choice).
fn core(e: mtl.ComputeEncoder, mm: affine_mm.Pipes, io: *const Io, bm: u32, glu: bool, side: usize) void {
    if (bm == 0) {
        if (glu) return affine_mm.gatherGlu(e, mm, io.x, io.sums, io.gate, io.up, io.offs, io.order, io.act[side], io.n, E, topk, 10.0);
        return affine_mm.gatherTo(e, mm, io.act[side], io.sums, io.down, io.offs, io.yp[side], io.order, io.n, E);
    }
    const t = @min(io.n, (io.n + bm - 1) / bm + E - 1);
    if (glu) {
        e.setPipeline(mm[if (bm == 64) 7 else 6]);
        inline for (.{ io.x, io.gate.w, io.gate.s, io.gate.b, io.sums }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue(affine_mm.Args{ .rows = @intCast(io.n), .n = I, .k = D, .experts = E }, 5);
        inline for (.{ io.act[side], io.offs, io.order, io.up.w, io.up.s, io.up.b }, 6..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue([2]f32{ topk, 10.0 }, 12);
        return e.dispatchGroups(mtl.Size.of(I / 32, t, 1), mtl.Size.of(32, 2, 2));
    }
    e.setPipeline(mm[if (bm == 64) 5 else 4]);
    inline for (.{ io.act[side], io.down.w, io.down.s, io.down.b, io.sums }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(affine_mm.Args{ .rows = @intCast(io.n), .n = D, .k = I, .experts = E }, 5);
    inline for (.{ io.yp[side], io.offs, io.order }, 6..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.dispatchGroups(mtl.Size.of(D / 64, t, 1), mtl.Size.of(32, 2, 2));
}

/// Ours at height `bm` (0: groups.zig's choice).
fn ours(e: mtl.ComputeEncoder, p: groups.Pipes, io: *const Io, bm: u32, glu: bool, side: usize) void {
    if (bm == 0) {
        if (glu) return groups.gatherGlu(e, p, io.x, io.sums, io.gate, io.up, io.offs, io.order, io.act[side], io.n, E, topk, 10.0);
        return groups.gatherTo(e, p, io.act[side], io.sums, io.down, io.offs, io.yp[side], io.order, io.n, E);
    }
    const t = @min(io.n, (io.n + bm - 1) / bm + E - 1);
    if (glu) {
        e.setPipeline(p[if (bm == 64) 1 else 0]);
        inline for (.{ io.x, io.gate.w, io.gate.s, io.gate.b, io.sums }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue(affine_mm.Args{ .rows = @intCast(io.n), .n = I, .k = D, .experts = E }, 5);
        inline for (.{ io.act[side], io.offs, io.order, io.up.w, io.up.s, io.up.b }, 6..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setValue([2]f32{ topk, 10.0 }, 12);
        return e.dispatchGroups(mtl.Size.of(t * (I / 32), 1, 1), mtl.Size.of(128, 1, 1));
    }
    e.setPipeline(p[if (bm == 64) 3 else 2]);
    inline for (.{ io.act[side], io.down.w, io.down.s, io.down.b, io.sums }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(affine_mm.Args{ .rows = @intCast(io.n), .n = D, .k = I, .experts = E }, 5);
    inline for (.{ io.yp[side], io.offs, io.order }, 6..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.dispatchGroups(mtl.Size.of(t * (D / 64), 1, 1), mtl.Size.of(128, 1, 1));
}

/// The routed experts as a prompt chunk runs them, today's way (side 0) or ours (side 1) at height `bm`.
fn pass(q: mtl.Queue, mm: affine_mm.Pipes, p: groups.Pipes, io: *const Io, bm: u32, side: usize) !void {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    affine_mm.rowSums(e, mm, 64, io.x, io.sums, io.rows, D);
    if (side == 0) core(e, mm, io, bm, true, 0) else ours(e, p, io, bm, true, 1);
    affine_mm.rowSums(e, mm, 64, io.act[side], io.sums, io.n, I);
    if (side == 0) core(e, mm, io, bm, false, 0) else ours(e, p, io, bm, false, 1);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

/// One launch's GPU ms (a command buffer of 3), the activations' sums made before.
fn once(q: mtl.Queue, mm: affine_mm.Pipes, p: groups.Pipes, io: *const Io, side: usize, glu: bool) !f64 {
    const n = 3;
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |_| if (side == 0) core(e, mm, io, 0, glu, 0) else ours(e, p, io, 0, glu, 1);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    return cb.gpuSeconds() * 1e3 / n;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    var mm: affine_mm.Pipes = undefined;
    var p: groups.Pipes = undefined;
    {
        const src = try groups.source(device, gpa);
        defer gpa.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (affine_mm.names, &mm) |name, *pp| pp.* = try mtl.Pipeline.init(device, lib, name, false);
        for (groups.names, &p) |name, *pp| pp.* = try mtl.Pipeline.init(device, lib, name, false);
    }
    var prng = std.Random.DefaultPrng.init(0x7e);
    const r = prng.random();
    const opts = mtl.ResourceOptions.shared;
    const max_rows = 16384;
    const max_n = max_rows * topk;
    var io: Io = undefined;
    io.gate = try matrix(device, r, E * I, I, D);
    io.up = try matrix(device, r, E * I, I, D);
    io.down = try matrix(device, r, E * D, D, I);
    io.x = .{ .buf = try device.buffer(max_rows * D * 2, opts) };
    io.sums = .{ .buf = try device.buffer(max_n * (I / 64) * 4, opts) };
    io.offs = .{ .buf = try device.buffer(E * 4, opts) };
    io.order = .{ .buf = try device.buffer(max_n * 4, opts) };
    for (&io.act) |*a| a.* = .{ .buf = try device.buffer(max_n * I * 2, opts) };
    for (&io.yp) |*y| y.* = .{ .buf = try device.buffer(max_n * D * 2, opts) };
    for (io.x.buf.slice(u16, max_rows * D), 0..) |*v, i| v.* = if (i % 499 == 0) 0x8000 else toBf((r.float(f32) - 0.5) * 3);
    const picks = try gpa.alloc(i32, max_n);
    defer gpa.free(picks);
    std.debug.print("GPU {s}: 288 experts, top-8, gate/up 4096 -> 2 x 2048, down 2048 -> 4096, 4-bit/64\n", .{device.name()});
    var row_list: std.ArrayList(u32) = .empty;
    for (args[1..]) |a| try row_list.append(gpa, try std.fmt.parseInt(u32, a, 10));
    if (row_list.items.len == 0) try row_list.appendSlice(gpa, &.{ 1, 37, 300, 2048, 7697, 8192, 13616, 16384 });
    var compared: usize = 0;
    for (row_list.items) |rows| {
        io.rows = rows;
        io.n = rows * topk;
        route(r, rows, picks[0..io.n]);
        var counts: [E]i32 = @splat(0);
        for (picks[0..io.n]) |pk| counts[@intCast(pk)] += 1;
        const offs = io.offs.buf.slice(i32, E);
        var at: i32 = 0;
        for (offs, counts) |*o, c| {
            o.* = at;
            at += c;
        }
        var fill = offs[0..E].*;
        const order = io.order.buf.slice(i32, io.n);
        for (picks[0..io.n], 0..) |pk, i| { // sorted pairs in pair order within an expert (the prompt's sort's)
            order[@intCast(fill[@intCast(pk)])] = @intCast(i);
            fill[@intCast(pk)] += 1;
        }
        for ([_]u32{ 32, 64, 0 }) |bm| { // both heights against today's at the same height, then each one's own choice
            for (0..2) |side| {
                @memset(io.act[side].buf.slice(u16, max_n * I), 0x7fc1);
                @memset(io.yp[side].buf.slice(u16, max_n * D), 0x7fc1);
            }
            try pass(queue, mm, p, &io, bm, 0);
            try pass(queue, mm, p, &io, bm, 1);
            if (!std.mem.eql(u16, io.act[0].buf.slice(u16, max_n * I), io.act[1].buf.slice(u16, max_n * I)) or
                !std.mem.eql(u16, io.yp[0].buf.slice(u16, max_n * D), io.yp[1].buf.slice(u16, max_n * D)))
            {
                std.debug.print("{d} rows, {d}-row tiles: BITS DIFFER\n", .{ rows, bm });
                return error.BitsDiffer;
            }
            compared += io.n * (I + D);
        }
        var lo: i32 = std.math.maxInt(i32);
        var hi: i32 = 0;
        for (counts) |c| {
            lo = @min(lo, c);
            hi = @max(hi, c);
        }
        std.debug.print("{d} rows ({d} pairs, {d}-{d} an expert): equal at 32 and 64 rows and each choice\n", .{ rows, io.n, lo, hi });
        if (rows < 128) continue;
        var hit: f64 = 0; // the experts a launch reads
        for (counts) |c| hit += @floatFromInt(@intFromBool(c > 0));
        var t: [2][2][7]f64 = undefined; // [side][glu, down][trial], interleaved
        for (0..7) |trial| for (0..2) |side| {
            t[side][0][trial] = try once(queue, mm, p, &io, side, true);
            t[side][1][trial] = try once(queue, mm, p, &io, side, false);
        };
        for (&t) |*s| for (s) |*v| std.mem.sort(f64, v, {}, std.sort.asc(f64));
        const fl = @as(f64, @floatFromInt(io.n)) * D * I * 2;
        std.debug.print("  gate/up {d:.3} -> {d:.3} ms ({d:.1} -> {d:.1} TFLOPS), down {d:.3} -> {d:.3} ms ({d:.1} -> {d:.1} TFLOPS)\n", .{ t[0][0][3], t[1][0][3], 2 * fl / t[0][0][3] / 1e9, 2 * fl / t[1][0][3] / 1e9, t[0][1][3], t[1][1][3], fl / t[0][1][3] / 1e9, fl / t[1][1][3] / 1e9 });
        const wb = hit * D * I * (0.5 + 4.0 / 64.0); // an expert matrix's codes, scales and biases
        std.debug.print("  weights read: gate/up {d:.0} -> {d:.0} GB/s, down {d:.0} -> {d:.0} GB/s\n", .{ 2 * wb / t[0][0][3] / 1e6, 2 * wb / t[1][0][3] / 1e6, wb / t[0][1][3] / 1e6, wb / t[1][1][3] / 1e6 });
    }
    for ([_][3]u32{ .{ 300, 4096, 8192 }, .{ 1000, 2240, 4096 }, .{ 8192, 4096, 8192 }, .{ 8192, 4096, 16384 }, .{ 8192, 24896, 4096 }, .{ 16384, 4096, 8192 } }) |sh| {
        compared += try denseCase(device, queue, mm, p, r, sh[0], sh[1], sh[2]);
    }
    std.debug.print("{d} outputs equal\n", .{compared});
}

/// A dense product at prompt.zig's pitches: affine_mm.dense against mm_groups.dense, bits, then GPU time at 8K rows on.
fn denseCase(device: mtl.Device, queue: mtl.Queue, mm: affine_mm.Pipes, p: groups.Pipes, r: std.Random, M: u32, N: u32, K: u32) !usize {
    const opts = mtl.ResourceOptions.shared;
    const MK = @as(usize, M) * K;
    const x: B = .{ .buf = try device.buffer(MK * 2, opts) };
    defer x.buf.deinit();
    for (x.buf.slice(u16, MK), 0..) |*v, i| v.* = if (i % 997 == 0) 0x8000 else toBf((r.float(f32) - 0.5) * 3);
    const q: Mat = try matrix(device, r, N, N, K);
    defer for ([_]B{ q.w, q.s, q.b }) |b| b.buf.deinit();
    const sums: B = .{ .buf = try device.buffer(MK / 64 * 4, opts) };
    defer sums.buf.deinit();
    const ys = [2]B{ .{ .buf = try device.buffer(@as(usize, M) * N * 2, opts) }, .{ .buf = try device.buffer(@as(usize, M) * N * 2, opts) } };
    defer for (ys) |y| y.buf.deinit();
    for (ys) |y| @memset(y.buf.slice(u16, @as(usize, M) * N), 0x7fc1);
    const st: affine_mm.Strides = .{ .x_row = @intCast(K), .y_row = @intCast(N) };
    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        affine_mm.rowSums(e, mm, 64, x, sums, M, K);
        affine_mm.dense(e, mm, x, sums, q, ys[0], M);
        groups.dense(e, p, x, sums, q, ys[1], M, st);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }
    if (!std.mem.eql(u16, ys[0].buf.slice(u16, @as(usize, M) * N), ys[1].buf.slice(u16, @as(usize, M) * N))) {
        std.debug.print("dense {d} x {d} x {d}: BITS DIFFER\n", .{ M, N, K });
        return error.BitsDiffer;
    }
    std.debug.print("dense {d} rows, {d} x {d}: equal", .{ M, N, K });
    if (M < 8192) {
        std.debug.print("\n", .{});
        return @as(usize, M) * N;
    }
    var t: [2][7]f64 = undefined;
    for (0..7) |trial| for (0..2) |side| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..3) |_| if (side == 0) affine_mm.dense(e, mm, x, sums, q, ys[0], M) else groups.dense(e, p, x, sums, q, ys[1], M, st);
        e.end();
        cb.commit();
        cb.wait();
        t[side][trial] = cb.gpuSeconds() * 1e3 / 3;
    };
    for (&t) |*v| std.mem.sort(f64, v, {}, std.sort.asc(f64));
    const fl = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    std.debug.print("; {d:.3} -> {d:.3} ms ({d:.1} -> {d:.1} TFLOPS)\n", .{ t[0][3], t[1][3], fl / t[0][3] / 1e9, fl / t[1][3] / 1e9 });
    return @as(usize, M) * N;
}
