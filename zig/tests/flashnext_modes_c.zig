const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;

const opts = fz.opts;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const CAP = fz.CAP;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const Buf = fz.Buf;
const Entry = fz.Entry;
const Variant = fz.Variant;
const Site = fz.Site;
const readAll = fz.readAll;
const Run = fz.Run;
const TOP = fz.TOP;
const KW = fz.KW;
const Select = fz.Select;
const GSelect = fz.GSelect;
const Hc = fz.Hc;
const Layer = fz.Layer;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const Ple = fz.Ple;
const Tmp = fz.Tmp;
const f32Buf = fz.f32Buf;
const i32Buf = fz.i32Buf;
const Mtp = fz.Mtp;
const Model = fz.Model;
const DepthRule = fz.DepthRule;
const copyDrafts = fz.copyDrafts;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const jsonInt = fz.jsonInt;

// 6-bit tensor-unit projections for prompt chunks: checked and timed
pub fn qmm6(r: *Run, m: *Model, device: mtl.Device, arena: std.mem.Allocator) !void {
    const lib = try mtl.Library.fromSource(device, try fz.frags.source(device, arena, ks.flashnext_qmm6), mtl.CompileOptions.mlx());
    const qmm_pipe = try mtl.Pipeline.init(device, lib, "tf_qmm6_t_nax", false);
    const off_pipe = try mtl.Pipeline.init(device, lib, "tf_expert_offsets6", false);
    const g64_pipe = try mtl.Pipeline.init(device, lib, "tf_gather_qmm6_nax_64", false);
    const W = m.layers[0].ex[0];
    const S = m.layers[0].ex[1];
    const Bi = m.layers[0].ex[2];
    const K: usize = 2560;
    const WPR = K * 6 / 32;
    const KG = K / 32;
    const Ref = struct {
        fn bf(v: u16) f64 {
            return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
        }
        fn code(w: []const u32, j: usize) u32 { // value j of a row's little-endian 6-bit stream
            const bit = 6 * j;
            const word = bit / 32;
            const off: u5 = @intCast(bit % 32);
            var v = w[word] >> off;
            if (@as(usize, off) > 26) v |= w[word + 1] << @intCast(32 - @as(usize, off));
            return v & 63;
        }
        fn dot(x: []const u16, w: []const u32, sc: []const u16, bi: []const u16, k: usize) f64 {
            var acc: f64 = 0;
            for (0..k) |j| acc += bf(x[j]) * (bf(sc[j / 32]) * @as(f64, @floatFromInt(code(w, j))) + bf(bi[j / 32]));
            return acc;
        }
    };
    var seed: u64 = 12345;
    const Rand = struct {
        fn next(st: *u64) u32 {
            st.* = st.* *% 6364136223846793005 +% 1442695040888963407;
            return @intCast(st.* >> 33);
        }
        fn bf16(st: *u64) u16 {
            const f: f32 = (@as(f32, @floatFromInt(next(st) % 20001)) - 10000.0) / 10000.0;
            return @intCast(@as(u32, @bitCast(f)) >> 16);
        }
    };
    const wv = W.b.contents();
    const w32: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, wv[0..W.b.length()]));
    const s16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, S.b.contents()[0..S.b.length()]));
    const b16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, Bi.b.contents()[0..Bi.b.length()]));
    // dense: experts 0..15's gate rows as one [10240, 2560] projection of 512 prompt rows
    {
        const M: usize = 512;
        const N: usize = 10240;
        const X = try r.buffer(M * K * 2);
        const x16 = X.slice(u16, M * K);
        for (x16) |*v| v.* = Rand.bf16(&seed);
        const Y = try r.buffer(M * N * 2);
        const P = try i32Buf(r, &.{ @intCast(K), @intCast(N), @intCast(M) });
        var gpu: f64 = 0;
        for (0..21) |it| {
            const cb = r.queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(qmm_pipe);
            for ([_]Buf{ W, S, Bi, .{ .b = X }, P, .{ .b = Y } }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
            enc.dispatchThreads(mtl.Size.of(((N + 63) / 64) * 128, (M + 63) / 64, 1), mtl.Size.of(128, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("qmm6: {s}", .{msg});
                return error.GpuFailed;
            }
            if (it > 0) gpu += cb.gpuSeconds();
        }
        const y16 = Y.slice(u16, M * N);
        var worst: f64 = 0;
        for (0..64) |q| {
            const row = Rand.next(&seed) % M;
            const n = Rand.next(&seed) % N;
            const want_v = Ref.dot(x16[row * K .. (row + 1) * K], w32[n * WPR .. (n + 1) * WPR], s16[n * KG .. (n + 1) * KG], b16[n * KG .. (n + 1) * KG], K);
            const got = Ref.bf(y16[row * N + n]);
            const err = @abs(got - want_v) / @max(1.0, @abs(want_v));
            worst = @max(worst, err);
            _ = q;
        }
        const ms = gpu * 1e3 / 20;
        std.debug.print("qmm6 dense {d}x{d}x{d}: {d:.3} ms, {d:.1} TFLOP/s, worst rel err {e:.2}\n", .{ M, N, K, ms, 2.0 * @as(f64, @floatFromInt(M * N * K)) / (ms * 1e9), worst });
    }
    // gather: 512 prompt rows x 10 experts, pairs sorted by expert, through all 512 experts' gate projections
    {
        const M: usize = 512 * 10;
        const N: usize = 640;
        const E: usize = 512;
        const ids = try r.buffer(M * 4);
        const idv = ids.slice(u32, M);
        for (idv) |*v| v.* = @intCast(Rand.next(&seed) % E);
        std.mem.sort(u32, idv, {}, std.sort.asc(u32));
        const X = try r.buffer(M * K * 2);
        const x16 = X.slice(u16, M * K);
        for (x16) |*v| v.* = Rand.bf16(&seed);
        const Y = try r.buffer(M * N * 2);
        const O = try r.buffer((E + 1) * 4);
        const PO = try i32Buf(r, &.{@intCast(M)});
        const P = try i32Buf(r, &.{ @intCast(M), @intCast(N), @intCast(K), @intCast(E) });
        const tiles = M / 64 + E;
        var gpu: f64 = 0;
        for (0..21) |it| {
            const cb = r.queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(off_pipe);
            for ([_]Buf{ .{ .b = ids }, PO, .{ .b = O } }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
            enc.dispatchThreads(mtl.Size.of(E, 1, 1), mtl.Size.of(256, 1, 1));
            enc.setPipeline(g64_pipe);
            for ([_]Buf{ .{ .b = X }, W, S, Bi, .{ .b = O }, P, .{ .b = Y } }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
            enc.dispatchThreads(mtl.Size.of(((N + 63) / 64) * 128, tiles, 1), mtl.Size.of(128, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("gather6: {s}", .{msg});
                return error.GpuFailed;
            }
            if (it > 0) gpu += cb.gpuSeconds();
        }
        const y16 = Y.slice(u16, M * N);
        var worst: f64 = 0;
        for (0..64) |_| {
            const row = Rand.next(&seed) % M;
            const n = Rand.next(&seed) % N;
            const g = idv[row] * N + n;
            const want_v = Ref.dot(x16[row * K .. (row + 1) * K], w32[g * WPR .. (g + 1) * WPR], s16[g * KG .. (g + 1) * KG], b16[g * KG .. (g + 1) * KG], K);
            const got = Ref.bf(y16[row * N + n]);
            worst = @max(worst, @abs(got - want_v) / @max(1.0, @abs(want_v)));
        }
        const ms = gpu * 1e3 / 20;
        std.debug.print("gather6 {d} pairs x{d}x{d} over {d} experts: {d:.3} ms, {d:.1} TFLOP/s, worst rel err {e:.2}\n", .{ M, N, K, E, ms, 2.0 * @as(f64, @floatFromInt(M * N * K)) / (ms * 1e9), worst });
    }
    return;
}

// how many experts a window's rows share, layer by layer
pub fn overlap(r: *Run, m: *Model, ref: std.json.Value, gpa: std.mem.Allocator) !void {
    const pr = ref.object.get("prompt").?.array.items;
    const want0 = ref.object.get("tokens").?.array.items;
    r.ar = .{ .b = try r.buffer(64) };
    r.ar.b.slice(i32, 1)[0] = 1;
    const pb: Buf = .{ .b = try r.buffer(LAYERS * MAXR * 10 * 4) };
    var pk: [MAXR]u32 = undefined;
    m.reset();
    for (pr) |x| {
        try m.window(&.{@intCast(x.integer)}, &pk);
        m.keepRows(&.{@intCast(x.integer)}, 1);
    }
    r.probe = pb;
    const picks = pb.b.slice(u32, LAYERS * MAXR * 10);
    const Count = struct {
        fn distinct(sets: []std.AutoHashMap(u32, void)) f64 {
            var n: usize = 0;
            for (sets) |*u| n += u.count();
            return @as(f64, @floatFromInt(n)) / LAYERS;
        }
        fn add(sets: []std.AutoHashMap(u32, void), all: []const u32, rows: usize) !void {
            for (0..LAYERS) |l| for (all[l * MAXR * 10 .. l * MAXR * 10 + rows * 10]) |e| try sets[l].put(e, {});
        }
    };
    var chain_sum: [4]f64 = @splat(0);
    var sib_sum: [4]f64 = @splat(0);
    var far: [LAYERS]std.AutoHashMap(u32, void) = undefined;
    for (&far) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
    const starts = [_]usize{ 0, 8, 16, 24 };
    for (starts) |s0| {
        for ([_]usize{ 8, 4, 2, 1 }, 0..) |rows, ri| { // chain windows from the pending token, widest first
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[s0 + i].integer);
            try m.window(toks[0..rows], &pk);
            var sets: [LAYERS]std.AutoHashMap(u32, void) = undefined;
            for (&sets) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
            try Count.add(&sets, picks, rows);
            chain_sum[3 - ri] += Count.distinct(&sets);
            if (rows == 1) try Count.add(&far, picks, 1);
        }
        // the target's 8 best tokens for the next position, each a row from the same state (tree siblings)
        const lg = m.t.logits.b.slice(u16, VOCAB);
        var best: [8]u32 = undefined;
        var bestv: [8]f32 = @splat(-std.math.inf(f32));
        for (lg, 0..) |raw, id| {
            const v: f32 = @bitCast(@as(u32, raw) << 16);
            if (v <= bestv[7]) continue;
            var k: usize = 7;
            while (k > 0 and v > bestv[k - 1]) : (k -= 1) {
                bestv[k] = bestv[k - 1];
                best[k] = best[k - 1];
            }
            bestv[k] = v;
            best[k] = @intCast(id);
        }
        m.keepRows(&.{@intCast(want0[s0].integer)}, 1);
        var sib: [LAYERS]std.AutoHashMap(u32, void) = undefined;
        for (&sib) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
        for (best, 0..) |cand, ci| {
            try m.window(&.{cand}, &pk);
            try Count.add(&sib, picks, 1);
            const at: ?usize = switch (ci) {
                0 => 0,
                1 => 1,
                3 => 2,
                7 => 3,
                else => null,
            };
            if (at) |j| sib_sum[j] += Count.distinct(&sib);
        }
        var toks: [MAXR]u32 = undefined;
        for (0..7) |i| toks[i] = @intCast(want0[s0 + 1 + i].integer);
        try m.window(toks[0..7], &pk);
        m.keepRows(toks[0..7], 7);
    }
    const nstarts: f64 = @floatFromInt(starts.len);
    for ([_]usize{ 1, 2, 4, 8 }, 0..) |rows, j| {
        std.debug.print("{d} rows (of {d} picks a layer): chain {d:.1} distinct, siblings {d:.1}\n", .{ rows, rows * 10, chain_sum[j] / nstarts, sib_sum[j] / nstarts });
    }
    std.debug.print("4 rows 8 tokens apart: {d:.1} distinct\n", .{Count.distinct(&far)});
    return;
}
