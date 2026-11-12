//! Qwen3-8B's projections on the dot2 and gfx12 WMMA tiles against float64: the BF16 outputs alike, fp32 recorded.
const std = @import("std");
const hip = @import("hip");
const qwen3 = @import("qwen3");
const harness = @import("forward_test.zig");

const t = std.testing;
const gpa = t.allocator;
const Tile = hip.launches.dense.Tile;

/// How much worse the WMMA tile's BF16 outputs may be than dot2's, aggregate and worst column: rounding flips only.
const allowance = 0.01;

fn widen(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// A weight matrix (n, k) as BF16 bits (the checkpoint's own: fp32 read back exactly) and as f64.
const Weight = struct {
    bits: []u16,
    n: usize,
    k: usize,

    fn read(a: std.mem.Allocator, dir: []const u8, name: []const u8, n: usize, k: usize) !Weight {
        const key = try qwen3.scopedKey(a, t.io, dir, name);
        const rows = try a.alloc(u32, n);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        const wide = try qwen3.readRows(gpa, t.io, dir, key, rows);
        defer gpa.free(wide);
        if (wide.len != n * k) return error.UnexpectedShape;
        const bits = try a.alloc(u16, n * k);
        for (bits, wide) |*b, f| b.* = @intCast(@as(u32, @bitCast(f)) >> 16);
        return .{ .bits = bits, .n = n, .k = k };
    }

    fn at(w: Weight, j: usize, i: usize) f64 {
        return widen(w.bits[j * w.k + i]);
    }
};

/// The float64 product of BF16 operands, y (m, n).
fn exact(a: std.mem.Allocator, x: []const u16, w: Weight) ![]f64 {
    const m = x.len / w.k;
    const y = try a.alloc(f64, m * w.n);
    const xs = try a.alloc(f64, x.len);
    for (xs, x) |*v, b| v.* = widen(b);
    for (0..m) |r| for (0..w.n) |j| {
        var s: f64 = 0;
        for (xs[r * w.k ..][0..w.k], w.bits[j * w.k ..][0..w.k]) |p, q| s += p * widen(q);
        y[r * w.n + j] = s;
    };
    return y;
}

/// Errors against float64: aggregate relative (root sum of squares) and the worst column's.
const Errors = struct { aggregate: f64, column: f64 };

fn errorsOf(got: []const f64, want: []const f64, n: usize) Errors {
    var e2: f64 = 0;
    var w2: f64 = 0;
    var worst: f64 = 0;
    for (0..n) |j| {
        var ce: f64 = 0;
        var cw: f64 = 0;
        var r: usize = 0;
        while (r * n + j < got.len) : (r += 1) {
            const d = got[r * n + j] - want[r * n + j];
            ce += d * d;
            cw += want[r * n + j] * want[r * n + j];
        }
        e2 += ce;
        w2 += cw;
        worst = @max(worst, @sqrt(ce / @max(cw, 1e-300)));
    }
    return .{ .aggregate = @sqrt(e2 / @max(w2, 1e-300)), .column = worst };
}

/// One class through `tile`: fp32 out and BF16 out, widened.
fn run(g: *hip.Launcher, r: *const hip.Runtime, s: hip.Stream, tile: Tile, x: []const u16, w: Weight, wide: bool) ![]f64 {
    const m = x.len / w.k;
    var dx = try hip.DeviceBuffer.fromHost(r, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try hip.DeviceBuffer.fromHost(r, std.mem.sliceAsBytes(w.bits));
    defer dw.free();
    var dy = try hip.DeviceBuffer.alloc(r, m * w.n * @as(usize, if (wide) 4 else 2));
    defer dy.free();
    const y: u64 = @intFromPtr(dy.ptr);
    try g.dense.gemm(r, tile, @intFromPtr(dx.ptr), @intFromPtr(dw.ptr), if (wide) y else 0, if (wide) 0 else y, @intCast(m), @intCast(w.n), @intCast(w.k), s.handle);
    try s.synchronize();
    const out = try gpa.alloc(f64, m * w.n);
    if (wide) {
        const h = try gpa.alloc(f32, m * w.n);
        defer gpa.free(h);
        try dy.download(0, std.mem.sliceAsBytes(h));
        for (out, h) |*o, v| o.* = v;
    } else {
        const h = try gpa.alloc(u16, m * w.n);
        defer gpa.free(h);
        try dy.download(0, std.mem.sliceAsBytes(h));
        for (out, h) |*o, v| o.* = widen(v);
    }
    return out;
}

test "Qwen3 projections: the WMMA tile's BF16 outputs are as close to float64 as the dot2 tile's, class by class" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rerank = try harness.env(a, "TF_QWEN3_RERANK_DIR");
    const dir = rerank orelse try harness.env(a, "TF_QWEN3_EMBED_DIR") orelse return error.SkipZigTest;
    const ref_path = try harness.env(a, if (rerank != null) "TF_QWEN3_RERANK_REF" else "TF_QWEN3_EMBED_REF") orelse return error.SkipZigTest;
    const ref = try harness.readJson(a, ref_path);
    // the reference's longest prompt, its first 1,024 tokens at most: a real prompt's rows, both tile geometries' sizes
    var longest: []i32 = &.{};
    const prompts = if (rerank != null) ref.object.get("cases").?.array.items else ref.object.get("ids").?.array.items;
    for (prompts) |p| {
        const one = try harness.idsOf(a, if (rerank != null) p.object.get("ids").? else p);
        if (one.len > longest.len) longest = one;
    }
    const ids = longest[0..@min(longest.len, 1024)];
    try t.expect(ids.len >= 64);
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{ .products = .{ .matrix = .off } }, hip.kernels.images);
    defer l.unload();
    if (!l.dense.wmma12) return error.SkipZigTest;
    var ar = try hip.Arena.init(&r, 2 << 30);
    defer ar.deinit();
    var m = try qwen3.Model.load(gpa, t.io, &r, dir, .plain);
    defer m.deinit();
    const spec = m.spec;
    const d = spec.hidden;
    const qw = spec.heads * spec.head_dim;
    const kv = spec.kv_heads * spec.head_dim;
    const n = ids.len;
    // layer 0's projection inputs as the forward makes them (dot2 serving), BF16 and the same for both tiles
    var dids = try hip.DeviceBuffer.fromHost(&r, std.mem.sliceAsBytes(ids));
    defer dids.free();
    var bufs: [4]hip.DeviceBuffer = undefined;
    for (&bufs, [_]usize{ d, qw, d, spec.inter }) |*b, w| b.* = try hip.DeviceBuffer.alloc(&r, n * w * 2);
    defer for (&bufs) |*b| b.free();
    const ops: hip.ops.Ops = .{ .l = &l, .bf16 = true, .stream = s.handle, .arena = &ar };
    try qwen3.layerInputs(ops, &m, @intFromPtr(dids.ptr), @intCast(n), .{ .normed = @intFromPtr(bufs[0].ptr), .attn = @intFromPtr(bufs[1].ptr), .post = @intFromPtr(bufs[2].ptr), .act = @intFromPtr(bufs[3].ptr) });
    try s.synchronize();
    var inputs: [4][]u16 = undefined;
    for (&inputs, bufs) |*x, b| {
        x.* = try a.alloc(u16, b.len / 2);
        try b.download(0, std.mem.sliceAsBytes(x.*));
        for (x.*) |bits| try t.expect(std.math.isFinite(widen(bits)));
    }
    const Class = struct { name: []const u8, x: []const u16, w: Weight };
    const classes = [_]Class{
        .{ .name = "q", .x = inputs[0], .w = try Weight.read(a, dir, "layers.0.self_attn.q_proj.weight", qw, d) },
        .{ .name = "k", .x = inputs[0], .w = try Weight.read(a, dir, "layers.0.self_attn.k_proj.weight", kv, d) },
        .{ .name = "v", .x = inputs[0], .w = try Weight.read(a, dir, "layers.0.self_attn.v_proj.weight", kv, d) },
        .{ .name = "o", .x = inputs[1], .w = try Weight.read(a, dir, "layers.0.self_attn.o_proj.weight", d, qw) },
        .{ .name = "gate", .x = inputs[2], .w = try Weight.read(a, dir, "layers.0.mlp.gate_proj.weight", spec.inter, d) },
        .{ .name = "up", .x = inputs[2], .w = try Weight.read(a, dir, "layers.0.mlp.up_proj.weight", spec.inter, d) },
        .{ .name = "down", .x = inputs[3], .w = try Weight.read(a, dir, "layers.0.mlp.down_proj.weight", d, spec.inter) },
    };
    var worse: usize = 0;
    for (classes) |c| {
        const want = try exact(a, c.x, c.w);
        for (want) |v| try t.expect(std.math.isFinite(v));
        for ([_]bool{ true, false }) |wide| {
            var errs: [2]Errors = undefined;
            for ([_]Tile{ .dot2, .wmma12 }, &errs) |tile, *er| {
                const got = try run(&l, &r, s, tile, c.x, c.w, wide);
                defer gpa.free(got);
                for (got) |v| try t.expect(std.math.isFinite(v));
                er.* = errorsOf(got, want, c.w.n);
                try t.expect(std.math.isFinite(er.aggregate) and std.math.isFinite(er.column));
            }
            // the engine writes BF16 projections: those are gated; fp32 sums (WMMA's 16-product steps) are recorded
            const ok = wide or (errs[1].aggregate <= errs[0].aggregate * (1 + allowance) and errs[1].column <= errs[0].column * (1 + allowance));
            const ratio = if (errs[0].aggregate > 0) errs[1].aggregate / errs[0].aggregate else 1;
            const column = if (errs[0].column > 0) errs[1].column / errs[0].column else 1;
            if (!ok) worse += 1;
            std.debug.print("projection {s:>4} ({d} x {d} x {d}, {s} out): dot2 {e:.4} / column {e:.4}; wmma12 {e:.4} / column {e:.4}; ratio {d:.4} / {d:.4}{s}\n", .{
                c.name, n, c.w.n, c.w.k, if (wide) "fp32" else "bf16", errs[0].aggregate, errs[0].column, errs[1].aggregate, errs[1].column, ratio, column, if (ok) "" else "  WORSE",
            });
        }
    }
    try t.expectEqual(@as(usize, 0), worse);
}
