//! core/affine_mm with tensor units on tile_qmm_check's synthetic matrices: rows, gather and GLU bits, float64.
const std = @import("std");
const mtl = @import("metal");
const tile = @import("tile_qmm");
const am = @import("affine_mm");
const check = @import("tile_qmm_check.zig");
const Synth = check.Synth;
const Gpu = check.Gpu;
const Ref = check.Ref;
const opts = check.opts;
const max_rows = check.max_rows;
const toBf = check.toBf;
const fromBf = check.fromBf;
/// Experts of this section's matrices (the starts are a fixed array).
const tensor_experts = 4;

/// bf16 of `v` as the GPU rounds it: subnormals flushed to zero.
fn toBfFtz(v: f32) u16 {
    return if (@abs(v) < std.math.floatMin(f32)) 0 else toBf(v);
}

/// mlx-lm's compiled SwiGLU on bf16 rows (the tensor GLU's epilogue): silu(g) * u, bf16 at each step.
fn swiglu(g: u16, u: u16) u16 {
    const gf = fromBf(g);
    const sg = toBfFtz(1.0 / (1.0 + @exp(-gf)));
    const silu = toBfFtz(gf * fromBf(sg));
    return toBfFtz(fromBf(silu) * fromBf(u));
}

/// A gather over sorted (token, slot) pairs: slot order, each expert's first sorted row, and x gathered by that order.
fn routing(device: mtl.Device, s: Synth, tokens: usize, topk: usize, order: []u32, starts: []i32) !mtl.Buffer {
    const k = s.k;
    const experts = s.experts;
    var at: usize = 0;
    for (0..experts) |e| {
        starts[e] = @intCast(at);
        for (0..tokens * topk) |slot| if ((slot / topk * 7 + slot % topk * 5 + slot / 10) % experts == e) {
            order[at] = @intCast(slot);
            at += 1;
        };
    }
    const sorted = try device.buffer(tokens * topk * k * 2, opts);
    const x = s.x.slice(u16, max_rows * k);
    for (order[0..at], 0..) |slot, j| @memcpy(sorted.slice(u16, at * k)[j * k ..][0..k], x[(slot / topk) * k ..][0..k]);
    return sorted;
}

/// core/affine_mm with tensor units: rows, gather and GLU bit-checks, float64 error and the twin gap, per width.
pub fn tensor(g: Gpu, bits: u8, seed: u64, a: std.mem.Allocator) !void {
    const cols = 128;
    const s = try Synth.init(g.device, bits, seed, 1536, tensor_experts, cols, true);
    defer s.deinit();
    const k = s.k;
    const experts = s.experts;
    const src = try am.source(g.device, a, .{ .bits = bits, .group = 64, .glu = .plain });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(g.device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var p: am.Pipes = undefined;
    for (am.names, 0..) |nm, i| p[i] = try mtl.Pipeline.init(g.device, lib, nm, false);
    defer for (p) |x| x.deinit();
    const sums = try g.device.buffer(max_rows * (k / 64) * 4, opts);
    defer sums.deinit();
    const wide = try g.device.buffer(max_rows * cols * 2, opts);
    defer wide.deinit();
    const part = try g.device.buffer(max_rows * cols * 2, opts);
    defer part.deinit();
    const Dense = struct {
        p: am.Pipes,
        s: Synth,
        e: usize,
        x: mtl.Buffer,
        y: mtl.Buffer,
        rows: usize,
        pub fn encode(d: @This(), enc: mtl.ComputeEncoder, _: *const tile.Pipelines) !void {
            am.dense(enc, d.p, Ref{ .buf = d.x }, d.s.affine(d.e), Ref{ .buf = d.y }, @intCast(d.rows));
        }
    };
    try g.run(Dense{ .p = p, .s = s, .e = 1, .x = s.x, .y = wide, .rows = max_rows });
    for ([_]usize{ 1, 17, 64 }) |rows| {
        try g.run(Dense{ .p = p, .s = s, .e = 1, .x = s.x, .y = part, .rows = rows });
        const alone = part.slice(u16, rows * cols);
        if (!std.mem.eql(u16, wide.slice(u16, rows * cols), alone)) return error.TensorRowsDiffer;
    }
    // The tiles (the reference) and the twin on the same matrix and rows: float64 error output by output, and bits.
    const tiles_y = try g.device.buffer(max_rows * cols * 2, opts);
    defer tiles_y.deinit();
    try g.run(check.Tiles{ .m = s.matrix(1), .x = s.x, .sums = sums, .y = tiles_y, .first = 0, .rows = max_rows });
    const twin = try g.device.buffer(max_rows * cols * 2, opts);
    defer twin.deinit();
    try g.run(check.Twin{ .m = s.matrix(1), .x = s.x, .y = twin, .first = 0, .rows = 32 });
    var worst = [2]f64{ 0, 0 };
    var total = [2]f64{ 0, 0 };
    var tensor_worse: usize = 0;
    var tiles_worse: usize = 0;
    var twin_differs: usize = 0;
    var twin_tiles: usize = 0;
    for (0..max_rows) |r| for (0..cols) |c| {
        const ref = s.reference(1, r, c);
        const at = r * cols + c;
        const y = [2]u16{ tiles_y.slice(u16, max_rows * cols)[at], wide.slice(u16, max_rows * cols)[at] };
        var err: [2]f64 = undefined;
        for (0..2) |i| {
            err[i] = @abs(@as(f64, fromBf(y[i])) - ref[0]) / @max(ref[1], 1e-30);
            worst[i] = @max(worst[i], err[i]);
            total[i] += err[i];
        }
        tensor_worse += @intFromBool(err[1] > err[0]);
        tiles_worse += @intFromBool(err[0] > err[1]);
        if (r < 32) {
            const t = twin.slice(u16, max_rows * cols)[r * cols + c];
            twin_differs += @intFromBool(t != y[1]);
            twin_tiles += @intFromBool(t != y[0]);
        }
    };
    if (twin_tiles != 0) return error.TwinDiffers;
    // The gather over 24 tokens x 2 slots sorted by expert equals each expert's dense product; GLU the host's SwiGLU.
    const tokens = 24;
    const topk = 2;
    var order: [tokens * topk]u32 = undefined;
    var starts: [tensor_experts]i32 = undefined;
    const sorted = try routing(g.device, s, tokens, topk, &order, &starts);
    defer sorted.deinit();
    const offsets = try g.device.buffer(experts * 4, opts);
    defer offsets.deinit();
    @memcpy(offsets.slice(i32, experts), &starts);
    const slots_buf = try g.device.buffer(tokens * topk * 4, opts);
    defer slots_buf.deinit();
    @memcpy(slots_buf.slice(u32, tokens * topk), &order);
    const out = try g.device.buffer(tokens * topk * cols * 2, opts);
    defer out.deinit();
    const Gather = struct {
        p: am.Pipes,
        s: Synth,
        x: mtl.Buffer,
        offsets: mtl.Buffer,
        order: mtl.Buffer,
        y: mtl.Buffer,
        glu: bool = false,
        pub fn encode(d: @This(), enc: mtl.ComputeEncoder, _: *const tile.Pipelines) !void {
            const rows: u32 = tokens * topk;
            const x = Ref{ .buf = d.x };
            const ro = Ref{ .buf = d.offsets };
            const y = Ref{ .buf = d.y };
            const m = d.s.affine(0);
            const ex: u32 = @intCast(d.s.experts);
            if (d.glu) return am.gatherGlu(enc, d.p, x, m, m, ro, Ref{ .buf = d.order }, y, rows, ex, topk, 0);
            am.gather(enc, d.p, x, m, ro, y, rows, ex);
        }
    };
    var gather = Gather{ .p = p, .s = s, .x = sorted, .offsets = offsets, .order = slots_buf, .y = out };
    try g.run(gather);
    for (0..experts) |e| {
        try g.run(Dense{ .p = p, .s = s, .e = e, .x = s.x, .y = wide, .rows = tokens });
        const last: usize = if (e + 1 < experts) @intCast(starts[e + 1]) else tokens * topk;
        for (@as(usize, @intCast(starts[e]))..last) |j| {
            const want = wide.slice(u16, tokens * cols)[(order[j] / topk) * cols ..][0..cols];
            const got = out.slice(u16, tokens * topk * cols)[j * cols ..][0..cols];
            if (!std.mem.eql(u16, got, want)) return error.TensorGatherDiffers;
        }
    }
    const glu_out = try g.device.buffer(tokens * topk * cols * 2, opts);
    defer glu_out.deinit();
    gather.x = s.x;
    gather.y = glu_out;
    gather.glu = true;
    try g.run(gather);
    // GLU rows against the host's SwiGLU of the gather's rows (each its expert's dense product, checked above).
    var glu_exact: usize = 0;
    var glu_ulp: u32 = 0;
    for (0..tokens * topk) |j| for (0..cols) |c| {
        const gv = out.slice(u16, tokens * topk * cols)[j * cols + c];
        const got = glu_out.slice(u16, tokens * topk * cols)[j * cols + c];
        const want = swiglu(gv, gv);
        glu_exact += @intFromBool(got == want);
        glu_ulp = @max(glu_ulp, @abs(@as(i32, got) - @as(i32, want)));
    };
    const outputs = max_rows * cols;
    std.debug.print("{d}-bit tensor units {d}x{d}: rows alone equal rows together, gather equals dense, " ++
        "gate-up SwiGLU within {d} bf16 ulp of the host's ({d}/{d} exact); error vs float64 / terms' magnitude " ++
        "on the same {d} outputs: tiles worst {e:.2} mean {e:.2}, tensor worst {e:.2} mean {e:.2}, tensor worse on " ++
        "{d}, tiles worse on {d}; decode (twin) rows equal the tiles' and differ from the tensor rows on {d}/{d}\n", .{
        bits,
        cols,
        k,
        glu_ulp,
        glu_exact,
        tokens * topk * cols,
        outputs,
        worst[0],
        total[0] / @as(f64, @floatFromInt(outputs)),
        worst[1],
        total[1] / @as(f64, @floatFromInt(outputs)),
        tensor_worse,
        tiles_worse,
        twin_differs,
        32 * cols,
    });
    if (glu_ulp > 2 or glu_exact * 100 < tokens * topk * cols * 99) return error.TensorGluDiffers;
    // tensor rows take each weight as bf16(fma(s, q, b)), as MLX's quantized matmul does: held to the bound
    if (worst[1] > 1.0 / 128.0) return error.TensorFarFromFloat64;
}
