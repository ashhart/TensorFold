//! Softmax top-k routing (cuda/moe.zig) and the SwiGLU pair form of the affine-4 experts against host references.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn f32bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn rbf(x: f32) f32 {
    return bf16f(f32bf16(x));
}

fn randBf16(r: std.Random, scale: f32) u16 {
    return f32bf16((r.float(f32) * 2.0 - 1.0) * scale);
}

fn upload(gpu: Gpu, bytes: []const u8) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}

/// The router's fp32 logits: one fma chain a (row, expert) in index order, as tf_moe_router sums.
fn router(gpu: Gpu, f: cuda.moe.Fns, s: cuda.Stream, r: std.Random) !void {
    const a = gpu.gpa;
    const rows = 5;
    const dims = 2048;
    const count = 257;
    const x = try a.alloc(u16, rows * dims);
    defer a.free(x);
    const w = try a.alloc(u16, count * dims);
    defer a.free(w);
    for (x) |*v| v.* = randBf16(r, 1.0);
    for (w) |*v| v.* = randBf16(r, 0.1);
    const want = try a.alloc(f32, rows * count);
    defer a.free(want);
    for (0..rows) |i| for (0..count) |e| {
        var acc: f32 = 0;
        for (0..dims) |k| acc = @mulAdd(f32, bf16f(x[i * dims + k]), bf16f(w[e * dims + k]), acc);
        want[i * count + e] = acc;
    };
    var dx = try upload(gpu, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try upload(gpu, std.mem.sliceAsBytes(w));
    defer dw.free();
    var dl = try cuda.DeviceBuffer.alloc(gpu.d, rows * count * 4);
    defer dl.free();
    try f.route(s, dx.ptr, dims, rows, dw.ptr, dl.ptr, dims, count);
    try s.synchronize();
    const got = try check.download(gpu, dl);
    defer a.free(got);
    try check.sameBytes("router logits", got, std.mem.sliceAsBytes(want));
    check.pass("moe router: {d} rows x {d} experts over {d} inputs equal the host's fma chains", .{ rows, count, dims });
}

/// Each row's picks and weights by cuda/moe.py's rule: top-k by logit, ties to the lower id, exp(l - top) / sum.
fn topk(gpu: Gpu, f: cuda.moe.Fns, s: cuda.Stream, r: std.Random) !void {
    const a = gpu.gpa;
    const rows = 7;
    const experts = 256;
    const top_k = 8;
    const slots = top_k + 1;
    const logits = try a.alloc(f32, rows * (experts + 1));
    defer a.free(logits);
    for (logits) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * 4.0;
    // a planted tie, a row of equal logits and a NaN that never wins
    logits[1 * (experts + 1) + 3] = logits[1 * (experts + 1) + 200];
    for (0..experts) |e| logits[2 * (experts + 1) + e] = 0.5;
    logits[3 * (experts + 1) + 7] = std.math.nan(f32);
    const want_ids = try a.alloc(i32, rows * slots);
    defer a.free(want_ids);
    const want_w = try a.alloc(f32, rows * slots);
    defer a.free(want_w);
    for (0..rows) |i| {
        const row = logits[i * (experts + 1) ..][0..experts];
        var v: [experts]f32 = undefined;
        for (&v, row) |*d, l| d.* = if (l != l) -std.math.inf(f32) else l;
        var top = -std.math.inf(f32);
        for (v) |l| top = @max(top, l);
        var total: f32 = 0;
        var ex: [top_k]f32 = undefined;
        for (0..top_k) |k| {
            var bi: usize = 0;
            var bv = -std.math.inf(f32);
            for (v, 0..) |l, e| if (l > bv) {
                bv = l;
                bi = e;
            };
            ex[k] = @exp(bv - top);
            total += ex[k];
            want_ids[i * slots + k] = @intCast(bi);
            v[bi] = -std.math.inf(f32);
        }
        for (0..top_k) |k| want_w[i * slots + k] = rbf(ex[k] / total);
        want_ids[i * slots + top_k] = experts;
        const sg = rbf(logits[i * (experts + 1) + experts]);
        want_w[i * slots + top_k] = rbf(1.0 / (1.0 + @exp(-sg)));
    }
    var dl = try upload(gpu, std.mem.sliceAsBytes(logits));
    defer dl.free();
    var dp = try cuda.DeviceBuffer.alloc(gpu.d, rows * slots * 4);
    defer dp.free();
    var dw = try cuda.DeviceBuffer.alloc(gpu.d, rows * slots * 4);
    defer dw.free();
    try f.pick(s, dl.ptr, dp.ptr, dw.ptr, rows, experts, top_k);
    try s.synchronize();
    const got_ids = try check.download(gpu, dp);
    defer a.free(got_ids);
    try check.sameBytes("top-k ids", got_ids, std.mem.sliceAsBytes(want_ids));
    const got_w = try check.download(gpu, dw);
    defer a.free(got_w);
    var off: usize = 0;
    for (std.mem.bytesAsSlice(f32, got_w), want_w) |g, wv| {
        // exp on the device and the host may differ in the last fp32 place: the bf16 weight then moves one step at most
        if (g != wv and @abs(g - wv) > @abs(wv) * (1.0 / 128.0)) off += 1;
    }
    try check.expect(off == 0, "top-k weights: {d} of {d} differ by more than a bf16 step", .{ off, want_w.len });
    check.pass("moe top-k: {d} rows' picks of {d} over {d} experts equal the host's (a tie, a flat row and a NaN included), weights within a bf16 step", .{ rows, top_k, experts });
}

/// The weighted sum of a row's slots in slot order, fma in fp32, rounded once to bf16.
fn combine(gpu: Gpu, f: cuda.moe.Fns, s: cuda.Stream, r: std.Random) !void {
    const a = gpu.gpa;
    const rows = 3;
    const slots = 4;
    const dims = 2048;
    const y = try a.alloc(f32, rows * slots * dims);
    defer a.free(y);
    const w = try a.alloc(f32, rows * slots);
    defer a.free(w);
    for (y) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * 3.0;
    for (w) |*v| v.* = rbf(r.float(f32));
    const want = try a.alloc(u16, rows * dims);
    defer a.free(want);
    for (0..rows) |i| for (0..dims) |d| {
        var acc: f32 = 0;
        for (0..slots) |k| acc = @mulAdd(f32, w[i * slots + k], y[(i * slots + k) * dims + d], acc);
        want[i * dims + d] = f32bf16(acc);
    };
    var dy = try upload(gpu, std.mem.sliceAsBytes(y));
    defer dy.free();
    var dw = try upload(gpu, std.mem.sliceAsBytes(w));
    defer dw.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, rows * dims * 2);
    defer dout.free();
    try f.combine(s, dy.ptr, true, dw.ptr, dout.ptr, rows, slots, dims);
    try s.synchronize();
    const got = try check.download(gpu, dout);
    defer a.free(got);
    try check.sameBytes("combined rows", got, std.mem.sliceAsBytes(want));
    check.pass("moe combine: {d} rows of {d} slots equal the host's fma chains rounded to bf16", .{ rows, slots });
}

/// Random affine-4 experts (words, scales, biases) as the checkpoint stores them.
const Table = struct { words: []u32, scales: []u16, biases: []u16, count: usize, n: usize, k: usize };

fn table(a: std.mem.Allocator, r: std.Random, count: usize, n: usize, k: usize) !Table {
    const t: Table = .{ .words = try a.alloc(u32, count * n * k / 8), .scales = try a.alloc(u16, count * n * k / 64), .biases = try a.alloc(u16, count * n * k / 64), .count = count, .n = n, .k = k };
    for (t.words) |*w| w.* = r.int(u32);
    for (t.scales) |*v| v.* = f32bf16(0.02 + r.float(f32) * 0.03);
    for (t.biases) |*v| v.* = randBf16(r, 0.2);
    return t;
}

/// Weight (e, j, i) dequantized: scale * code + bias.
fn weight(t: Table, e: usize, j: usize, i: usize) f64 {
    const row = (e * t.n + j);
    const code: f64 = @floatFromInt((t.words[row * (t.k / 8) + i / 8] >> @intCast(4 * (i % 8))) & 15);
    return code * bf16f(t.scales[row * (t.k / 64) + i / 64]) + bf16f(t.biases[row * (t.k / 64) + i / 64]);
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// The pair form: gate and up interleaved per group, SwiGLU with the kernel's bf16 rounding points, down to fp32 sums.
fn pairExperts(gpu: Gpu, s: cuda.Stream, r: std.Random) !void {
    const a = gpu.gpa;
    var mods: [3]cuda.Module = undefined;
    mods[0] = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mods[0].unload();
    mods[1] = try cuda.Module.load(gpu.d, cuda.kernels.experts_prefill);
    defer mods[1].unload();
    mods[2] = try cuda.Module.load(gpu.d, cuda.kernels.experts_pack);
    defer mods[2].unload();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const ex = try cuda.experts.Affine4Experts.resolve(mods[0], mods[1], mods[2], sms);
    const plan_fns = try cuda.grouped.Router.resolve(mods[0]);
    const count = 6;
    const ni = 64;
    const dims = 128;
    const rows = 3;
    const slots = 4;
    const pairs = rows * slots;
    const gate = try table(a, r, count, ni, dims);
    defer {
        a.free(gate.words);
        a.free(gate.scales);
        a.free(gate.biases);
    }
    const up = try table(a, r, count, ni, dims);
    defer {
        a.free(up.words);
        a.free(up.scales);
        a.free(up.biases);
    }
    const down = try table(a, r, count, dims, ni);
    defer {
        a.free(down.words);
        a.free(down.scales);
        a.free(down.biases);
    }
    const x = try a.alloc(u16, rows * dims);
    defer a.free(x);
    for (x) |*v| v.* = randBf16(r, 1.0);
    const picks = [pairs]i32{ 0, 2, 5, 5, 1, 2, 3, 5, 4, 4, 0, 5 };
    // the host reference
    const want = try a.alloc(f64, pairs * dims);
    defer a.free(want);
    const act = try a.alloc(f32, ni);
    defer a.free(act);
    for (0..pairs) |p| {
        const e: usize = @intCast(picks[p]);
        const xr = x[(p / slots) * dims ..][0..dims];
        for (0..ni) |j| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..dims) |i| {
                g += weight(gate, e, j, i) * bf16f(xr[i]);
                u += weight(up, e, j, i) * bf16f(xr[i]);
            }
            const gv = rbf(@floatCast(g));
            const uv = rbf(@floatCast(u));
            act[j] = rbf(rbf(silu(gv)) * uv);
        }
        for (0..dims) |d| {
            var y: f64 = 0;
            for (0..ni) |j| y += weight(down, e, d, j) * act[j];
            want[p * dims + d] = y;
        }
    }
    // the device: packed tables, the plan, up (pair) then down
    var dpair = try cuda.DeviceBuffer.alloc(gpu.d, cuda.experts.Affine4Experts.pairBytes(count, ni, dims));
    defer dpair.free();
    var ddown = try cuda.DeviceBuffer.alloc(gpu.d, cuda.experts.Affine4Experts.bytes(count, dims, ni));
    defer ddown.free();
    for ([_]struct { Table, u64, usize, usize }{ .{ gate, dpair.ptr, 2, 0 }, .{ up, dpair.ptr, 2, 1 }, .{ down, ddown.ptr, 1, 0 } }) |item| {
        const t = item[0];
        var w = try upload(gpu, std.mem.sliceAsBytes(t.words));
        defer w.free();
        var sc = try upload(gpu, std.mem.sliceAsBytes(t.scales));
        defer sc.free();
        var bi = try upload(gpu, std.mem.sliceAsBytes(t.biases));
        defer bi.free();
        try ex.packInto(s, w.ptr, sc.ptr, bi.ptr, item[1], t.count, t.n, t.k, item[2], item[3]);
        try s.synchronize();
    }
    var dx = try upload(gpu, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dpicks = try upload(gpu, std.mem.sliceAsBytes(&picks));
    defer dpicks.free();
    var members = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 4);
    defer members.free();
    var items = try cuda.DeviceBuffer.alloc(gpu.d, cuda.grouped.maxItems(pairs, count, 16) * 12);
    defer items.free();
    var counts = try cuda.DeviceBuffer.alloc(gpu.d, 8);
    defer counts.free();
    var dact = try cuda.DeviceBuffer.alloc(gpu.d, pairs * ni * 2);
    defer dact.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, pairs * dims * 4);
    defer dy.free();
    const plan: cuda.grouped.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = 0, .hist = 0 };
    try plan_fns.route(s, dpicks.ptr, pairs, count, 16, plan);
    const layer: cuda.experts.Layer = .{ .format = .affine4, .up = dpair.ptr, .down = ddown.ptr, .width = ni, .dims = dims, .experts = count, .swiglu = true };
    const grouped: cuda.experts.Grouped = .{ .affine4 = &ex };
    const items_n = cuda.grouped.maxItems(pairs, count, 16);
    try grouped.decode(s, .up, .{ .x = dx.ptr, .stride = dims, .slots = slots }, layer, plan, dact.ptr, items_n, -1);
    try grouped.decode(s, .down_f32, .{ .x = dact.ptr, .stride = ni }, layer, plan, dy.ptr, items_n, -1);
    try s.synchronize();
    const got = try check.download(gpu, dy);
    defer a.free(got);
    var worst: f64 = 0;
    var scale: f64 = 0;
    for (std.mem.bytesAsSlice(f32, got), want) |g, wv| {
        worst = @max(worst, @abs(@as(f64, g) - wv));
        scale = @max(scale, @abs(wv));
    }
    try check.expect(worst <= scale * 2e-3, "pair experts: worst error {e} against a largest sum of {e}", .{ worst, scale });
    check.pass("moe experts: {d} pairs over {d} SwiGLU experts (gate and up interleaved) within {e} of the fp64 reference (largest sum {e})", .{ pairs, count, worst, scale });
}

pub fn run(gpu: Gpu) !void {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const r = prng.random();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.moe);
    defer mod.unload();
    const f = try cuda.moe.Fns.resolve(mod);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    try router(gpu, f, s, r);
    try topk(gpu, f, s, r);
    try combine(gpu, f, s, r);
    try pairExperts(gpu, s, r);
}
