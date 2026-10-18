//! route_bolt: the nemotron_h_puzzle router top-k at a runtime expert count. The key check is the pad-to-32 fix —
//! at NE=240 (NE % 32 == 16, as every Bolt layer is) the last slice [224,240) must still be reachable, which the
//! baked NE/32 striping dropped. Ids and weights are checked against a CPU reference (sigmoid + bias, normalised).
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Gpu = common.Gpu;
const Size = mtl.Size;

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16); // exact for the small integers planted below
}

fn sigmoid(v: f32) f32 {
    return 1.0 / (1.0 + @exp(-v));
}

pub fn routeBolt(gpu: Gpu) !void {
    const E: usize = 240; // a real Bolt layer's expert count: 240 % 32 == 16
    const K: usize = 9; // Bolt's uniform top-k
    const rows: usize = 2;
    const scaling: f32 = 2.5;

    var logit: [rows * E]f32 = undefined;
    @memset(&logit, 0.0); // the bulk sit at sigmoid(0) = 0.5 and are never picked
    // row 0: the top nine logits land in the last partial slice [224, 233) — unreachable if NE/32 truncates
    for (0..K) |j| logit[0 * E + 224 + j] = 20.0 - @as(f32, @floatFromInt(j));
    // row 1: nine high logits scattered across every slice, including the last expert 239
    const hi = [K]usize{ 239, 200, 160, 128, 96, 64, 32, 16, 1 };
    for (hi, 0..) |e, j| logit[1 * E + e] = 18.0 - @as(f32, @floatFromInt(j));

    const g = try gpu.buffer(u16, rows * E);
    defer g.deinit();
    const bias = try gpu.buffer(f32, E);
    defer bias.deinit();
    const idx = try gpu.buffer(u32, rows * K);
    defer idx.deinit();
    const wt = try gpu.buffer(f32, rows * K);
    defer wt.deinit();
    for (g.slice(u16, rows * E), logit) |*d, v| d.* = bf16(v);
    @memset(bias.slice(f32, E), 0.0);

    const p = try gpu.pipeline("tf_nemotron_route_bolt", false);
    defer p.deinit();
    const cb = gpu.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(g, 0, 0);
    enc.setBuffer(bias, 0, 1);
    enc.setValue(scaling, 2);
    enc.setBuffer(idx, 0, 3);
    enc.setBuffer(wt, 0, 4);
    enc.setValue(@as(u32, @intCast(E)), 5);
    enc.setValue(@as(u32, @intCast(K)), 6);
    enc.dispatchThreads(Size.of(32 * rows, 1, 1), Size.of(32, 1, 1));
    enc.end();
    try common.run(cb);

    const got_idx = idx.slice(u32, rows * K);
    const got_wt = wt.slice(f32, rows * K);
    for (0..rows) |r| {
        // CPU reference: top-k by sigmoid(logit) + bias, lowest id winning a tie (as simd_min does)
        var used: [E]bool = @splat(false);
        var total: f32 = 0;
        var ref_id: [K]u32 = undefined;
        var ref_prob: [K]f32 = undefined;
        for (0..K) |k| {
            var best: f32 = -std.math.inf(f32);
            var best_e: usize = E;
            for (0..E) |e| {
                if (used[e]) continue;
                const sel = sigmoid(logit[r * E + e]) + bias.slice(f32, E)[e];
                if (sel > best) {
                    best = sel;
                    best_e = e;
                }
            }
            used[best_e] = true;
            ref_id[k] = @intCast(best_e);
            ref_prob[k] = sigmoid(logit[r * E + best_e]);
            total += ref_prob[k];
        }
        for (0..K) |k| {
            if (got_idx[r * K + k] != ref_id[k]) {
                std.debug.print("  row {d} pick {d}: got expert {d}, reference {d}\n", .{ r, k, got_idx[r * K + k], ref_id[k] });
                return error.WrongExpert;
            }
            const want = ref_prob[k] / (total + 1e-20) * scaling;
            if (@abs(got_wt[r * K + k] - want) > 1e-3) {
                std.debug.print("  row {d} pick {d}: weight {d} vs reference {d}\n", .{ r, k, got_wt[r * K + k], want });
                return error.WrongWeight;
            }
        }
    }
    // the pad-to-32 guarantee, stated plainly: row 0's winners are exactly experts 224..232
    for (0..K) |k| if (got_idx[k] != 224 + k) return error.LastSliceDropped;
}

fn f32of(w: u16) f32 {
    return @bitCast(@as(u32, w) << 16);
}

/// dense_bf16: y = x . w^T over bf16 (no accumulate), the gate and latent projections. Checked against a CPU
/// reference that reads back the same bf16 inputs and accumulates in fp32, to a bf16 output tolerance.
pub fn denseBf16(gpu: Gpu) !void {
    const M: usize = 5; // rows (or picks)
    const N: usize = 20; // output width
    const K: usize = 64; // input width (multiple of 4 for the bfloat4 loads)

    const x = try gpu.buffer(u16, M * K);
    defer x.deinit();
    const w = try gpu.buffer(u16, N * K);
    defer w.deinit();
    const y = try gpu.buffer(u16, M * N);
    defer y.deinit();
    var prng = std.Random.DefaultPrng.init(0x5017d00d);
    const rnd = prng.random();
    const xv = x.slice(u16, M * K);
    const wv = w.slice(u16, N * K);
    for (xv) |*v| v.* = bf16(rnd.float(f32) - 0.5);
    for (wv) |*v| v.* = bf16(rnd.float(f32) - 0.5);
    @memset(y.slice(u16, M * N), 0xffff); // garbage, so a missed write is caught

    const p = try gpu.pipeline("tf_nemotron_dense_bf16", false);
    defer p.deinit();
    const cb = gpu.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(w, 0, 1);
    enc.setBuffer(y, 0, 2);
    enc.setValue([3]u32{ @intCast(M), @intCast(N), @intCast(K) }, 3);
    enc.dispatchThreads(Size.of(32 * N, (M + 7) / 8, 1), Size.of(256, 1, 1));
    enc.end();
    try common.run(cb);

    const got = y.slice(u16, M * N);
    for (0..M) |m| for (0..N) |j| {
        var sum: f32 = 0;
        for (0..K) |i| sum += f32of(xv[m * K + i]) * f32of(wv[j * K + i]);
        const g = f32of(got[m * N + j]);
        if (@abs(g - sum) > 0.01 * @abs(sum) + 1e-2) {
            std.debug.print("  dense_bf16 [{d},{d}]: got {d}, reference {d}\n", .{ m, j, g, sum });
            return error.WrongProduct;
        }
    };
}

/// add_norm_moe_xs_bolt is the Lightning add_norm kernel at <D=4096, T=512, E=9> (896 does not divide 4096; 512
/// does). The code is golden-verified; the only new risk is the T=512 threadgroup reduction and the rmsnorm. With
/// no routed picks (WE=0) and no shared (SH=0) the block reduces to OUT = rmsnorm(H) * W, which this checks.
pub fn addNormBolt(gpu: Gpu) !void {
    const D: usize = 4096; // Bolt hidden, baked into the instantiation
    const E: usize = 9;
    const rows: usize = 2;
    const mp: usize = 16; // rows padded to 16, the XS stride
    const eps: f32 = 1e-5;

    const h = try gpu.buffer(u16, rows * D);
    defer h.deinit();
    const yexp = try gpu.buffer(u16, rows * E * D);
    defer yexp.deinit();
    const we = try gpu.buffer(f32, rows * E);
    defer we.deinit();
    const sh = try gpu.buffer(u16, rows * D);
    defer sh.deinit();
    const wnorm = try gpu.buffer(u16, D);
    defer wnorm.deinit();
    const hn = try gpu.buffer(u16, rows * D);
    defer hn.deinit();
    const out = try gpu.buffer(u16, rows * D);
    defer out.deinit();
    const xs = try gpu.buffer(f32, D / 64 * mp);
    defer xs.deinit();

    var prng = std.Random.DefaultPrng.init(0xadd_0_117);
    const rnd = prng.random();
    const hv = h.slice(u16, rows * D);
    for (hv) |*v| v.* = bf16(rnd.float(f32) - 0.5);
    @memset(yexp.slice(u16, rows * E * D), 0); // no routed contribution
    @memset(we.slice(f32, rows * E), 0);
    @memset(sh.slice(u16, rows * D), 0); // no shared contribution
    for (wnorm.slice(u16, D)) |*v| v.* = bf16(1.0); // W = 1, so OUT is the pure rmsnorm
    @memset(hn.slice(u16, rows * D), 0xffff);
    @memset(out.slice(u16, rows * D), 0xffff);

    const p = try gpu.pipeline("tf_nemotron_add_norm_moe_xs_bolt", false);
    defer p.deinit();
    const cb = gpu.queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(h, 0, 0);
    enc.setBuffer(yexp, 0, 1);
    enc.setBuffer(we, 0, 2);
    enc.setBuffer(sh, 0, 3);
    enc.setBuffer(wnorm, 0, 4);
    enc.setValue(eps, 5);
    enc.setValue([8]i32{ @intCast(rows), @intCast(mp), 0, 0, 0, 0, 0, 0 }, 6);
    enc.setBuffer(hn, 0, 7);
    enc.setBuffer(out, 0, 8);
    enc.setBuffer(xs, 0, 9);
    enc.dispatchThreads(Size.of(512 * mp, 1, 1), Size.of(512, 1, 1));
    enc.end();
    try common.run(cb);

    const got_hn = hn.slice(u16, rows * D);
    const got_out = out.slice(u16, rows * D);
    const got_xs = xs.slice(f32, D / 64 * mp);
    for (0..rows) |r| {
        var ss: f32 = 0;
        for (0..D) |c| ss += f32of(hv[r * D + c]) * f32of(hv[r * D + c]);
        const scale = 1.0 / @sqrt(ss / @as(f32, D) + eps);
        for (0..D) |c| {
            if (got_hn[r * D + c] != hv[r * D + c]) return error.HnChanged; // delta is 0, so HN == H
            const want = f32of(hv[r * D + c]) * scale;
            if (@abs(f32of(got_out[r * D + c]) - want) > 0.02 * @abs(want) + 1e-2) {
                std.debug.print("  add_norm_bolt out[{d},{d}]: got {d}, reference {d}\n", .{ r, c, f32of(got_out[r * D + c]), want });
                return error.WrongNorm;
            }
        }
        // XS[g] is the sum of OUT over each 64-wide group
        for (0..D / 64) |g| {
            var acc: f32 = 0;
            for (0..64) |i| acc += f32of(got_out[r * D + g * 64 + i]);
            if (@abs(got_xs[g * mp + r] - acc) > 0.05 * @abs(acc) + 1e-2) return error.WrongSums;
        }
    }
}
