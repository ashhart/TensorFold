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
