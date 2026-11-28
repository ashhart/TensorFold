//! Synthetic GPU parity tests for mixed 4/64 routed + 8/64 shared fused MoE (batch and pick paths).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const kernels = @import("kernels.zig");
const moe = @import("moe.zig");
const wide = @import("wide.zig");
const moe_route = @import("../../core/moe_route.zig");

const bf16_round = @import("bf16_round");
const bf16 = bf16_round.bits;
fn ref(b: mtl.Buffer) wts.Ref {
    return .{ .buf = b };
}

fn fillAffine(rnd: std.Random, w_buf: mtl.Buffer, s_buf: mtl.Buffer, b_buf: mtl.Buffer, n: usize, k: usize, bits: usize) void {
    rnd.bytes(w_buf.slice(u8, n * k * bits / 8));
    const div: f32 = if (bits == 8) 512.0 else 32.0;
    for (s_buf.slice(u16, n * (k / 64)), b_buf.slice(u16, n * (k / 64))) |*s, *b| {
        s.* = bf16((rnd.float(f32) + 0.5) / div);
        b.* = bf16(-(rnd.float(f32) + 0.5) * 0.15);
    }
}

test "mixed 4/64 routed + 8/64 shared moe_0_sh8 (and pick at row 1) matches wide + moe_2 bit-for-bit" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const k = try kernels.load(a, device);
    defer {
        k.deinit();
        a.destroy(k);
    }
    try k.widen(a, device);
    const D: usize = 4096;
    const N: usize = 2048;
    const E: usize = 8;
    const MAXE: usize = kernels.route_shape.experts;
    const TOPK: usize = 8;
    const MAXR: usize = 16;
    const opts = mtl.ResourceOptions.shared;
    const g_w = try device.buffer(E * N * D / 2, opts);
    const g_s = try device.buffer(E * N * (D / 64) * 2, opts);
    const g_b = try device.buffer(E * N * (D / 64) * 2, opts);
    const u_w = try device.buffer(E * N * D / 2, opts);
    const u_s = try device.buffer(E * N * (D / 64) * 2, opts);
    const u_b = try device.buffer(E * N * (D / 64) * 2, opts);
    const d_w = try device.buffer(E * D * N / 2, opts);
    const d_s = try device.buffer(E * D * (N / 64) * 2, opts);
    const d_b = try device.buffer(E * D * (N / 64) * 2, opts);
    const sg_w = try device.buffer(2 * N * D, opts);
    const sg_s = try device.buffer(2 * N * (D / 64) * 2, opts);
    const sg_b = try device.buffer(2 * N * (D / 64) * 2, opts);
    const sd_w = try device.buffer(D * N, opts);
    const sd_s = try device.buffer(D * (N / 64) * 2, opts);
    const sd_b = try device.buffer(D * (N / 64) * 2, opts);
    const bufs = [_]mtl.Buffer{ g_w, g_s, g_b, u_w, u_s, u_b, d_w, d_s, d_b, sg_w, sg_s, sg_b, sd_w, sd_s, sd_b };
    defer for (bufs) |b| b.deinit();
    var prng = std.Random.DefaultPrng.init(0x5802);
    const rnd = prng.random();
    fillAffine(rnd, g_w, g_s, g_b, E * N, D, 4);
    fillAffine(rnd, u_w, u_s, u_b, E * N, D, 4);
    fillAffine(rnd, d_w, d_s, d_b, E * D, N, 4);
    fillAffine(rnd, sg_w, sg_s, sg_b, 2 * N, D, 8);
    fillAffine(rnd, sd_w, sd_s, sd_b, D, N, 8);
    const x_in = try device.buffer(MAXR * D * 2, opts);
    const logits_r = try device.buffer(MAXR * MAXE * 4, opts);
    const bias = try device.buffer(MAXE * 4, opts);
    const rl = try device.buffer(TOPK * MAXR * 4 * 2 + 128, opts);
    const uids = try device.buffer(MAXE * 4, opts);
    const umem = try device.buffer(MAXE * MAXR * 4, opts);
    const ucount = try device.buffer(64, opts);
    const pick = try device.buffer(MAXR * TOPK * 4, opts);
    const wts_b = try device.buffer(MAXR * TOPK * 4, opts);
    const act = try device.buffer(MAXR * (TOPK + 1) * N * 2, opts);
    const acts = try device.buffer(MAXR * N * 2, opts);
    const ye = try device.buffer(MAXR * (TOPK + 1) * D * 2, opts);
    const ys = try device.buffer(MAXR * D * 2, opts);
    const out_fused = try device.buffer(MAXR * D * 2, opts);
    const out_ref = try device.buffer(MAXR * D * 2, opts);
    const gu = try device.buffer(MAXR * 2 * N * 2, opts);
    const sc_bufs = [_]mtl.Buffer{ x_in, logits_r, bias, rl, uids, umem, ucount, pick, wts_b, act, acts, ye, ys, out_fused, out_ref, gu };
    defer for (sc_bufs) |b| b.deinit();
    for (x_in.slice(u16, MAXR * D)) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (wts_b.slice(f32, MAXR * TOPK)) |*v| v.* = rnd.float(f32) * 0.25 + 0.05;
    for (logits_r.slice(f32, MAXE)) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
    const bi = bias.slice(f32, MAXE);
    @memset(bi, -std.math.inf(f32));
    for (bi[0..E]) |*v| v.* = rnd.float(f32) * 0.1;
    const c_cfg = cfg.Config{ .layers = 45, .run = 45, .tp = 1, .experts = MAXE };
    const moe_w: wts.Moe = .{
        .router = ref(uids),
        .bias = ref(bias),
        .gate = .{ .w = ref(g_w), .s = ref(g_s), .b = ref(g_b), .n = N, .k = D, .bits = 4, .group = 64 },
        .up = .{ .w = ref(u_w), .s = ref(u_s), .b = ref(u_b), .n = N, .k = D, .bits = 4, .group = 64 },
        .down = .{ .w = ref(d_w), .s = ref(d_s), .b = ref(d_b), .n = D, .k = N, .bits = 4, .group = 64 },
        .sh_gate_up = .{ .w = ref(sg_w), .s = ref(sg_s), .b = ref(sg_b), .n = 2 * N, .k = D, .bits = 8, .group = 64 },
        .sh_down = .{ .w = ref(sd_w), .s = ref(sd_s), .b = ref(sd_b), .n = D, .k = N, .bits = 8, .group = 64 },
    };
    try std.testing.expect(moe.canBoth(&moe_w));
    for ([_]bool{ false, true }) |fused_pick| {
        const row_cases: []const u32 = if (fused_pick) &.{1} else &.{ 1, 2, 5, 8, 15 };
        for (row_cases) |rows| {
            const ui = uids.slice(i32, E);
            const um = umem.slice(i32, MAXE * MAXR);
            @memset(um, -1);
            for (0..E) |e| {
                ui[e] = @intCast(e);
                for (0..rows) |r| um[e * MAXR + r] = @intCast(r * TOPK + e);
            }
            ucount.slice(i32, 1)[0] = @intCast(E);
            const pk = pick.slice(i32, rows * TOPK);
            for (0..rows) |r| for (0..TOPK) |s| {
                pk[r * TOPK + s] = @intCast(s);
            };
            var sc: st.Scratch = undefined;
            sc.logits_r = ref(logits_r);
            sc.rl = ref(rl);
            sc.uids = ref(uids);
            sc.umem = ref(umem);
            sc.ucount = ref(ucount);
            sc.pick = ref(pick);
            sc.wts = ref(wts_b);
            sc.act = ref(act);
            sc.acts = ref(acts);
            sc.ye = ref(ye);
            sc.ys = ref(ys);
            sc.gu = ref(gu);
            sc.branch = ref(out_fused);
            sc.none = ref(ucount);
            var state_stub: st.State = .{ .cap = 128 };
            var ctx = fwd.Ctx{ .k = k, .c = &c_cfg, .w = undefined, .s = &state_stub, .sc = &sc, .skip = fwd.Class.route, .fused_route = fused_pick };
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            moe.moe(&ctx, enc, &moe_w, ref(x_in), rows);
            enc.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
            const wp = &k.wide.?;
            const cb2 = queue.commandBuffer();
            const enc2 = cb2.compute(.serial);
            if (fused_pick) {
                const T: usize = TOPK * MAXR * 4;
                moe_route.select(enc2, k.route_select, ref(logits_r), ref(bias), c_cfg.routed_scale, rows, .{ 0, MAXE }, .{ .pick = ref(pick), .wts = ref(wts_b), .ids = ref(uids), .members = ref(umem), .count = ref(ucount), .mine = ref(rl), .theirs = ref(rl).at(T), .counts = ref(rl).at(2 * T), .word = ref(rl).at(2 * T + 64) });
            }
            wide.glu(wp, enc2, moe_w.sh_gate_up, N, ref(x_in), ref(acts), rows, c_cfg.swiglu_limit);
            wide.dense(wp, enc2, moe_w.sh_down, .{ .in = ref(acts), .ldx = N, .out = ref(ys), .ldo = D, .rows = rows });
            enc2.setPipeline(k.moe_gateup_2);
            fwd.bind(enc2, 0, .{ref(x_in)});
            fwd.shape(enc2, 1, .{ rows, D });
            fwd.bind(enc2, 2, .{ moe_w.gate.w, moe_w.gate.s, moe_w.gate.b, moe_w.up.w, moe_w.up.s, moe_w.up.b, moe_w.sh_gate_up.w, moe_w.sh_gate_up.s, moe_w.sh_gate_up.b, ref(uids), ref(umem), ref(ucount) });
            enc2.setValue(c_cfg.swiglu_limit, 14);
            fwd.bind(enc2, 15, .{ref(act)});
            enc2.dispatchThreads(fwd.size(32 * rows, N / 4, rows * TOPK), fwd.size(32 * rows, 1, 1));
            enc2.setPipeline(k.moe_down_2);
            fwd.bind(enc2, 0, .{ref(act)});
            fwd.shape(enc2, 1, .{ rows, TOPK, N });
            fwd.bind(enc2, 2, .{ moe_w.down.w, moe_w.down.s, moe_w.down.b, moe_w.sh_down.w, moe_w.sh_down.s, moe_w.sh_down.b, ref(uids), ref(umem), ref(ucount), ref(ye) });
            enc2.dispatchThreads(fwd.size(32 * rows, D / 4, rows * TOPK), fwd.size(32 * rows, 1, 1));
            enc2.setPipeline(k.moe_combine);
            fwd.bind(enc2, 0, .{ ref(ys), ref(ye), ref(wts_b) });
            fwd.shape(enc2, 3, .{ rows, TOPK });
            fwd.bind(enc2, 4, .{ref(out_ref)});
            enc2.dispatchThreads(fwd.size(rows * D, 1, 1), fwd.size(256, 1, 1));
            enc2.end();
            cb2.commit();
            cb2.wait();
            try std.testing.expect(cb2.failure() == null);
            try std.testing.expectEqualSlices(u16, out_ref.slice(u16, rows * D), out_fused.slice(u16, rows * D));
        }
    }
}
