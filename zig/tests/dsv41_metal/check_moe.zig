//! The FFN half of a layer's float64 checks: hc mixes, the route, routed and shared experts, the join, hc post.
const std = @import("std");
const pkg = @import("dsv41_metal");
const ref = @import("ref.zig");
const probe = @import("probe.zig");
const check_attn = @import("check_attn.zig");

const Probe = probe.Probe;
const Ref = probe.Ref;
const u = ref.u;

/// The FFN half of the launch: every stage against float64 on the GPU's own inputs; x_out is the layer's output.
pub fn check(p: *Probe, pre_out: Ref, x_out: Ref) !void {
    const c = &p.r.c;
    const s = &p.r.s;
    const w = p.w;
    const H: usize = c.hidden;
    const all = probe.all(p.rows);
    const x1 = p.bfs(s.x1, p.rows * 4 * H);
    try check_attn.hcMix(p, "hc mix (ffn)", x1, w.hc_ffn, .{ pre_out, s.mix_f[1], s.mix_f[2] });
    try check_attn.hcPre(p, "hc pre (ffn, attn's pre)", x1, p.f32s(s.mix_a[0], p.rows * 4), s.h2);
    try p.norm("ffn_norm", s.h2, w.ffn_norm, s.hn2, H, all);
    try p.mv("gate logits", s.hn2, w.gate_w, s.logits, H, p.experts(), true, all);
    try route(p);
    try routed(p);
    try p.actqRows("act_quant(x) shared", s.hn2, s.hq2, H);
    try p.proj("shared w1", s.hq2, w.shared[0], s.sg, false, all);
    try p.proj("shared w3", s.hq2, w.shared[2], s.su, false, all);
    try swiglu(p);
    try p.actqRows("act_quant(shared act)", s.sact, s.sactq, c.expert_width);
    try p.proj("shared w2", s.sactq, w.shared[1], s.sout, false, all);
    const js = try p.stage("join routed + shared");
    const yr = p.f32s(s.yr, p.rows * H);
    const so = p.bfs(s.sout, p.rows * H);
    const mo = p.bfs(s.mout, p.rows * H);
    for (0..p.rows * H) |i| {
        const want = @as(f64, yr[i]) + ref.bf(so[i]);
        js.add(ref.bf(mo[i]), want, ref.bfBound(want, @abs(yr[i]) + @abs(ref.bf(so[i])), 1), i / H, i % H);
    }
    try check_attn.hcPost(p, "hc post (ffn)", s.mout, x1, .{ pre_out, s.mix_f[1], s.mix_f[2] }, x_out);
}

fn softplus(z: f64) f64 {
    return if (z > 20) z else std.math.log1p(@exp(z));
}

/// The route: sqrt(softplus) scores, the top-k of score plus bias (lower id on a tie), weights over their sum.
fn route(p: *Probe) !void {
    const picks = try p.stage("route picks");
    const wts = try p.stage("route weights");
    const c = &p.r.c;
    const E = p.experts();
    const K = p.topk();
    const lg = p.f32s(p.r.s.logits, p.rows * E);
    const bias = p.f32s(p.w.gate_b, E);
    const got = p.i32s(p.r.s.picks, p.rows * K);
    const gw = p.f32s(p.r.s.wts, p.rows * K);
    var sc: [1024]f64 = undefined;
    var used: [1024]bool = undefined;
    for (0..p.rows) |r| {
        for (0..E) |e| {
            sc[e] = @sqrt(softplus(lg[r * E + e]));
            used[e] = false;
        }
        var sum: f64 = 0;
        for (0..K) |k| {
            var best: usize = E;
            for (0..E) |e| if (!used[e] and (best == E or sc[e] + bias[e] > sc[best] + bias[best])) {
                best = e;
            };
            used[best] = true;
            const g: usize = @intCast(got[r * K + k]);
            const tie = @abs((sc[g] + bias[g]) - (sc[best] + bias[best])) <= 64 * u * (@abs(sc[best]) + @abs(bias[best]));
            picks.same(g == best or tie, r, k);
            sum += sc[g];
        }
        for (0..K) |k| {
            const g: usize = @intCast(got[r * K + k]);
            const want = sc[g] / (sum + 1e-20) * c.routed_scale;
            wts.add(gw[r * K + k], want, 32 * u * @abs(want), r, k);
        }
    }
}

const Mx = struct { codes: []const u8, scales: []const u8, k: usize };

/// An expert's MXFP4 matrix from the GPU table's addresses into the layer's arena.
fn expertMat(p: *const Probe, e: usize, which: usize, n: usize, k: usize) Mx {
    const t = p.w.table.slice(pkg.pipelines.experts.ExpertPtrs, p.experts())[e];
    const ptr = [6]u64{ t.w1p, t.w1s, t.w3p, t.w3s, t.w2p, t.w2s };
    return .{ .codes = p.w.bytesAt(ptr[2 * which])[0 .. n * k / 2], .scales = p.w.bytesAt(ptr[2 * which + 1])[0 .. n * k / 32], .k = k };
}

fn mxDot(m: Mx, j: usize, x: []const f64) [2]f64 {
    var s: f64 = 0;
    var mag: f64 = 0;
    for (x, 0..) |v, i| {
        const b = m.codes[j * m.k / 2 + i / 2];
        const t = v * ref.e2m1(if (i % 2 == 0) b & 15 else b >> 4) * ref.e8m0(m.scales[j * m.k / 32 + i / 32]);
        s += t;
        mag += @abs(t);
    }
    return .{ s, mag };
}

/// Routed experts: gate/up on act_quant(x) at bf16, clamped SwiGLU times the route weight, act_quant; down; the sum.
fn routed(p: *Probe) !void {
    const up_st = try p.stage("experts up (act, one E4M3 step)");
    const down_st = try p.stage("experts down");
    const sum_st = try p.stage("experts combine");
    const c = &p.r.c;
    const H: usize = c.hidden;
    const W: usize = c.expert_width;
    const K = p.topk();
    const a = p.a;
    const picks = p.i32s(p.r.s.picks, p.rows * K);
    const where = p.i32s(p.r.s.plan[3], p.rows * K);
    const wts = p.f32s(p.r.s.wts, p.rows * K);
    const act = p.bfs(p.r.s.act, p.rows * K * W);
    const eout = p.bfs(p.r.s.eout, p.rows * K * H);
    const yr = p.f32s(p.r.s.yr, p.rows * H);
    const hn2 = p.bfs(p.r.s.hn2, p.rows * H);
    const xq = try a.alloc(f64, H);
    defer a.free(xq);
    const xa = try a.alloc(f64, W);
    defer a.free(xa);
    const v16 = try a.alloc(u16, W);
    defer a.free(v16);
    const q16 = try a.alloc(u16, W);
    defer a.free(q16);
    var blk: [32]u16 = undefined;
    for (0..p.rows) |r| {
        for (0..H / 32) |b| {
            ref.actq32(hn2[r * H + 32 * b ..][0..32], &blk);
            for (0..32) |j| xq[32 * b + j] = ref.bf(blk[j]);
        }
        for (0..K) |k| {
            const e: usize = @intCast(picks[r * K + k]);
            const at: usize = @intCast(where[r * K + k]);
            const w1 = expertMat(p, e, 0, W, H);
            const w3 = expertMat(p, e, 1, W, H);
            const w2 = expertMat(p, e, 2, H, W);
            for (0..W) |j| {
                const g = @min(ref.bf(ref.toBf(@floatCast(mxDot(w1, j, xq)[0]))), c.swiglu_limit);
                const uu = std.math.clamp(ref.bf(ref.toBf(@floatCast(mxDot(w3, j, xq)[0]))), -c.swiglu_limit, c.swiglu_limit);
                const v = @as(f64, g) / (1 + @exp(-@as(f64, g))) * uu * wts[r * K + k];
                v16[j] = ref.toBf(@floatCast(v));
            }
            for (0..W / 32) |b| ref.actq32(v16[32 * b ..][0..32], q16[32 * b ..][0..32]);
            for (0..W) |j| {
                const want: f64 = ref.bf(q16[j]);
                up_st.add(ref.bf(act[at * W + j]), want, ref.quantStep(want) + ref.halfUlp(want), r, k * W + j);
                xa[j] = ref.bf(act[at * W + j]);
            }
            for (0..H) |j| {
                const d = mxDot(w2, j, xa);
                const depth: f64 = @floatFromInt(32 + W / 256 + 6);
                down_st.add(ref.bf(eout[at * H + j]), d[0], ref.bfBound(d[0], d[1], depth), r, k * H + j);
            }
        }
        for (0..H) |j| {
            var want: f64 = 0;
            var mag: f64 = 0;
            for (0..K) |k| {
                const v: f64 = ref.bf(eout[@as(usize, @intCast(where[r * K + k])) * H + j]);
                want += v;
                mag += @abs(v);
            }
            sum_st.add(yr[r * H + j], want, ref.fBound(want, mag, @floatFromInt(K + 1)), r, j);
        }
    }
}

/// The shared expert's SwiGLU: gate clamped above, up both ways, silu(gate) * up, bf16.
fn swiglu(p: *Probe) !void {
    const st = try p.stage("shared swiglu");
    const W: usize = p.r.c.expert_width;
    const lim: f64 = p.r.c.swiglu_limit;
    const g = p.bfs(p.r.s.sg, p.rows * W);
    const uv = p.bfs(p.r.s.su, p.rows * W);
    const y = p.bfs(p.r.s.sact, p.rows * W);
    for (0..p.rows * W) |i| {
        const gg = @min(@as(f64, ref.bf(g[i])), lim);
        const uu = std.math.clamp(@as(f64, ref.bf(uv[i])), -lim, lim);
        const want = gg / (1 + @exp(-gg)) * uu;
        st.add(ref.bf(y[i]), want, ref.bfBound(want, @abs(want), 8), i / W, i % W);
    }
}
