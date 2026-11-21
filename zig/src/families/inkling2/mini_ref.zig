//! Mini's forward in float64 on the host, unrounded, weights decoded exactly: the reference for GPU greedy tokens.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const names = @import("names.zig");

const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;

/// Each position's top token, margin over the runner-up, top logit, five best ids and logits (ties: the lower id).
pub const Pick = struct { id: u32, margin: f64, top: f64, ids: [5]u32 = @splat(0), logits: [5]f64 = @splat(-std.math.inf(f64)) };

const e4m3_table = blk: {
    @setEvalBranchQuota(100000);
    var t: [256]f64 = undefined;
    for (0..256) |i| t[i] = core.expert_decode.e4m3(@intCast(i));
    break :blk t;
};

fn bf(bytes: []const u8, i: usize) f64 {
    const u: u32 = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
    return @as(f32, @bitCast(u << 16));
}

fn f32At(bytes: []const u8, i: usize) f64 {
    return std.mem.bytesToValue(f32, bytes[4 * i ..][0..4]);
}

fn dotBf(w: []const u8, row: usize, k: usize, x: []const f64) f64 {
    var s: f64 = 0;
    const base = row * k;
    for (0..k) |i| s += bf(w, base + i) * x[i];
    return s;
}

/// y[n] = W[n, k] (bf16) x.
fn matvec(w: Tensor, x: []const f64, y: []f64) void {
    const k = x.len;
    for (y, 0..) |*o, j| o.* = dotBf(w.bytes, j, k, x);
}

fn rmsInv(x: []const f64, eps: f64) f64 {
    var s: f64 = 0;
    for (x) |v| s += v * v;
    return 1.0 / @sqrt(s / @as(f64, @floatFromInt(x.len)) + eps);
}

fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

fn mat2(a: std.mem.Allocator, n: usize, w: usize) ![][]f64 {
    const x = try a.alloc([]f64, n);
    for (x) |*row| row.* = try a.alloc(f64, w);
    return x;
}

/// x * scale (bf16 [.], from `at`), clipped to L2 norm `thr`, times tau.
fn clipScaled(x: []f64, scale: ?[]const u8, at: usize, thr: f64, tau: f64) void {
    if (scale) |sb| for (x, 0..) |*v, i| {
        v.* *= bf(sb, at + i);
    };
    var s: f64 = 0;
    for (x) |v| s += v * v;
    const f = @min(thr / (@sqrt(s) + 1e-6), 1);
    for (x) |*v| v.* *= f * tau;
}

/// A causal short conv in place: static [C][4] taps over sqrt(5), or `dyn` taps on static_w over 3; plus x.
fn convTime(x: [][]f64, w: Tensor, dyn: ?[][]f64, heads: usize) !void {
    const n = x.len;
    if (n == 0) return;
    const c = x[0].len;
    var prev: [3][]f64 = undefined;
    var buf: [3][8192]f64 = undefined;
    if (c > 8192) return error.Inkling2RefWidth;
    for (0..3) |k| {
        prev[k] = buf[k][0..c];
        @memset(prev[k], 0);
    }
    for (0..n) |t| {
        const cur = x[t];
        for (0..c) |ch| {
            const h0 = prev[0][ch];
            const h1 = prev[1][ch];
            const h2 = prev[2][ch];
            const xi = cur[ch];
            var y: f64 = undefined;
            if (dyn) |d| {
                const hd = ch / (c / heads);
                const dd = d[t][hd * 4 ..][0..4];
                y = (bf(w.bytes, ch) + dd[0]) * xi + (bf(w.bytes, c + ch) + dd[1]) * h2 + (bf(w.bytes, 2 * c + ch) + dd[2]) * h1 + (bf(w.bytes, 3 * c + ch) + dd[3]) * h0;
                y = (y + xi) / 3.0;
            } else {
                y = bf(w.bytes, ch * 4) * h0 + bf(w.bytes, ch * 4 + 1) * h1 + bf(w.bytes, ch * 4 + 2) * h2 + bf(w.bytes, ch * 4 + 3) * xi;
                y = (y + xi) / @sqrt(5.0);
            }
            prev[0][ch] = h1;
            prev[1][ch] = h2;
            prev[2][ch] = xi;
            cur[ch] = y;
        }
    }
}

/// Row `row` of one expert's stored matrix (k inputs) dotted with x: NVFP4 codes x E4M3 block scales, or MXFP8.
fn expertDot(nv: bool, codes: []const u8, scales: []const u8, k: usize, row: usize, x: []const f64) f64 {
    var s: f64 = 0;
    if (nv) {
        const cr = codes[row * (k / 2) ..][0 .. k / 2];
        const sr = scales[row * (k / 16) ..][0 .. k / 16];
        for (0..k) |i| s += core.expert_decode.e2m1(core.expert_decode.nibble(cr, i)) * e4m3_table[sr[i / 16]] * x[i];
    } else {
        const cr = codes[row * k ..][0..k];
        const sr = scales[row * (k / 32) ..][0 .. k / 32];
        for (0..k) |i| s += e4m3_table[cr[i]] * core.expert_decode.e8m0(sr[i / 32]) * x[i];
    }
    return s;
}

pub const Ref = struct {
    gpa: std.mem.Allocator,
    c: config.Config,
    p: plan_.Plan,
    ck: *Checkpoint,
    plant: bool = false, // a planted control: the cross-pass messages dropped (the check must then go red)

    fn get(r: *Ref, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [128]u8 = undefined;
        return r.ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    }

    /// The residual mix of `query` (fp32 rows) over sources (blocks then the partial) at one position, no norm.
    fn mixOne(h: usize, sources: []const []const f64, query: []const u8, qrow: usize, out: []f64) void {
        var w: [16]f64 = undefined;
        var inv: [16]f64 = undefined;
        var m: f64 = -std.math.inf(f64);
        for (sources, 0..) |s, j| {
            inv[j] = rmsInv(s, 1e-6);
            var d: f64 = 0;
            for (0..h) |i| d += f32At(query, qrow * h + i) * s[i];
            w[j] = d * inv[j] / @sqrt(@as(f64, @floatFromInt(h)));
            m = @max(m, w[j]);
        }
        var l: f64 = 0;
        for (sources, 0..) |_, j| {
            w[j] = @exp(w[j] - m);
            l += w[j];
        }
        @memset(out, 0);
        for (sources, 0..) |s, j| for (0..h) |i| {
            out[i] += w[j] / l * inv[j] * s[i];
        };
    }

    fn normInto(x: []f64, gamma: Tensor, eps: f64) void {
        const r = rmsInv(x, eps);
        for (x, 0..) |*v, i| v.* = v.* * r * bf(gamma.bytes, i);
    }

    /// A gate: (1 - tanh(Wf d)) h + tanh(Wg d) m with d = Wdown [unit(h), unit(m)], then normed by `gamma`.
    fn gate(r: *Ref, h: []const f64, msg: []const f64, ctl_layer: u32, comptime name: []const u8, gamma: Tensor, out: []f64) !void {
        const hd = h.len;
        const rank: usize = r.c.text.loop.?.rank;
        const eps: f64 = r.c.text.eps;
        const down = try r.get("model.llm.layers.{d}.controllers." ++ name ++ ".down_weight", .{ctl_layer});
        const forget = try r.get("model.llm.layers.{d}.controllers." ++ name ++ ".forget_weight", .{ctl_layer});
        const gain = try r.get("model.llm.layers.{d}.controllers." ++ name ++ ".gain_weight", .{ctl_layer});
        const cat = try r.gpa.alloc(f64, 2 * hd);
        defer r.gpa.free(cat);
        const ih = rmsInv(h, eps);
        const im = rmsInv(msg, eps);
        for (0..hd) |i| {
            cat[i] = h[i] * ih;
            cat[hd + i] = msg[i] * im;
        }
        var d: [config.max_gate_rank]f64 = undefined;
        for (0..rank) |j| d[j] = dotBf(down.bytes, j, 2 * hd, cat);
        for (0..hd) |i| {
            const f = dotBf(forget.bytes, i, rank, d[0..rank]);
            const g = dotBf(gain.bytes, i, rank, d[0..rank]);
            out[i] = (1 - std.math.tanh(f)) * h[i] + std.math.tanh(g) * msg[i];
        }
        normInto(out, gamma, eps);
    }

    /// Teacher-forced picks of every position of `ids` (`out.len == ids.len`).
    pub fn run(r: *Ref, ids: []const u32, out: []Pick) !void {
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const t = &r.c.text;
        const n = ids.len;
        const h: usize = t.hidden;
        const nblocks = (t.layers + t.block - 1) / t.block;
        var hcur = try mat2(a, n, h);
        var hnext = try mat2(a, n, h);
        var blocks: [16][][]f64 = undefined;
        for (0..nblocks) |b| blocks[b] = try mat2(a, n, h);
        const bank = try mat2(a, n, h);
        const bank_h = try mat2(a, n, h);
        const embed = try r.get("model.llm.embed.weight", .{});
        const enorm = try r.get("model.llm.embed_norm.weight", .{});
        for (ids, 0..) |id, i| {
            for (0..h) |c| hcur[i][c] = bf(embed.bytes, @as(usize, id) * h + c);
            normInto(hcur[i], enorm, t.eps);
        }
        for (r.p.visits.slice(), 0..) |_, v| {
            try r.visit(a, @intCast(v), ids.len, hcur, hnext, &blocks, bank, bank_h);
            std.mem.swap([][]f64, &hcur, &hnext);
        }
        const oq = try r.get("model.llm.attn_res.output_query", .{});
        const norm = try r.get("model.llm.norm.weight", .{});
        const unembed = try r.get("model.llm.unembed.weight", .{});
        const xf = try a.alloc(f64, h);
        var src: [17][]const f64 = undefined;
        for (0..n) |i| {
            const nb = r.p.final_sources - 1;
            for (0..nb) |b| src[b] = blocks[b][i];
            src[nb] = hcur[i];
            mixOne(h, src[0 .. nb + 1], oq.bytes, 0, xf);
            normInto(xf, norm, t.eps);
            var best: Pick = .{ .id = 0, .margin = 0, .top = 0 };
            for (0..t.unpadded_vocab) |vid| {
                const l = t.logit_cap * std.math.tanh(dotBf(unembed.bytes, vid, h, xf) / t.logit_cap);
                if (l <= best.logits[4]) continue;
                var at: usize = 4;
                while (at > 0 and l > best.logits[at - 1]) : (at -= 1) {
                    best.logits[at] = best.logits[at - 1];
                    best.ids[at] = best.ids[at - 1];
                }
                best.logits[at] = l;
                best.ids[at] = @intCast(vid);
            }
            best.id = best.ids[0];
            best.top = best.logits[0];
            best.margin = best.logits[0] - best.logits[1];
            out[i] = best;
        }
    }

    fn visit(r: *Ref, a: std.mem.Allocator, v: u32, n: usize, hin: [][]f64, hout: [][]f64, blocks: *[16][][]f64, bank: [][]f64, bank_h: [][]f64) !void {
        const t = &r.c.text;
        const p = &r.p;
        const i = p.visits.steps[v].layer;
        const spec = p.at(i);
        const h: usize = t.hidden;
        const global = spec.kind == .global;
        const msgs = p.messages(v) and !r.plant;
        const L = .{i};
        if (p.bank_blocks > 1) return error.Inkling2RefBank; // one banked block (Mini), or none (the tiny checkpoint)
        if (p.restarts(v)) for (0..n) |row| {
            for (0..p.bank_blocks) |b| @memcpy(bank[row], blocks[p.pre_blocks + 1 + b][row]);
            @memcpy(bank_h[row], hin[row]);
            @memcpy(hin[row], blocks[p.pre_blocks][row]);
        };
        const xin = [3][][]f64{ try a.alloc([]f64, n), try a.alloc([]f64, n), try a.alloc([]f64, n) };
        const msg = [4][][]f64{ try a.alloc([]f64, n), try a.alloc([]f64, n), try a.alloc([]f64, n), try a.alloc([]f64, n) };
        var norms: [3]Tensor = undefined;
        if (global) {
            norms = .{ try r.get("model.llm.layers.{d}.q_input_norm.weight", L), try r.get("model.llm.layers.{d}.k_input_norm.weight", L), try r.get("model.llm.layers.{d}.v_input_norm.weight", L) };
        } else {
            const one = try r.get("model.llm.layers.{d}.attn_norm.weight", L);
            norms = .{ one, one, one };
        }
        const aq: ?Tensor = if (i > 0) try r.get("model.llm.attn_res.attn_queries.{d}", .{i - 1}) else null;
        const core_j = p.coreIndex(v);
        const cross = .{ try r.get("model.llm.middle_cycle_cross_pass_queries", .{}), try r.get("model.llm.middle_cycle_cross_pass_kv_queries", .{}) };
        var src: [17][]const f64 = undefined;
        for (0..n) |row| {
            for (0..3) |q| xin[q][row] = try a.alloc(f64, h);
            if (msgs) {
                for (0..4) |q| msg[q][row] = try a.alloc(f64, h);
                for (0..p.bank_blocks) |b| src[b] = bank[row];
                src[p.bank_blocks] = bank_h[row];
                const sources = src[0 .. p.bank_blocks + 1];
                // the message queries: main 2j (q), kv 2j (k), kv 2j + 1 (v), main 2j + 1 (mlp)
                mixOne(h, sources, cross[0].bytes, 2 * core_j, msg[0][row]);
                mixOne(h, sources, cross[1].bytes, 2 * core_j, msg[1][row]);
                mixOne(h, sources, cross[1].bytes, 2 * core_j + 1, msg[2][row]);
                mixOne(h, sources, cross[0].bytes, 2 * core_j + 1, msg[3][row]);
            }
            const nq: usize = spec.attn_queries;
            for (0..spec.blocks_in) |b| src[b] = blocks[b][row];
            src[spec.blocks_in] = hin[row];
            for (0..@max(nq, 1)) |q| {
                if (i == 0) {
                    @memcpy(xin[q][row], hin[row]);
                } else mixOne(h, src[0 .. spec.blocks_in + 1], aq.?.bytes, q, xin[q][row]);
            }
            if (msgs) {
                var y: [3][]f64 = undefined;
                inline for (.{ "attn", "k", "v" }, 0..) |name, g| {
                    y[g] = try a.alloc(f64, h);
                    try r.gate(xin[if (nq == 3) g else 0][row], msg[g][row], i, name, norms[g], y[g]);
                }
                for (0..3) |g| @memcpy(xin[g][row], y[g]);
            } else for (0..@max(nq, 1)) |q| normInto(xin[q][row], norms[q], t.eps);
            if (!global and !msgs) for (1..3) |q| @memcpy(xin[q][row], xin[0][row]);
        }
        // the attention, its conv into the residual, the block boundary, the MLP input, experts and the MLP conv
        const h32 = try mat2(a, n, h);
        try r.attention(a, v, n, xin, h32);
        try convTime(h32, try r.get("model.llm.layers.{d}.attn_sconv.weight", L), null, 1);
        for (0..n) |row| if (!spec.boundary) for (0..h) |c| {
            h32[row][c] += hin[row][c];
        };
        if (spec.boundary and !p.restarts(v)) for (0..n) |row| @memcpy(blocks[i / t.block][row], hin[row]);
        const nb = spec.blocks_in + @intFromBool(spec.boundary);
        const mq = try r.get("model.llm.attn_res.mlp_queries.{d}", L);
        const mnorm = try r.get("model.llm.layers.{d}.mlp_norm.weight", L);
        const moe = try mat2(a, n, h);
        const xm = try a.alloc(f64, h);
        for (0..n) |row| {
            for (0..nb) |b| src[b] = blocks[b][row];
            src[nb] = h32[row];
            mixOne(h, src[0 .. nb + 1], mq.bytes, 0, xm);
            if (msgs) {
                const y = try a.alloc(f64, h);
                try r.gate(xm, msg[3][row], i, "mlp", mnorm, y);
                @memcpy(xm, y);
            } else normInto(xm, mnorm, t.eps);
            try r.experts(a, i, xm, moe[row]);
        }
        try convTime(moe, try r.get("model.llm.layers.{d}.mlp_sconv.weight", L), null, 1);
        for (0..n) |row| for (0..h) |c| {
            hout[row][c] = h32[row][c] + moe[row][c];
        };
    }

    /// The routed experts at one position: sigmoid scores, top-k by score + bias, the soft-capped SwiGLU, weighted sum.
    fn experts(r: *Ref, a: std.mem.Allocator, i: u32, x: []const f64, out: []f64) !void {
        const t = &r.c.text;
        const h: usize = t.hidden;
        const m: usize = t.expert_inter;
        const L = .{i};
        const router = try r.get("model.llm.layers.{d}.mlp.gate.weight", L);
        const bias = try r.get("model.llm.layers.{d}.mlp.gate.bias", L);
        var score: [512]f64 = undefined;
        var sel: [512]f64 = undefined;
        for (0..t.experts) |e| {
            score[e] = sigmoid(dotBf(router.bytes, e, h, x));
            sel[e] = score[e] + f32At(bias.bytes, e);
        }
        var pick: [32]usize = undefined;
        var total: f64 = 0;
        for (0..t.topk) |k| {
            var best: usize = 0;
            var bv = -std.math.inf(f64);
            for (0..t.experts) |e| if (sel[e] > bv) {
                bv = sel[e];
                best = e;
            };
            pick[k] = best;
            sel[best] = -std.math.inf(f64);
            total += score[best];
        }
        const pre = "model.llm.layers.{d}.mlp.experts.";
        const w13 = try r.get(pre ++ "w13_weight", L);
        const s13 = try r.get(pre ++ "w13_weight_scale", L);
        const w2 = try r.get(pre ++ "w2_weight", L);
        const s2 = try r.get(pre ++ "w2_weight_scale", L);
        const nv = w13.dtype == .u8;
        var f13: f64 = 1;
        var f2: f64 = 1;
        const act = try a.alloc(f64, m);
        @memset(out, 0);
        for (pick[0..t.topk]) |e| {
            if (nv) {
                const den: f32 = @floatCast(f32At((try r.get(pre ++ "w13_weight_outer_scale_denominator", L)).bytes, 0));
                f13 = @import("cuda").experts_mx.nvfp4Factor(@floatCast(f32At((try r.get(pre ++ "w13_weight_amax", L)).bytes, e)), den);
                f2 = @import("cuda").experts_mx.nvfp4Factor(@floatCast(f32At((try r.get(pre ++ "w2_weight_amax", L)).bytes, e)), den);
            }
            const c13 = w13.bytes[e * (w13.bytes.len / t.experts) ..][0 .. w13.bytes.len / t.experts];
            const sc13 = s13.bytes[e * (s13.bytes.len / t.experts) ..][0 .. s13.bytes.len / t.experts];
            for (0..m) |j| {
                var gu: [2]f64 = .{ 0, 0 };
                for (0..2) |half| gu[half] = expertDot(nv, c13, sc13, h, 2 * j + half, x) * f13; // rows interleave gate, up
                act[j] = sigmoid(gu[0]) * t.gate_cap * std.math.tanh(gu[0] / t.gate_cap) * t.up_cap * std.math.tanh(gu[1] / t.up_cap);
            }
            const c2 = w2.bytes[e * (w2.bytes.len / t.experts) ..][0 .. w2.bytes.len / t.experts];
            const sc2 = s2.bytes[e * (s2.bytes.len / t.experts) ..][0 .. s2.bytes.len / t.experts];
            const w = score[e] / (total + 1e-20) * t.route_scale;
            for (0..h) |row| out[row] += w * expertDot(nv, c2, sc2, m, row, act) * f2;
        }
    }

    /// The attention of every position at visit `v` from its inputs (q from plane 0, k from 1, v from 2) into `y`.
    fn attention(r: *Ref, a: std.mem.Allocator, v: u32, n: usize, xin: [3][][]f64, y: [][]f64) !void {
        const t = &r.c.text;
        const p = &r.p;
        const i = p.visits.steps[v].layer;
        const global = p.at(i).kind == .global;
        const L = .{i};
        const hd = t.local;
        const qw: usize = @as(usize, hd.q) * hd.dim;
        const kvw: usize = @as(usize, hd.kv) * hd.dim;
        const dr: usize = t.d_rel;
        const q = try mat2(a, n, qw);
        const k = try mat2(a, n, kvw);
        const vv = try mat2(a, n, kvw);
        const rel = try mat2(a, n, hd.q * dr);
        const gate_ = try mat2(a, n, qw);
        const sink = try mat2(a, n, hd.q);
        const w = .{
            try r.get("model.llm.layers.{d}.attn.wq_du.weight", L),     try r.get("model.llm.layers.{d}.attn.wk_dv.weight", L),
            try r.get("model.llm.layers.{d}.attn.wv_dv.weight", L),     try r.get("model.llm.layers.{d}.attn.wr_du.weight", L),
            try r.get("model.llm.layers.{d}.attn.gate_attn.weight", L), try r.get("model.llm.layers.{d}.attn.attn_sink_proj.weight", L),
        };
        for (0..n) |row| {
            matvec(w[0], xin[0][row], q[row]);
            matvec(w[1], xin[1][row], k[row]);
            matvec(w[2], xin[2][row], vv[row]);
            matvec(w[3], xin[0][row], rel[row]);
            matvec(w[4], xin[0][row], gate_[row]);
            matvec(w[5], xin[0][row], sink[row]);
        }
        if (global) {
            const taps = kvw / hd.dim * 4;
            const dk = try mat2(a, n, taps);
            const dv = try mat2(a, n, taps);
            const pk = try r.get("model.llm.layers.{d}.attn.k_sconv.weight_proj", L);
            const pv = try r.get("model.llm.layers.{d}.attn.v_sconv.weight_proj", L);
            for (0..n) |row| {
                matvec(pk, xin[1][row], dk[row]);
                matvec(pv, xin[2][row], dv[row]);
            }
            try convTime(k, try r.get("model.llm.layers.{d}.attn.k_sconv.static_w", L), dk, hd.kv);
            try convTime(vv, try r.get("model.llm.layers.{d}.attn.v_sconv.static_w", L), dv, hd.kv);
        } else {
            try convTime(k, try r.get("model.llm.layers.{d}.attn.k_sconv.weight", L), null, 1);
            try convTime(vv, try r.get("model.llm.layers.{d}.attn.v_sconv.weight", L), null, 1);
        }
        const qs = try r.get("model.llm.layers.{d}.attn.q_l2_clip.scale", L);
        const rs = try r.get("model.llm.layers.{d}.attn.rel_l2_clip.scale", L);
        const proj = try r.get("model.llm.layers.{d}.attn.rel_logits_proj.proj", L);
        const extent: usize = proj.dim(1);
        for (0..n) |row| {
            const tau: f64 = if (global) 1 + t.log_alpha * @log(@max(@as(f64, @floatFromInt(row + 1)) / @as(f64, @floatFromInt(t.log_floor)), 1)) else 1;
            for (0..hd.q) |hh| {
                clipScaled(q[row][hh * hd.dim ..][0..hd.dim], qs.bytes, hh * hd.dim, t.qk_clip, tau);
                clipScaled(rel[row][hh * dr ..][0..dr], rs.bytes, hh * dr, t.rel_clip, tau);
            }
            for (0..hd.kv) |hh| {
                clipScaled(k[row][hh * hd.dim ..][0..hd.dim], null, 0, t.qk_clip, 1);
                const vr = vv[row][hh * hd.dim ..][0..hd.dim];
                const inv = rmsInv(vr, t.eps);
                for (vr) |*x| x.* *= inv;
            }
        }
        const att = try mat2(a, n, qw);
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(hd.dim)));
        const group = hd.q / hd.kv;
        const sc = try a.alloc(f64, n);
        for (0..n) |row| for (0..hd.q) |hh| {
            const kh = hh / group;
            const lo = if (global) 0 else (row + 1) -| t.window;
            var mx: f64 = sink[row][hh];
            for (lo..row + 1) |j| {
                var s: f64 = 0;
                for (0..hd.dim) |d| s += q[row][hh * hd.dim + d] * k[j][kh * hd.dim + d];
                const dist = row - j;
                var bias: f64 = 0;
                if (dist < extent) for (0..dr) |d| {
                    bias += rel[row][hh * dr + d] * bf(proj.bytes, d * extent + dist);
                };
                sc[j] = s * scale + bias;
                mx = @max(mx, sc[j]);
            }
            var l: f64 = @exp(sink[row][hh] - mx);
            const o = att[row][hh * hd.dim ..][0..hd.dim];
            @memset(o, 0);
            for (lo..row + 1) |j| {
                const e = @exp(sc[j] - mx);
                l += e;
                for (0..hd.dim) |d| o[d] += e * vv[j][kh * hd.dim + d];
            }
            for (o, 0..) |*x, d| x.* = x.* / l * sigmoid(gate_[row][hh * hd.dim + d]);
        };
        const wo = try r.get("model.llm.layers.{d}.attn.wo_ud.weight", L);
        for (0..n) |row| matvec(wo, att[row], y[row]);
    }
};
