//! Inkling 2 Mini's DSpark head for tests: the checkpoint's own structure (3 layers with the first and last sliding, 4
//! logical taps, 3 query heads to a key/value head, rnn_hidden Markov head) at test widths with random bf16 values, and
//! an independent f64 reference of a block that reads the head's tensors from their file, whatever its widths.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const draft = @import("metal_draft.zig");

pub const head = 128;
pub const max_layers = 8;
pub const max_taps = 16;

pub fn bf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn unbf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

pub fn rb(x: f64) f64 {
    return unbf(bf(@floatCast(x)));
}

/// The widths of a head, read from the config (the reference never hard-codes Mini's own).
pub const Dims = struct {
    D: usize,
    V: usize,
    unpadded: usize,
    rank: usize,
    taps: usize,
    layers: usize,
    heads: usize,
    kv: usize,
    inter: usize,
    block: usize,
    window: usize,
    theta: f64,
    cap: f64,
    eps: f64,
    local: [max_layers]bool,

    pub fn of(c: *const config.Config) Dims {
        const m = c.mtp.?;
        return .{ .D = c.text.hidden, .V = c.text.vocab, .unpadded = c.text.unpadded_vocab, .rank = m.rank, .taps = m.taps_n, .layers = m.layers, .heads = m.q_heads, .kv = m.kv_heads, .inter = m.inter, .block = m.block, .window = m.window, .theta = m.rope_theta, .cap = c.text.logit_cap, .eps = m.eps, .local = m.local };
    }
};

/// The real Mini config.json (inkling2_loop, taps [5, 23, 41, 48], 24 physical layers in 48 logical visits) narrowed to
/// test widths: same layer plan, window, block, taps and head ratio; 6 query heads over 2 (as 24 over 8), width 128.
pub fn tinyConfig(a: std.mem.Allocator) !config.Config {
    var why: config.Why = .{};
    var c = try config.parse(a, @embedFile("fixtures/mini_media_config.json"), &why);
    const m = &c.mtp.?;
    c.text.hidden = 128;
    m.q_heads = 6;
    m.kv_heads = 2;
    m.inter = 128;
    m.rank = 8;
    c.vision.text_hidden = 128;
    return c;
}

/// The head's tensors as host views, the target's embedding and unembedding, and the tap mixtures' softmax.
pub const Reference = struct {
    d: Dims,
    ckpt: ck.Checkpoint,
    owned: bool = false,
    embed: []const u16,
    embed_norm: []const u16,
    unembed: []const u16,
    alpha: [max_layers + 1][max_taps]f64 = undefined,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, c: *const config.Config, target: draft.Target) !Reference {
        var store = ck.Checkpoint.init(gpa);
        errdefer store.deinit();
        const joined = try std.fs.path.join(gpa, &.{ dir, "mtp.safetensors" });
        defer gpa.free(joined);
        const path = try gpa.dupeSentinel(u8, joined, 0);
        defer gpa.free(path);
        try store.addFile(device, path, "");
        var r = fromCheckpoint(&store, c, target);
        r.owned = true;
        return r;
    }

    pub fn fromCheckpoint(store: *const ck.Checkpoint, c: *const config.Config, target: draft.Target) Reference {
        var r: Reference = .{ .d = Dims.of(c), .ckpt = store.*, .embed = target.embed.host(u16), .embed_norm = target.embed_norm.host(u16), .unembed = target.unembed.host(u16) };
        const fuse = r.w("fuse_weight", .{});
        for (0..r.d.layers + 1) |row| {
            var top: f64 = -std.math.inf(f64);
            for (0..r.d.taps) |t| top = @max(top, unbf(fuse[row * r.d.taps + t]));
            var z: f64 = 0;
            for (0..r.d.taps) |t| z += @exp(unbf(fuse[row * r.d.taps + t]) - top);
            for (0..r.d.taps) |t| r.alpha[row][t] = rb(@exp(unbf(fuse[row * r.d.taps + t]) - top) / z);
        }
        return r;
    }

    pub fn deinit(r: *Reference) void {
        if (r.owned) r.ckpt.deinit();
    }

    pub fn w(r: *const Reference, comptime fmt: []const u8, args: anytype) []const u16 {
        var buf: [96]u8 = undefined;
        return (r.ckpt.get(std.fmt.bufPrint(&buf, fmt, args) catch unreachable) catch unreachable).host(u16);
    }

    fn rope(r: *const Reference, v: []const f64, pos: usize, out: []f64) void {
        for (0..head) |d| {
            const f: f32 = @floatCast(std.math.pow(f64, r.d.theta, -2.0 * @as(f64, @floatFromInt(d & 63)) / head));
            const ang: f32 = @as(f32, @floatFromInt(pos)) * f;
            const c: f64 = @cos(ang);
            const s: f64 = @sin(ang);
            out[d] = rb(if (d < 64) v[d] * c - v[d + 64] * s else v[d] * c + v[d - 64] * s);
        }
    }

    /// One head vector: bf16(rms norm with weight), then the rotation at `pos`.
    fn normRope(r: *const Reference, x: []const f64, wv: []const u16, pos: usize, out: []f64) void {
        var ss: f64 = 0;
        for (x) |e| ss += e * e;
        const inv = 1.0 / @sqrt(ss / head + r.d.eps);
        var v: [head]f64 = undefined;
        for (0..head) |d| v[d] = rb(x[d] * inv * unbf(wv[d]));
        r.rope(&v, pos, out);
    }

    fn rms(r: *const Reference, x: []const f64, wv: []const u16, out: []f64) void {
        var ss: f64 = 0;
        for (x) |e| ss += e * e;
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + r.d.eps);
        for (x, out, wv) |e, *o, g| o.* = rb(e * inv * unbf(g));
    }

    /// The mix-normed tap row `row` of `feats` ([rows, taps, D]) for alpha row `layer`.
    fn mixRow(r: *const Reference, a: std.mem.Allocator, feats: []const u16, row: usize, layer: usize, wv: []const u16, out: []f64) !void {
        const D = r.d.D;
        const m = try a.alloc(f64, D);
        defer a.free(m);
        for (0..D) |i| {
            var s: f64 = 0;
            for (0..r.d.taps) |t| s += unbf(feats[(row * r.d.taps + t) * D + i]) * r.alpha[layer][t];
            m[i] = s;
        }
        r.rms(m, wv, out);
    }

    /// The final-normed rows [block, D] of the block run after `p` committed rows (`feats`), the anchor token first.
    pub fn blockRows(r: *const Reference, a: std.mem.Allocator, feats: []const u16, p: usize, anchor: u32) ![]f64 {
        const d = r.d;
        const D = d.D;
        const block = d.block;
        const resid = try a.alloc(f64, block * D);
        const x = try a.alloc(f64, block * D);
        const e = try a.alloc(f64, D);
        for (0..D) |i| e[i] = unbf(r.embed[@as(usize, anchor) * D + i]);
        var ss: f64 = 0;
        for (e) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(D)) + d.eps);
        for (0..D) |i| resid[i] = rb(unbf(r.embed_norm[i]) * (e[i] * inv));
        for (1..block) |row| for (0..D) |i| {
            resid[row * D + i] = unbf(r.w("mask_embed", .{})[i]);
        };
        for (0..block) |row| r.rms(resid[row * D ..][0..D], r.w("layers.0.input_layernorm.weight", .{}), x[row * D ..][0..D]);
        const kvw = d.kv * head;
        const qw = d.heads * head;
        const q = try a.alloc(f64, block * qw);
        const kraw = try a.alloc(f64, block * kvw);
        const vraw = try a.alloc(f64, block * kvw);
        const attn = try a.alloc(f64, block * qw);
        const y = try a.alloc(f64, block * D);
        const g = try a.alloc(f64, block * d.inter);
        const u = try a.alloc(f64, block * d.inter);
        const c = try a.alloc(f64, D);
        for (0..d.layers) |l| {
            matNT(x, block, D, r.w("layers.{d}.self_attn.q_proj.weight", .{l}), qw, q);
            matNT(x, block, D, r.w("layers.{d}.self_attn.k_proj.weight", .{l}), kvw, kraw);
            matNT(x, block, D, r.w("layers.{d}.self_attn.v_proj.weight", .{l}), kvw, vraw);
            for (0..block) |row| {
                var o: [head]f64 = undefined;
                for (0..d.heads) |h| {
                    r.normRope(q[row * qw + h * head ..][0..head], r.w("layers.{d}.self_attn.q_norm.weight", .{l}), p + row, &o);
                    @memcpy(q[row * qw + h * head ..][0..head], &o);
                }
                for (0..d.kv) |h| {
                    r.normRope(kraw[row * kvw + h * head ..][0..head], r.w("layers.{d}.self_attn.k_norm.weight", .{l}), p + row, &o);
                    @memcpy(kraw[row * kvw + h * head ..][0..head], &o);
                }
            }
            // a sliding layer keeps the last window of committed rows, the full layer all of them
            const lo: usize = if (d.local[l]) p -| d.window else 0;
            const nctx = p - lo;
            const ck_ = try a.alloc(f64, @max(nctx, 1) * kvw);
            const cv = try a.alloc(f64, @max(nctx, 1) * kvw);
            const kproj = r.w("layers.{d}.self_attn.k_proj_ctx.weight", .{l});
            const vproj = r.w("layers.{d}.self_attn.v_proj_ctx.weight", .{l});
            const knorm = r.w("layers.{d}.self_attn.k_norm_ctx.weight", .{l});
            const dnorm = r.w("layers.{d}.dflare_norm.weight", .{l});
            for (0..nctx) |i| {
                try r.mixRow(a, feats, lo + i, l, dnorm, c);
                for (0..kvw) |o| {
                    var sk: f64 = 0;
                    var sv: f64 = 0;
                    for (0..D) |j| {
                        sk += c[j] * unbf(kproj[o * D + j]);
                        sv += c[j] * unbf(vproj[o * D + j]);
                    }
                    ck_[i * kvw + o] = rb(sk);
                    cv[i * kvw + o] = rb(sv);
                }
                var o: [head]f64 = undefined;
                for (0..d.kv) |h| {
                    r.normRope(ck_[i * kvw + h * head ..][0..head], knorm, lo + i, &o);
                    @memcpy(ck_[i * kvw + h * head ..][0..head], &o);
                }
            }
            const total = nctx + block;
            const sc = try a.alloc(f64, total);
            for (0..block) |row| for (0..d.heads) |h| {
                const grp = h / (d.heads / d.kv);
                var top: f64 = -std.math.inf(f64);
                for (0..total) |j| {
                    const kk = if (j < nctx) ck_[j * kvw + grp * head ..] else kraw[(j - nctx) * kvw + grp * head ..];
                    var s: f64 = 0;
                    for (0..head) |t| s += q[row * qw + h * head + t] * kk[t];
                    sc[j] = s / @sqrt(@as(f64, head));
                    top = @max(top, sc[j]);
                }
                var z: f64 = 0;
                for (sc) |*s| {
                    s.* = @exp(s.* - top);
                    z += s.*;
                }
                for (0..head) |t| {
                    var o: f64 = 0;
                    for (0..total) |j| {
                        const vv = if (j < nctx) cv[j * kvw + grp * head + t] else vraw[(j - nctx) * kvw + grp * head + t];
                        o += sc[j] / z * vv;
                    }
                    attn[row * qw + h * head + t] = rb(o);
                }
            };
            matNT(attn, block, qw, r.w("layers.{d}.self_attn.o_proj.weight", .{l}), D, y);
            r.addNorm(resid, y, r.w("layers.{d}.post_attention_layernorm.weight", .{l}), x);
            matNT(x, block, D, r.w("layers.{d}.mlp.gate_proj.weight", .{l}), d.inter, g);
            matNT(x, block, D, r.w("layers.{d}.mlp.up_proj.weight", .{l}), d.inter, u);
            for (g, u) |*gv, uv| gv.* = rb(gv.* / (1.0 + @exp(-gv.*)) * uv);
            matNT(g, block, d.inter, r.w("layers.{d}.mlp.down_proj.weight", .{l}), D, y);
            const next = if (l + 1 < d.layers) r.w("layers.{d}.input_layernorm.weight", .{l + 1}) else r.w("norm.weight", .{});
            r.addNorm(resid, y, next, x);
        }
        return x;
    }

    /// resid = bf16(resid + delta); x = bf16(norm of the unrounded sum * w), per row.
    fn addNorm(r: *const Reference, resid: []f64, delta: []const f64, wv: []const u16, x: []f64) void {
        const D = r.d.D;
        for (0..r.d.block) |row| {
            var ss: f64 = 0;
            for (0..D) |i| {
                const s = resid[row * D + i] + delta[row * D + i];
                ss += s * s;
            }
            const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(D)) + r.d.eps);
            for (0..D) |i| {
                const s = resid[row * D + i] + delta[row * D + i];
                resid[row * D + i] = rb(s);
                x[row * D + i] = rb(s * inv * unbf(wv[i]));
            }
        }
    }

    /// The Markov steps teacher-forced on the GPU's choices: per step the best corrected logit over `cand` (null: the
    /// whole unpadded vocabulary) and the chosen token's. The folded unembedding is formed for the tokens read only.
    pub fn steps(r: *const Reference, a: std.mem.Allocator, x: []const f64, feats_last: []const u16, anchor: u32, chosen: []const u32, cand: ?[]const u32, best: []f64, got: []f64) !void {
        const d = r.d;
        const D = d.D;
        const rank = d.rank;
        const ho_w = r.w("markov_head.hidden_out.weight", .{});
        const ho_b = r.w("markov_head.hidden_out.bias", .{});
        const w1 = r.w("markov_head.markov_w1.weight", .{});
        const rw = r.w("markov_head.rnn_proj.weight", .{});
        const rbias = r.w("markov_head.rnn_proj.bias", .{});
        const frnn = try a.alloc(f64, D);
        try r.mixRow(a, feats_last, 0, d.layers, r.w("markov_head.dflare_norm.weight", .{}), frnn);
        var toks: std.ArrayList(u32) = .empty;
        if (cand) |list| {
            try toks.appendSlice(a, list);
            try toks.appendSlice(a, chosen);
        } else for (0..d.unpadded) |v| try toks.append(a, @intCast(v));
        const fold_w = try a.alloc(f64, toks.items.len * rank);
        const fold_b = try a.alloc(f64, toks.items.len);
        for (toks.items, 0..) |v, ti| {
            var sb: f64 = 0;
            const sw = fold_w[ti * rank ..][0..rank];
            @memset(sw, 0);
            for (0..D) |k| {
                const uv = unbf(r.unembed[@as(usize, v) * D + k]);
                sb += uv * unbf(ho_b[k]);
                for (0..rank) |c| sw[c] += uv * unbf(ho_w[k * rank + c]);
            }
            fold_b[ti] = rb(sb);
            for (sw) |*s| s.* = rb(s.*);
        }
        const state = try a.alloc(f64, rank);
        @memset(state, 0);
        const z = try a.alloc(f64, 2 * rank + 2 * D);
        const raw = try a.alloc(f64, 3 * rank);
        const latent = try a.alloc(f64, rank);
        var prev = anchor;
        for (0..chosen.len) |i| {
            for (0..rank) |k| z[k] = state[k];
            for (0..rank) |k| z[rank + k] = unbf(w1[@as(usize, prev) * rank + k]);
            for (0..D) |k| z[2 * rank + k] = x[(i + 1) * D + k];
            for (0..D) |k| z[2 * rank + D + k] = frnn[k];
            for (0..3 * rank) |o| {
                var s: f64 = 0;
                for (z, 0..) |zv, k| s += zv * unbf(rw[o * z.len + k]);
                raw[o] = rb(s + unbf(rbias[o]));
            }
            for (0..rank) |j| {
                const gate = rb(1.0 / (1.0 + @exp(-raw[j])));
                const keep = rb(gate * state[j]);
                const add = rb(rb(1.0 - gate) * rb(std.math.tanh(raw[rank + j])));
                state[j] = rb(keep + add);
                latent[j] = rb(std.math.tanh(raw[2 * rank + j]));
            }
            var top: f64 = -std.math.inf(f64);
            got[i] = -std.math.inf(f64);
            for (toks.items, 0..) |v, ti| {
                if (v >= d.unpadded) continue;
                var base: f64 = 0;
                for (0..D) |k| base += x[(i + 1) * D + k] * unbf(r.unembed[@as(usize, v) * D + k]);
                var s: f64 = fold_b[ti];
                for (0..rank) |c| s += latent[c] * fold_w[ti * rank + c];
                const logit = rb(d.cap * std.math.tanh(rb(rb(base) + rb(s)) / d.cap));
                top = @max(top, logit);
                if (v == chosen[i]) got[i] = logit;
            }
            best[i] = top;
            prev = chosen[i];
        }
    }
};

/// out [rows, n] = bf16(x [rows, k] w^T) for w [n, k].
fn matNT(x: []const f64, rows: usize, k: usize, wv: []const u16, n: usize, out: []f64) void {
    for (0..rows) |r| for (0..n) |o| {
        var s: f64 = 0;
        for (0..k) |i| s += x[r * k + i] * unbf(wv[o * k + i]);
        out[r * n + o] = rb(s);
    };
}

/// Random target tap rows [rows, taps, D] as bf16.
pub fn features(a: std.mem.Allocator, d: Dims, rows: usize, seed: u64) ![]u16 {
    const out = try a.alloc(u16, rows * d.taps * d.D);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (out) |*h| h.* = bf(rng.float(f32) * 2 - 1);
    return out;
}

pub fn tensorOf(b: mtl.Buffer, dims: []const usize) ck.Tensor {
    var t: ck.Tensor = .{ .buffer = b, .offset = 0, .bytes = 2, .dtype = .bf16, .rank = dims.len };
    for (dims, 0..) |dim, i| {
        t.shape[i] = dim;
        t.bytes *= dim;
    }
    return t;
}

/// Random target tensors the head shares (embedding, its norm, unembedding), on the device.
pub const Target = struct {
    embed: mtl.Buffer,
    embed_norm: mtl.Buffer,
    unembed: mtl.Buffer,

    pub fn init(device: mtl.Device, V: usize, D: usize) !Target {
        const embed = try random(device, V * D, 1.0, 0);
        errdefer embed.deinit();
        const norm = try random(device, D, 0.1, 1.0);
        errdefer norm.deinit();
        return .{ .embed = embed, .embed_norm = norm, .unembed = try random(device, V * D, 0.08, 0) };
    }

    pub fn deinit(t: Target) void {
        t.embed.deinit();
        t.embed_norm.deinit();
        t.unembed.deinit();
    }

    pub fn tensors(t: Target, V: usize, D: usize) draft.Target {
        return .{ .embed = tensorOf(t.embed, &.{ V, D }), .embed_norm = tensorOf(t.embed_norm, &.{D}), .unembed = tensorOf(t.unembed, &.{ V, D }) };
    }
};

fn random(device: mtl.Device, n: usize, scale: f32, bias: f32) !mtl.Buffer {
    const b = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    var prng = std.Random.DefaultPrng.init(n ^ 0x5eed);
    const rng = prng.random();
    for (b.slice(u16, n)) |*h| h.* = bf(bias + scale * (rng.float(f32) * 2 - 1));
    return b;
}

/// The synthetic head: a tiny Mini config, its mtp.safetensors in a temporary directory, random target tensors, and the
/// reference over both.
pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    dir: []u8,
    cfg: config.Config,
    d: Dims,
    target: Target,
    ref: Reference,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device) !*Fixture {
        const fx = try gpa.create(Fixture);
        errdefer gpa.destroy(fx);
        fx.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer fx.arena.deinit();
        fx.tmp = std.testing.tmpDir(.{});
        errdefer fx.tmp.cleanup();
        fx.dir = try fx.tmp.dir.realPathFileAlloc(io, ".", fx.arena.allocator());
        fx.cfg = try tinyConfig(fx.arena.allocator());
        fx.d = Dims.of(&fx.cfg);
        try fx.generate(io);
        fx.target = try Target.init(device, fx.d.V, fx.d.D);
        errdefer fx.target.deinit();
        fx.ref = try Reference.init(gpa, device, fx.dir, &fx.cfg, fx.tensors());
        return fx;
    }

    pub fn deinit(fx: *Fixture) void {
        const gpa = fx.arena.child_allocator;
        fx.ref.deinit();
        fx.target.deinit();
        fx.tmp.cleanup();
        fx.arena.deinit();
        gpa.destroy(fx);
    }

    /// The target the head shares, as tensors over the fixture's buffers.
    pub fn tensors(fx: *const Fixture) draft.Target {
        return fx.target.tensors(fx.d.V, fx.d.D);
    }

    fn generate(fx: *Fixture, io: std.Io) !void {
        const a = fx.arena.allocator();
        const d = fx.d;
        var header: std.ArrayList(u8) = .empty;
        var data: std.ArrayList(u8) = .empty;
        try header.append(a, '{');
        var prng = std.Random.DefaultPrng.init(0x1d5a);
        const rng = prng.random();
        const Spec = struct { name: []const u8, dims: []const usize, scale: f32, bias: f32 };
        var specs: std.ArrayList(Spec) = .empty;
        const qw = d.heads * head;
        const kvw = d.kv * head;
        try specs.append(a, .{ .name = "fuse_weight", .dims = &.{ d.layers + 1, d.taps }, .scale = 1.5, .bias = 0 });
        try specs.append(a, .{ .name = "mask_embed", .dims = &.{d.D}, .scale = 1.0, .bias = 0 });
        try specs.append(a, .{ .name = "norm.weight", .dims = &.{d.D}, .scale = 0.1, .bias = 1 });
        try specs.append(a, .{ .name = "markov_head.dflare_norm.weight", .dims = &.{d.D}, .scale = 0.1, .bias = 1 });
        try specs.append(a, .{ .name = "markov_head.hidden_out.weight", .dims = &.{ d.D, d.rank }, .scale = 0.3, .bias = 0 });
        try specs.append(a, .{ .name = "markov_head.hidden_out.bias", .dims = &.{d.D}, .scale = 0.3, .bias = 0 });
        try specs.append(a, .{ .name = "markov_head.markov_w1.weight", .dims = &.{ d.V, d.rank }, .scale = 1.0, .bias = 0 });
        try specs.append(a, .{ .name = "markov_head.rnn_proj.weight", .dims = &.{ 3 * d.rank, 2 * d.D + 2 * d.rank }, .scale = 0.15, .bias = 0 });
        try specs.append(a, .{ .name = "markov_head.rnn_proj.bias", .dims = &.{3 * d.rank}, .scale = 0.2, .bias = 0 });
        for (0..d.layers) |l| {
            const norms = [_]struct { []const u8, usize }{ .{ "dflare_norm.weight", d.D }, .{ "input_layernorm.weight", d.D }, .{ "post_attention_layernorm.weight", d.D }, .{ "self_attn.q_norm.weight", head }, .{ "self_attn.k_norm.weight", head }, .{ "self_attn.k_norm_ctx.weight", head } };
            for (norms) |nm| try specs.append(a, .{ .name = try std.fmt.allocPrint(a, "layers.{d}.{s}", .{ l, nm[0] }), .dims = try a.dupe(usize, &.{nm[1]}), .scale = 0.1, .bias = 1 });
            const proj = [_]struct { []const u8, usize, usize }{ .{ "self_attn.q_proj.weight", qw, d.D }, .{ "self_attn.k_proj.weight", kvw, d.D }, .{ "self_attn.v_proj.weight", kvw, d.D }, .{ "self_attn.o_proj.weight", d.D, qw }, .{ "self_attn.k_proj_ctx.weight", kvw, d.D }, .{ "self_attn.v_proj_ctx.weight", kvw, d.D }, .{ "mlp.gate_proj.weight", d.inter, d.D }, .{ "mlp.up_proj.weight", d.inter, d.D }, .{ "mlp.down_proj.weight", d.D, d.inter } };
            for (proj) |p| try specs.append(a, .{ .name = try std.fmt.allocPrint(a, "layers.{d}.{s}", .{ l, p[0] }), .dims = try a.dupe(usize, &.{ p[1], p[2] }), .scale = 0.12, .bias = 0 });
        }
        for (specs.items, 0..) |s, i| {
            var n: usize = 1;
            for (s.dims) |dim| n *= dim;
            const v = try a.alloc(u16, n);
            for (v) |*h| h.* = bf(s.bias + s.scale * (rng.float(f32) * 2 - 1));
            if (i != 0) try header.append(a, ',');
            try header.print(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{s.name});
            for (s.dims, 0..) |dim, j| try header.print(a, "{s}{d}", .{ if (j != 0) "," else "", dim });
            try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ data.items.len, data.items.len + n * 2 });
            try data.appendSlice(a, std.mem.sliceAsBytes(v));
        }
        try header.append(a, '}');
        while (header.items.len % 8 != 0) try header.append(a, ' ');
        var out: std.ArrayList(u8) = .empty;
        try out.resize(a, 8);
        std.mem.writeInt(u64, out.items[0..8], header.items.len, .little);
        try out.appendSlice(a, header.items);
        try out.appendSlice(a, data.items);
        try fx.tmp.dir.writeFile(io, .{ .sub_path = "mtp.safetensors", .data = out.items });
    }
};
