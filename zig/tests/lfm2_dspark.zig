//! LFM2's DSpark drafter on the GPU against an fp32 host reference of the released model, on the real weights.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const c = q.config;
const st = q.state;
const ck = tf.checkpoint;
const quant = q.dspark_quant;

const hd = c.head_dim;
const kw = c.kv_heads * hd;
const qw = c.query_heads * hd;

/// The drafter's tensors as fp32 on the host, in the checkpoint's own layout (interleaved RoPE pairs).
const Ref = struct {
    gpa: std.mem.Allocator,
    check: ck.Checkpoint,
    cfg: q.dspark_config.Config,

    fn get(r: *const Ref, name: []const u8) ![]f32 {
        const t = try r.check.get(name);
        const src = t.host(u16);
        const out = try r.gpa.alloc(f32, src.len);
        for (out, src) |*o, v| o.* = quant.toF32(v);
        return out;
    }

    fn layer(r: *const Ref, buf: []u8, i: usize, part: []const u8) ![]f32 {
        return r.get(try std.fmt.bufPrint(buf, "layers.{d}.{s}.weight", .{ i, part }));
    }
};

/// y[n] = W[n, k] x[k] for each of `rows` rows, on every core.
fn matmul(gpa: std.mem.Allocator, w: []const f32, x: []const f32, y: []f32, n: usize, k: usize, rows: usize) !void {
    const Job = struct {
        fn run(w_: []const f32, x_: []const f32, y_: []f32, n_: usize, k_: usize, rows_: usize, from: usize, to: usize) void {
            for (0..rows_) |r| for (from..to) |col| {
                var s: f32 = 0;
                const wr = w_[col * k_ ..][0..k_];
                const xr = x_[r * k_ ..][0..k_];
                for (wr, xr) |a, b| s += a * b;
                y_[r * n_ + col] = s;
            };
        }
    };
    const threads = @min(std.Thread.getCpuCount() catch 1, 16);
    const per = (n + threads - 1) / threads;
    const hs = try gpa.alloc(std.Thread, threads);
    defer gpa.free(hs);
    var made: usize = 0;
    defer for (hs[0..made]) |h| h.join(); // a failed spawn still joins the workers already running
    var at: usize = 0;
    while (at < n) : (at += per) {
        hs[made] = try std.Thread.spawn(.{}, Job.run, .{ w, x, y, n, k, rows, at, @min(n, at + per) });
        made += 1;
    }
}

fn rms(x: []f32, gain: []const f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + c.eps);
    for (x, gain) |*v, g| v.* = v.* * inv * g;
}

/// Interleaved RoPE at `pos` over each head of `x` (pairs 2i, 2i + 1).
fn rope(x: []f32, pos: u64, theta: f32) void {
    var h: usize = 0;
    while (h < x.len) : (h += hd) for (0..hd / 2) |i| {
        const angle = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, theta, -@as(f32, @floatFromInt(i)) / (hd / 2));
        const a = x[h + 2 * i];
        const b = x[h + 2 * i + 1];
        x[h + 2 * i] = a * @cos(angle) - b * @sin(angle);
        x[h + 2 * i + 1] = a * @sin(angle) + b * @cos(angle);
    };
}

/// The target's embedding row `id`, dequantized from its codes at the checkpoint's own width (scale * q + bias).
fn embedRow(m: *q.Model, id: u32, out: []f32) void {
    const e = m.weights.embedding;
    const bits: usize = e.bits;
    const row_bytes = c.hidden * bits / 8;
    const codes = (e.w.contents() + e.w_off)[@as(usize, id) * row_bytes ..][0..row_bytes];
    const sc = @as([*]const u16, @ptrCast(@alignCast(e.scales.contents() + e.s_off)))[@as(usize, id) * c.hidden / 64 ..][0 .. c.hidden / 64];
    const bi = @as([*]const u16, @ptrCast(@alignCast(e.biases.contents() + e.b_off)))[@as(usize, id) * c.hidden / 64 ..][0 .. c.hidden / 64];
    for (0..c.hidden) |j| {
        // codes are a little-endian bit stream: code j takes bits [j * bits, (j + 1) * bits)
        const at = j * bits;
        var word: u32 = codes[at / 8];
        if (at / 8 + 1 < row_bytes) word |= @as(u32, codes[at / 8 + 1]) << 8;
        const qv = (word >> @intCast(at % 8)) & ((@as(u32, 1) << @intCast(bits)) - 1);
        out[j] = quant.toF32(sc[j / 64]) * @as(f32, @floatFromInt(qv)) + quant.toF32(bi[j / 64]);
    }
}

/// GPU bf16 row against the reference's row + bias: cosine and worst difference.
fn compare(label: []const u8, gpu: []const u16, row: []const f32, bias: []const f32) void {
    var dot: f64 = 0;
    var gg: f64 = 0;
    var rr: f64 = 0;
    var worst: f32 = 0;
    for (gpu, row, bias) |g, b, m2| {
        const x = quant.toF32(g);
        const y = b + m2;
        dot += x * y;
        gg += x * x;
        rr += y * y;
        worst = @max(worst, @abs(x - y));
    }
    std.debug.print("{s}: cosine {d:.5}, worst difference {d:.3}\n", .{ label, dot / @sqrt(gg * rr), worst });
}

fn absorb(d: *q.dspark.Drafter, piece: q.dspark_block.Piece) !void {
    const run = q.dspark_block.begin(d);
    try q.dspark_block.absorbPass(d, run.e, &.{piece});
    try q.dspark_block.end(run.cb, run.e);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4 or args.len > 6) return error.ExpectedModelDrafterPromptLengthRowsBits;
    const bits: u8 = if (args.len == 6) try std.fmt.parseInt(u8, args[5], 10) else 0;
    const length = try std.fmt.parseInt(usize, args[3], 10);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    const d = try q.dspark.Drafter.init(gpa, init.io, m, args[2], .{ .capacity = length + 64, .bits = bits });
    defer d.deinit();
    const cfg = d.config;
    // the compared block's rows (a hold of depth d runs a d-row block)
    const block: u32 = if (args.len >= 5) try std.fmt.parseInt(u32, args[4], 10) else cfg.block;
    if (block == 0 or block > cfg.block) return error.BadBlockRows;
    var scratch = try st.Scratch.init(gpa, m.device, &m.config.attention, length);
    defer scratch.deinit();
    const width = cfg.tap_count * c.hidden;
    var taps: st.Taps = .{ .count = cfg.tap_count, .buf = try scratch.memory.alloc(length * width * 2) };
    @memcpy(taps.layers[0..cfg.tap_count], cfg.taps[0..cfg.tap_count]);
    scratch.taps = taps;
    var cache = try st.Cache.init(gpa, m.device, &m.config.attention, length + 16);
    defer cache.deinit();
    const ids = try gpa.alloc(u32, length);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % 120000);
    try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = length }}, ids, false, .last);
    // the context: rows [0, length - 1); the pending token is the last id, at position length - 1
    const ctx = length - 1;
    const lane = try d.newLane(length + 32);
    defer d.dropLane(lane);
    var at: usize = 0;
    while (at < ctx) : (at += q.dspark.absorb_rows) {
        const rows = @min(q.dspark.absorb_rows, ctx - at);
        try absorb(d, .{ .lane = lane, .src = .{ .buf = taps.buf, .off = at * width * 2 }, .rows = rows, .p0 = at });
    }
    lane.end = ctx;
    _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = block }}, null);
    var held: [16]u32 = undefined;
    @memcpy(held[0..block], lane.held[0..block]);
    const gpu = held[0..block];
    for ([_]usize{ 1, 4, 16 }) |rows| {
        var best: f64 = std.math.floatMax(f64);
        for (0..5) |_| {
            const t0 = std.Io.Clock.awake.now(init.io);
            try absorb(d, .{ .lane = lane, .src = .{ .buf = taps.buf, .off = (ctx - rows) * width * 2 }, .rows = rows, .p0 = ctx - rows });
            best = @min(best, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e6);
        }
        std.debug.print("absorb of {d} rows: {d:.2} ms (best of 5)\n", .{ rows, best });
    }
    const depths = [_]u32{ 1, 3, cfg.block };
    for (depths) |depth| {
        var best: f64 = std.math.floatMax(f64);
        for (0..5) |_| {
            const t0 = std.Io.Clock.awake.now(init.io);
            _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = depth }}, null);
            best = @min(best, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds())) / 1e6);
        }
        std.debug.print("hold at depth {d}: {d:.2} ms (best of 5)\n", .{ depth, best });
    }
    _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = ids[ctx], .anchor = ctx, .depth = block }}, null);
    // the host reference, fp32 throughout
    var r: Ref = .{ .gpa = gpa, .check = ck.Checkpoint.init(gpa), .cfg = cfg };
    defer r.check.deinit();
    const path = try std.fs.path.joinZ(gpa, &.{ args[2], "model.safetensors" });
    defer gpa.free(path);
    try r.check.addFile(m.device, path, "");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    r.gpa = arena.allocator();
    const a = r.gpa;
    const n = block;
    const tap_rows = taps.buf.slice(u16, length * width);
    const feats = try a.alloc(f32, ctx * width);
    for (feats, tap_rows[0 .. ctx * width]) |*f, v| f.* = quant.toF32(v);
    const hctx = try a.alloc(f32, ctx * c.hidden);
    try matmul(gpa, try r.get("fc.weight"), feats, hctx, c.hidden, width, ctx);
    const hn = try r.get("hidden_norm.weight");
    for (0..ctx) |j| rms(hctx[j * c.hidden ..][0..c.hidden], hn);
    const h = try a.alloc(f32, n * c.hidden);
    for (0..n) |i| embedRow(m, if (i == 0) ids[ctx] else cfg.mask, h[i * c.hidden ..][0..c.hidden]);
    const keys = ctx + n;
    var name: [96]u8 = undefined;
    const x = try a.alloc(f32, n * c.hidden);
    for (0..cfg.layers) |li| {
        const kmat = try r.layer(&name, li, "self_attn.k_proj");
        const vmat = try r.layer(&name, li, "self_attn.v_proj");
        const qn = try r.layer(&name, li, "self_attn.q_norm");
        const kn = try r.layer(&name, li, "self_attn.k_norm");
        const kk = try a.alloc(f32, keys * kw);
        const vv = try a.alloc(f32, keys * kw);
        try matmul(gpa, kmat, hctx, kk, kw, c.hidden, ctx);
        try matmul(gpa, vmat, hctx, vv, kw, c.hidden, ctx);
        @memcpy(x, h);
        for (0..n) |i| rms(x[i * c.hidden ..][0..c.hidden], try r.layer(&name, li, "input_layernorm"));
        const qq = try a.alloc(f32, n * qw);
        try matmul(gpa, try r.layer(&name, li, "self_attn.q_proj"), x, qq, qw, c.hidden, n);
        try matmul(gpa, kmat, x, kk[ctx * kw ..], kw, c.hidden, n);
        try matmul(gpa, vmat, x, vv[ctx * kw ..], kw, c.hidden, n);
        for (0..keys) |j| for (0..c.kv_heads) |kh| {
            const row = kk[j * kw + kh * hd ..][0..hd];
            rms(row, kn);
            rope(row, j, cfg.theta);
        };
        for (0..n) |i| for (0..c.query_heads) |qh| {
            const row = qq[i * qw + qh * hd ..][0..hd];
            rms(row, qn);
            rope(row, ctx + i, cfg.theta);
        };
        const o = try a.alloc(f32, n * qw);
        const sc = try a.alloc(f32, keys);
        for (0..n) |i| for (0..c.query_heads) |qh| {
            const kh = qh / (c.query_heads / c.kv_heads);
            var top: f32 = -std.math.inf(f32);
            for (0..keys) |j| {
                var s: f32 = 0;
                for (0..hd) |t| s += qq[i * qw + qh * hd + t] * kk[j * kw + kh * hd + t];
                sc[j] = s / @sqrt(@as(f32, hd));
                top = @max(top, sc[j]);
            }
            var sum: f32 = 0;
            for (sc) |*s| {
                s.* = @exp(s.* - top);
                sum += s.*;
            }
            for (0..hd) |t| {
                var acc: f32 = 0;
                for (0..keys) |j| acc += sc[j] * vv[j * kw + kh * hd + t];
                o[i * qw + qh * hd + t] = acc / sum;
            }
        };
        const y = try a.alloc(f32, n * c.hidden);
        try matmul(gpa, try r.layer(&name, li, "self_attn.o_proj"), o, y, c.hidden, qw, n);
        for (h, y) |*v, w| v.* += w;
        @memcpy(x, h);
        for (0..n) |i| rms(x[i * c.hidden ..][0..c.hidden], try r.layer(&name, li, "post_attention_layernorm"));
        const gate = try a.alloc(f32, n * cfg.intermediate);
        const up = try a.alloc(f32, n * cfg.intermediate);
        try matmul(gpa, try r.layer(&name, li, "mlp.gate_proj"), x, gate, cfg.intermediate, c.hidden, n);
        try matmul(gpa, try r.layer(&name, li, "mlp.up_proj"), x, up, cfg.intermediate, c.hidden, n);
        for (gate, up) |*g, u| g.* = g.* / (1 + @exp(-g.*)) * u;
        try matmul(gpa, try r.layer(&name, li, "mlp.down_proj"), gate, y, c.hidden, cfg.intermediate, n);
        for (h, y) |*v, w| v.* += w;
    }
    const fin = try r.get("norm.weight");
    for (0..n) |i| rms(h[i * c.hidden ..][0..c.hidden], fin);
    const head = try a.alloc(f32, c.vocab * c.hidden);
    for (0..c.vocab) |v| embedRow(m, @intCast(v), head[v * c.hidden ..][0..c.hidden]);
    const base = try a.alloc(f32, n * c.vocab);
    try matmul(gpa, head, h, base, c.vocab, c.hidden, n);
    const w1 = try r.get("markov_head.markov_w1.weight");
    const w2 = try r.get("markov_head.markov_w2.weight");
    const bias = try a.alloc(f32, c.vocab);
    var prev = ids[ctx];
    var same: usize = 0;
    var worst: f32 = 0;
    for (0..n) |i| {
        try matmul(gpa, w2, w1[@as(usize, prev) * cfg.rank ..][0..cfg.rank], bias, c.vocab, cfg.rank, 1);
        const row = base[i * c.vocab ..][0..c.vocab];
        var best: usize = 0;
        var second: usize = 1;
        for (row, bias, 0..) |b, m2, v| {
            const s = b + m2;
            if (s > row[best] + bias[best]) {
                second = best;
                best = v;
            } else if (v != best and s > row[second] + bias[second]) second = v;
        }
        const margin = (row[best] + bias[best]) - (row[second] + bias[second]);
        if (i == 0) { // the GPU keeps the base logits (the Markov step adds its bias on the way to the pick)
            const zero = try a.alloc(f32, c.vocab);
            @memset(zero, 0);
            compare("row 0 base logits", d.s.logits.slice(u16, c.vocab), row, zero);
        }
        const ok = gpu[i] == best;
        same += @intFromBool(ok);
        std.debug.print("step {d}: reference {d} (margin {d:.3}, second {d}), GPU {d} {s}\n", .{ i, best, margin, second, gpu[i], if (ok) "equal" else "DIFFERENT" });
        if (!ok) worst = @max(worst, margin);
        prev = gpu[i]; // teacher-forced: step i + 1 follows the GPU's pick, so each step is compared alone
    }
    std.debug.print("the reference's largest margin over a different GPU pick: {d:.3}\n", .{worst});
    std.debug.print("{d} of {d} drafts equal the fp32 reference's picks on the same chain ({d}-token context)\n", .{ same, n, ctx });
}
