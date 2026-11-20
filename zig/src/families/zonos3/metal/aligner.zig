//! The release's forced aligner (FuseAlign behind a Dia text encoder) on Metal: 30 s and 1022 bytes a forward.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;
const lm = @import("kernels.zig");
const codec = @import("codec_kernels.zig");
const ek = @import("encoder_kernels.zig");
const text = @import("../align_words.zig");
const store = @import("aligner_weights.zig");
const dim = store.dim;
const ffn = store.ffn;
const feat = store.feat;
const hop = store.hop;
const heads = 16;
const hd = dim / heads;
const dia_heads = 16;
const dia_hd = 128;
const dia_width = dia_heads * dia_hd;
const eps: f32 = 1e-6;
const dia_eps: f32 = 1e-5;
const theta: f32 = 10000.0;

/// A stage of a forward for a check: fp32 rows [rows][width] on the host.
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, values: []const f32, rows: usize, width: usize) anyerror!void,
};

/// What the model says of each word: its start and end on the 1 ms grid, and the presence logit.
pub const Found = struct { start_ms: []u32, end_ms: []u32, presence: []f32 };

pub const Aligner = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
    w: store.Weights,
    scratch: u8 = 0, // the owner of a forward's buffers, released when it returns
    no_distance: u64, // fp32 [73][64] of zeros: the teacher's attention without its relative term

    /// On `g`; `dir` is the release's web/aligner folder. Every linear is the codec's kernel, in a fixed order.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, dir: []const u8) !*Aligner {
        const a = try gpa.create(Aligner);
        errdefer gpa.destroy(a);
        a.* = undefined;
        a.gpa = gpa;
        a.g = g;
        a.scratch = 0;
        const owner = g.own(a);
        defer _ = g.own(owner);
        errdefer g.release(a);
        a.w = try store.load(gpa, io, g, dir);
        errdefer a.w.deinit();
        a.no_distance = try g.alloc(73 * 64 * 4);
        return a;
    }

    pub fn destroy(a: *Aligner) void {
        a.g.release(a);
        a.w.deinit();
        a.gpa.destroy(a);
    }

    /// Device bytes of the weights held.
    pub fn bytes(a: *const Aligner) usize {
        return a.w.bytes;
    }

    fn alloc(a: *Aligner, n: usize) !u64 {
        return a.g.alloc(n);
    }

    fn up(a: *Aligner, values: []const f32) !u64 {
        return a.g.upload(std.mem.sliceAsBytes(values));
    }

    fn down(a: *Aligner, ptr: u64, out: []f32) !void {
        try a.g.get(ptr, std.mem.sliceAsBytes(out));
    }

    /// y [rows][n] = x [rows][k] W^T (+ b); row r reads x[r * xs ..][0..k], so rows may overlap.
    fn lin(a: *Aligner, l: store.Linear, x: u64, xs: usize, y: u64, ys: usize, rows: usize) !void {
        codec.linear(a.g, l.w, l.b, x, xs, l.k, l.n, y, ys, rows);
    }

    fn window(a: *Aligner, l: store.Linear, x: u64, xs: usize, y: u64, ys: usize, rows: usize) !void {
        codec.linear(a.g, l.w, l.b, x, xs, l.k, l.n, y, ys, rows);
    }

    fn sum(a: *Aligner, x: u64, y: u64, n: usize) !void {
        ek.add(a.g, x, y, n);
    }

    fn norm(a: *Aligner, x: u64, y: u64, rows: usize, e: f32) !void {
        lm.rmsnorm(a.g, x, dim, 0, y, dim, dim, e, rows);
    }

    /// Dia's encoder over the ids: rows [ids][1024] in bf16 values; linears on the device, RoPE and attention on host.
    fn encode(a: *Aligner, s: std.mem.Allocator, ids: []const u32) ![]f32 {
        const n = ids.len;
        const first = try s.alloc(f32, n * dim);
        for (ids, 0..) |id, i| @memcpy(first[i * dim ..][0..dim], a.w.embedding[id * dim ..][0..dim]);
        const x = try a.up(first);
        const h = try a.alloc(n * dim * 4);
        const delta = try a.alloc(n * dim * 4);
        const qkv = try a.alloc(n * 3 * dia_width * 4);
        const att = try a.alloc(n * dia_width * 4);
        const gate = try a.alloc(n * 2 * ffn * 4);
        const act = try a.alloc(n * ffn * 4);
        const positions = try s.alloc(f32, n);
        for (positions, 0..) |*p, i| p.* = @floatFromInt(i);
        const tables = try text.ropeTables(s, positions, dia_hd, theta, true);
        const host = try s.alloc(f32, n * 3 * dia_width);
        const mixed = try s.alloc(f32, n * dia_width);
        for (a.w.dia, 0..) |L, i| {
            if (i != 0) try a.sum(x, delta, n * dim);
            try a.norm(x, h, n, dia_eps);
            try a.lin(L.qkv, h, dim, qkv, 3 * dia_width, n);
            try a.down(qkv, host);
            text.rope(host, 3 * dia_width, 0, dia_heads, dia_hd, tables[0], tables[1]);
            text.rope(host, 3 * dia_width, dia_width, dia_heads, dia_hd, tables[0], tables[1]);
            attend(host, mixed, n);
            try a.g.put(att, std.mem.sliceAsBytes(mixed));
            try a.lin(L.o, att, dia_width, delta, dim, n);
            try a.sum(x, delta, n * dim);
            try a.norm(x, h, n, dia_eps);
            try a.lin(L.gate_up, h, dim, gate, 2 * ffn, n);
            lm.mulSilu(a.g, gate, 2 * ffn, ffn, 0, ffn, act, n);
            try a.lin(L.down, act, ffn, delta, dim, n);
        }
        try a.sum(x, delta, n * dim);
        try a.norm(x, h, n, dia_eps);
        const out = try s.alloc(f32, n * dim);
        try a.down(h, out);
        for (0..n) |r| for (out[r * dim ..][0..dim], a.w.dia_norm) |*v, g| {
            v.* = text.bf16(v.* * g);
        };
        return out;
    }

    /// One forward: `pcm` is 16 kHz mono of at least a patch, `words` the transcript's words with a letter or digit.
    pub fn forward(a: *Aligner, gpa: std.mem.Allocator, pcm: []const f32, words: []const []const u8, probe: ?Probe) !Found {
        if (pcm.len < 2 * hop or pcm.len > text.max_samples or words.len == 0) return error.BadAlignerInput;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const s = arena.allocator();
        const owner = a.g.own(&a.scratch);
        defer _ = a.g.own(owner);
        defer a.g.release(&a.scratch);
        const script = try text.transcript(s, words);
        const feats = try a.encode(s, script.ids);
        const t = script.ids.len - 2;
        if (probe) |p| try p.f(p.ctx, "text_feats", feats[2 * dim ..], t, dim);
        const patches = (pcm.len - 2 * hop) / hop + 1;
        const n = patches + t;

        // the joint sequence: a patch a row, then a transcript byte a row
        const x = try a.alloc(n * dim * 4);
        const standard = try text.standardize(s, pcm);
        const frames = try a.up(standard[0 .. (patches + 1) * hop]);
        try a.window(a.w.patch, frames, hop, x, dim, patches);
        const text_rows = try a.up(feats[2 * dim ..]);
        try a.lin(a.w.text_proj, text_rows, dim, x + patches * dim * 4, dim, t);
        const tables = try text.ropeTables(s, try text.jointPositions(s, patches, t), hd, theta, false);
        const h = try a.alloc(n * dim * 4);
        const delta = try a.alloc(n * dim * 4);
        const qkv = try a.alloc(n * 3 * dim * 4);
        const att = try a.alloc(n * dim * 4);
        const wide = try a.alloc(n * 2 * ffn * 4);
        const act = try a.alloc(n * ffn * 4);
        const host = try s.alloc(f32, n * 3 * dim);
        for (a.w.fa, 0..) |L, i| {
            if (i != 0) try a.sum(x, delta, n * dim);
            try a.norm(x, h, n, eps);
            try a.lin(L.qkv, h, dim, qkv, 3 * dim, n);
            // rotate_half on the queries and keys, by each row's own position
            try a.down(qkv, host);
            text.rope(host, 3 * dim, 0, heads, hd, tables[0], tables[1]);
            text.rope(host, 3 * dim, dim, heads, hd, tables[0], tables[1]);
            try a.g.put(qkv, std.mem.sliceAsBytes(host));
            ek.attention(a.g, qkv, dim, n, a.no_distance, att);
            try a.lin(L.out, att, dim, delta, dim, n);
            try a.sum(x, delta, n * dim);
            try a.norm(x, h, n, eps);
            try a.lin(L.w13, h, dim, wide, 2 * ffn, n);
            lm.mulSilu(a.g, wide, 2 * ffn, ffn, 0, ffn, act, n);
            try a.lin(L.w2, act, ffn, delta, dim, n);
        }
        try a.sum(x, delta, n * dim);
        try a.norm(x, h, n, eps); // the final norm's weight is in the layers that read h
        if (probe) |p| {
            const joint = try s.alloc(f32, n * dim);
            try a.down(h, joint);
            for (0..n) |r| for (joint[r * dim ..][0..dim], a.w.final_norm) |*v, g| {
                v.* *= g;
            };
            try p.f(p.ctx, "joint", joint, n, dim);
        }

        // each word: the mean of its bytes' rows, then the three small networks
        const rows = try s.alloc(f32, t * dim);
        try a.down(h + patches * dim * 4, rows);
        const pooled = try s.alloc(f32, words.len * dim);
        @memset(pooled, 0);
        for (script.spans, 0..) |span, w| {
            const inv = 1.0 / @as(f32, @floatFromInt(@max(span[1] - span[0], 1)));
            for (span[0]..span[1]) |j| for (pooled[w * dim ..][0..dim], rows[j * dim ..][0..dim]) |*o, v| {
                o.* += v * inv;
            };
        }
        const wv = try a.up(pooled);
        const hm = try a.alloc(words.len * 3 * dim * 4);
        try a.lin(a.w.mlp0, wv, dim, hm, 3 * dim, words.len);
        codec.gelu(a.g, hm, words.len * 3 * dim);
        const vec = try a.alloc(2 * words.len * feat * 4);
        const presence_at = try a.alloc(words.len * 4);
        try a.lin(a.w.start2, hm, 3 * dim, vec, feat, words.len);
        try a.lin(a.w.end2, hm + dim * 4, 3 * dim, vec + words.len * feat * 4, feat, words.len);
        try a.lin(a.w.presence2, hm + 2 * dim * 4, 3 * dim, presence_at, 1, words.len);
        const found: Found = .{ .start_ms = try gpa.alloc(u32, words.len), .end_ms = try gpa.alloc(u32, words.len), .presence = try gpa.alloc(f32, words.len) };
        errdefer {
            gpa.free(found.start_ms);
            gpa.free(found.end_ms);
            gpa.free(found.presence);
        }
        try a.down(presence_at, found.presence);

        // the refinement heads: the patches' rows up to the 1 ms grid, then every word against every frame
        const grid = patches << store.ups;
        const ping = try a.alloc((grid + 4) * feat * 4);
        const pong = try a.alloc((grid + 4) * feat * 4);
        const smooth = try a.alloc(grid * feat * 4);
        const last = try a.alloc(grid * feat * 4);
        const logits = try a.alloc(words.len * grid * 4);
        const scores = try s.alloc(f32, words.len * grid);
        for ([_]*const store.Head{ &a.w.start, &a.w.end }, [_][]u32{ found.start_ms, found.end_ms }, 0..) |head, ms, which| {
            var src = ping;
            var dst = pong;
            var m = patches;
            // a zero row before and after the rows: the transposed convolutions read one past each end
            try a.zero(src, feat);
            try a.zero(src + (m + 1) * feat * 4, feat);
            try a.lin(head.in, h, dim, src + feat * 4, feat, patches);
            for (head.up, 0..) |u, k| {
                // ConvTranspose1d(4, stride 2, padding 1): row 2m from rows m - 1 and m, row 2m + 1 from m and m + 1
                const pad: usize = if (k + 1 == store.ups) 2 else 1;
                try a.zero(dst, pad * feat);
                try a.zero(dst + (2 * m + pad) * feat * 4, pad * feat);
                try a.window(u.even, src, feat, dst + pad * feat * 4, 2 * feat, m);
                try a.window(u.odd, src + feat * 4, feat, dst + (pad + 1) * feat * 4, 2 * feat, m);
                codec.gelu(a.g, dst + pad * feat * 4, 2 * m * feat);
                m *= 2;
                std.mem.swap(u64, &src, &dst);
            }
            try a.window(head.smooth, src, feat, smooth, feat, grid);
            codec.gelu(a.g, smooth, grid * feat);
            try a.lin(head.out, smooth, feat, last, feat, grid);
            // logits [words][grid]: the grid's rows are the weights of a linear over the words' vectors
            try a.lin(.{ .w = last, .b = 0, .n = grid, .k = feat }, vec + which * words.len * feat * 4, feat, logits, grid, words.len);
            try a.down(logits, scores);
            for (ms, 0..) |*at, w| {
                var best: usize = 0;
                for (scores[w * grid ..][0..grid], 0..) |v, f| {
                    if (v > scores[w * grid + best]) best = f;
                }
                at.* = @intCast(best);
            }
        }
        for (found.end_ms, found.start_ms) |*e, b| e.* = @max(e.*, b);
        return found;
    }

    fn zero(a: *Aligner, ptr: u64, values: usize) !void {
        const none: [feat]f32 = @splat(0);
        var at: usize = 0;
        while (at < values) : (at += feat) {
            const n = @min(feat, values - at);
            try a.g.put(ptr + at * 4, std.mem.sliceAsBytes(none[0..n]));
        }
    }
};

/// Dia's attention on the host: every row's heads over all rows, the scores unscaled (DiaSelfAttention's scaling 1).
fn attend(qkv: []const f32, out: []f32, n: usize) void {
    const cores = @min(16, std.Thread.getCpuCount() catch 1, @max(1, n / 16));
    var threads: [16]?std.Thread = @splat(null);
    const share = (n + cores - 1) / cores;
    for (0..cores) |i| {
        const lo = @min(n, i * share);
        const hi = @min(n, lo + share);
        if (i + 1 < cores) threads[i] = std.Thread.spawn(.{}, attendRows, .{ qkv, out, n, lo, hi }) catch null;
        if (threads[i] == null) attendRows(qkv, out, n, lo, hi);
    }
    for (threads) |th| if (th) |t| t.join();
}

fn attendRows(qkv: []const f32, out: []f32, n: usize, lo: usize, hi: usize) void {
    const stride = 3 * dia_width;
    var scores: [text.max_bytes + 2]f32 = undefined;
    for (lo..hi) |r| for (0..dia_heads) |head| {
        const q = qkv[r * stride + head * dia_hd ..][0..dia_hd];
        var top: f32 = -std.math.inf(f32);
        for (scores[0..n], 0..) |*sc, j| {
            const k = qkv[j * stride + dia_width + head * dia_hd ..][0..dia_hd];
            var acc: f32 = 0;
            for (q, k) |x, y| acc += x * y;
            sc.* = acc;
            top = @max(top, acc);
        }
        var total: f32 = 0;
        for (scores[0..n]) |*sc| {
            sc.* = @exp(sc.* - top);
            total += sc.*;
        }
        const o = out[r * dia_width + head * dia_hd ..][0..dia_hd];
        @memset(o, 0);
        for (scores[0..n], 0..) |sc, j| {
            const v = qkv[j * stride + 2 * dia_width + head * dia_hd ..][0..dia_hd];
            const weight = sc / total;
            for (o, v) |*x, y| x.* += weight * y;
        }
    };
}
