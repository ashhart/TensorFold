//! Inkling 2 Mini's shared DSpark pass: one projection and one batched kernel launch per layer across active streams.
const std = @import("std");
const cuda = @import("cuda");
const dspark = @import("dspark.zig");
const dsp = @import("dspark_ops.zig");
const ops = @import("ops.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const H = dspark.hidden;
const Q = dspark.query;
const KV = dspark.kv;
const I = dspark.inter;
const R = dspark.rank;
const T = dspark.rows;
const max_streams = 16;

pub const Head = struct {
    gpa: std.mem.Allocator,
    w: *const dspark.Weights,
    k: *const dsp.Kernels,
    t: dspark.Target,
    glue: *const ops.Kernels,
    d: *const cuda.Driver,
    cap: usize,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    fused: u64 = 0,
    ctx_kv: u64 = 0,
    resid: u64 = 0,
    x: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    po: u64 = 0,
    pml: u64 = 0,
    att: u64 = 0,
    y: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    context: u64 = 0,
    state: u64 = 0,
    latent: u64 = 0,
    hidden: u64 = 0,
    logits: u64 = 0,
    picks: u64 = 0,
    walk: u64 = 0,
    table: u64 = 0,
    at: u64 = 0,
    part: u64 = 0,

    fn alloc(h: *Head, len: usize) !u64 {
        var b = try DeviceBuffer.alloc(h.d, @max(len, 256));
        errdefer b.free();
        try h.buffers.append(h.gpa, b);
        return b.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const dspark.Weights, k: *const dsp.Kernels, glue: *const ops.Kernels, t: dspark.Target, cap: usize) !Head {
        var h: Head = .{ .gpa = gpa, .w = w, .k = k, .glue = glue, .t = t, .d = d, .cap = cap };
        errdefer h.deinit(gpa);
        const batch = max_streams * T;
        const maxc = cap / dsp.chunk + 2;
        var partial: usize = t.lane.partBytes(max_streams, t.unembed);
        for (w.proj) |p| for ([_]cuda.qmmf.Weight{ p.qkv, p.ctx_kv, p.o, p.gu, p.down }) |weight| {
            partial = @max(partial, t.lane.partBytes(batch, weight));
        };
        h.fused = try h.alloc(4 * dspark.absorb_rows * H * 2);
        h.ctx_kv = try h.alloc(dspark.absorb_rows * 2 * KV * 2);
        h.resid = try h.alloc(batch * H * 2);
        h.x = try h.alloc(batch * H * 2);
        h.qkv = try h.alloc(batch * (Q + 2 * KV) * 2);
        h.q = try h.alloc(batch * Q * 2);
        h.po = try h.alloc(max_streams * maxc * T * 24 * dsp.head_dim * 4);
        h.pml = try h.alloc(max_streams * maxc * T * 24 * 2 * 4);
        h.att = try h.alloc(batch * Q * 2);
        h.y = try h.alloc(batch * H * 4);
        h.gu = try h.alloc(batch * 2 * I * 2);
        h.act = try h.alloc(batch * I * 2);
        h.context = try h.alloc(max_streams * T * 3 * R * 4);
        h.state = try h.alloc(max_streams * R * 2);
        h.latent = try h.alloc(max_streams * R * 2);
        h.hidden = try h.alloc(max_streams * H * 2);
        h.logits = try h.alloc(max_streams * t.w.c.text.unpadded_vocab * 4);
        h.picks = try h.alloc(max_streams * 16);
        h.walk = try h.alloc(max_streams * T * 4);
        h.table = try h.alloc(max_streams * @sizeOf(dsp.Stream));
        h.at = try h.alloc(batch * @sizeOf([2]i32));
        h.part = try h.alloc(partial);
        return h;
    }

    pub fn deinit(h: *Head, gpa: std.mem.Allocator) void {
        for (h.buffers.items) |*b| b.free();
        h.buffers.deinit(gpa);
    }

    fn gemm(h: *const Head, s: cuda.Stream, x: u64, rows: usize, weight: cuda.qmmf.Weight, out: u64, kind: cuda.qmmf.Out) !void {
        try h.t.lane.matmulOut(s, x, weight.k, rows, weight, out, kind, h.part);
    }

    /// Absorb committed rows only; a rejected span never changes the stream's published position.
    pub fn absorb(h: *const Head, s: cuda.Stream, c: *dspark.Context, taps: u64, start: u64, n: usize) !void {
        const span = try c.admit(start, n);
        var at: usize = 0;
        while (at < n) : (at += dspark.absorb_rows) {
            const count: usize = @min(dspark.absorb_rows, n - at);
            const input = taps + at * 4 * H * 2;
            for (0..4) |f| {
                try dsp.mix(h.k, s, input, h.w.alpha + f * 4 * 4, h.w.dflare + f * H * 2, h.fused + f * count * H * 2, count, 4, H, 1e-6);
            }
            try h.d.check(h.d.api.cuMemcpyDtoDAsync_v2(c.newest.ptr, h.fused + (3 * count + count - 1) * H * 2, H * 2, s.handle), "cuMemcpyDtoDAsync");
            for (h.w.proj, h.w.norms, 0..) |p, norms, layer| {
                try h.gemm(s, h.fused + layer * count * H * 2, count, p.ctx_kv, h.ctx_kv, .bf16);
                try dsp.qk(h.k, s, .{
                    .x = h.ctx_kv,
                    .qw = 0,
                    .kw = norms.k_ctx,
                    .q = 0,
                    .kc = c.k[layer].ptr,
                    .vc = c.v[layer].ptr,
                    .ld = 2 * KV,
                    .hq = 0,
                    .hk = 8,
                    .ring = @intCast(c.rings[layer]),
                    .pos0 = @intCast(start + at),
                    .eps = 1e-6,
                    .theta = 1e6,
                }, count);
            }
        }
        c.from, c.end = .{ span.from, span.end };
    }

    /// A batched block pass shares weights; Q/K and attention use #4858's stream-table kernels.
    pub fn draftMany(h: *const Head, s: cuda.Stream, contexts: []const *dspark.Context, pending: []const u32, max_depth: usize, out: []u32) !void {
        const n = contexts.len;
        if (n == 0 or n > max_streams or pending.len != n or out.len != n * T or max_depth == 0 or max_depth > dspark.depth)
            return error.DSparkHoldGeometry;
        const vocab = h.t.w.c.text.unpadded_vocab;
        var table: [max_streams]dsp.Stream = undefined;
        var at: [max_streams * T][2]i32 = undefined;
        var walk: [max_streams * T]u32 = @splat(0);
        for (contexts, pending, 0..) |c, token, stream| {
            if (c.end <= c.from or c.end + T > h.cap or token >= h.t.w.c.text.vocab) return error.DSparkHoldGeometry;
            walk[stream * T] = token;
            const p: i32 = @intCast(c.end);
            table[stream] = .{ .frnn = c.newest.ptr, .p = p };
            for (c.rings, 0..) |ring, layer| {
                table[stream].kc[layer] = c.k[layer].ptr;
                table[stream].vc[layer] = c.v[layer].ptr;
                const window: u64 = if (layer == 1) ring else 1024;
                table[stream].lo[layer] = @intCast(@max(c.from, c.end + T -| window));
            }
            for (0..T) |row| at[stream * T + row] = .{ @intCast(stream), p + @as(i32, @intCast(row)) };
        }
        try cuda.copy_source.enqueue(h.d, h.table, std.mem.sliceAsBytes(table[0..n]), s);
        try cuda.copy_source.enqueue(h.d, h.at, std.mem.sliceAsBytes(at[0 .. n * T]), s);
        try cuda.copy_source.enqueue(h.d, h.walk, std.mem.sliceAsBytes(walk[0 .. n * T]), s);
        // Embed the first token exactly as the target does; masks are the remaining uncommitted rows.
        for (0..n) |stream| {
            try ops.embed(h.glue, s, h.walk + stream * T * 4, 1, h.t.w.embed, h.t.w.embed_norm, h.resid + stream * T * H * 2, H, h.t.w.c.text.eps);
            for (1..T) |row| try h.d.check(h.d.api.cuMemcpyDtoDAsync_v2(h.resid + (stream * T + row) * H * 2, h.w.mask, H * 2, s.handle), "cuMemcpyDtoDAsync");
        }
        try dsp.addNorm(h.k, s, h.resid, 0, 0, n * T * H, h.w.norms[0].input, h.x, n * T, H, 1e-6);
        const tab: dsp.Table = .{ .dev = h.table, .host = table[0..n] };
        const row_ids: dsp.Rows = .{ .dev = h.at, .host = at[0 .. n * T] };
        for (h.w.proj, h.w.norms, 0..) |p, norms, layer| {
            try h.gemm(s, h.x, n * T, p.qkv, h.qkv, .bf16);
            try dsp.qkMany(h.k, s, .{
                .x = h.qkv,
                .qw = norms.q,
                .kw = norms.k,
                .q = h.q,
                .kc = 0,
                .vc = 0,
                .ld = Q + 2 * KV,
                .hq = 24,
                .hk = 8,
                .ring = @intCast(contexts[0].rings[layer]),
                .pos0 = 0,
                .eps = 1e-6,
                .theta = 1e6,
            }, tab, row_ids, layer);
            var maxc: usize = 1;
            for (table[0..n]) |item| maxc = @max(maxc, dsp.chunks(@intCast(item.lo[layer]), @intCast(item.p + T - 1)));
            try dsp.attendMany(h.k, s, .{
                .q = h.q,
                .kc = 0,
                .vc = 0,
                .po = h.po,
                .pml = h.pml,
                .T = T,
                .hq = 24,
                .hk = 8,
                .ring = @intCast(contexts[0].rings[layer]),
                .lo = 0,
                .hi = 0,
                .c0 = 0,
                .scale = 1.0 / @sqrt(@as(f32, dsp.head_dim)),
            }, tab, layer, maxc, n * (h.cap / dsp.chunk + 2), h.att);
            try h.gemm(s, h.att, n * T, p.o, h.y, .f32);
            try dsp.addNorm(h.k, s, h.resid, h.y, 1, n * T * H, norms.post, h.x, n * T, H, 1e-6);
            try h.gemm(s, h.x, n * T, p.gu, h.gu, .bf16);
            try dsp.silu(h.k, s, h.gu, h.act, n * T, I);
            try h.gemm(s, h.act, n * T, p.down, h.y, .f32);
            try dsp.addNorm(h.k, s, h.resid, h.y, 1, n * T * H, if (layer < 2) h.w.norms[layer + 1].input else h.w.norm, h.x, n * T, H, 1e-6);
        }
        try dsp.ctxMany(h.k, s, h.x + H * 2, T * H, h.table, h.w.rnn, h.w.rnn_b, h.context, n, T - 1, H, R);
        try h.d.check(h.d.api.cuMemsetD8Async(h.state, 0, n * R * 2, s.handle), "cuMemsetD8Async");
        for (0..max_depth) |step| {
            try dsp.rnnMany(h.k, s, h.context + step * 3 * R * 4, (T - 1) * 3 * R, h.w.rnn, h.w.markov, h.walk, T, step, h.state, h.latent, n, R, H);
            try dsp.miniHidden(h.k, s, h.x + (step + 1) * H * 2, T * H, h.latent, h.w.hidden_w, h.w.hidden_b, h.hidden, n, H, R);
            try h.gemm(s, h.hidden, n, h.t.unembed, h.logits, .f32);
            try ops.softcap(h.glue, s, h.logits, n * vocab, h.t.w.c.text.logit_cap);
            try dsp.pickMany(h.k, s, h.logits, vocab, vocab, 0, h.table, step, h.picks, n);
            try dsp.pickRows(h.k, s, h.picks, 1, n, h.walk, T, step);
        }
        try s.synchronize();
        try (DeviceBuffer{ .d = h.d, .ptr = h.walk, .len = n * T * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(out));
    }
};
