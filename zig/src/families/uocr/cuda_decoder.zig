//! Unlimited-OCR-2's decoder on CUDA (decoder.zig's steps): 12 layers of page2d-rotary MHA under DASH's block-sparse
//! reads (a 128-key local window plus 12 routed 32-key blocks) and a 64-expert MoE, over bf16 caches. A prompt's rows
//! take cuBLASLt's GEMMs; one token's take the matvecs, its experts picked on the GPU.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");
const prompt_mod = @import("prompt.zig");
const T = wts.T;

const H = cfg.hidden;
const D = cfg.router_dim;
const B = cfg.block;
const slots = cfg.top_k;

/// An expert's three projections as device addresses (uocr.cu's UdExpert).
const UdExpert = extern struct { gate: u64, up: u64, down: u64 };

/// A decode step's per-token words, uploaded in one copy: tok, page, qpos, qpage.
const StepWords = extern struct { tok: u32, page: u32, qpos: u32, qpage: i32 };

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const ks.Kernels,
    gemms: *cuda.vit.Gemms,
    w: *const wts.Weights,
    /// Tokens the caches hold, and tokens in them now.
    cap: usize,
    len: usize = 0,
    rows: usize,
    caches: cuda.DeviceBuffer,
    blocks: cuda.DeviceBuffer,
    consts: cuda.DeviceBuffer,
    experts: cuda.DeviceBuffer,
    scratch: Scratch,
    /// Host copies of the block page ranges (uploaded at prefill), and the MoE's prompt row lists.
    blk: []i32,
    ids: []u32,
    erows: []u32,
    edest: []u32,
    last_rows: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const ks.Kernels, gemms: *cuda.vit.Gemms, w: *const wts.Weights, cap: usize, rows: usize) !Decoder {
        if (cap == 0 or cap > cfg.max_context or rows == 0 or rows > cap) return error.DecoderSize;
        const cap_b = std.mem.alignForward(usize, cap, B);
        var dec: Decoder = .{ .gpa = gpa, .d = d, .s = s, .k = k, .gemms = gemms, .w = w, .cap = cap_b, .rows = rows, .caches = undefined, .blocks = undefined, .consts = undefined, .experts = undefined, .scratch = undefined, .blk = undefined, .ids = undefined, .erows = undefined, .edest = undefined };
        dec.caches = try cuda.DeviceBuffer.alloc(d, cacheBytes(cap_b));
        errdefer dec.caches.free();
        dec.blocks = try cuda.DeviceBuffer.alloc(d, 2 * (cap_b / B) * 4);
        errdefer dec.blocks.free();
        const inv = cfg.invTok();
        dec.consts = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&inv));
        errdefer dec.consts.free();
        const tab = try gpa.alloc(UdExpert, cfg.layers * cfg.experts);
        defer gpa.free(tab);
        @memset(tab, .{ .gate = 0, .up = 0, .down = 0 });
        for (w.layers, 0..) |L, l| {
            if (L.gate == null) continue;
            for (L.experts, 0..) |ex, x| tab[l * cfg.experts + x] = .{ .gate = ex.gate.dev, .up = ex.up.dev, .down = ex.down.dev };
        }
        dec.experts = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(tab));
        errdefer dec.experts.free();
        dec.scratch = try Scratch.make(d, rows);
        errdefer dec.scratch.buf.free();
        dec.blk = try gpa.alloc(i32, 2 * (cap_b / B));
        errdefer gpa.free(dec.blk);
        dec.ids = try gpa.alloc(u32, rows * slots);
        errdefer gpa.free(dec.ids);
        dec.erows = try gpa.alloc(u32, rows * slots);
        errdefer gpa.free(dec.erows);
        dec.edest = try gpa.alloc(u32, rows * slots);
        return dec;
    }

    pub fn deinit(dec: *Decoder) void {
        for ([_][]u32{ dec.ids, dec.erows, dec.edest }) |b| dec.gpa.free(b);
        dec.gpa.free(dec.blk);
        dec.scratch.buf.free();
        dec.experts.free();
        dec.consts.free();
        dec.blocks.free();
        dec.caches.free();
    }

    /// The cache bytes for `cap` tokens (a multiple of 32): bf16 keys and values, fp32 router keys and block keys.
    pub fn cacheBytes(cap: usize) usize {
        return cfg.layers * cap * (2 * H * 2 + D * 4) + cfg.layers * (cap / B) * D * 4;
    }
    fn kc(dec: *const Decoder, l: usize) u64 {
        return dec.caches.ptr + l * dec.cap * H * 2;
    }
    fn vc(dec: *const Decoder, l: usize) u64 {
        return dec.caches.ptr + (cfg.layers + l) * dec.cap * H * 2;
    }
    fn idxK(dec: *const Decoder, l: usize) u64 {
        return dec.caches.ptr + 2 * cfg.layers * dec.cap * H * 2 + l * dec.cap * D * 4;
    }
    fn kbar(dec: *const Decoder, l: usize) u64 {
        return dec.caches.ptr + cfg.layers * dec.cap * (2 * H * 2 + D * 4) + l * (dec.cap / B) * D * 4;
    }
    fn bmin(dec: *const Decoder) u64 {
        return dec.blocks.ptr;
    }
    fn bmax(dec: *const Decoder) u64 {
        return dec.blocks.ptr + (dec.cap / B) * 4;
    }

    fn upload(dec: *const Decoder, dst: u64, bytes: []const u8) !void {
        try dec.d.check(dec.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, dec.s.handle), "cuMemcpyHtoDAsync");
    }

    fn download(dec: *const Decoder, out: []u8, src: u64) !void {
        try dec.d.check(dec.d.api.cuMemcpyDtoHAsync_v2(out.ptr, src, out.len, dec.s.handle), "cuMemcpyDtoHAsync");
        try dec.s.synchronize();
    }

    /// The prompt's prefill: its embeddings with the image rows spliced in at its image tokens, then every layer.
    /// `image_rows` holds one 1280-wide bf16 row per image token, in order.
    pub fn prefill(dec: *Decoder, p: prompt_mod.Prompt, image_rows: []const u16) !void {
        const n = p.len();
        if (n > dec.cap) return error.PromptTooLong;
        var images: usize = 0;
        for (p.ids) |id| images += @intFromBool(id == cfg.image_token);
        if (image_rows.len != images * H) return error.ImageRows;
        dec.len = 0;
        const nb = dec.cap / B;
        @memset(dec.blk, -1);
        for (0..(n + B - 1) / B) |c| {
            const r = p.blockPages(c);
            dec.blk[c] = r[0];
            dec.blk[nb + c] = r[1];
        }
        try dec.upload(dec.blocks.ptr, std.mem.sliceAsBytes(dec.blk));
        const sc = dec.scratch;
        const qpos = try dec.gpa.alloc(u32, dec.rows);
        defer dec.gpa.free(qpos);
        var img: usize = 0;
        var start: usize = 0;
        while (start < n) {
            const m = @min(dec.rows, n - start);
            try dec.s.synchronize(); // the host arrays below are reused chunk to chunk
            const ids = p.ids[start..][0..m];
            try dec.upload(sc.at(.ids_in), std.mem.sliceAsBytes(ids));
            try dec.k.run("embed", dec.s, .{ m, ks.blocks(H), 1 }, 256, .{ sc.at(.ids_in), dec.w.embed.dev, sc.at(.x), @as(u32, cfg.image_token) });
            // the chunk's image rows, a run of image tokens at a time
            var i: usize = 0;
            while (i < m) {
                if (ids[i] != cfg.image_token) {
                    i += 1;
                    continue;
                }
                var j = i;
                while (j < m and ids[j] == cfg.image_token) j += 1;
                try dec.upload(sc.at(.x) + i * H * 2, std.mem.sliceAsBytes(image_rows[img * H ..][0 .. (j - i) * H]));
                img += j - i;
                i = j;
            }
            for (qpos[0..m], 0..) |*q, r| q.* = @intCast(start + r);
            try dec.upload(sc.at(.tok), std.mem.sliceAsBytes(p.tok[start..][0..m]));
            try dec.upload(sc.at(.page), std.mem.sliceAsBytes(p.page[start..][0..m]));
            try dec.upload(sc.at(.qpos), std.mem.sliceAsBytes(qpos[0..m]));
            try dec.upload(sc.at(.qpage), std.mem.sliceAsBytes(p.qpage[start..][0..m]));
            try dec.forward(start, m, true);
            start += m;
            dec.len = start;
        }
    }

    /// One decode step: `id` fed at the next position with its rotary coordinates and router query page.
    pub fn step(dec: *Decoder, id: u32, tok: u32, page: u32, qpage: i32) !void {
        if (dec.len >= dec.cap) return error.ContextFull;
        const sc = dec.scratch;
        try dec.d.check(dec.d.api.cuMemcpyDtoDAsync_v2(sc.at(.x), dec.w.embed.dev + @as(u64, id) * H * 2, H * 2, dec.s.handle), "cuMemcpyDtoDAsync");
        // the four words land in .tok, .page, .qpos, .qpage, each at its row 0
        const words = StepWords{ .tok = tok, .page = page, .qpos = @intCast(dec.len), .qpage = qpage };
        try dec.upload(sc.at(.tok), std.mem.asBytes(&words.tok));
        try dec.upload(sc.at(.page), std.mem.asBytes(&words.page));
        try dec.upload(sc.at(.qpos), std.mem.asBytes(&words.qpos));
        try dec.upload(sc.at(.qpage), std.mem.asBytes(&words.qpage));
        try dec.forward(dec.len, 1, false);
        dec.len += 1;
    }

    /// The last forward's final row through the norm and lm_head: bf16 logits over the vocabulary, on the device.
    fn head(dec: *Decoder) !void {
        if (dec.last_rows == 0) return error.NoForward;
        const sc = dec.scratch;
        try dec.rmsNorm(sc.at(.x) + (dec.last_rows - 1) * H * 2, dec.w.norm, sc.at(.h), 1);
        try dec.lin(sc.at(.h), dec.w.head, sc.at(.logits), 1, cfg.vocab, H, true);
    }

    /// bf16 logits over the vocabulary, copied to `out` (vocab values).
    pub fn logits(dec: *Decoder, out: []u16) !void {
        try dec.head();
        try dec.download(std.mem.sliceAsBytes(out[0..cfg.vocab]), dec.scratch.at(.logits));
    }

    /// The greedy pick over the logits on the device (the first maximum; EOS skipped when `bar_eos`).
    pub fn argmax(dec: *Decoder, bar_eos: bool) !u32 {
        try dec.head();
        const sc = dec.scratch;
        try dec.k.run("argmax", dec.s, .{ 1, 1, 1 }, 1024, .{ sc.at(.logits), cfg.vocab, @as(c_int, @intFromBool(bar_eos)), sc.at(.pick) });
        var id: u32 = 0;
        try dec.download(std.mem.asBytes(&id), sc.at(.pick));
        return id;
    }

    /// The hidden rows after the last forward (the residual stream past layer 11), for checks.
    pub fn hidden(dec: *Decoder, out: []u16) !void {
        try dec.download(std.mem.sliceAsBytes(out), dec.scratch.at(.x));
    }

    /// Rows start..start+n through every layer; the caches take their keys at those rows.
    fn forward(dec: *Decoder, start: usize, n: usize, prefill_rows: bool) !void {
        const sc = dec.scratch;
        const k = dec.k;
        const s = dec.s;
        const scale = 1.0 / @sqrt(@as(f32, cfg.head_dim));
        const done_before = start / B;
        const done_after = (start + n) / B;
        const nblk = (start + n + B - 1) / B;
        if (nblk > 4096) return error.ContextFull;
        dec.last_rows = n;
        const vec = !prefill_rows;
        const inv = dec.consts.ptr;
        for (dec.w.layers, 0..) |L, l| {
            try dec.rmsNorm(sc.at(.x), L.in_norm, sc.at(.h), n);
            const krows = dec.kc(l) + start * H * 2;
            try dec.lin(sc.at(.h), L.q, sc.at(.q), n, H, H, vec);
            try dec.lin(sc.at(.h), L.k, krows, n, H, H, vec);
            try dec.lin(sc.at(.h), L.v, dec.vc(l) + start * H * 2, n, H, H, vec);
            for ([_]u64{ sc.at(.q), krows }) |t| try k.run("rope", s, .{ n, cfg.heads, 1 }, 128, .{ t, sc.at(.tok), sc.at(.page), inv, H });
            // the router's index q and k, the newly complete blocks' summary keys, and each query's reads
            try dec.lin(sc.at(.h), L.router.q, sc.at(.rq), n, D, H, vec);
            try dec.lin(sc.at(.h), L.router.k, sc.at(.rk), n, D, H, vec);
            try k.run("router_norm", s, .{ n, 1, 1 }, 128, .{ sc.at(.rq), L.router.q_norm.dev, sc.at(.tok), sc.at(.page), inv, sc.at(.iq) });
            try k.run("router_norm", s, .{ n, 1, 1 }, 128, .{ sc.at(.rk), L.router.k_norm.dev, sc.at(.tok), sc.at(.page), inv, dec.idxK(l) + start * D * 4 });
            if (done_after > done_before)
                try k.run("block_keys", s, .{ done_after - done_before, 1, 1 }, 32, .{ dec.idxK(l), L.router.head_cls.dev, dec.kbar(l), done_before, 1.0 / @sqrt(@as(f32, D)) });
            try k.run("route", s, .{ n, 1, 1 }, 256, .{ sc.at(.iq), dec.kbar(l), dec.bmin(), dec.bmax(), sc.at(.qpage), sc.at(.qpos), sc.at(.sel), sc.at(.bias), nblk, done_after, 1.0 / @sqrt(@as(f32, D)), cfg.router_scale, cfg.router_floor, @as(c_int, @intFromBool(prefill_rows)) });
            try k.run("attend", s, .{ n, cfg.heads, 1 }, 256, .{ sc.at(.q), dec.kc(l), dec.vc(l), sc.at(.sel), sc.at(.bias), sc.at(.qpos), sc.at(.att), H, scale, cfg.local });
            try dec.lin(sc.at(.att), L.o, sc.at(.tmp), n, H, H, vec);
            try dec.add(sc.at(.x), sc.at(.tmp), sc.at(.x), n * H);
            try dec.rmsNorm(sc.at(.x), L.post_norm, sc.at(.h), n);
            if (L.gate) |gate| try dec.moe(L, l, gate, n, vec) else try dec.mlp(L.ffn, sc.at(.h), sc.at(.act), sc.at(.tmp), n, cfg.dense_ffn, vec);
            try dec.add(sc.at(.x), sc.at(.tmp), sc.at(.x), n * H);
        }
    }

    /// The routed layer's FFN into `.tmp`: the gate on the GPU, the experts' row lists here (a prompt's) or on the GPU
    /// (one token's), the shared expert, each expert on its gathered rows, and the combine in gate order.
    fn moe(dec: *Decoder, L: wts.Layer, l: usize, gate: T, n: usize, vec: bool) !void {
        const sc = dec.scratch;
        const k = dec.k;
        const s = dec.s;
        try k.run("moe_gate", s, .{ n, 1, 1 }, 64, .{ sc.at(.h), gate.dev, sc.at(.ids), sc.at(.wts) });
        if (vec) {
            const tab = dec.experts.ptr + l * cfg.experts * @sizeOf(UdExpert);
            try k.run("expert_gu", s, .{ cfg.expert_ffn / 32, slots, 1 }, 256, .{ sc.at(.h), tab, sc.at(.ids), sc.at(.act), cfg.expert_ffn });
            try k.run("expert_down", s, .{ H / 32, slots, 1 }, 256, .{ sc.at(.act), tab, sc.at(.ids), sc.at(.slots), cfg.expert_ffn });
            try dec.mlp(L.ffn, sc.at(.h), sc.at(.xg), sc.at(.shared), 1, cfg.shared_ffn, true);
            try k.run("moe_combine", s, .{ 1, ks.blocks(H), 1 }, 256, .{ sc.at(.slots), sc.at(.wts), sc.at(.shared), sc.at(.tmp) });
            return;
        }
        const ids = dec.ids[0 .. n * slots];
        try dec.download(std.mem.sliceAsBytes(ids), sc.at(.ids));
        var counts: [cfg.experts]usize = @splat(0);
        for (ids) |x| counts[x] += 1;
        var first: [cfg.experts + 1]usize = undefined;
        first[0] = 0;
        for (0..cfg.experts) |x| first[x + 1] = first[x] + counts[x];
        var fill = first;
        for (ids, 0..) |x, i| {
            dec.erows[fill[x]] = @intCast(i / slots);
            dec.edest[fill[x]] = @intCast(i);
            fill[x] += 1;
        }
        try dec.upload(sc.at(.erows), std.mem.sliceAsBytes(dec.erows[0 .. n * slots]));
        try dec.upload(sc.at(.edest), std.mem.sliceAsBytes(dec.edest[0 .. n * slots]));
        try dec.mlp(L.ffn, sc.at(.h), sc.at(.act), sc.at(.shared), n, cfg.shared_ffn, false);
        for (L.experts, 0..) |ex, x| {
            const c = counts[x];
            if (c == 0) continue;
            try k.run("gather", s, .{ c, ks.blocks(H), 1 }, 256, .{ sc.at(.h), sc.at(.erows) + first[x] * 4, sc.at(.xg) });
            try dec.mlp(ex, sc.at(.xg), sc.at(.act), sc.at(.ey), c, cfg.expert_ffn, false);
            try k.run("scatter", s, .{ c, ks.blocks(H), 1 }, 256, .{ sc.at(.ey), sc.at(.edest) + first[x] * 4, sc.at(.slots) });
        }
        try k.run("moe_combine", s, .{ n, ks.blocks(H), 1 }, 256, .{ sc.at(.slots), sc.at(.wts), sc.at(.shared), sc.at(.tmp) });
    }

    /// y [M, N] = bf16(x w^T): cuBLASLt for a prompt's rows, the matvec for one token's.
    fn lin(dec: *Decoder, x: u64, w: T, y: u64, M: usize, N: usize, K: usize, vec: bool) !void {
        if (!vec) return dec.gemms.run(dec.s, x, w.dev, y, M, N, K, .bf16);
        std.debug.assert(M == 1);
        try dec.k.run("matvec", dec.s, .{ (N + 31) / 32, 1, 1 }, 256, .{ x, w.dev, y, N, K });
    }

    /// down(bf16(silu(gate(x))) * up(x)) for `n` rows into `y`; `act` holds the activations.
    fn mlp(dec: *Decoder, w: wts.Expert, x: u64, act: u64, y: u64, n: usize, width: usize, vec: bool) !void {
        const sc = dec.scratch;
        try dec.lin(x, w.gate, sc.at(.g), n, width, H, vec);
        try dec.lin(x, w.up, sc.at(.u), n, width, H, vec);
        try dec.k.run("silu_mul", dec.s, .{ ks.blocks(n * width), 1, 1 }, 256, .{ sc.at(.g), sc.at(.u), act, @as(i64, @intCast(n * width)) });
        try dec.lin(act, w.down, y, n, H, width, vec);
    }

    fn rmsNorm(dec: *Decoder, x: u64, w: T, y: u64, n: usize) !void {
        try dec.k.run("rms_norm", dec.s, .{ n, 1, 1 }, 256, .{ x, w.dev, y, H, cfg.eps });
    }

    fn add(dec: *Decoder, a: u64, b: u64, y: u64, n: usize) !void {
        try dec.k.run("add", dec.s, .{ ks.blocks(n), 1, 1 }, 256, .{ a, b, y, @as(i64, @intCast(n)) });
    }
};

/// One device buffer's parts for `rows` tokens, each 256-aligned.
const Scratch = struct {
    const Part = enum { x, h, q, att, tmp, shared, rq, rk, iq, sel, bias, tok, page, qpos, qpage, ids_in, g, u, act, ids, wts, erows, edest, slots, xg, ey, logits, pick };
    buf: cuda.DeviceBuffer,
    offs: [std.enums.values(Part).len]usize,

    fn bytes(p: Part, r: usize) usize {
        return switch (p) {
            .x, .h, .q, .att, .tmp, .shared, .xg, .ey => r * H * 2,
            .rq, .rk => r * D * 2,
            .iq => r * D * 4,
            .sel, .bias => r * cfg.read_blocks * 4,
            .tok, .page, .qpos, .qpage, .ids_in => r * 4,
            .g, .u, .act => r * cfg.dense_ffn * 2,
            .ids, .wts, .erows, .edest => r * slots * 4,
            .slots => r * slots * H * 2,
            .logits => cfg.vocab * 2,
            .pick => 4,
        };
    }

    fn make(d: *const cuda.Driver, rows: usize) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, rows), 256);
        }
        s.buf = try cuda.DeviceBuffer.alloc(d, total);
        return s;
    }

    fn at(s: Scratch, p: Part) u64 {
        return s.buf.ptr + s.offs[@backingInt(p)];
    }
};
