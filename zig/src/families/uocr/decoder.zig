//! Unlimited-OCR-2's decoder on Metal: 12 layers of page2d-rotary MHA under DASH's block-sparse reads (a 128-key
//! local window plus 12 routed 32-key blocks) and a 64-expert MoE, over bf16 caches in the model's precision.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const ks = @import("kernels.zig");
const prompt_mod = @import("prompt.zig");
const Ref = ks.Ref;

const H = cfg.hidden;
const D = cfg.router_dim;
const B = cfg.block;
const slots = cfg.top_k;

const UdNorm = extern struct { width: u32, eps: f32 };
const UdRope = extern struct { heads: u32, stride: u32 };
const UdBlocks = extern struct { first: u32, scale: f32 };
const UdRoute = extern struct { nblk: u32, complete: u32, scale: f32, idx_scale: f32, floor_: f32, round_bias: u32 };
const UdAttend = extern struct { width: u32, scale: f32, local: u32 };
const UdMv = extern struct { n: u32, k: u32 };
const UdExpert = extern struct { gate: u64, up: u64, down: u64 };
const UdExpertArgs = extern struct { ffn: u32 };

/// One sequence's caches and the scratch for a forward of up to `rows` tokens.
pub const Decoder = struct {
    gpa: std.mem.Allocator,
    queue: mtl.Queue,
    k: *const ks.Kernels,
    w: *const wts.Weights,
    /// Tokens the caches hold, and tokens in them now.
    cap: usize,
    len: usize = 0,
    rows: usize,
    caches: mtl.Buffer,
    blocks: mtl.Buffer,
    consts: mtl.Buffer,
    /// Each routed layer's 64 experts as GPU pointers (gate, up, down), for decode's on-GPU routing, and the
    /// buffers those pointers reach (made resident on each encoder).
    experts: mtl.Buffer,
    weight_bufs: std.ArrayList(mtl.Buffer) = .empty,
    scratch: Scratch,
    /// For checks: each layer's output rows are copied here after a forward's layer (one slice a layer).
    trace: ?[][]u16 = null,
    /// For checks: each layer's attention (after o_proj) and FFN outputs, beside `trace`.
    trace_parts: ?[2][][]u16 = null,
    /// Pending command buffer and its encoder.
    cb: ?mtl.CommandBuffer = null,
    enc: ?mtl.ComputeEncoder = null,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: *const ks.Kernels, w: *const wts.Weights, cap: usize, rows: usize) !Decoder {
        if (cap == 0 or cap > cfg.max_context or rows == 0 or rows > cap) return error.DecoderSize;
        const cap_b = std.mem.alignForward(usize, cap, B);
        var d: Decoder = .{ .gpa = gpa, .queue = queue, .k = k, .w = w, .cap = cap_b, .rows = rows, .caches = undefined, .blocks = undefined, .consts = undefined, .experts = undefined, .scratch = undefined };
        d.caches = try device.buffer(cacheBytes(cap_b), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer d.caches.deinit();
        d.blocks = try device.buffer(2 * (cap_b / B) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer d.blocks.deinit();
        d.consts = try device.buffer(256, mtl.ResourceOptions.shared);
        errdefer d.consts.deinit();
        const inv = cfg.invTok();
        @memcpy(@as([*]f32, @ptrCast(@alignCast(d.consts.contents())))[0..inv.len], &inv);
        d.experts = try device.buffer(cfg.layers * cfg.experts * @sizeOf(UdExpert), mtl.ResourceOptions.shared);
        errdefer d.experts.deinit();
        errdefer d.weight_bufs.deinit(gpa);
        const tab = @as([*]UdExpert, @ptrCast(@alignCast(d.experts.contents())))[0 .. cfg.layers * cfg.experts];
        for (w.layers, 0..) |L, l| for (L.experts, 0..) |ex, x| {
            tab[l * cfg.experts + x] = .{ .gate = address(ex.gate), .up = address(ex.up), .down = address(ex.down) };
            for ([_]Ref{ ex.gate, ex.up, ex.down }) |r| {
                for (d.weight_bufs.items) |b| {
                    if (b.id == r.buf.id) break;
                } else try d.weight_bufs.append(gpa, r.buf);
            }
        };
        d.scratch = try Scratch.make(device, rows);
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        d.scratch.buf.deinit();
        d.weight_bufs.deinit(d.gpa);
        d.experts.deinit();
        d.consts.deinit();
        d.blocks.deinit();
        d.caches.deinit();
    }

    fn cacheBytes(cap: usize) usize {
        return cfg.layers * cap * (2 * H * 2 + D * 4) + cfg.layers * (cap / B) * D * 4;
    }
    fn kc(d: *const Decoder, l: usize) Ref {
        return .{ .buf = d.caches, .off = l * d.cap * H * 2 };
    }
    fn vc(d: *const Decoder, l: usize) Ref {
        return .{ .buf = d.caches, .off = (cfg.layers + l) * d.cap * H * 2 };
    }
    fn idxK(d: *const Decoder, l: usize) Ref {
        return .{ .buf = d.caches, .off = 2 * cfg.layers * d.cap * H * 2 + l * d.cap * D * 4 };
    }
    fn kbar(d: *const Decoder, l: usize) Ref {
        return .{ .buf = d.caches, .off = cfg.layers * d.cap * (2 * H * 2 + D * 4) + l * (d.cap / B) * D * 4 };
    }
    fn bmin(d: *const Decoder) Ref {
        return .{ .buf = d.blocks, .off = 0 };
    }
    fn bmax(d: *const Decoder) Ref {
        return .{ .buf = d.blocks, .off = (d.cap / B) * 4 };
    }
    fn invTok(d: *const Decoder) Ref {
        return .{ .buf = d.consts, .off = 0 };
    }

    /// The prompt's prefill: its embeddings with the image rows spliced in at its image tokens, then every layer.
    /// `image_rows` holds one 1280-wide bf16 row per image token, in order.
    pub fn prefill(d: *Decoder, p: prompt_mod.Prompt, image_rows: []const u16) !void {
        const n = p.len();
        if (n > d.rows or n > d.cap) return error.PromptTooLong;
        var images: usize = 0;
        for (p.ids) |id| images += @intFromBool(id == cfg.image_token);
        if (image_rows.len != images * H) return error.ImageRows;
        d.len = 0;
        const blk_lo = @as([*]i32, @ptrCast(@alignCast(d.blocks.contents())))[0 .. d.cap / B];
        const blk_hi = @as([*]i32, @ptrCast(@alignCast(d.blocks.contents() + (d.cap / B) * 4)))[0 .. d.cap / B];
        @memset(blk_lo, -1);
        @memset(blk_hi, -1);
        for (0..(n + B - 1) / B) |c| {
            const r = p.blockPages(c);
            blk_lo[c] = r[0];
            blk_hi[c] = r[1];
        }
        const s = d.scratch;
        const x = halves(s.at(.x), n * H);
        var img: usize = 0;
        for (p.ids, 0..) |id, i| {
            const row = x[i * H ..][0..H];
            if (id == cfg.image_token) {
                @memcpy(row, image_rows[img * H ..][0..H]);
                img += 1;
            } else @memcpy(row, halves(d.w.embed.at(@as(usize, id) * H * 2), H));
        }
        @memcpy(words(s.at(.tok), n), p.tok);
        @memcpy(words(s.at(.page), n), p.page);
        for (words(s.at(.qpos), n), 0..) |*q, i| q.* = @intCast(i);
        @memcpy(@as([*]i32, @ptrCast(@alignCast(s.at(.qpage).addr())))[0..n], p.qpage);
        try d.forward(0, n, true);
        d.len = n;
    }

    /// One decode step: `id` fed at the next position with its rotary coordinates and router query page.
    pub fn step(d: *Decoder, id: u32, tok: u32, page: u32, qpage: i32) !void {
        if (d.len >= d.cap) return error.ContextFull;
        const s = d.scratch;
        @memcpy(halves(s.at(.x), H), halves(d.w.embed.at(@as(usize, id) * H * 2), H));
        words(s.at(.tok), 1)[0] = tok;
        words(s.at(.page), 1)[0] = page;
        words(s.at(.qpos), 1)[0] = @intCast(d.len);
        @as([*]i32, @ptrCast(@alignCast(s.at(.qpage).addr())))[0] = qpage;
        try d.forward(d.len, 1, false);
        d.len += 1;
    }

    /// The last forward's final row through the norm and lm_head: bf16 logits over the vocabulary.
    pub fn logits(d: *Decoder, row: usize) ![]const u16 {
        const s = d.scratch;
        const e = d.encoder();
        rmsNorm(e, d.k, s.at(.x).at(row * H * 2), d.w.norm, s.at(.h), 1);
        lin(e, d.k, s.at(.h), d.w.head, s.at(.logits), 1, cfg.vocab, H);
        try d.flush();
        return halves(s.at(.logits), cfg.vocab);
    }

    /// The hidden rows after the last forward (the residual stream past layer 11), for checks.
    pub fn hidden(d: *Decoder, n: usize) ![]const u16 {
        try d.flush();
        return halves(d.scratch.at(.x), n * H);
    }

    fn encoder(d: *Decoder) mtl.ComputeEncoder {
        if (d.enc) |e| return e;
        const cb = d.queue.commandBuffer();
        d.cb = cb;
        const e = cb.compute(.serial);
        for (d.weight_bufs.items) |b| e.useResource(b, mtl.ResourceUsage.read);
        d.enc = e;
        return e;
    }

    fn flush(d: *Decoder) !void {
        const e = d.enc orelse return;
        e.end();
        const cb = d.cb.?;
        cb.commit();
        cb.wait();
        d.enc = null;
        d.cb = null;
        if (cb.failure()) |msg| {
            std.log.err("unlimited ocr: the decoder failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
    }

    /// Rows start..start+n through every layer; the caches take their keys at those rows.
    fn forward(d: *Decoder, start: usize, n: usize, prefill_rows: bool) !void {
        const s = d.scratch;
        const k = d.k;
        const scale = 1.0 / @sqrt(@as(f32, cfg.head_dim));
        const done_before = start / B;
        const done_after = (start + n) / B;
        const nblk = (start + n + B - 1) / B;
        for (d.w.layers, 0..) |L, l| {
            const e = d.encoder();
            rmsNorm(e, k, s.at(.x), L.in_norm, s.at(.h), n);
            const krows = d.kc(l).at(start * H * 2);
            lin(e, k, s.at(.h), L.q, s.at(.q), n, H, H);
            lin(e, k, s.at(.h), L.k, krows, n, H, H);
            lin(e, k, s.at(.h), L.v, d.vc(l).at(start * H * 2), n, H, H);
            for ([_]Ref{ s.at(.q), krows }) |t| {
                ks.Kernels.bind(e, k.get("ud_rope"), &.{ t, s.at(.tok), s.at(.page), d.invTok() }, UdRope{ .heads = cfg.heads, .stride = H });
                e.dispatchThreads(mtl.Size.of(cfg.head_dim, cfg.heads, n), mtl.Size.of(cfg.head_dim, 1, 1));
            }
            // the router's index q and k, the newly complete blocks' summary keys, and each query's reads
            lin(e, k, s.at(.h), L.router.q, s.at(.rq), n, D, H);
            lin(e, k, s.at(.h), L.router.k, s.at(.rk), n, D, H);
            for ([_][3]Ref{ .{ s.at(.rq), L.router.q_norm, s.at(.iq) }, .{ s.at(.rk), L.router.k_norm, d.idxK(l).at(start * D * 4) } }) |r| {
                ks.Kernels.bind(e, k.get("ud_router_norm"), &.{ r[0], r[1], s.at(.tok), s.at(.page), d.invTok(), r[2] }, {});
                e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(D, 1, 1));
            }
            if (done_after > done_before) {
                ks.Kernels.bind(e, k.get("ud_block_keys"), &.{ d.idxK(l), L.router.head_cls, d.kbar(l) }, UdBlocks{ .first = @intCast(done_before), .scale = 1.0 / @sqrt(@as(f32, D)) });
                e.dispatchGroups(mtl.Size.of(done_after - done_before, 1, 1), mtl.Size.of(B, 1, 1));
            }
            ks.Kernels.bind(e, k.get("ud_route"), &.{ s.at(.iq), d.kbar(l), d.bmin(), d.bmax(), s.at(.qpage), s.at(.qpos), s.at(.sel), s.at(.bias) }, UdRoute{
                .nblk = @intCast(nblk),
                .complete = @intCast(done_after),
                .scale = 1.0 / @sqrt(@as(f32, D)),
                .idx_scale = cfg.router_scale,
                .floor_ = cfg.router_floor,
                .round_bias = @intFromBool(prefill_rows),
            });
            e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
            ks.Kernels.bind(e, k.get("ud_attend"), &.{ s.at(.q), d.kc(l), d.vc(l), s.at(.sel), s.at(.bias), s.at(.qpos), s.at(.att) }, UdAttend{ .width = H, .scale = scale, .local = cfg.local });
            e.dispatchGroups(mtl.Size.of(n, cfg.heads, 1), mtl.Size.of(256, 1, 1));
            lin(e, k, s.at(.att), L.o, s.at(.tmp), n, H, H);
            if (d.trace_parts) |tp| {
                try d.flush();
                @memcpy(tp[0][l][0 .. n * H], halves(s.at(.tmp), n * H));
            }
            add(d.encoder(), k, s.at(.x), s.at(.tmp), s.at(.x), n * H);
            rmsNorm(d.encoder(), k, s.at(.x), L.post_norm, s.at(.h), n);
            if (L.gate) |gate| try d.moe(L, l, gate, n) else mlp(d.encoder(), k, L.ffn, s.at(.h), s.at(.g), s.at(.u), s.at(.act), s.at(.tmp), n, cfg.dense_ffn);
            if (d.trace_parts) |tp| {
                try d.flush();
                @memcpy(tp[1][l][0 .. n * H], halves(s.at(.tmp), n * H));
            }
            add(d.encoder(), k, s.at(.x), s.at(.tmp), s.at(.x), n * H);
            if (d.trace) |t| {
                try d.flush();
                @memcpy(t[l][0 .. n * H], halves(s.at(.x), n * H));
            }
        }
    }

    /// The routed layer's FFN into `.tmp`: the gate on the GPU, the experts' row lists here, then the shared expert,
    /// each expert on its gathered rows, and the combine in gate order.
    fn moe(d: *Decoder, L: wts.Layer, l: usize, gate: Ref, n: usize) !void {
        const s = d.scratch;
        const k = d.k;
        var e = d.encoder();
        ks.Kernels.bind(e, k.get("ud_moe_gate"), &.{ s.at(.h), gate, s.at(.ids), s.at(.wts) }, {});
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(cfg.experts, 1, 1));
        if (n == 1) {
            // one token: its experts straight from the gate's ids, no wait on the host
            const tab = Ref{ .buf = d.experts, .off = l * cfg.experts * @sizeOf(UdExpert) };
            const args = UdExpertArgs{ .ffn = cfg.expert_ffn };
            ks.Kernels.bind(e, k.get("ud_expert_gu"), &.{ s.at(.h), tab, s.at(.ids), s.at(.act) }, args);
            e.dispatchGroups(mtl.Size.of(cfg.expert_ffn / 32, slots, 1), mtl.Size.of(256, 1, 1));
            ks.Kernels.bind(e, k.get("ud_expert_down"), &.{ s.at(.act), tab, s.at(.ids), s.at(.slots) }, args);
            e.dispatchGroups(mtl.Size.of(H / 32, slots, 1), mtl.Size.of(256, 1, 1));
            mlp(e, k, L.ffn, s.at(.h), s.at(.g), s.at(.u), s.at(.xg), s.at(.shared), 1, cfg.shared_ffn);
            ks.Kernels.bind(e, k.get("ud_moe_combine"), &.{ s.at(.slots), s.at(.wts), s.at(.shared), s.at(.tmp) }, {});
            e.dispatchThreads(mtl.Size.of(H, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        try d.flush();
        const ids = words(s.at(.ids), n * slots);
        var counts: [cfg.experts]usize = @splat(0);
        for (ids) |x| counts[x] += 1;
        var first: [cfg.experts + 1]usize = undefined;
        first[0] = 0;
        for (0..cfg.experts) |x| first[x + 1] = first[x] + counts[x];
        var fill = first;
        const rows = words(s.at(.erows), n * slots);
        const dest = words(s.at(.edest), n * slots);
        for (ids, 0..) |x, i| {
            rows[fill[x]] = @intCast(i / slots);
            dest[fill[x]] = @intCast(i);
            fill[x] += 1;
        }
        e = d.encoder();
        mlp(e, k, L.ffn, s.at(.h), s.at(.g), s.at(.u), s.at(.act), s.at(.shared), n, cfg.shared_ffn);
        for (L.experts, 0..) |ex, x| {
            const c = counts[x];
            if (c == 0) continue;
            ks.Kernels.bind(e, k.get("ud_gather"), &.{ s.at(.h), s.at(.erows).at(first[x] * 4), s.at(.xg) }, {});
            e.dispatchThreads(mtl.Size.of(H, c, 1), mtl.Size.of(256, 1, 1));
            mlp(e, k, ex, s.at(.xg), s.at(.g), s.at(.u), s.at(.act), s.at(.ey), c, cfg.expert_ffn);
            ks.Kernels.bind(e, k.get("ud_scatter"), &.{ s.at(.ey), s.at(.edest).at(first[x] * 4), s.at(.slots) }, {});
            e.dispatchThreads(mtl.Size.of(H, c, 1), mtl.Size.of(256, 1, 1));
        }
        ks.Kernels.bind(e, k.get("ud_moe_combine"), &.{ s.at(.slots), s.at(.wts), s.at(.shared), s.at(.tmp) }, {});
        e.dispatchThreads(mtl.Size.of(H, n, 1), mtl.Size.of(256, 1, 1));
    }
};

/// y [M, N] = bf16(x w^T): the tensor-unit GEMM for a prompt's rows, the matvec for one token's.
fn lin(e: mtl.ComputeEncoder, k: *const ks.Kernels, x: Ref, w: Ref, y: Ref, M: usize, N: usize, K: usize) void {
    if (M > 1) return k.linear(e, x, w, y, M, N, K);
    ks.Kernels.bind(e, k.get("ud_matvec"), &.{ x, w, y }, UdMv{ .n = @intCast(N), .k = @intCast(K) });
    e.dispatchGroups(mtl.Size.of((N + 31) / 32, 1, 1), mtl.Size.of(256, 1, 1));
}

fn address(r: Ref) u64 {
    return r.buf.gpuAddress() + r.off;
}

/// down(bf16(silu(gate(x))) * up(x)) for `n` rows into `y`.
fn mlp(e: mtl.ComputeEncoder, k: *const ks.Kernels, w: wts.Expert, x: Ref, g: Ref, u: Ref, act: Ref, y: Ref, n: usize, width: usize) void {
    lin(e, k, x, w.gate, g, n, width, H);
    lin(e, k, x, w.up, u, n, width, H);
    ks.Kernels.bind(e, k.get("ud_silu_mul"), &.{ g, u, act }, @as(u32, @intCast(n * width)));
    e.dispatchThreads(mtl.Size.of(n * width, 1, 1), mtl.Size.of(256, 1, 1));
    lin(e, k, act, w.down, y, n, H, width);
}

fn rmsNorm(e: mtl.ComputeEncoder, k: *const ks.Kernels, x: Ref, w: Ref, y: Ref, n: usize) void {
    ks.Kernels.bind(e, k.get("ud_rms_norm"), &.{ x, w, y }, UdNorm{ .width = H, .eps = cfg.eps });
    e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

fn add(e: mtl.ComputeEncoder, k: *const ks.Kernels, a: Ref, b: Ref, y: Ref, n: usize) void {
    ks.Kernels.bind(e, k.get("ud_add"), &.{ a, b, y }, @as(u32, @intCast(n)));
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

fn halves(r: Ref, n: usize) []u16 {
    return @as([*]u16, @ptrCast(@alignCast(r.addr())))[0..n];
}

fn words(r: Ref, n: usize) []u32 {
    return @as([*]u32, @ptrCast(@alignCast(r.addr())))[0..n];
}

/// One buffer's parts for `rows` tokens, each 256-aligned.
const Scratch = struct {
    const Part = enum { x, h, q, att, tmp, shared, rq, rk, iq, sel, bias, tok, page, qpos, qpage, g, u, act, ids, wts, erows, edest, slots, xg, ey, logits };
    buf: mtl.Buffer,
    offs: [std.enums.values(Part).len]usize,

    fn bytes(p: Part, r: usize) usize {
        return switch (p) {
            .x, .h, .q, .att, .tmp, .shared, .xg, .ey => r * H * 2,
            .rq, .rk => r * D * 2,
            .iq => r * D * 4,
            .sel, .bias => r * cfg.read_blocks * 4,
            .tok, .page, .qpos, .qpage => r * 4,
            .g, .u, .act => r * cfg.dense_ffn * 2,
            .ids, .wts, .erows, .edest => r * slots * 4,
            .slots => r * slots * H * 2,
            .logits => cfg.vocab * 2,
        };
    }

    fn make(device: mtl.Device, rows: usize) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, rows), 256);
        }
        s.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return s;
    }

    fn at(s: Scratch, p: Part) Ref {
        return .{ .buf = s.buf, .off = s.offs[@backingInt(p)] };
    }
};
