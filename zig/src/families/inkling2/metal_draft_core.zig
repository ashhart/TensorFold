//! The DSpark draft head's device side: kernels, scratch, per-stream context caches, the absorb and block sequences.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const gm = @import("metal_vision_gemm.zig");
const wt = @import("metal_draft_weights.zig");

const Ref = gm.Ref;
pub const head_dim = wt.head_dim;

/// Target rows one absorb pass projects at most.
pub const absorb_rows: u32 = 1024;
/// Rows a lane keeps on the host before they must be absorbed.
pub const pending_rows: u32 = 32;
/// Splits an attention pass divides its keys into at most (the partials' scratch), and the keys a split reads at least.
const max_splits: u32 = 128;
const min_per_split: u32 = 32;

/// The target tensors the draft head shares: its embedding table and norm, and the unembedding (all bf16).
pub const Target = struct { embed: ck.Tensor, embed_norm: ck.Tensor, unembed: ck.Tensor };

const kernel_names = [_][]const u8{ "embed_block", "rms", "add_rms", "mix_rms", "qk_prep", "ctx_store", "attn_split", "attn_merge", "swiglu", "rnn", "cell", "step_logits", "pick", "fold_b", "alpha" };
const Kern = enum { embed_block, rms, add_rms, mix_rms, qk_prep, ctx_store, attn_split, attn_merge, swiglu, rnn, cell, step_logits, pick, fold_b, alpha };

/// Kernel constants (the structs of kernels/metal/inkling2/draft.metal).
const Rows = extern struct { rows: u32, width: u32, eps: f32, pitch: u32 = 0, taps: u32 = 0 };
const Prep = extern struct { rows: u32, pos0: u32, heads_q: u32, heads_kv: u32, eps: f32 };
const Ctx = extern struct { rows: u32, pos0: u32, ring: u32, groups: u32, eps: f32 };
const Attn = extern struct { rows: u32, heads: u32, groups: u32, lo: u32, nctx: u32, ring: u32, splits: u32, per: u32, scale: f32 };
const Rnn = extern struct { width: u32, rank: u32, step: u32 };
const Step = extern struct { vocab: u32, unpadded: u32, rank: u32, cap: f32, lanes: u32, base_stride: u32, groups: u32 };
const Fold = extern struct { vocab: u32, width: u32 };

const scratch_names = [_][]const u8{ "resid", "x", "y", "qraw", "q", "attn", "kraw", "kb", "vb", "pm", "pl", "po", "g", "u", "act", "base", "raw", "state", "latent", "best", "best_id", "cbuf", "kcr", "vcr" };

/// One stream's context: each layer's keys and values for committed positions, and the Markov head's feature vector.
pub const Lane = struct {
    k: [wt.max_layers]mtl.Buffer,
    v: [wt.max_layers]mtl.Buffer,
    f: mtl.Buffer, // f_rnn: the mixed tap vector of the last committed row (bf16 [hidden])
    end: u64 = 0, // committed positions with features: the next block's anchor
    from: u64 = 0, // the first position the global layer holds (rows before it were restored, never absorbed)
    lfrom: u64 = 0, // the first position the sliding layers' rings still hold intact
    seq: mtl.Buffer, // pending token and proposals, shared by the GPU and the verified lane's consumer
    held_len: u32 = 0,
    pend: mtl.Buffer, // tap rows of decode rounds kept on the host (bf16 [pending_rows, taps * hidden]) until a hold needs them
    pend_rows: u32 = 0, // rows in `pend`: positions [end - pend_rows, end) of the context, not yet absorbed on the GPU
};

pub const Core = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    m: config.Mtp,
    hidden: u32,
    vocab: u32,
    unpadded: u32,
    cap: f32, // the target's final_logit_softcapping, applied to the corrected logits
    eps: f32,
    capacity: u32,
    batch: u32, // lanes one `blocks` pass takes at most (the scratch holds that many blocks)
    w: wt.Weights,
    target: Target,
    lib: mtl.Library,
    gemm: gm.Gemm,
    pipes: [kernel_names.len]mtl.Pipeline,
    s: [scratch_names.len]mtl.Buffer,
    alpha: mtl.Buffer, // bf16 softmax of fuse_weight rows
    inv: mtl.Buffer, // f32 [64] rotary inverse frequencies
    fold_w: mtl.Buffer, // bf16 [vocab, rank] = unembed x hidden_out.weight, rounded once
    fold_b: mtl.Buffer, // bf16 [vocab] = unembed x hidden_out.bias
    staging: ?mtl.Buffer = null, // host features' device copy (absorb_rows rows), made at first use

    pub fn init(gpa: std.mem.Allocator, dir: []const u8, device: mtl.Device, queue: mtl.Queue, c: *const config.Config, target: Target, capacity: u32, batch: u32) !*Core {
        const m = c.mtp orelse return error.NoDraftHead;
        const h: usize = c.text.hidden;
        const V: usize = c.text.vocab;
        for ([_]ck.Tensor{ target.embed, target.unembed }) |t| if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != V or t.shape[1] != h) return error.UnexpectedDraftTarget;
        if (target.embed_norm.dtype != .bf16 or target.embed_norm.rank != 1 or target.embed_norm.shape[0] != h)
            return error.UnexpectedDraftTarget;
        if (c.text.unpadded_vocab > c.text.vocab or capacity == 0 or batch == 0) return error.UnexpectedDraftTarget;
        const k = try gpa.create(Core);
        errdefer gpa.destroy(k);
        k.* = .{ .gpa = gpa, .device = device, .queue = queue, .m = m, .hidden = c.text.hidden, .vocab = c.text.vocab, .unpadded = c.text.unpadded_vocab, .cap = c.text.logit_cap, .eps = m.eps, .capacity = capacity, .batch = batch, .w = try wt.Weights.load(gpa, dir, device, c), .target = target, .lib = undefined, .gemm = .{}, .pipes = undefined, .s = undefined, .alpha = undefined, .inv = undefined, .fold_w = undefined, .fold_b = undefined };
        errdefer k.w.deinit();
        const src = try frags.source(device, gpa, ks.inkling2_draft);
        defer gpa.free(src);
        k.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer k.lib.deinit();
        k.gemm = try gm.Gemm.init(device, k.lib, &.{ .bf16_nt, .bf16_nn });
        errdefer k.gemm.deinit();
        var made: usize = 0;
        errdefer for (k.pipes[0..made]) |p| p.deinit();
        inline for (kernel_names, 0..) |name, i| {
            k.pipes[i] = try mtl.Pipeline.init(device, k.lib, "ik2d_" ++ name, false);
            made += 1;
        }
        try k.allocate();
        errdefer k.release();
        try k.derive();
        return k;
    }

    pub fn deinit(k: *Core) void {
        k.release();
        for (k.pipes) |p| p.deinit();
        k.gemm.deinit();
        k.lib.deinit();
        k.w.deinit();
        k.gpa.destroy(k);
    }

    fn release(k: *Core) void {
        for (k.s) |b| b.deinit();
        k.alpha.deinit();
        k.inv.deinit();
        k.fold_w.deinit();
        k.fold_b.deinit();
        if (k.staging) |b| b.deinit();
        k.staging = null;
    }

    /// The sliding layers' ring slots: the window and a block past it, whole 128-position runs.
    pub fn ring(m: config.Mtp) u32 {
        return std.mem.alignForward(u32, m.window + m.block, 128);
    }

    /// Slots layer `l`'s cache holds for a stream: a ring, or the whole capacity for a full-attention layer.
    pub fn slots(k: *const Core, l: usize) u32 {
        return if (k.m.local[l]) ring(k.m) else k.capacity;
    }

    pub fn laneBytes(m: config.Mtp, hidden: u32, capacity: u32) u64 {
        const kvw: u64 = @as(u64, m.kv_heads) * head_dim;
        var total: u64 = 0;
        for (0..m.layers) |l| total += 2 * 2 * kvw * (if (m.local[l]) ring(m) else capacity);
        return total + @as(u64, hidden) * 2 + @as(u64, m.block) * @sizeOf(u32) + @as(u64, pending_rows) * m.taps_n * hidden * 2;
    }

    /// Device bytes of everything but the streams' lanes.
    pub fn bytes(k: *const Core) u64 {
        var total = k.w.bytes() + k.alpha.length() + k.inv.length() + k.fold_w.length() + k.fold_b.length();
        for (k.s) |b| total += b.length();
        if (k.staging) |b| total += b.length();
        return total;
    }

    fn sizes(k: *const Core) [scratch_names.len]usize {
        const R: usize = wt.max_block;
        const D: usize = k.hidden;
        const V: usize = k.vocab;
        const qw: usize = @as(usize, k.m.q_heads) * head_dim;
        const kvw: usize = @as(usize, k.m.kv_heads) * head_dim;
        const nbest = (V + 7) / 8;
        const B: usize = k.batch;
        const part: usize = @as(usize, max_splits) * k.m.q_heads * R * B;
        const A: usize = absorb_rows;
        const RB = R * B;
        return .{ RB * D * 2, RB * D * 2, RB * D * 2, RB * qw * 2, RB * qw * 2, RB * qw * 2, RB * kvw * 2, RB * kvw * 2, RB * kvw * 2, part * 4, part * 4, part * head_dim * 4, RB * k.m.inter * 2, RB * k.m.inter * 2, RB * k.m.inter * 2, (RB - 1) * V * 2, B * 3 * k.m.rank * 2, B * k.m.rank * 2, B * k.m.rank * 2, B * nbest * 4, B * nbest * 4, A * D * 2, A * kvw * 2, A * kvw * 2 };
    }

    fn allocate(k: *Core) !void {
        var made: usize = 0;
        errdefer for (k.s[0..made]) |b| b.deinit();
        for (k.sizes(), 0..) |n, i| {
            k.s[i] = try k.device.buffer(@max(n, 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        const taps: usize = k.m.taps_n;
        k.alpha = try k.device.buffer((k.m.layers + 1) * taps * 2, mtl.ResourceOptions.shared);
        errdefer k.alpha.deinit();
        k.inv = try k.device.buffer(head_dim / 2 * 4, mtl.ResourceOptions.shared);
        errdefer k.inv.deinit();
        k.fold_w = try k.device.buffer(@as(usize, k.vocab) * k.m.rank * 2, mtl.ResourceOptions.shared);
        errdefer k.fold_w.deinit();
        k.fold_b = try k.device.buffer(@as(usize, k.vocab) * 2, mtl.ResourceOptions.shared);
    }

    pub fn buf(k: *const Core, comptime name: []const u8) mtl.Buffer {
        inline for (scratch_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.s[i];
        @compileError("no draft buffer " ++ name);
    }

    fn r(k: *const Core, comptime name: []const u8) Ref {
        return .{ .buf = k.buf(name) };
    }

    fn pipe(k: *const Core, kern: Kern) mtl.Pipeline {
        return k.pipes[@backingInt(kern)];
    }

    /// The load-time tables: alpha = bf16(softmax(fuse_weight)) a row, fold_w and fold_b (unembed x hidden_out).
    fn derive(k: *Core) !void {
        const inv = k.inv.slice(f32, head_dim / 2);
        for (inv, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, @as(f64, k.m.rope_theta), -2.0 * @as(f64, @floatFromInt(i)) / head_dim));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = k.queue.commandBuffer();
        const e = cb.compute(.serial);
        const D = k.hidden;
        k.bind(e, .alpha, &.{ k.w.fuse, .{ .buf = k.alpha } }, [2]u32{ k.m.layers + 1, k.m.taps_n });
        e.dispatchThreads(mtl.Size.of(k.m.layers + 1, 1, 1), mtl.Size.of(64, 1, 1));
        const U = Ref{ .buf = k.target.unembed.buffer, .off = k.target.unembed.offset };
        k.gemm.run(e, .bf16_nn, U, k.w.hidden_out_w, .{ .buf = k.fold_w }, .{ .m = k.vocab, .n = k.m.rank, .k = D, .lda = D, .ldb = k.m.rank, .ldd = k.m.rank });
        k.bind(e, .fold_b, &.{ U, k.w.hidden_out_b, .{ .buf = k.fold_b } }, Fold{ .vocab = k.vocab, .width = D });
        e.dispatchGroups(mtl.Size.of((k.vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
    }

    pub fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("inkling2 draft: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }

    fn bind(k: *const Core, e: mtl.ComputeEncoder, kern: Kern, refs: []const Ref, value: anytype) void {
        e.setPipeline(k.pipe(kern));
        for (refs, 0..) |x, i| e.setBuffer(x.buf, x.off, i);
        e.setValue(value, refs.len);
    }

    fn rowsDispatch(e: mtl.ComputeEncoder, n: u32) void {
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// A lane whose full-attention layer holds `global_slots` positions (the warm-up's tiny one; streams: `capacity`).
    pub fn newLane(k: *Core, global_slots: u32) !*Lane {
        const l = try k.gpa.create(Lane);
        errdefer k.gpa.destroy(l);
        const kvw: usize = @as(usize, k.m.kv_heads) * head_dim;
        l.* = .{ .k = undefined, .v = undefined, .f = try k.device.buffer(@as(usize, k.hidden) * 2, mtl.ResourceOptions.shared), .seq = undefined, .pend = undefined };
        @memset(l.f.slice(u8, @as(usize, k.hidden) * 2), 0);
        errdefer l.f.deinit();
        l.seq = try k.device.buffer(@as(usize, k.m.block) * @sizeOf(u32), mtl.ResourceOptions.shared);
        errdefer l.seq.deinit();
        l.pend = try k.device.buffer(@as(usize, pending_rows) * k.m.taps_n * k.hidden * 2, mtl.ResourceOptions.shared);
        errdefer l.pend.deinit();
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..k.m.layers) |i| {
            const bytes_per = @as(usize, if (k.m.local[i]) ring(k.m) else global_slots) * kvw * 2;
            l.k[i] = try k.device.buffer(bytes_per, mtl.ResourceOptions.private);
            l.v[i] = k.device.buffer(bytes_per, mtl.ResourceOptions.private) catch |e| {
                l.k[i].deinit();
                return e;
            };
            made += 1;
        }
        return l;
    }

    pub fn dropLane(k: *Core, l: *Lane) void {
        for (0..k.m.layers) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.f.deinit();
        l.seq.deinit();
        l.pend.deinit();
        k.gpa.destroy(l);
    }

    /// One lane's committed rows: bf16 taps [n, taps * hidden] at `feats` become context positions [start, start + n).
    pub const Piece = struct { l: *Lane, feats: Ref, start: u64, n: u32 };

    /// Every piece's rows (at most `absorb_rows` together) in one pass: each projection reads its weights once.
    pub fn absorb(k: *const Core, e: mtl.ComputeEncoder, pieces: []const Piece) !void {
        var total: u64 = 0;
        for (pieces) |p| {
            if (p.n == 0) return error.DraftAbsorbRows;
            if (p.start + p.n > k.capacity) return error.DraftContextFull;
            total += p.n;
        }
        if (total > absorb_rows) return error.DraftAbsorbRows;
        const D = k.hidden;
        const taps = k.m.taps_n;
        const pitch: u32 = taps * D;
        const kvw: u32 = k.m.kv_heads * head_dim;
        for (0..k.m.layers) |li| {
            const L = &k.w.layers[li];
            const slots_n = k.slots(li);
            var rows: u32 = 0;
            for (pieces) |p| {
                // a sliding layer reads the last window and a block: older rows of a long pass are overwritten anyway
                const rows_n: u32 = if (k.m.local[li]) @min(p.n, slots_n) else p.n;
                const first: u32 = p.n - rows_n;
                k.bind(e, .mix_rms, &.{ p.feats.at(@as(usize, first) * pitch * 2), .{ .buf = k.alpha, .off = li * taps * 2 }, L.ctx_norm, k.r("cbuf").at(@as(usize, rows) * D * 2) }, Rows{ .rows = rows_n, .width = D, .eps = k.eps, .pitch = pitch, .taps = taps });
                rowsDispatch(e, rows_n);
                rows += rows_n;
            }
            k.gemm.run(e, .bf16_nt, k.r("cbuf"), L.k_ctx, k.r("kcr"), .{ .m = rows, .n = kvw, .k = D, .lda = D, .ldb = D, .ldd = kvw });
            k.gemm.run(e, .bf16_nt, k.r("cbuf"), L.v_ctx, k.r("vcr"), .{ .m = rows, .n = kvw, .k = D, .lda = D, .ldb = D, .ldd = kvw });
            rows = 0;
            for (pieces) |p| {
                const rows_n: u32 = if (k.m.local[li]) @min(p.n, slots_n) else p.n;
                const at = @as(usize, rows) * kvw * 2;
                k.bind(e, .ctx_store, &.{ k.r("kcr").at(at), k.r("vcr").at(at), L.k_norm_ctx, .{ .buf = k.inv }, .{ .buf = p.l.k[li] }, .{ .buf = p.l.v[li] } }, Ctx{ .rows = rows_n, .pos0 = @intCast(p.start + p.n - rows_n), .ring = slots_n, .groups = k.m.kv_heads, .eps = k.eps });
                e.dispatchGroups(mtl.Size.of(k.m.kv_heads, rows_n, 1), mtl.Size.of(head_dim, 1, 1));
                rows += rows_n;
            }
        }
        const last = k.m.layers;
        for (pieces) |p| {
            k.bind(e, .mix_rms, &.{ p.feats.at(@as(usize, p.n - 1) * pitch * 2), .{ .buf = k.alpha, .off = last * taps * 2 }, k.w.markov_norm, .{ .buf = p.l.f } }, Rows{ .rows = 1, .width = D, .eps = k.eps, .pitch = pitch, .taps = taps });
            rowsDispatch(e, 1);
        }
    }

    /// The blocks of `ls` (at most `batch`) at `end` ..: seq[0] the anchor, the Markov steps fill seq[1 .. 1 + steps].
    pub fn blocks(k: *const Core, e: mtl.ComputeEncoder, ls: []const *const Lane, steps: u32) !void {
        if (ls.len == 0 or ls.len > k.batch) return error.DraftBatch;
        for (ls) |l| if (l.end + k.m.block > std.math.maxInt(u32)) return error.DraftContextFull;
        const m = k.m;
        const n: u32 = m.block;
        const lanes_n: u32 = @intCast(ls.len);
        const rows: u32 = lanes_n * n;
        const D = k.hidden;
        const qw: u32 = m.q_heads * head_dim;
        const kvw: u32 = m.kv_heads * head_dim;
        const V = k.vocab;
        const per_group = m.q_heads / m.kv_heads;
        const rows_c = Rows{ .rows = rows, .width = D, .eps = k.eps };
        const part: usize = @as(usize, max_splits) * m.q_heads * wt.max_block;
        const embed: [2]Ref = .{ .{ .buf = k.target.embed.buffer, .off = k.target.embed.offset }, .{ .buf = k.target.embed_norm.buffer, .off = k.target.embed_norm.offset } };
        for (ls, 0..) |l, i| {
            k.bind(e, .embed_block, &.{ .{ .buf = l.seq }, embed[0], embed[1], k.w.mask, k.r("resid").at(rowOff(i, n, D)) }, Rows{ .rows = n, .width = D, .eps = k.eps });
            rowsDispatch(e, n);
        }
        k.bind(e, .rms, &.{ k.r("resid"), k.w.layers[0].in_norm, k.r("x") }, rows_c);
        rowsDispatch(e, rows);
        for (0..m.layers) |li| {
            const L = &k.w.layers[li];
            k.gemm.run(e, .bf16_nt, k.r("x"), L.q, k.r("qraw"), .{ .m = rows, .n = qw, .k = D, .lda = D, .ldb = D, .ldd = qw });
            k.gemm.run(e, .bf16_nt, k.r("x"), L.k, k.r("kraw"), .{ .m = rows, .n = kvw, .k = D, .lda = D, .ldb = D, .ldd = kvw });
            k.gemm.run(e, .bf16_nt, k.r("x"), L.v, k.r("vb"), .{ .m = rows, .n = kvw, .k = D, .lda = D, .ldb = D, .ldd = kvw });
            for (ls, 0..) |l, i| {
                const anchor = l.end;
                const prep = Prep{ .rows = n, .pos0 = @intCast(anchor), .heads_q = m.q_heads, .heads_kv = m.kv_heads, .eps = k.eps };
                k.bind(e, .qk_prep, &.{ k.r("qraw").at(rowOff(i, n, qw)), k.r("kraw").at(rowOff(i, n, kvw)), L.q_norm, L.k_norm, .{ .buf = k.inv }, k.r("q").at(rowOff(i, n, qw)), k.r("kb").at(rowOff(i, n, kvw)) }, prep);
                e.dispatchGroups(mtl.Size.of(m.q_heads + m.kv_heads, n, 1), mtl.Size.of(head_dim, 1, 1));
                // keys: a sliding layer's last window or the full layer's context before the anchor, then the block's
                const floor: u64 = if (m.local[li]) @max(l.lfrom, anchor -| m.window) else l.from;
                const lo: u64 = @min(floor, anchor);
                const nctx: u32 = @intCast(anchor - lo);
                const total: u32 = nctx + n;
                const splits = std.math.clamp((total + min_per_split - 1) / min_per_split, 1, max_splits);
                const a = Attn{ .rows = n, .heads = m.q_heads, .groups = m.kv_heads, .lo = @intCast(lo), .nctx = nctx, .ring = k.slots(li), .splits = splits, .per = (total + splits - 1) / splits, .scale = 1.0 / @sqrt(@as(f32, head_dim)) };
                const pm = k.r("pm").at(i * part * 4);
                const pl = k.r("pl").at(i * part * 4);
                const po = k.r("po").at(i * part * head_dim * 4);
                k.bind(e, .attn_split, &.{ k.r("q").at(rowOff(i, n, qw)), .{ .buf = l.k[li] }, .{ .buf = l.v[li] }, k.r("kb").at(rowOff(i, n, kvw)), k.r("vb").at(rowOff(i, n, kvw)), pm, pl, po }, a);
                e.dispatchGroups(mtl.Size.of(m.kv_heads, splits, 1), mtl.Size.of(32 * per_group, 1, 1));
                k.bind(e, .attn_merge, &.{ pm, pl, po, k.r("attn").at(rowOff(i, n, qw)) }, a);
                e.dispatchGroups(mtl.Size.of(m.q_heads, n, 1), mtl.Size.of(head_dim, 1, 1));
            }
            k.gemm.run(e, .bf16_nt, k.r("attn"), L.o, k.r("y"), .{ .m = rows, .n = D, .k = qw, .lda = qw, .ldb = qw, .ldd = D });
            k.bind(e, .add_rms, &.{ k.r("resid"), k.r("y"), L.post_norm, k.r("x") }, rows_c);
            rowsDispatch(e, rows);
            k.gemm.run(e, .bf16_nt, k.r("x"), L.gate, k.r("g"), .{ .m = rows, .n = m.inter, .k = D, .lda = D, .ldb = D, .ldd = m.inter });
            k.gemm.run(e, .bf16_nt, k.r("x"), L.up, k.r("u"), .{ .m = rows, .n = m.inter, .k = D, .lda = D, .ldb = D, .ldd = m.inter });
            k.bind(e, .swiglu, &.{ k.r("g"), k.r("u"), k.r("act") }, m.inter);
            e.dispatchThreads(mtl.Size.of(m.inter, rows, 1), mtl.Size.of(256, 1, 1));
            k.gemm.run(e, .bf16_nt, k.r("act"), L.down, k.r("y"), .{ .m = rows, .n = D, .k = m.inter, .lda = m.inter, .ldb = m.inter, .ldd = D });
            if (li + 1 < m.layers) {
                k.bind(e, .add_rms, &.{ k.r("resid"), k.r("y"), k.w.layers[li + 1].in_norm, k.r("x") }, rows_c);
                rowsDispatch(e, rows);
            }
        }
        // the final norm of rows 1 .. steps only, lane after lane: just those meet the unembedding (i * used + j)
        const used: u32 = @min(steps, n - 1);
        if (used == 0) return;
        const last_rows = Rows{ .rows = used, .width = D, .eps = k.eps };
        for (0..ls.len) |i| {
            const from = rowOff(i, n, D) + @as(usize, D) * 2;
            k.bind(e, .add_rms, &.{ k.r("resid").at(from), k.r("y").at(from), k.w.norm, k.r("x").at(i * used * D * 2) }, last_rows);
            rowsDispatch(e, used);
        }
        const U = Ref{ .buf = k.target.unembed.buffer, .off = k.target.unembed.offset };
        k.gemm.run(e, .bf16_nt, k.r("x"), U, k.r("base"), .{ .m = lanes_n * used, .n = V, .k = D, .lda = D, .ldb = D, .ldd = V });
        const nbest: u32 = (V + 7) / 8;
        for (0..used) |i| {
            const step: u32 = @intCast(i);
            const rnn = Rnn{ .width = D, .rank = m.rank, .step = step };
            for (ls, 0..) |l, j| {
                const at_rank = j * m.rank * 2;
                k.bind(e, .rnn, &.{ k.r("state").at(at_rank), k.w.w1, .{ .buf = l.seq }, k.r("x").at((j * used + i) * D * 2), .{ .buf = l.f }, k.w.rnn_w, k.w.rnn_b, k.r("raw").at(3 * at_rank) }, rnn);
                e.dispatchGroups(mtl.Size.of(3 * m.rank, 1, 1), mtl.Size.of(256, 1, 1));
                k.bind(e, .cell, &.{ k.r("raw").at(3 * at_rank), k.r("state").at(at_rank), k.r("latent").at(at_rank) }, rnn);
                e.dispatchThreads(mtl.Size.of(m.rank, 1, 1), mtl.Size.of(64, 1, 1));
            }
            // every lane's winners in one pass over fold_w
            const logits = Step{ .vocab = V, .unpadded = k.unpadded, .rank = m.rank, .cap = k.cap, .lanes = lanes_n, .base_stride = used * V, .groups = nbest };
            k.bind(e, .step_logits, &.{ k.r("base").at(i * V * 2), k.r("latent"), .{ .buf = k.fold_w }, .{ .buf = k.fold_b }, k.r("best"), k.r("best_id") }, logits);
            e.dispatchGroups(mtl.Size.of(nbest, 1, 1), mtl.Size.of(256, 1, 1));
            for (ls, 0..) |l, j| {
                const at_best = j * nbest * 4;
                k.bind(e, .pick, &.{ k.r("best").at(at_best), k.r("best_id").at(at_best), .{ .buf = l.seq } }, [2]u32{ nbest, step });
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
            }
        }
    }

    /// Bytes before lane `i`'s rows in a bf16 [lanes * block, width] scratch.
    fn rowOff(i: usize, n: u32, width: u32) usize {
        return i * n * width * 2;
    }
};
