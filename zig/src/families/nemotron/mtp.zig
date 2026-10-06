//! The MTP draft head on Metal: kept rows absorbed into a stream's head cache, then one-row steps chained into drafts.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const layers = @import("layers.zig");
const head_block = @import("head_block.zig");

const Buffer = mtl.Buffer;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Token i + 2's draft from hidden row i and token i + 1: drafts never change verified tokens, only their rounds.
pub const Head = struct {
    w: *const wts.Mtp,
    scratch: st.Scratch,
    cat: Buffer, // [rows, 2D]: enorm(embedding) | hnorm(hidden)
    cat_xs: Buffer, // [2D / 64, MP]
    hid: Buffer, // [1, D]: the last step's output, the next chained step's hidden row
    hids: Buffer, // [1 + rows, D]: a tree's step outputs (the root's, then each stepped lane's)
    pass_h: Buffer, // [rows, D]: a tree depth's hidden inputs
    pass_ids: Buffer, // u32 [rows]: stepped lanes' tokens
    c: cfg.Config,
    k: *const kern.Kernels,
    weights: *const wts.Weights,
    pool: *st.Pool,
    topk: bool = false, // each drafted level's best 4 and their probabilities into the stream's top-k
    gpu: ?layers.GpuArgs = null, // the next step's kv write and attention dims, written on the GPU (a GPU-side round)
    keep_hid: bool = true, // copy each step's output row to `hid` (false: the next step reads scratch.x)
    fused: bool = true, // prep through the fused one-row kernels (nemotron_head.metal)
    geometry: usize = 1, // the routed experts' kernel geometry (bit-identical): (2, 2), the fastest at one row
    absorbed: bool = false, // the next step's row was prepped and written by absorb() (its h and q/k/v gathered to row 0)
    pick: bool = true, // false: a step stops at its draft logits (scratch.logits), its caller drafts from them
    block: ?*const head_block.Weights = null, // greedy chains drafted in one pass (row 0 the stock first level)

    pub fn init(device: mtl.Device, c: cfg.Config, k: *const kern.Kernels, weights: *const wts.Weights, pool: *st.Pool, rows: usize, capacity: usize) !Head {
        const mp = 16 * ((rows + 15) / 16);
        return .{
            .w = &weights.mtp.?,
            .scratch = try st.Scratch.init(device, c, rows, capacity),
            .cat = try device.buffer(rows * 2 * c.hidden * 2, opts),
            .cat_xs = try device.buffer(2 * c.hidden / 64 * mp * 4, opts),
            .hid = try device.buffer(c.hidden * 2, opts),
            .hids = try device.buffer((rows + 1) * c.hidden * 2, opts),
            .pass_h = try device.buffer(rows * c.hidden * 2, opts),
            .pass_ids = try device.buffer(@max(rows, 8) * 4, opts),
            .c = c,
            .k = k,
            .weights = weights,
            .pool = pool,
            .fused = k.rows == null,
        };
    }

    pub fn deinit(self: *Head) void {
        self.pass_ids.deinit();
        self.pass_h.deinit();
        self.hids.deinit();
        self.hid.deinit();
        self.cat_xs.deinit();
        self.cat.deinit();
        self.scratch.deinit();
    }

    pub fn forward(self: *Head) fwd.Forward {
        return .{ .c = self.c, .k = self.k, .w = self.weights, .s = &self.scratch, .pool = self.pool, .geometry = self.geometry };
    }

    /// x = eh_proj([enorm(embedding(ids)), hnorm(h)]) into scratch.h, then the attention input norm and its sums.
    pub fn prep(self: *Head, e: *fwd.Enc, rows: usize, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize) void {
        const f = self.forward();
        const d = self.c.hidden;
        const s = &self.scratch;
        const sk = f.splitsAt("eh", self.w.eh, rows);
        if (self.fused and sk > 0) {
            // embedding, both input norms and their sums in one kernel; eh's K slices; their sum, the norm and its sums
            const t = d / 4;
            e.pipe(self.k.get("tf_head_prep"));
            e.buf(ids, ids_off, 0);
            for (0..3) |i| e.buf(self.weights.embed[i].buffer, self.weights.embed[i].offset, 1 + i);
            e.buf(self.w.enorm.buffer, self.w.enorm.offset, 4);
            e.buf(self.w.hnorm.buffer, self.w.hnorm.offset, 5);
            e.buf(h, h_off, 6);
            e.buf(self.cat, 0, 7);
            e.buf(self.cat_xs, 0, 8);
            e.bytes(self.c.eps, 9);
            e.bytes([2]u32{ @intCast(d), @intCast(fwd.Forward.mp(rows)) }, 10);
            e.run(.{ t * rows, 1, 1 }, .{ t, 1, 1 });
            f.coopParts(e, "eh", self.w.eh, self.cat, 0, self.cat_xs, rows, sk);
            e.pipe(self.k.get("tf_head_norm"));
            e.buf(s.part, 0, 0);
            e.buf(self.w.norm.buffer, self.w.norm.offset, 1);
            e.buf(s.h, 0, 2);
            e.buf(s.x, 0, 3);
            e.buf(s.xs, 0, 4);
            e.bytes(self.c.eps, 5);
            e.bytes([4]u32{ @intCast(d), @intCast(fwd.Forward.mp(rows)), @intCast(sk), 0 }, 6);
            e.run(.{ t * rows, 1, 1 }, .{ t, 1, 1 });
            return;
        }
        f.embed(e, rows, ids, ids_off, s.delta);
        f.rms(e, rows, s.delta, 0, d, self.w.enorm, self.cat, 0, 2 * d);
        f.rms(e, rows, h, h_off, d, self.w.hnorm, self.cat, d * 2, 2 * d);
        f.xsum(e, "xsum_5376", 2 * d, self.cat, 0, self.cat_xs, rows);
        f.coop(e, "eh", self.w.eh, self.cat, 0, self.cat_xs, s.h, rows);
        f.rms(e, rows, s.h, 0, d, self.w.norm, s.x, 0, d);
        f.xsum(e, "xsum_2688", d, s.x, 0, s.xs, rows);
    }

    /// Rows' keys and values into the stream's head cache (hidden rows at `h` + `h_off`, next tokens at `ids`).
    pub fn absorb(self: *Head, e: *fwd.Enc, cache: *st.Cache, rows: usize, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize) void {
        if (rows == 0) return;
        const f = self.forward();
        self.prep(e, rows, h, h_off, ids, ids_off);
        f.coop(e, "qkv", self.w.attention.qkv, self.scratch.x, 0, self.scratch.xs, self.scratch.qkv, rows);
        layers.kvWrite(f, e, cache.mtp.?, cache.mtp_len, 0, rows);
        cache.mtp_len += rows;
    }

    /// absorb() with the cache row it writes from read on the GPU (a kv write's 4 u32 at `args_off` in `args`).
    pub fn absorbAt(self: *Head, e: *fwd.Enc, cache: *st.Cache, rows: usize, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize, args: Buffer, args_off: usize) void {
        const f = self.forward();
        self.prep(e, rows, h, h_off, ids, ids_off);
        f.coop(e, "qkv", self.w.attention.qkv, self.scratch.x, 0, self.scratch.xs, self.scratch.qkv, rows);
        layers.kvWriteAt(f, e, cache.mtp.?, args, args_off, 0, rows);
    }

    /// One row through the whole head at the cache's length, its draft token into `out` at `out_off`.
    pub fn step(self: *Head, e: *fwd.Enc, cache: *st.Cache, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize, how: fwd.Draw, out: Buffer, out_off: usize) void {
        self.stepAt(e, cache, h, h_off, ids, ids_off, how, out, out_off, null);
    }

    /// step(), and with `level` the draft logits' best 4 and their probabilities into the stream's top-k at that level.
    pub fn stepAt(self: *Head, e: *fwd.Enc, cache: *st.Cache, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize, how: fwd.Draw, out: Buffer, out_off: usize, level: ?usize) void {
        const f = self.forward();
        const s = &self.scratch;
        const eps = self.c.eps;
        const kvs = [_]layers.Kvs{.{ .kv = cache.mtp.?, .len = cache.mtp_len, .rows = 1, .gpu = self.gpu }};
        if (self.absorbed) {
            self.absorbed = false;
            layers.attend(f, e, self.w.attention, &kvs, 1);
        } else {
            self.prep(e, 1, h, h_off, ids, ids_off);
            layers.attention(f, e, self.w.attention, &kvs, 1);
        }
        f.norm(e, s.delta, self.w.norm2, 1, eps);
        layers.moe(f, e, self.w.moe, self.w.final, 1, eps);
        if (self.keep_hid) {
            e.pipe(self.k.get("tf_copy_rows"));
            e.buf(s.x, 0, 0);
            e.buf(self.hid, 0, 1);
            e.run(.{ self.c.hidden, 1, 1 }, .{ 256, 1, 1 });
        }
        f.coop(e, "draft", self.w.draft, s.x, 0, s.xs, s.logits, 1);
        if (!self.pick) return;
        f.pick(e, 1, self.w.vocab, self.w.ids, how, out, out_off);
        if (level) |l| {
            e.alongside();
            self.topRows(e, cache, 1, l);
        }
    }

    /// The draft logits' best 4 and their chances for `rows` rows into the stream's top-k from slot `slot`.
    pub fn topRows(self: *Head, e: *fwd.Enc, cache: *st.Cache, rows: usize, slot: usize) void {
        e.pipe(self.k.get("tf_topk_probs"));
        e.buf(self.scratch.logits, 0, 0);
        e.buf(self.w.ids, 0, 1);
        e.buf(cache.topk, slot * 16, 2);
        e.buf(cache.topk, st.max_levels * 16 + slot * 16, 3);
        e.bytes(@as(u32, @intCast(self.w.vocab)), 4);
        e.run(.{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 });
    }

    /// `depth` drafts into `out`: the first step's cache row stays, the chained steps' rows go after the chain.
    pub fn chain(self: *Head, e: *fwd.Enc, cache: *st.Cache, depth: usize, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize, how: fwd.Draw, out: Buffer, out_off: usize) void {
        cache.levels = 0;
        if (depth == 0) return self.absorb(e, cache, 1, h, h_off, ids, ids_off);
        if (self.block) |bw| if (how == .greedy and depth > 1) return self.blockPass(e, cache, @min(depth, bw.lanes + 1), bw, h, h_off, ids, ids_off, out, out_off);
        const kept = cache.mtp_len + 1;
        const top = self.topk and depth <= st.max_levels;
        self.stepAt(e, cache, h, h_off, ids, ids_off, at(how, 0), out, out_off, if (top) 0 else null);
        for (1..depth) |j| {
            cache.mtp_len += 1;
            self.stepAt(e, cache, self.hid, 0, out, out_off + (j - 1) * 4, at(how, j), out, out_off + j * 4, if (top) j else null);
        }
        cache.levels = if (top) depth else 0;
        cache.trunk = std.simd.iota(u8, st.max_levels);
        cache.mtp_len = kept;
    }

    /// `rows` drafts in one pass: row 0 is level 0's row (same kernels, row-exact), rows 1.. the block's placeholder lanes.
    pub fn blockPass(self: *Head, e: *fwd.Enc, cache: *st.Cache, rows: usize, bw: *const head_block.Weights, h: Buffer, h_off: usize, ids: Buffer, ids_off: usize, out: Buffer, out_off: usize) void {
        const f = self.forward();
        const s = &self.scratch;
        const d = self.c.hidden;
        const kept = cache.mtp_len + 1;
        self.prep(e, 1, h, h_off, ids, ids_off);
        e.pipe(self.k.get("tf_copy_rows"));
        e.buf(bw.rows, 0, 0);
        e.buf(s.h, d * 2, 1);
        e.run(.{ d, rows - 1, 1 }, .{ 256, 1, 1 });
        f.rms(e, rows, s.h, 0, d, self.w.norm, s.x, 0, d);
        f.xsum(e, "xsum_2688", d, s.x, 0, s.xs, rows);
        const kvs = [_]layers.Kvs{.{ .kv = cache.mtp.?, .len = cache.mtp_len, .rows = rows }};
        layers.attention(f, e, self.w.attention, &kvs, rows);
        f.norm(e, s.delta, self.w.norm2, rows, self.c.eps);
        layers.moe(f, e, self.w.moe, self.w.final, rows, self.c.eps);
        f.coop(e, "draft", self.w.draft, s.x, 0, s.xs, s.logits, rows);
        f.pick(e, rows, self.w.vocab, self.w.ids, .greedy, out, out_off);
        cache.trunk = std.simd.iota(u8, st.max_levels);
        cache.mtp_len = kept; // row 0 is the next position's row; the placeholder rows' keys and values drop
    }

    /// Draft j's draw: the stream's sampler keyed at the chain's first position + j.
    fn at(how: fwd.Draw, j: usize) fwd.Draw {
        return switch (how) {
            .greedy => .greedy,
            .sampled => |d| .{ .sampled = .{ .sampling = d.sampling, .position = d.position + j } },
        };
    }
};
