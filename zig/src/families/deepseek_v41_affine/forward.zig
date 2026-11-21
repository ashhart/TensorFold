//! One stream's rows through the backbone in one command buffer, with the caches DeepSeek's decode keeps.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const am = @import("../../core/affine_mm.zig");
const idx = @import("index.zig");
const ffn = @import("moe.zig");
const dspark = @import("draft.zig");
const layer = @import("layer.zig");
const kern = @import("kernels.zig");
const ahead = @import("ahead.zig");
const HcOverlap = @import("hc_overlap_live.zig");
pub const HcJob = HcOverlap.Job;
const Buffer = mtl.Buffer;

pub const Head = enum { last, all, greedy, none };

/// One DSpark round (draft.zig).
pub const draft = dspark.draft;
pub const draftPrefix = dspark.draftPrefix;
pub const draftScheduled = dspark.draftScheduled;

/// Per-layer outputs a check reads back (the streams after each block), or null.
pub const Dump = struct { layers: []Buffer };

/// A prompt's image spans: each position's code (-1 text, an aligner row, -2/-3/-4 start/end/break), their rows.
pub const Vis = struct {
    codes: []const i32,
    rows: Buffer,

    pub fn code(v: ?*const Vis, pos: usize) i32 {
        const x = v orelse return -1;
        return if (pos < x.codes.len) x.codes[pos] else -1;
    }
};

const HcArgs = extern struct { eps: f32, hc_eps: f32, iters: i32 };
const RowsArgs = extern struct { n: i32, x_stride: i32, y_stride: i32, eps: f32 };
pub const GemvArgs = extern struct { rows: i32, n: i32, k: i32, group_n: i32, x_stride: i32, y_stride: i32 };
const RopeArgs = extern struct { heads: i32, stride: i32, offset: i32, inverse: i32 };
const CopyArgs = extern struct { width: i32, ring: i32 };
const ClippedMergeArgs = extern struct { written: i32, full: i32, first_pos: i32, window: i32, picks: i32 };
const AttnArgs = extern struct { first_pos: i32, ring: i32, window: i32, picks: i32, scale: f32, window_end: i32 = -1, window_floor: i32 = 0 };
/// Rows from which a call's projections take the tensor units (core/affine_mm), on GPUs that have them.
pub const nax_rows = 16;
/// A buffer and byte offset as core/affine_mm binds them.
pub const Ref = struct { buf: Buffer, off: usize = 0 };

pub fn matrix(w: row.Weights) struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 } {
    return .{ .w = .{ .buf = w.w, .off = w.w_off }, .s = .{ .buf = w.scales, .off = w.s_off }, .b = .{ .buf = w.biases, .off = w.b_off }, .n = @intCast(w.n), .k = @intCast(w.k) };
}

pub const gemv_sgs: usize = 8;

/// Heads a split-attention threadgroup takes (v41_attend.metal's HG).
const split_heads = 16;
/// The next dispatch runs alongside the one before it (`Enc.with`): no barrier between them.
var joined = false;
/// A sublayer's branch waits to be expanded: the next mixing boundary does it; other stream readers flush it.
pub var pending_expand = false;
/// The pending branch is the MoE's, still to be combined from its routed (this many a row) and shared rows.
pub var pending_topk: i32 = 0;
pub const index_nax = true;

/// The chunk after this call's, its engram rows gathered on the host while this one runs on the GPU (`runChunk`).
pub const Next = struct { tokens: []const u32, vis: ?*const Vis = null };

pub const Enc = struct {
    ep: *mtl.ComputeEncoder,
    cbp: *mtl.CommandBuffer,
    m: *Model,
    s: *st.Scratch,
    rows: usize,
    base: usize, // the first row's position (a stream's view; meaningless across streams)
    route_trace_layer: i32 = -1,
    short_index: bool = false,
    wo_decode: bool = false,
    // Pruned prompt layers compact their activations. Positions, picks and

    // candidate blocks use the original row coordinates for the whole chunk.
    token_skip: usize = 0,
    window_floor: usize = 0,
    expand_skip: usize = 0,
    overlap_hc: bool = false,
    r0: usize = 0, // a stream's view: its first scratch row, where its row-indexed buffers bind
    units: bool = false, // a lone stream's prompt chunk: the tensor units (shared forwards keep the row kernels)
    segs: []const Seg = &.{}, // the streams of the forward, in scratch row order

    /// The view of one stream's rows: its position, its scratch rows.
    pub fn of(self: Enc, g: Seg) Enc {
        if (self.token_skip != 0) {
            std.debug.assert(self.segs.len == 1 and g.r0 == 0);
            return self;
        }
        var v = self;
        v.rows = g.tokens.len;
        v.base = g.cache.len;
        v.r0 = g.r0;
        return v;
    }

    /// `b` bound at this view's first row, `row_bytes` a row.
    pub fn at(self: Enc, b: Buffer, row_bytes: usize, slot: usize) void {
        const source_row = self.r0 + if (b.id == self.s.pos.id) self.token_skip else 0;
        self.buf(b, source_row * row_bytes, slot);
    }

    /// The tensor-unit pipelines for `w` when this call is a lone stream's prompt chunk.
    pub fn nax(self: Enc, w: row.Weights) ?am.Pipes {
        if (!self.units) return null;
        return self.m.kernels.naxPipes(.{ .bits = w.bits, .group = @intCast(w.group) });
    }

    pub fn fence(self: Enc) void {
        if (joined) {
            joined = false;
            return;
        }
        self.ep.*.barrier();
    }

    pub fn with(self: Enc) void {
        _ = self;
        joined = true;
    }
    pub fn pipe(self: Enc, comptime key: []const u8) void {
        self.ep.*.setPipeline(self.m.kernels.get(key));
    }

    pub fn buf(self: Enc, b: Buffer, off: usize, slot: usize) void {
        self.ep.*.setBuffer(b, off, slot);
    }
    pub fn tensor(self: Enc, t: wts.Tensor, slot: usize) void {
        self.ep.*.setBuffer(t.buffer, t.offset, slot);
    }
    pub fn args(self: Enc, v: anytype, slot: usize) void {
        self.ep.*.setBytes(std.mem.asBytes(&v), slot);
    }
    pub fn threads(self: Enc, grid: [3]usize, group: [3]usize) void {
        self.fence();
        self.ep.*.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    pub fn groups(self: Enc, grid: [3]usize, group: [3]usize) void {
        self.fence();
        self.ep.*.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }

    /// y[rows, n] = x[rows, k] W^T: the tensor units for a prompt chunk's rows, else the row kernels.
    pub fn linear(self: Enc, w: wts.Linear, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        if (rows >= nax_rows) if (self.nax(w)) |p| {
            const xr = Ref{ .buf = x, .off = x_off };
            // one beside the projection before it (`with`) keeps its group sums apart: the buffer's second half
            const half = self.s.sums.length() / 2 & ~@as(usize, 15);
            std.debug.assert(!joined or rows * (w.k / w.group) * 4 <= half);
            const sums = Ref{ .buf = self.s.sums, .off = if (joined) half else 0 };
            self.fence();
            am.rowSums(self.ep.*, p, @intCast(w.group), xr, sums, @intCast(rows), @intCast(w.k));
            self.fence();
            am.dense(self.ep.*, p, xr, sums, matrix(w), Ref{ .buf = y, .off = y_off }, @intCast(rows));
            return;
        };
        // a window's rows on the tiles: each weight read once for 8 rows, a row's sums the same at any width
        if (self.m.kernels.tiles) |*t| if (kern.tiled(w)) {
            if (row.simd_bits.call(t, w.bits, w.n, w.k, rows, w.n, rows <= 2, w.n)) |call| {
                const m = row.simd_bits.Matrix{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off };
                self.fence();
                row.simd_bits.encode(self.ep.*, call, m, x, x_off, y, y_off);
                return;
            } else |_| {}
        };
        const call = row.call(&self.m.kernels.rows, w, rows, false) catch unreachable;
        self.ep.*.setPipeline(call.pipeline);
        self.buf(x, x_off, 0);
        self.buf(w.w, w.w_off, 1);
        self.buf(w.scales, w.s_off, 2);
        self.buf(w.biases, w.b_off, 3);
        self.args(call.dims, 4);
        self.buf(y, y_off, 5);
        self.threads(.{ call.groups * call.threads, 1, 1 }, .{ call.threads, 1, 1 });
    }

    /// Slot i takes expert ids[i] on input row i / repeat.
    pub fn indexed(self: Enc, x: row.Experts, slots: usize, in: Buffer, ids: Buffer, out: Buffer) void {
        const call = row.callIndexed(&self.m.kernels.rows, x, slots, false) catch unreachable;
        self.ep.*.setPipeline(call.pipeline);
        self.buf(in, 0, 0);
        self.buf(x.one.w, x.one.w_off, 1);
        self.buf(x.one.scales, x.one.s_off, 2);
        self.buf(x.one.biases, x.one.b_off, 3);
        self.args(call.dims, 4);
        self.buf(out, 0, 5);
        self.buf(ids, 0, 6);
        self.threads(.{ call.groups * call.threads, call.slots, 1 }, .{ call.threads, 1, 1 });
    }

    /// s.oa [rows, 8 x 1024] = each group's 4,096 of s.o [rows, 64 x 512] times its block of wo_a.
    pub fn woA(self: Enc, w: wts.WoA, rows: usize) void {
        const s = self.s;
        switch (w) {
            .bf16 => |t| self.gemv(false, t, s.o, 0, c.heads * c.head_dim, s.oa, 0, rows, c.o_groups * c.o_rank, c.group_in, c.o_rank),
            .affine => |x| {
                if (self.wo_decode) if (self.m.woa_batch) |p| {
                    if (p.forward(self, x, rows)) return;
                };
                if (rows >= nax_rows) if (self.nax(x.one)) |p| {
                    const g: i32 = c.o_groups;
                    const xr = Ref{ .buf = s.o };
                    const sums = Ref{ .buf = s.sums };
                    self.fence();
                    am.rowSums(self.ep.*, p, @intCast(x.one.group), xr, sums, @intCast(rows * c.o_groups), c.group_in);
                    self.fence();
                    am.denseBatch(self.ep.*, p, xr, sums, matrix(x.one), Ref{ .buf = s.oa }, @intCast(rows), c.o_groups, .{ .x_row = g * c.group_in, .y_row = g * c.o_rank, .sums_row = g, .x_batch = c.group_in, .y_batch = c.o_rank, .sums_batch = 1, .w_batch = c.o_rank });
                    return;
                };
                self.indexed(x, rows * c.o_groups, s.o, s.wo_ids, s.oa);
            },
        }
    }

    /// y = x W^T with a bf16 W [n, k]; `group_n` outputs share one k-wide slice of each x row.
    pub fn gemv(self: Enc, comptime out_f32: bool, w: wts.Tensor, x: Buffer, x_off: usize, x_stride: usize, y: Buffer, y_off: usize, rows: usize, n: usize, k: usize, group_n: usize) void {
        if (out_f32) self.pipe("ds_gemv_bf_f32") else self.pipe("ds_gemv_bf_bf");
        self.buf(x, x_off, 0);
        self.tensor(w, 1);
        self.buf(y, y_off, 2);
        self.args(GemvArgs{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .group_n = @intCast(group_n), .x_stride = @intCast(x_stride), .y_stride = @intCast(n) }, 3);
        self.groups(.{ (n + gemv_sgs - 1) / gemv_sgs, (rows + 7) / 8, 1 }, .{ 32 * gemv_sgs, 1, 1 });
    }

    /// The current rows' RoPE table when it matches the positions and frequency buffer.
    fn trig(self: Enc, freq: Buffer, positions: Buffer) ?Ref {
        const s = self.s;
        if (s.trig_rows == 0 or positions.id != s.pos.id) return null;
        const variant: usize = if (freq.id == self.m.freq_compressed.id) 1 else if (freq.id == self.m.freq_plain.id) 0 else return null;
        return .{ .buf = s.trig, .off = (variant * s.trig_rows + self.r0 + self.token_skip) * 32 * 8 };
    }
    /// Attention over rows' windows and `picks` compressed keys (RoPE in and out, fp8): split by keys, or by row.
    pub fn attend(self: Enc, q: Buffer, ring: Buffer, comp: Buffer, picks_buf: Buffer, picks: usize, sink: wts.Tensor, o: Buffer, window_end: i32, freq: Buffer, ratio: usize) void {
        const s = self.s;
        const aa = AttnArgs{ .first_pos = @intCast(self.base), .ring = st.ring, .window = c.window, .picks = @intCast(picks), .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .window_end = window_end, .window_floor = @intCast(self.window_floor) };
        const qo = c.heads * c.head_dim * 2;
        if (self.units and self.rows >= nax_rows) {
            self.rope(q, self.rows, c.heads, c.head_dim, s.pos, freq, false);
            const tensor_attend = self.m.kernels.attend_nax; // on the tensor units when the GPU has them
            if (tensor_attend) |p| {
                self.ep.*.setPipeline(p);
            } else self.pipe("ds_attend_rows");
            self.buf(q, 0, 0);
            self.buf(ring, 0, 1);
            self.buf(comp, 0, 2);
            self.buf(picks_buf, self.token_skip * picks * @sizeOf(i32), 3);
            self.tensor(sink, 4);
            self.buf(o, 0, 5);
            self.args(aa, 6);
            if (tensor_attend != null) self.groups(.{ c.heads / 16, self.rows, 1 }, .{ 128, 1, 1 }) else self.groups(.{ c.heads / 8, self.rows, 1 }, .{ 512, 1, 1 });
            self.rope(o, self.rows, c.heads, c.head_dim, s.pos, freq, true);
            self.fp8(o, self.rows, c.heads * c.head_dim);
            return;
        }
        const full: usize = (c.window + picks + 15) / 16;
        var chunks = full;
        if (self.m.clip_attention and self.segs.len == 1 and self.rows <= 1 + c.draft_block and ratio > 0 and picks > 0) {
            const latest = self.base + self.rows;
            const visible = latest / ratio;
            if (visible > 0) chunks = (@min(c.window, latest) + @min(picks, visible) + 15) / 16;
        }
        const clipped = chunks < full;
        const small = self.m.attention_eight and self.segs.len == 1 and self.rows <= c.draft_block and
            (self.rows > 1 or chunks <= 24);
        const clip_audit = clipped and self.m.audit_attention_clip;
        const geometry_audit = small and self.m.attention_eight_audit;
        const audit = clip_audit or geometry_audit;
        const tb = self.trig(freq, s.pos);
        if (audit) {
            const reference_chunks = if (clip_audit) full else chunks;
            self.attendSplit(q, ring, comp, picks_buf, aa, reference_chunks, tb, freq, false);
            self.attendMerge(s.attention_ref, 0, sink, comp, aa, reference_chunks, full, tb, freq);
        }
        self.attendSplit(q, ring, comp, picks_buf, aa, chunks, tb, freq, small);
        const stream = self.m.attention_stream_merge and tb != null and picks > 0 and self.segs.len == 1 and
            self.rows <= 1 + c.draft_block;
        const stream_audit = stream and self.m.attention_stream_audit;
        if (stream) {
            if (stream_audit) self.attendMerge(s.attention_ref, 0, sink, comp, aa, chunks, full, tb, freq);
            self.attendStreamMerge(o, self.r0 * qo, sink, comp, aa, chunks, full, tb.?);
        } else {
            self.attendMerge(o, self.r0 * qo, sink, comp, aa, chunks, full, tb, freq);
        }
        if (audit or stream_audit) {
            const values = self.rows * c.heads * c.head_dim;
            self.pipe("ds_attend_clipped_audit");
            self.at(o, qo, 0);
            self.buf(s.attention_ref, 0, 1);
            self.buf(s.attention_differs, 0, 2);
            self.threads(.{ values, 1, 1 }, .{ 256, 1, 1 });
            if (clip_audit) self.segs[0].cache.attention_values_checked += values;
            if (geometry_audit) self.segs[0].cache.attention_eight_checked += values;
            if (stream_audit) self.segs[0].cache.attention_stream_checked += values;
        }
    }

    fn attendSplit(self: Enc, q: Buffer, ring: Buffer, comp: Buffer, picks_buf: Buffer, aa: AttnArgs, chunks: usize, tb: ?Ref, freq: Buffer, small: bool) void {
        const s = self.s;
        const qo = c.heads * c.head_dim * 2;
        if (small) {
            if (tb != null) self.pipe("tf_attend_heads8_t") else self.pipe("tf_attend_heads8");
        } else if (tb != null) self.pipe("ds_attend_split_t") else self.pipe("ds_attend_split");
        self.at(q, qo, 0);
        self.buf(ring, 0, 1);
        self.buf(comp, 0, 2);
        self.at(picks_buf, @as(usize, @intCast(aa.picks)) * 4, 3);
        self.buf(s.attn_po, 0, 4);
        self.buf(s.attn_ml, 0, 5);
        self.args(aa, 6);
        self.at(s.pos, 4, 7);
        if (tb) |t| self.buf(t.buf, t.off, 8) else self.buf(freq, 0, 8);
        const tile_heads: usize = if (small) 8 else split_heads;
        self.groups(.{ c.heads / tile_heads, chunks, self.rows }, .{ tile_heads * 32, 1, 1 });
    }

    fn attendStreamMerge(self: Enc, out: Buffer, offset: usize, sink: wts.Tensor, comp: Buffer, aa: AttnArgs, chunks: usize, full: usize, tb: Ref) void {
        const s = self.s;
        const wide = self.rows == 1 + c.draft_block;
        if (wide) self.pipe("tf_stream_merge_4") else self.pipe("tf_stream_merge_2");
        self.buf(s.attn_po, 0, 0);
        self.buf(s.attn_ml, 0, 1);
        self.tensor(sink, 2);
        self.buf(out, offset, 3);
        self.args(ClippedMergeArgs{ .written = @intCast(chunks), .full = @intCast(full), .first_pos = aa.first_pos, .window = aa.window, .picks = aa.picks }, 4);
        self.at(s.pos, 4, 5);
        self.buf(tb.buf, tb.off, 6);
        self.buf(comp, 0, 7);
        self.groups(.{ c.heads, self.rows, 1 }, .{ if (wide) c.head_dim / 4 else c.head_dim / 2, 1, 1 });
    }

    fn attendMerge(self: Enc, out: Buffer, offset: usize, sink: wts.Tensor, comp: Buffer, aa: AttnArgs, chunks: usize, full: usize, tb: ?Ref, freq: Buffer) void {
        const s = self.s;
        if (chunks < full) {
            if (self.m.tight_attention_merge) {
                if (tb != null) self.pipe("ds_attend_merge_tight_t") else self.pipe("ds_attend_merge_tight");
            } else {
                if (tb != null) self.pipe("ds_attend_merge_clipped_t") else self.pipe("ds_attend_merge_clipped");
            }
            self.buf(comp, 0, 7);
            self.args(ClippedMergeArgs{ .written = @intCast(chunks), .full = @intCast(full), .first_pos = aa.first_pos, .window = aa.window, .picks = aa.picks }, 4);
        } else {
            if (tb != null) self.pipe("ds_attend_merge_t") else self.pipe("ds_attend_merge");
            self.args(@as(i32, @intCast(chunks)), 4);
        }
        self.buf(s.attn_po, 0, 0);
        self.buf(s.attn_ml, 0, 1);
        self.tensor(sink, 2);
        self.buf(out, offset, 3);
        self.at(s.pos, 4, 5);
        if (tb) |t| self.buf(t.buf, t.off, 6) else self.buf(freq, 0, 6);
        self.groups(.{ c.heads, self.rows, 1 }, .{ c.head_dim, 1, 1 });
    }

    pub fn rms(self: Enc, x: Buffer, x_off: usize, w: wts.Tensor, y: Buffer, y_off: usize, rows: usize, n: usize) void {
        self.pipe("ds_rms");
        self.buf(x, x_off, 0);
        self.tensor(w, 1);
        self.buf(y, y_off, 2);
        self.args(RowsArgs{ .n = @intCast(n), .x_stride = @intCast(n), .y_stride = @intCast(n), .eps = c.norm_eps }, 3);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
    }

    pub fn rope(self: Enc, x: Buffer, rows: usize, heads: usize, stride: usize, positions: Buffer, freq: Buffer, inverse: bool) void {
        const tb = self.trig(freq, positions);
        if (tb != null) self.pipe("ds_rope_t") else self.pipe("ds_rope");
        self.at(x, heads * stride * 2, 0);
        self.at(positions, 4, 1);
        if (tb) |t| self.buf(t.buf, t.off, 2) else self.buf(freq, 0, 2);
        self.args(RopeArgs{ .heads = @intCast(heads), .stride = @intCast(stride), .offset = @intCast(stride - c.rope_dim), .inverse = @intFromBool(inverse) }, 3);
        self.threads(.{ c.rope_dim / 2, heads, rows }, .{ c.rope_dim / 2, 1, 1 });
    }

    pub fn fp8(self: Enc, x: Buffer, rows: usize, width: usize) void {
        self.pipe("ds_fp8_qdq");
        self.at(x, width * 2, 0);
        self.threads(.{ rows * width / 32, 1, 1 }, .{ 64, 1, 1 });
    }

    pub fn fp4(self: Enc, x: Buffer, rows: usize, width: usize, e4m3_scale: bool) void {
        self.pipe("ds_fp4_qdq");
        self.at(x, width * 2, 0);
        self.args(@as(i32, @intFromBool(e4m3_scale)), 1);
        self.threads(.{ rows * width / @as(usize, if (e4m3_scale) 16 else 32), 1, 1 }, .{ 64, 1, 1 });
    }

    /// The window keys' tail in place (kv_norm, RoPE, fp8), then into `ring` at their positions when given.
    pub fn kvFinish(self: Enc, kv: Buffer, norm: wts.Tensor, freq: Buffer, ring: ?Buffer) void {
        const tb = self.trig(freq, self.s.pos);
        if (tb != null) self.pipe("ds_kv_finish_t") else self.pipe("ds_kv_finish");
        self.at(kv, c.head_dim * 2, 0);
        self.tensor(norm, 1);
        self.at(self.s.pos, 4, 2);
        if (tb) |t| self.buf(t.buf, t.off, 3) else self.buf(freq, 0, 3);
        if (ring) |r| self.buf(r, 0, 4) else self.at(kv, c.head_dim * 2, 4);
        self.args([2]f32{ c.norm_eps, if (ring != null) @floatFromInt(st.ring) else 0 }, 5);
        self.groups(.{ self.rows, 1, 1 }, .{ 256, 1, 1 });
    }

    pub fn put(self: Enc, src: Buffer, positions: Buffer, dst: Buffer, rows: usize, width: usize, ring: usize) void {
        self.pipe("ds_put_rows");
        self.at(src, width * 2, 0);
        self.at(positions, 4, 1);
        self.buf(dst, 0, 2);
        self.args(CopyArgs{ .width = @intCast(width), .ring = @intCast(ring) }, 3);
        self.threads(.{ width, rows, 1 }, .{ @min(width, 256), 1, 1 });
    }

    pub fn putF32(self: Enc, src: Buffer, positions: Buffer, dst: Buffer, rows: usize, width: usize, ring: usize) void {
        self.pipe("ds_put_rows_f32");
        self.at(src, width * 4, 0);
        self.at(positions, 4, 1);
        self.buf(dst, 0, 2);
        self.args(CopyArgs{ .width = @intCast(width), .ring = @intCast(ring) }, 3);
        self.threads(.{ width, rows, 1 }, .{ @min(width, 256), 1, 1 });
    }

    /// A boundary: (`mixes`) its pre, post and comb, and the streams collapsed with `pre_in`, normed into `normed`.
    pub fn boundary(self: Enc, hc: ?wts.Hc, pre_in: Buffer, pre_out: Buffer, norm: wts.Tensor) void {
        _ = self.boundaryForLayer(hc, pre_in, pre_out, norm, false);
    }

    pub fn boundaryForLayer(self: Enc, hc: ?wts.Hc, pre_in: Buffer, pre_out: Buffer, norm: wts.Tensor, allow: bool) ?HcJob {
        const s = self.s;
        if (hc) |h| {
            if (pending_expand) {
                pending_expand = false;
                const by_rows = self.rows >= nax_rows; // a chunk's rows four to a threadgroup, FN read once for them
                if (by_rows) self.pipe("ds_hc_expand_mix_rows") else self.pipe("ds_hc_expand_mix");
                const row_offset = self.expand_skip;
                std.debug.assert(row_offset == 0 or (by_rows and pending_topk == c.topk));
                self.buf(s.x, row_offset * c.hc * c.hidden * 2, 0);
                self.buf(s.branch, row_offset * c.hidden * 2, 1);
                self.buf(s.post, row_offset * c.hc * 4, 2);
                self.buf(s.comb, row_offset * c.hc * c.hc * 4, 3);
                self.tensor(h.fn_, 4);
                self.buf(s.hc_part, 0, 5);
                self.buf(s.x2, 0, 6);
                self.buf(s.experts, row_offset * c.topk * c.hidden * 2, 7);
                self.buf(s.shared, row_offset * c.hidden * 2, 8);
                self.args([2]i32{ pending_topk, @intCast(self.rows) }, 9);
                pending_topk = 0;
                const grid: [2]usize = if (by_rows) .{ c.hidden / 1024, (self.rows + 3) / 4 } else .{ c.hc * c.hidden / 1024, self.rows };
                self.groups(.{ grid[0], grid[1], 1 }, .{ 256, 1, 1 });
                std.mem.swap(Buffer, &s.x, &s.x2);
            } else {
                self.pipe("ds_hc_mix_part");
                self.buf(s.x, 0, 0);
                self.tensor(h.fn_, 1);
                self.buf(s.hc_part, 0, 2);
                self.groups(.{ c.hc * c.hidden / 1024, self.rows, 1 }, .{ 256, 1, 1 });
            }
        } else self.flush();
        if (allow and self.overlap_hc) if (hc) |h| {
            if (self.m.hc_overlap.?.audit != null) {
                self.encodeBoundary(h, pre_in, s.hc_overlap_ref_pre, norm, s.hc_overlap_ref_post, s.hc_overlap_ref_comb, s.hc_overlap_ref_norm);
                self.with();
            }
            self.encodeBoundary(null, pre_in, pre_out, norm, s.post, s.comb, s.normed);
            return .{ .weights = h, .pre = pre_out };
        };
        self.encodeBoundary(hc, pre_in, pre_out, norm, s.post, s.comb, s.normed);
        return null;
    }

    fn encodeBoundary(self: Enc, hc: ?wts.Hc, pre_in: Buffer, pre_out: Buffer, norm: wts.Tensor, post: Buffer, comb: Buffer, normed: Buffer) void {
        const s = self.s;
        if (hc != null) self.pipe("ds_hc_boundary") else self.pipe("ds_hc_collapse");
        self.buf(s.x, 0, 0);
        if (hc) |h| {
            self.buf(s.hc_part, 0, 1);
            self.tensor(h.base, 2);
            self.tensor(h.scale, 3);
        } else {
            self.buf(s.x, 0, 1);
            self.buf(s.x, 0, 2);
            self.buf(s.x, 0, 3);
        }
        self.buf(pre_in, self.expand_skip * c.hc * 4, 4);
        self.tensor(norm, 5);
        self.args(HcArgs{ .eps = c.norm_eps, .hc_eps = c.hc_eps, .iters = c.sinkhorn }, 6);
        self.buf(pre_out, 0, 7);
        self.buf(post, 0, 8);
        self.buf(comb, 0, 9);
        self.buf(normed, 0, 10);
        self.groups(.{ self.rows, 1, 1 }, .{ 1024, 1, 1 });
    }

    pub fn completeBoundary(self: Enc, pending: ?HcJob) void {
        const job = pending orelse return;
        const p = self.m.hc_overlap.?;
        self.with();
        self.fence();
        p.issue(self.ep.*, self.s, job, self.rows);
        if (p.audit) |audit| {
            self.fence();
            HcOverlap.Kernels.check(audit, self.ep.*, self.s, job.pre, self.rows);
            self.segs[0].cache.hc_overlap_checked += self.rows * (c.hidden + c.hc_mix);
            self.fence();
        }
    }

    /// The sublayer's branch into the streams, deferred to the next boundary (or `flush`).
    pub fn expand(self: Enc) void {
        _ = self;
        pending_expand = true;
    }

    /// A deferred expand, now: before anything but a mixing boundary reads the streams.
    pub fn flush(self: Enc) void {
        if (!pending_expand) return;
        pending_expand = false;
        const s = self.s;
        self.pipe("ds_hc_expand");
        self.buf(s.x, 0, 0);
        self.buf(s.branch, 0, 1);
        self.buf(s.post, 0, 2);
        self.buf(s.comb, 0, 3);
        self.buf(s.experts, 0, 4);
        self.buf(s.shared, 0, 5);
        self.args(pending_topk, 6);
        pending_topk = 0;
        self.threads(.{ c.hidden, self.rows, 1 }, .{ 256, 1, 1 });
    }
};

const execute = @import("execute.zig");
pub const Seg = execute.Seg;
pub const completed = execute.completed;
pub const run = execute.run;
pub const runWith = execute.runWith;
pub const runChunk = execute.runChunk;
pub const many = execute.many;
