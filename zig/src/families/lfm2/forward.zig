//! One command buffer a call: segment-local conv state and causal attention, every row's arithmetic its own.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const tile = @import("../../core/tile_qmm.zig");
const Buffer = mtl.Buffer;
pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { last, all };
/// Rows from which a call takes the tiles (the twin's bits, each weight read once for every row).
const tile_rows = 33;

const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32 };
const AttnArgs = extern struct { len: u32, cap: u32, scale: f32, subs: u32 };

const Encoder = struct {
    e: mtl.ComputeEncoder,
    m: *Model,
    s: *st.Scratch,
    /// Prompt calls and windows of tile_rows or more take the tiles; fewer decode rows the twin (the same bits).
    prompt: bool,

    fn pipe(self: Encoder, comptime key: []const u8) void {
        self.e.setPipeline(self.m.kernels.get(key));
    }
    fn tensor(self: Encoder, t: wts.Tensor, slot: usize) void {
        self.e.setBuffer(t.buffer, t.offset, slot);
    }
    fn threads(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    fn groups(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    fn weights(self: Encoder, w: row.Weights) void {
        self.e.setBuffer(w.w, w.w_off, 1);
        self.e.setBuffer(w.scales, w.s_off, 2);
        self.e.setBuffer(w.biases, w.b_off, 3);
    }
    /// y[rows] = x[rows] W^T: the prompt tiles (on x's group sums), or their twin at decode and window rows.
    fn projection(self: Encoder, w: row.Weights, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        const t = &self.m.kernels.tiles;
        if (self.prompt or !self.m.kernels.twin) {
            tile.encodeSums(self.e, t, w.k, x, x_off, self.s.sums, rows);
            tile.encode(self.e, t, matrix(w), x, x_off, self.s.sums, 0, y, y_off, rows) catch unreachable;
        } else tile.encodeTwin(self.e, t, matrix(w), x, x_off, y, y_off, rows) catch unreachable;
    }
    /// h += r unless first, then x = norm(h); `combine`: r is the decode experts' combine, taken here instead.
    fn norm(self: Encoder, s: *st.Scratch, weight: wts.Tensor, first: bool, rows: usize, combine: bool) void {
        if (combine) {
            self.pipe("lfm2_combine_norm");
            self.e.setBuffer(s.experts, 0, 0);
            self.e.setBuffer(s.weights, 0, 1);
            self.e.setBuffer(s.h, 0, 2);
            self.tensor(weight, 3);
            self.e.setValue(c.eps, 4);
            self.e.setBuffer(s.x, 0, 5);
            self.groups(.{ rows, 1, 1 }, .{ c.hidden / 4, 1, 1 });
            return;
        }
        self.pipe("lfm2_norm");
        self.e.setBuffer(s.h, 0, 0);
        self.e.setBuffer(s.r, 0, 1);
        self.tensor(weight, 2);
        self.e.setValue(c.eps, 3);
        self.e.setValue(@as(u32, @intFromBool(!first)), 4);
        self.e.setBuffer(s.x, 0, 5);
        self.groups(.{ rows, 1, 1 }, .{ c.hidden / 4, 1, 1 });
    }
    fn swiglu(self: Encoder, s: *st.Scratch, count: usize) void {
        self.pipe("lfm2_swiglu");
        self.e.setBuffer(s.gate, 0, 0);
        self.e.setBuffer(s.up, 0, 1);
        self.e.setBuffer(s.act, 0, 2);
        self.e.setValue(@as(u32, @intCast(count)), 3);
        self.threads(.{ count / 8, 1, 1 }, .{ 256, 1, 1 });
    }
};

pub fn matrix(w: row.Weights) tile.Matrix {
    return .{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k, .bits = w.bits };
}

pub fn run(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head) !void {
    if (tokens.len == 0 or tokens.len > s.rows or (record and tokens.len > st.batch_rows)) return error.LfmRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.LfmContextFull;
        total += seg.rows;
    }
    if (total != tokens.len or (head == .all and total > st.batch_rows) or (!record and segments.len != 1)) return error.InvalidLfmSegments;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidLfmToken;
    @memcpy(s.ids.slice(u32, total), tokens);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = Encoder{ .e = cb.compute(.serial), .m = m, .s = s, .prompt = (!record and head == .last) or total >= tile_rows };
    const e = enc.e;
    const embedding = m.weights.embedding;
    e.setPipeline(m.kernels.embed(embedding.bits));
    e.setBuffer(s.ids, 0, 0);
    enc.weights(embedding);
    e.setBuffer(s.h, 0, 4);
    const units = c.hidden / @as(usize, if (embedding.bits == 6) 4 else 8 / embedding.bits);
    enc.threads(.{ units, total, 1 }, .{ 256, 1, 1 });
    var combine = false; // the previous block's decode experts, combined by the next norm
    for (m.weights.blocks, 0..) |block, i| {
        enc.norm(s, block.operator_norm, i == 0, total, combine);
        combine = false;
        switch (block.mixer) {
            .conv => |cv| conv(enc, s, cv, i, segments, total),
            .attention => |a| attention(enc, s, a, &m.config, segments, i, total),
        }
        enc.norm(s, block.ffn_norm, false, total, false);
        switch (block.ffn) {
            .dense => |d| {
                enc.projection(d.gate, s.x, 0, s.gate, 0, total);
                enc.projection(d.up, s.x, 0, s.up, 0, total);
                enc.swiglu(s, total * c.intermediate);
                enc.projection(d.down, s.act, 0, s.r, 0, total);
            },
            .moe => |x| combine = moe(enc, s, x, m.config.scale, total),
        }
    }
    enc.norm(s, m.weights.norm, false, total, combine);
    if (head == .all) enc.projection(embedding, s.x, 0, s.logits, 0, total) else {
        enc.projection(embedding, s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("LFM2 forward: {s}\n", .{msg});
        return error.GpuFailed;
    }
    var base: usize = 0;
    for (segments) |seg| {
        const logit_row = if (head == .all) base + seg.rows - 1 else 0;
        @memcpy(seg.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2]);
        seg.cache.commit(s, base, seg.rows, record);
        base += seg.rows;
    }
}

fn conv(enc: Encoder, s: *st.Scratch, cv: wts.Conv, layer: usize, segments: []const Segment, total: usize) void {
    const e = enc.e;
    enc.projection(cv.in, s.x, 0, s.bcx, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        enc.pipe("lfm2_conv");
        e.setBuffer(s.bcx, base * 3 * c.hidden * 2, 0);
        e.setBuffer(seg.cache.blocks[layer].conv, 0, 1);
        enc.tensor(cv.weight, 2);
        e.setBuffer(s.y, base * c.hidden * 2, 3);
        e.setBuffer(s.snapshots[layer].?, base * st.conv_bytes, 4);
        enc.threads(.{ c.hidden, seg.rows, 1 }, .{ 256, 1, 1 });
        base += seg.rows;
    }
    enc.projection(cv.out, s.y, 0, s.r, 0, total);
}

fn attention(enc: Encoder, s: *st.Scratch, a: wts.Attention, config: *const c.Config, segments: []const Segment, layer: usize, total: usize) void {
    const e = enc.e;
    enc.projection(a.q, s.x, 0, s.q, 0, total);
    enc.projection(a.k, s.x, 0, s.k, 0, total);
    enc.projection(a.v, s.x, 0, s.v, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer].attention;
        const len: u32 = @intCast(seg.cache.len);
        const cap: u32 = @intCast(seg.cache.capacity);
        enc.pipe("lfm2_qk");
        e.setBuffer(s.q, base * c.hidden * 2, 0);
        e.setBuffer(s.k, base * c.kv_heads * c.head_dim * 2, 1);
        e.setBuffer(s.v, base * c.kv_heads * c.head_dim * 2, 2);
        enc.tensor(a.q_norm, 3);
        enc.tensor(a.k_norm, 4);
        e.setBytes(std.mem.asBytes(&QkArgs{ .len = len, .cap = cap, .eps = c.eps, .log2_theta = std.math.log2(config.theta) }), 5);
        e.setBuffer(s.queries, base * c.hidden * 2, 6);
        e.setBuffer(cache.keys, 0, 7);
        e.setBuffer(cache.values, 0, 8);
        enc.groups(.{ c.query_heads + c.kv_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        const subs: u32 = if (seg.rows <= st.split_rows) st.subs else 1;
        const args = AttnArgs{ .len = len, .cap = cap, .scale = 1.0 / @sqrt(@as(f32, c.head_dim)), .subs = subs };
        const used = (seg.cache.len + seg.rows + st.chunk_keys - 1) / st.chunk_keys * subs;
        const wide = seg.rows > st.split_rows;
        if (wide) enc.pipe("lfm2_attn_mma_r2") else enc.pipe("lfm2_attn_mma");
        e.setBuffer(s.queries, base * c.hidden * 2, 0);
        e.setBuffer(cache.keys, 0, 1);
        e.setBuffer(cache.values, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(s.pm, 0, 4);
        e.setBuffer(s.pl, 0, 5);
        e.setBuffer(s.po, 0, 6);
        e.setValue(@as(u32, @intCast(seg.rows)), 7);
        const per_tile: usize = if (wide) st.attn_rows else 8;
        enc.groups(.{ (seg.rows + per_tile - 1) / per_tile, c.kv_heads, used }, .{ 128, 1, 1 });
        enc.pipe("lfm2_attn_merge");
        e.setBuffer(s.pm, 0, 0);
        e.setBuffer(s.pl, 0, 1);
        e.setBuffer(s.po, 0, 2);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(s.y, base * c.hidden * 2, 4);
        enc.groups(.{ c.query_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        base += seg.rows;
    }
    enc.projection(a.out, s.y, 0, s.r, 0, total);
}

/// True when the picks' combine is left to the next norm (decode rows).
fn moe(enc: Encoder, s: *st.Scratch, x: wts.Moe, scale: f32, total: usize) bool {
    const e = enc.e;
    const slots = total * c.topk;
    enc.pipe("lfm2_route_logits");
    e.setBuffer(s.x, 0, 0);
    enc.tensor(x.router, 1);
    e.setBuffer(s.route, 0, 2);
    enc.groups(.{ c.experts / 8, total, 1 }, .{ 256, 1, 1 });
    enc.pipe("lfm2_route_select");
    e.setBuffer(s.route, 0, 0);
    enc.tensor(x.bias, 1);
    e.setValue(scale, 2);
    e.setBuffer(s.picks, 0, 3);
    e.setBuffer(s.weights, 0, 4);
    enc.groups(.{ total, 1, 1 }, .{ 32, 1, 1 });
    const t = &enc.m.kernels.tiles;
    if (!enc.prompt and enc.m.kernels.twin) {
        // Wide windows read each picked expert once for its rows (a row's own picks are distinct experts).
        const groups: ?Buffer = if (total >= tile.group_rows) s.groups else null;
        if (groups != null) tile.encodeTwinGroup(e, t, s.picks, slots, s.groups) catch unreachable;
        const gate = matrix(x.gate.one);
        tile.encodeTwinGlu(e, t, gate, matrix(x.up.one), s.x, s.act, s.picks, groups, slots, c.topk) catch unreachable;
        tile.encodeTwinIndexed(e, t, matrix(x.down.one), s.act, s.experts, s.picks, groups, slots, 1) catch unreachable;
        return true;
    }
    enc.pipe("lfm2_sort");
    e.setBuffer(s.picks, 0, 0);
    e.setValue(@as(u32, @intCast(slots)), 1);
    e.setBuffer(s.starts, 0, 2);
    e.setBuffer(s.order, 0, 3);
    e.setBuffer(s.pos, 0, 4);
    e.setBuffer(s.tokens, 0, 5);
    enc.groups(.{ 1, 1, 1 }, .{ c.experts * 16, 1, 1 });
    tile.encodeWork(e, &enc.m.kernels.tiles, s.starts, c.experts, s.work);
    tile.encodeSums(e, t, c.hidden, s.x, 0, s.sums, total);
    tile.encodeGroupedGlu(e, t, matrix(x.gate.one), matrix(x.up.one), c.experts, s.x, s.sums, s.tokens, s.act, slots, s.starts, s.work) catch unreachable;
    tile.encodeSums(e, t, c.moe_intermediate, s.act, 0, s.sums, slots);
    tile.encodeGrouped(e, t, matrix(x.down.one), c.experts, s.act, s.sums, s.ident, s.experts, slots, s.starts, s.work) catch unreachable;
    enc.pipe("lfm2_combine_sorted");
    e.setBuffer(s.experts, 0, 0);
    e.setBuffer(s.weights, 0, 1);
    e.setBuffer(s.r, 0, 2);
    e.setBuffer(s.pos, 0, 3);
    enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    return false;
}
