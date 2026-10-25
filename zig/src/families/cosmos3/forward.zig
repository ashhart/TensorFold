//! The text model, a command buffer a call: per-stream row segments at M-RoPE positions, image rows, causal GQA.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const row = @import("../../core/row_projection.zig");
const tile = @import("../../core/tile_qmm.zig");
const am = @import("../../core/affine_mm.zig");
const Buffer = mtl.Buffer;

pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { last, all };
/// Image rows [n, hidden] bf16, and for every row of the call the image row it takes (-1: its token's embedding).
pub const Images = struct { rows: Buffer, map: []const i32 };
/// GPU seconds of every call so far.
pub var gpu_seconds: f64 = 0;
/// Rows from which a call takes the tiles (the twin's bits, each weight read once for every row).
const tile_rows = 33;
/// A buffer and byte offset as core/affine_mm binds them.
pub const Ref = struct { buf: Buffer, off: usize = 0 };

const QkArgs = extern struct { cap: u32, log2_theta: f32 };
pub const AttnArgs = extern struct { first: u32, keys: u32, causal: u32, group: u32, q_row: u32, k_row: u32, k_head: u32, o_row: u32, scale: f32 };
pub const LnArgs = extern struct { width: u32, eps: f32, add: u32 };
pub const BiasArgs = extern struct { width: u32, count: u32, act: u32 };
pub const Act = enum(u32) { none = 0, gelu_tanh = 1, gelu_erf = 2 };

pub const Encoder = struct {
    e: mtl.ComputeEncoder,
    m: *Model,
    sums: Buffer,
    /// Prompt calls and windows of tile_rows or more take the tiles; fewer decode rows the twin (the same bits).
    prompt: bool,

    pub fn pipe(self: Encoder, comptime key: []const u8) void {
        self.e.setPipeline(self.m.kernels.get(key));
    }
    pub fn tensor(self: Encoder, t: wts.Tensor, slot: usize) void {
        self.e.setBuffer(t.buffer, t.offset, slot);
    }
    pub fn threads(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    pub fn groups(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchGroups(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    fn nax(self: Encoder, bits: u8) ?am.Pipes {
        if (!self.prompt) return null;
        return self.m.kernels.naxPipes(bits);
    }
    /// y[rows] = x[rows] W^T: the tensor units or the prompt tiles (on x's group sums), or their twin at decode rows.
    pub fn projection(self: Encoder, w: row.Weights, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize) void {
        const t = &self.m.kernels.tiles;
        if (self.nax(w.bits)) |p| {
            const xr = Ref{ .buf = x, .off = x_off };
            const sums = Ref{ .buf = self.sums };
            am.rowSums(self.e, p, c.group, xr, sums, @intCast(rows), @intCast(w.k));
            am.dense(self.e, p, xr, sums, affine(w), Ref{ .buf = y, .off = y_off }, @intCast(rows));
            return;
        }
        if (self.prompt or !self.m.kernels.twin or rows > tile.twin_rows) {
            tile.encodeSums(self.e, t, w.k, x, x_off, self.sums, rows);
            tile.encode(self.e, t, matrix(w), x, x_off, self.sums, 0, y, y_off, rows) catch unreachable;
        } else tile.encodeTwin(self.e, t, matrix(w), x, x_off, y, y_off, rows) catch unreachable;
    }
    /// y = act(x W^T + bias) over [rows, n].
    pub fn biased(self: Encoder, w: wts.Biased, x: Buffer, x_off: usize, y: Buffer, y_off: usize, rows: usize, act: Act) void {
        self.projection(w.w, x, x_off, y, y_off, rows);
        self.pipe("cos3_bias");
        self.e.setBuffer(y, y_off, 0);
        self.tensor(w.bias, 1);
        const count = rows * w.w.n;
        self.e.setBytes(std.mem.asBytes(&BiasArgs{ .width = @intCast(w.w.n), .count = @intCast(count), .act = @backingInt(act) }), 2);
        self.threads(.{ count, 1, 1 }, .{ 256, 1, 1 });
    }
    /// h += r when `add`, then x = LayerNorm(h) over [rows, width].
    pub fn layerNorm(self: Encoder, n: wts.Norm, h: Ref, r: ?Ref, x: Ref, rows: usize, width: usize, eps: f32) void {
        self.pipe("cos3_layer_norm");
        self.e.setBuffer(h.buf, h.off, 0);
        const rr = r orelse h;
        self.e.setBuffer(rr.buf, rr.off, 1);
        self.tensor(n.w, 2);
        self.tensor(n.b, 3);
        self.e.setBytes(std.mem.asBytes(&LnArgs{ .width = @intCast(width), .eps = eps, .add = @intFromBool(r != null) }), 4);
        self.e.setBuffer(x.buf, x.off, 5);
        self.groups(.{ rows, 1, 1 }, .{ 256, 1, 1 });
    }
    /// h += r (RMS add) unless first, then x = RMSNorm(h).
    fn norm(self: Encoder, s: *st.Scratch, weight: wts.Tensor, first: bool, rows: usize) void {
        self.pipe("cos3_norm");
        self.e.setBuffer(s.h, 0, 0);
        self.e.setBuffer(s.r, 0, 1);
        self.tensor(weight, 2);
        self.e.setValue(c.eps, 3);
        self.e.setValue(@as(u32, @intFromBool(!first)), 4);
        self.e.setBuffer(s.x, 0, 5);
        self.groups(.{ rows, 1, 1 }, .{ c.hidden / 4, 1, 1 });
    }
};

/// A matrix as core/affine_mm binds it.
fn affine(w: row.Weights) struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 } {
    return .{
        .w = .{ .buf = w.w, .off = w.w_off },
        .s = .{ .buf = w.scales, .off = w.s_off },
        .b = .{ .buf = w.biases, .off = w.b_off },
        .n = @intCast(w.n),
        .k = @intCast(w.k),
    };
}

pub fn matrix(w: row.Weights) tile.Matrix {
    return .{ .w = w.w, .w_off = w.w_off, .scales = w.scales, .s_off = w.s_off, .biases = w.biases, .b_off = w.b_off, .n = w.n, .k = w.k, .bits = w.bits };
}

pub fn run(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head, images: ?Images) !void {
    if (tokens.len == 0 or tokens.len > s.rows or (record and tokens.len > st.batch_rows)) return error.CosmosRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.CosmosContextFull;
        total += seg.rows;
    }
    if (total != tokens.len or (head == .all and total > st.batch_rows) or (!record and segments.len != 1)) return error.InvalidCosmosSegments;
    if (images) |im| if (im.map.len != total) return error.InvalidCosmosSegments;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidCosmosToken;
    @memcpy(s.ids.slice(u32, total), tokens);
    {
        const pos = s.pos.slice(i32, 3 * total);
        const slots = s.slots.slice(u32, total);
        var r: usize = 0;
        for (segments) |seg| for (0..seg.rows) |i| {
            const p = seg.cache.position(seg.cache.len + i);
            @memcpy(pos[3 * r ..][0..3], &p);
            slots[r] = @intCast(seg.cache.len + i);
            r += 1;
        };
        if (images) |im| @memcpy(s.map.slice(i32, total), im.map);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    // a few rows (a prompt's tail after a kept state, decode, windows) read each weight once through the twin
    const prompt = total >= tile_rows;
    const enc = Encoder{ .e = cb.compute(.serial), .m = m, .sums = s.sums, .prompt = prompt };
    const e = enc.e;
    const embedding = m.weights.embedding;
    enc.pipe("cos3_embed_b8");
    e.setBuffer(s.ids, 0, 0);
    e.setBuffer(embedding.w, embedding.w_off, 1);
    e.setBuffer(embedding.scales, embedding.s_off, 2);
    e.setBuffer(embedding.biases, embedding.b_off, 3);
    e.setBuffer(s.h, 0, 4);
    enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    if (images) |im| {
        enc.pipe("cos3_place");
        e.setBuffer(s.h, 0, 0);
        e.setBuffer(im.rows, 0, 1);
        e.setBuffer(s.map, 0, 2);
        enc.threads(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    }
    for (m.weights.layers, 0..) |layer, i| {
        enc.norm(s, layer.input_norm, i == 0, total);
        attention(enc, s, layer, &m.config, segments, i, total);
        enc.norm(s, layer.post_norm, false, total);
        enc.projection(layer.fc1, s.x, 0, s.act, 0, total);
        enc.pipe("cos3_relu2");
        e.setBuffer(s.act, 0, 0);
        e.setValue(@as(u32, @intCast(total * c.intermediate)), 1);
        enc.threads(.{ total * c.intermediate, 1, 1 }, .{ 256, 1, 1 });
        enc.projection(layer.fc2, s.act, 0, s.r, 0, total);
    }
    enc.norm(s, m.weights.norm, false, total);
    if (head == .all) enc.projection(m.weights.head, s.x, 0, s.logits, 0, total) else {
        enc.projection(m.weights.head, s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
    }
    e.end();
    cb.commit();
    cb.wait();
    gpu_seconds += cb.gpuSeconds();
    if (cb.failure()) |msg| {
        std.debug.print("Cosmos3 forward: {s}\n", .{msg});
        return error.GpuFailed;
    }
    var base: usize = 0;
    for (segments) |seg| {
        const logit_row = if (head == .all) base + seg.rows - 1 else 0;
        @memcpy(seg.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2]);
        seg.cache.commit(base, seg.rows, record);
        base += seg.rows;
    }
}

fn attention(enc: Encoder, s: *st.Scratch, a: wts.Layer, config: *const c.Config, segments: []const Segment, layer: usize, total: usize) void {
    const e = enc.e;
    const kvd = c.kv_heads * c.head_dim;
    enc.projection(a.q, s.x, 0, s.q, 0, total);
    enc.projection(a.k, s.x, 0, s.k, 0, total);
    enc.projection(a.v, s.x, 0, s.v, 0, total);
    var base: usize = 0;
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer];
        const cap: u32 = @intCast(seg.cache.capacity);
        enc.pipe("cos3_qk");
        e.setBuffer(s.q, base * c.hidden * 2, 0);
        e.setBuffer(s.k, base * kvd * 2, 1);
        e.setBuffer(s.v, base * kvd * 2, 2);
        e.setBuffer(s.pos, base * 12, 3);
        e.setBuffer(s.slots, base * 4, 4);
        e.setBytes(std.mem.asBytes(&QkArgs{ .cap = cap, .log2_theta = std.math.log2(config.theta) }), 5);
        e.setBuffer(s.queries, base * c.hidden * 2, 6);
        e.setBuffer(cache.keys, 0, 7);
        e.setBuffer(cache.values, 0, 8);
        enc.groups(.{ c.query_heads + c.kv_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        const tiled = seg.rows >= 8;
        if (tiled) enc.pipe("cos3_flash_128") else enc.pipe("cos3_attn_128");
        e.setBuffer(s.queries, base * c.hidden * 2, 0);
        e.setBuffer(cache.keys, 0, 1);
        e.setBuffer(cache.values, 0, 2);
        const args = AttnArgs{
            .first = @intCast(seg.cache.len),
            .keys = 0,
            .causal = 1,
            .group = c.query_heads / c.kv_heads,
            .q_row = c.query_heads * c.head_dim,
            .k_row = c.head_dim,
            .k_head = cap * c.head_dim,
            .o_row = c.query_heads * c.head_dim,
            .scale = 1.0 / @sqrt(@as(f32, c.head_dim)),
        };
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(s.y, base * c.hidden * 2, 4);
        if (tiled) {
            e.setValue(@as(u32, @intCast(seg.rows)), 5);
            enc.groups(.{ c.query_heads, (seg.rows + 31) / 32, 1 }, .{ 128, 1, 1 });
        } else enc.groups(.{ c.query_heads, seg.rows, 1 }, .{ 32, 1, 1 });
        base += seg.rows;
    }
    enc.projection(a.o, s.y, 0, s.r, 0, total);
}
