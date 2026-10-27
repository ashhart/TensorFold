//! LFM2.5-24B-A2B's device state on CUDA: KV in 512-position pages, each slot's conv states, the scratch.
const std = @import("std");
const cuda = @import("cuda");
const c = @import("config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;

/// Rows a prompt chunk takes.
pub const prompt_rows = 1024;
/// Rows a round of windows takes over every stream.
pub const round_rows = 128;
/// Conv layers (every layer that is not attention) and attention layers of the admitted geometry.
pub const conv_layers = 30;
pub const attention_layers = c.layers - conv_layers;
/// Elements a KV row holds (8 heads of 64) and a page holds over every attention layer's keys and values.
pub const kv_width = c.kv_heads * c.head_dim;
pub const page_elems = attention_layers * 2 * kern.page * kv_width;
/// Elements a slot's conv states hold: two bx rows a conv layer.
pub const slot_elems = conv_layers * 2 * c.hidden;
/// Key chunks one attention launch walks in parallel at most (nemotron's chunk_blocks).
const chunk_blocks = 64;

pub const Scratch = struct {
    ids: u64, // int32 [R]: tokens, positions, slots, offsets in their stream's rows, R each
    pos: u64,
    slot: u64,
    off: u64,
    meta: u64, // int32 [round_rows]: each chain's first position (AttnSegs' META), its commit slot and row
    commit_slot: u64,
    commit_row: u64,
    h: u64, // bf16 [R, hidden]: the residual
    x: u64, // bf16 [R, hidden]: a norm's output
    xs: u64, // fp32 [R, intermediate / 64]: a projection input's 64-value sums
    r: u64, // bf16 [R, hidden]: a block's output
    wide: u64, // bf16 [R, 2 intermediate]: conv in (3 hidden), q|k|v, dense gate|up
    y: u64, // bf16 [R, intermediate]: conv out, attention out, dense act
    q: u64, // bf16 [R, hidden]: normed and rotated queries
    kv: u64, // bf16 [R, 2 kv_width]: keys then values a row
    picks: u64, // int32 [R, 4]
    wts: u64, // fp32 [R, 4]
    members: u64,
    items: u64,
    counts: u64,
    rank: u64,
    hist: u64,
    eact: u64, // bf16 [R * 4, moe_intermediate]
    ey: u64, // fp32 [R * 4, hidden] (bf16 at prompt rows)
    po: u64, // decode attention partials: o, m, l
    pm: u64,
    pl: u64,
    snaps: u64, // bf16 [conv_layers][round_rows][2][hidden]: each window row's conv state after it
    logits: u64, // bf16 [round_rows, vocab]
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    context: usize, // positions a slot holds: whole pages
    slots: usize,
    arena: u64 = 0, // [pages][attention layer][K, V][512][kv_width] bf16
    table: u64 = 0, // int32 [slots][pages a slot]: slot s's pages s * per .. (s + 1) * per
    states: u64 = 0, // bf16 [slots][conv layer][2][hidden]
    inv: u64 = 0, // fp32 [head_dim / 2]: RoPE's inverse frequencies
    s: Scratch = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated

    pub fn perSlot(m: *const Model) usize {
        return m.context / kern.page;
    }

    /// Positions a slot holds for a window of `context`: whole pages.
    pub fn positions(context: usize) usize {
        return (context + kern.page - 1) / kern.page * kern.page;
    }

    /// Bytes one stream's caches take at `context` positions: its pages and its conv states.
    pub fn streamBytes(context: usize) usize {
        return positions(context) / kern.page * page_elems * 2 + slot_elems * 2;
    }

    /// Decode attention's key chunks a row walks at most, and the blocks that walk them.
    pub fn chunks(m: *const Model) usize {
        return m.perSlot();
    }

    pub fn chunkBlocks(m: *const Model) usize {
        return @min(m.chunks(), chunk_blocks);
    }

    fn alloc(m: *Model, bytes: usize) !u64 {
        if (m.planned) |*p| {
            p.* += (@max(bytes, 256) + 255) / 256 * 256;
            return 0;
        }
        const b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize) !Model {
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .context = positions(context), .slots = slots };
        errdefer m.deinit();
        try m.build(s);
        return m;
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize) !usize {
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .context = positions(context), .slots = slots, .planned = 0 };
        defer m.deinit();
        try m.build(s);
        return m.planned.?;
    }

    fn build(m: *Model, s: cuda.Stream) !void {
        const per = m.perSlot();
        m.arena = try m.alloc(m.slots * per * page_elems * 2);
        m.table = try m.alloc(m.slots * per * 4);
        m.states = try m.alloc(m.slots * slot_elems * 2);
        m.inv = try m.alloc(c.head_dim / 2 * 4);
        const r = prompt_rows;
        const h = c.hidden;
        const pairs = r * c.topk;
        const items = cuda.grouped.maxItems(pairs, c.experts, 16);
        const parts = round_rows * m.chunks() * c.query_heads;
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .slot = try m.alloc(r * 4),
            .off = try m.alloc(r * 4),
            .meta = try m.alloc(round_rows * 4),
            .commit_slot = try m.alloc(round_rows * 4),
            .commit_row = try m.alloc(round_rows * 4),
            .h = try m.alloc(r * h * 2),
            .x = try m.alloc(r * h * 2),
            .xs = try m.alloc(r * (c.intermediate / 64) * 4),
            .r = try m.alloc(r * h * 2),
            .wide = try m.alloc(r * 2 * c.intermediate * 2),
            .y = try m.alloc(r * c.intermediate * 2),
            .q = try m.alloc(r * h * 2),
            .kv = try m.alloc(r * 2 * kv_width * 2),
            .picks = try m.alloc(pairs * 4),
            .wts = try m.alloc(pairs * 4),
            .members = try m.alloc(pairs * 4),
            .items = try m.alloc(3 * items * 4),
            .counts = try m.alloc(2 * 4),
            .rank = try m.alloc(pairs * 4),
            .hist = try m.alloc(((pairs + 1023) / 1024) * c.experts * 4),
            .eact = try m.alloc(pairs * c.moe_intermediate * 2),
            .ey = try m.alloc(pairs * h * 4),
            .po = try m.alloc(parts * c.head_dim * 4),
            .pm = try m.alloc(parts * 4),
            .pl = try m.alloc(parts * 4),
            .snaps = try m.alloc(conv_layers * round_rows * 2 * h * 2),
            .logits = try m.alloc(round_rows * c.vocab * 2),
        };
        if (m.planned != null) return;
        const ids = try m.gpa.alloc(i32, m.slots * per);
        defer m.gpa.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(i);
        try (DeviceBuffer{ .d = m.d, .ptr = m.table, .len = ids.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(ids));
        const inv = inverseFrequencies(m.w.config.theta);
        try (DeviceBuffer{ .d = m.d, .ptr = m.inv, .len = inv.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&inv));
        try (DeviceBuffer{ .d = m.d, .ptr = m.states, .len = m.slots * slot_elems * 2, .borrowed = true }).fill8(0, s.handle);
        try s.synchronize();
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    /// Attention layer a's key (or value) cache as the kernels take it: its part of page 0.
    pub fn cache(m: *const Model, a: usize, value: bool) u64 {
        return m.arena + ((2 * a + @intFromBool(value)) * kern.page * kv_width) * 2;
    }

    /// Slot s's page table row.
    pub fn tableOf(m: *const Model, slot: usize) u64 {
        return m.table + slot * m.perSlot() * 4;
    }

    /// Zero slot s's conv states, queued on `s` (a new stream starts from no history).
    pub fn clearSlot(m: *const Model, s: cuda.Stream, slot: usize) !void {
        try (DeviceBuffer{ .d = m.d, .ptr = m.states + slot * slot_elems * 2, .len = slot_elems * 2, .borrowed = true }).fill8(0, s.handle);
    }
};

/// RoPE's inverse frequencies as torch's float32 rows: 1 / theta^(2i / 64), the power in float64 rounded once.
pub fn inverseFrequencies(theta: f32) [c.head_dim / 2]f32 {
    var out: [c.head_dim / 2]f32 = undefined;
    for (&out, 0..) |*v, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, c.head_dim);
        const pw: f32 = @floatCast(std.math.pow(f64, theta, e));
        v.* = 1.0 / pw;
    }
    return out;
}

test "a page holds 512 positions of the ten attention layers' keys and values: 10 MiB" {
    try std.testing.expectEqual(@as(usize, 10 << 20), page_elems * 2);
    try std.testing.expectEqual(@as(usize, 4 * (10 << 20) + slot_elems * 2), Model.streamBytes(1537));
    try std.testing.expectEqual(@as(usize, 2048), Model.positions(1537));
    const inv = inverseFrequencies(1e7);
    try std.testing.expectEqual(@as(f32, 1), inv[0]);
    try std.testing.expect(@abs(inv[31] - 1.0 / std.math.pow(f64, 1e7, 62.0 / 64.0)) < 1e-12);
}
