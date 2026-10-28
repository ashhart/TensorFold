//! LFM2.5-24B-A2B's device state on CUDA: KV in 512-position pages, each slot's conv states, the scratch.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
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
/// Marks a prompt chunk taps (their conv states kept as the pass goes by); one more ends the chunk.
pub const max_taps = 4;
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
    tap: u64, // bf16 [max_taps][conv_layers][2][hidden]: the conv states after a chunk's tapped rows (a slot's layout)
    logits: u64, // bf16 [round_rows, vocab]
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    context: usize, // positions a slot holds: whole pages
    slots: usize,
    paged: bool = false, // slots lease pages of the core pool (attach makes the arena), else each owns a window
    pages: usize = 0, // pages in the arena
    arena: u64 = 0, // [pages][attention layer][K, V][512][kv_width] bf16
    table: u64 = 0, // int32 [slots][pages a slot]: slot s's page ids in position order
    states: u64 = 0, // bf16 [slots][conv layer][2][hidden]
    ids: []u32 = &.{}, // the tables on the host
    caps: []usize = &.{}, // positions slot s's pages cover
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

    /// What the host's pool plans: a stream's conv states and table row, a token its rows of the ten layers.
    pub fn layout(m: *const Model) kv.Layout {
        return .{ .fixed_bytes = slot_elems * 2 + m.perSlot() * 4, .token_bytes = page_elems * 2 / kern.page, .page_tokens = kern.page, .snap_bytes = 0 };
    }

    /// Slots a backend holds: the model's, or (paged) as many as a round takes.
    pub fn slotCap(m: *const Model) usize {
        return if (m.paged) round_rows else m.slots;
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

    /// `slots` streams, each owning a window of pages; paged: one slot's states now, `attach` makes the pages.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize, paged: bool) !Model {
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .context = positions(context), .slots = if (paged) 1 else slots, .paged = paged };
        errdefer m.deinit();
        try m.build(s);
        return m;
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize, paged: bool) !usize {
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .context = positions(context), .slots = if (paged) 1 else slots, .paged = paged, .planned = 0 };
        defer m.deinit();
        try m.build(s);
        return m.planned.?;
    }

    /// Makes the arena of `pool` pages to lease and the states and tables of `extra + 1` slots (no pages mapped).
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        if (!m.paged or m.pages != 0 or extra >= round_rows) return error.BadAttach;
        m.forget(m.table);
        m.forget(m.states);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
        m.ids = &.{};
        m.caps = &.{};
        m.slots = extra + 1;
        m.pages = pool;
        m.arena = try m.alloc(pool * page_elems * 2);
        try m.d.check(m.d.api.cuMemsetD8Async(m.arena, 0, pool * page_elems * 2, s.handle), "cuMemsetD8Async");
        try m.makeSlots(s);
        try s.synchronize();
    }

    /// Slot `slot`'s pages from table entry `from` on are `ids` (arena page numbers in position order).
    pub fn mapPages(m: *Model, s: cuda.Stream, slot: usize, ids: []const u32, from: usize) !void {
        const at = try m.note(slot, ids, from);
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(m.table + at * 4, ids.ptr, ids.len * 4, s.handle), "cuMemcpyHtoDAsync");
    }

    /// mapPages on the host: the table's copy and the slot's reach; the entry its device copy starts at.
    fn note(m: *Model, slot: usize, ids: []const u32, from: usize) !usize {
        const per = m.perSlot();
        if (slot >= m.slots or from + ids.len > per) return error.BadChain;
        for (ids) |id| if (id >= m.pages) return error.BadChain;
        const at = slot * per + from;
        @memcpy(m.ids[at..][0..ids.len], ids);
        m.caps[slot] = (from + ids.len) * kern.page;
        return at;
    }

    /// Slot `slot` holds no pages: its table entries stay, no forward may reach them.
    pub fn unmapPages(m: *Model, slot: usize) void {
        m.caps[slot] = 0;
    }

    /// Frees one buffer `alloc` made.
    fn forget(m: *Model, ptr: u64) void {
        for (m.buffers.items, 0..) |*b, i| if (b.ptr == ptr) {
            b.free();
            _ = m.buffers.swapRemove(i);
            return;
        };
    }

    /// The slots' tables and conv states; a window of pages a slot unless paged (their pages come from leases).
    fn makeSlots(m: *Model, s: cuda.Stream) !void {
        const per = m.perSlot();
        m.table = try m.alloc(m.slots * per * 4);
        m.states = try m.alloc(m.slots * slot_elems * 2);
        if (m.planned != null) return;
        m.ids = try m.gpa.alloc(u32, m.slots * per);
        m.caps = try m.gpa.alloc(usize, m.slots);
        for (m.ids, 0..) |*v, i| v.* = if (m.paged) 0 else @intCast(i);
        @memset(m.caps, if (m.paged) 0 else m.context);
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(m.table, m.ids.ptr, m.ids.len * 4, s.handle), "cuMemcpyHtoDAsync");
        try (DeviceBuffer{ .d = m.d, .ptr = m.states, .len = m.slots * slot_elems * 2, .borrowed = true }).fill8(0, s.handle);
        try s.synchronize();
    }

    fn build(m: *Model, s: cuda.Stream) !void {
        if (!m.paged) m.pages = m.slots * m.perSlot();
        if (m.pages > 0) m.arena = try m.alloc(m.pages * page_elems * 2);
        try m.makeSlots(s);
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
            .tap = try m.alloc(max_taps * slot_elems * 2),
            .logits = try m.alloc(round_rows * c.vocab * 2),
        };
        if (m.planned != null) return;
        const inv = inverseFrequencies(m.w.config.theta);
        try (DeviceBuffer{ .d = m.d, .ptr = m.inv, .len = inv.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&inv));
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
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

test "the pool's layout: 20 KiB a token, a stream's conv states and table row fixed, no snapshot slots" {
    var m: Model = undefined;
    m.context = Model.positions(32768);
    const l = m.layout();
    try std.testing.expectEqual(@as(u64, 10 << 20), l.pageBytes());
    try std.testing.expectEqual(@as(u64, 20 << 10), l.token_bytes);
    try std.testing.expectEqual(@as(u64, slot_elems * 2 + 64 * 4), l.fixed_bytes);
    // 40 GiB past the weights, 8 streams at 32k, the default 16 GiB cache: a window for the first, the rest pages
    const p = try kv.plan(l, .{ .room = 40 << 30, .engine = 0, .lanes = 8, .cache = 16 << 30, .window = 32768 + 16, .engine_paged = true });
    try std.testing.expectEqual(@as(u32, 8), p.lanes);
    try std.testing.expectEqual(@as(usize, 0), p.snaps);
    try std.testing.expect(p.pages * l.pageBytes() + p.copied + 7 * l.fixed_bytes <= 40 << 30);
    try std.testing.expect(p.pages >= l.pages(32768 + 16));
}

test "leases map a slot's pages in position order; its reach ends at them; an id past the arena is refused" {
    var ids: [8]u32 = @splat(0);
    var caps = [_]usize{ 0, 0 };
    var m: Model = undefined;
    m.context = 4 * kern.page;
    m.slots = 2;
    m.pages = 16;
    m.ids = &ids;
    m.caps = &caps;
    var pool = try kv.Pool.init(std.testing.allocator, 16, page_elems * 2);
    defer pool.deinit();
    var lease: kv.Lease = .{};
    try pool.promise(&lease, 3);
    const Rec = struct {
        m: *Model,
        fn write(ctx: *anyopaque, got: []const u32, from: usize) anyerror!void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            _ = try r.m.note(1, got[from..], from);
        }
    };
    var rec: Rec = .{ .m = &m };
    try pool.cover(&lease, 3, .{ .ctx = &rec, .write = Rec.write });
    try std.testing.expectEqualSlices(u32, lease.ids.items, ids[4..7]);
    try std.testing.expectEqual(@as(usize, 3 * kern.page), caps[1]);
    try std.testing.expectEqual(@as(usize, 13), pool.available());
    pool.finish(&lease);
    try std.testing.expect(pool.idle() and pool.available() == 16);
    try std.testing.expectError(error.BadChain, m.note(0, &.{16}, 0));
    try std.testing.expectError(error.BadChain, m.note(0, &.{ 1, 2, 3, 4, 5 }, 0));
}
