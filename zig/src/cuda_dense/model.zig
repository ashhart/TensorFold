//! The dense engine's device state: the forward's scratch, the page arena of every layer's KV, each slot's table.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const recipe = @import("recipe.zig");
const pages = @import("pages.zig");
const rope = @import("rope.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Geometry = recipe.Geometry;

/// Rows a prompt chunk takes.
pub const prompt_rows = 1024;
/// Rows a round of windows takes over every stream.
pub const round_rows = 128;
/// Key chunks one decode attention launch walks in parallel at most.
const chunk_blocks = 64;

/// The forward's buffers, `R` rows of each (a prompt chunk's); a recipe reads and writes them through its Pass.
pub const Scratch = struct {
    ids: u64, // int32 [R]: tokens
    pos: u64, // int32 [R]: each row's position
    meta: u64, // int32 [round_rows]: each chain's first position (AttnSegs' META)
    h: u64, // bf16 [R, hidden]: the residual
    x: u64, // bf16 [R, hidden]: a norm's output
    r: u64, // bf16 [R, hidden]: a block's output
    acc: u64, // fp32 [R, acc width]: a projection's sums before its bias
    wide: u64, // bf16 [R, wide]: gate|up
    y: u64, // bf16 [R, max(q width, mid)]: attention out, the MLP's activation
    q: u64, // bf16 [R, q width]: rotated queries
    kv: u64, // bf16 [R, 2 kv width]: keys then values a row
    xs: u64, // fp32 [round_rows, q width / 64]: decode attention's 64-value sums
    po: u64, // decode attention partials: o, m, l
    pm: u64,
    pl: u64,
    logits: u64, // bf16 [round_rows, vocab]
    part: u64, // the lane matmul's slice partials
};

/// bytes to dst on `s`: a private pinned copy kept until the stream completes, so the source may change at once.
pub fn put(d: *const cuda.Driver, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
    try cuda.copy_source.enqueue(d, dst, bytes, s);
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: Geometry,
    context: usize, // the served window: a request's prompt and reply fit in it
    slots: usize = 0, // streams with a table row: none until attach
    pages: usize = 0, // pages in the arena
    arena: u64 = 0, // [page][layer][K, V][512][kv width] bf16
    table: u64 = 0, // int32 [slots][perSlot()]: slot s's page ids in position order
    ids: []u32 = &.{}, // the tables on the host
    caps: []usize = &.{}, // positions slot s's pages cover
    inv: u64 = 0, // fp32 [head_dim / 2]: RoPE's inverse frequencies (a geometry with theta)
    acc_width: usize,
    part_bytes: usize,
    s: Scratch = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    planned: ?usize = null, // a plan: the bytes counted, nothing allocated

    pub fn perSlot(m: *const Model) usize {
        return pages.perSlot(m.context);
    }

    /// Positions a slot's table reaches: the window and its verify rows, in whole pages.
    pub fn capacity(m: *const Model) usize {
        return m.perSlot() * pages.page_rows;
    }

    pub fn pageElems(m: *const Model) usize {
        return pages.pageElems(m.g);
    }

    pub fn pageBytes(m: *const Model) u64 {
        return pages.pageBytes(m.g);
    }

    pub fn layout(m: *const Model) kv.Layout {
        return pages.layout(m.g, m.context);
    }

    pub fn streamBytes(m: *const Model) u64 {
        return pages.streamBytes(m.g, m.context);
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

    /// The f32 sums width every biased projection fits, and the lane's largest slice partials over the shapes.
    fn sizes(g: Geometry, lane: *const cuda.qmmf.Lane) struct { acc: usize, part: usize } {
        const shapes = [_][2]usize{ .{ g.acc, g.hidden }, .{ g.hidden, g.qWidth() }, .{ g.wide, g.hidden }, .{ g.hidden, g.mid }, .{ g.vocab, g.hidden } };
        var part: usize = 0;
        for (shapes) |nk| for (1..prompt_rows + 1) |rows| {
            part = @max(part, lane.partBytes(rows, cuda.modelopt.bf16Weight(0, nk[0], nk[1])));
        };
        return .{ .acc = @max(g.acc, g.wide, g.hidden), .part = part };
    }

    fn blank(gpa: std.mem.Allocator, d: *const cuda.Driver, g: Geometry, context: usize, lane: *const cuda.qmmf.Lane) Model {
        const z = sizes(g, lane);
        return .{ .gpa = gpa, .d = d, .g = g, .context = context, .acc_width = z.acc, .part_bytes = z.part };
    }

    /// The scratch and the RoPE table; the arena and tables come with attach, once the host has planned the pages.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, g: Geometry, context: usize, lane: *const cuda.qmmf.Lane) !Model {
        var m = blank(gpa, d, g, context, lane);
        errdefer m.deinit();
        errdefer cuda.completion.drain(d, s, "the dense engine's scratch uploads did not drain before their buffers were freed");
        try m.build(s);
        return m;
    }

    /// The device bytes `init` allocates, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, g: Geometry, context: usize, lane: *const cuda.qmmf.Lane) !usize {
        var m = blank(gpa, d, g, context, lane);
        m.planned = 0;
        defer m.deinit();
        try m.build(s);
        return m.planned.?;
    }

    fn build(m: *Model, s: cuda.Stream) !void {
        const g = m.g;
        const r = prompt_rows;
        const parts = round_rows * m.chunks() * g.heads;
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .meta = try m.alloc(round_rows * 4),
            .h = try m.alloc(r * g.hidden * 2),
            .x = try m.alloc(r * g.hidden * 2),
            .r = try m.alloc(r * g.hidden * 2),
            .acc = try m.alloc(r * m.acc_width * 4),
            .wide = try m.alloc(r * g.wide * 2),
            .y = try m.alloc(r * @max(g.qWidth(), g.mid) * 2),
            .q = try m.alloc(r * g.qWidth() * 2),
            .kv = try m.alloc(r * 2 * g.kvWidth() * 2),
            .xs = try m.alloc(round_rows * (g.qWidth() / 64) * 4),
            .po = try m.alloc(parts * g.head_dim * 4),
            .pm = try m.alloc(parts * 4),
            .pl = try m.alloc(parts * 4),
            .logits = try m.alloc(round_rows * g.vocab * 2),
            .part = try m.alloc(m.part_bytes),
        };
        const theta = g.theta orelse return;
        m.inv = try m.alloc(rope.count(g.head_dim) * 4);
        if (m.planned != null) return;
        const inv = try m.gpa.alloc(f32, rope.count(g.head_dim));
        defer m.gpa.free(inv);
        if (g.inv_freq) |table| {
            if (table.len != inv.len) return error.BadRopeTable;
            @memcpy(inv, table);
        } else rope.inverse(theta, g.head_dim, inv);
        try m.upload(s, m.inv, std.mem.sliceAsBytes(inv));
        try s.synchronize();
    }

    fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try put(m.d, s, dst, bytes);
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
    }

    /// The arena of `pool` pages (zeroed) and the tables of `extra + 1` slots, no page mapped.
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        if (m.pages != 0 or pool == 0 or extra >= round_rows or m.planned != null) return error.BadAttach;
        const mark = m.buffers.items.len; // the errdefer reads it at failure, when the list has grown
        errdefer m.undo(s, mark);
        const bytes = pool * m.pageBytes();
        m.arena = try m.alloc(bytes);
        try m.d.check(m.d.api.cuMemsetD8Async(m.arena, 0, bytes, s.handle), "cuMemsetD8Async");
        const slots = extra + 1;
        m.table = try m.alloc(slots * m.perSlot() * 4);
        m.ids = try m.gpa.alloc(u32, slots * m.perSlot());
        @memset(m.ids, 0);
        m.caps = try m.gpa.alloc(usize, slots);
        @memset(m.caps, 0);
        try m.upload(s, m.table, std.mem.sliceAsBytes(m.ids));
        try s.synchronize();
        m.slots = slots;
        m.pages = pool;
    }

    /// A failed attach: its fill and table upload proved complete, then its arena, tables and host rows gone.
    fn undo(m: *Model, s: cuda.Stream, mark: usize) void {
        cuda.completion.drain(m.d, s, "the dense engine's page arena fill did not drain before it was freed");
        while (m.buffers.items.len > mark) {
            var b = m.buffers.pop().?;
            b.free();
        }
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
        m.ids = &.{};
        m.caps = &.{};
        m.arena = 0;
        m.table = 0;
    }

    /// Slot `slot`'s pages from table entry `from` on are `ids` (arena page numbers in position order).
    pub fn mapPages(m: *Model, s: cuda.Stream, slot: usize, ids: []const u32, from: usize) !void {
        const per = m.perSlot();
        if (slot >= m.slots or from + ids.len > per) return error.BadChain;
        for (ids) |id| if (id >= m.pages) return error.BadChain;
        const at = slot * per + from;
        @memcpy(m.ids[at..][0..ids.len], ids);
        m.caps[slot] = (from + ids.len) * pages.page_rows;
        try m.upload(s, m.table + at * 4, std.mem.sliceAsBytes(m.ids[at..][0..ids.len]));
    }

    /// Slot `slot` holds no pages: its table entries stay, no forward may reach them.
    pub fn unmapPages(m: *Model, slot: usize) void {
        if (slot < m.slots) m.caps[slot] = 0;
    }

    /// Layer `layer`'s key (or value) cache as the kernels take it: its part of page 0.
    pub fn cache(m: *const Model, layer: usize, value: bool) u64 {
        return m.arena + ((2 * layer + @intFromBool(value)) * pages.page_rows * m.g.kvWidth()) * 2;
    }

    /// Slot s's page table row.
    pub fn tableOf(m: *const Model, slot: usize) u64 {
        return m.table + slot * m.perSlot() * 4;
    }
};
