//! LFM3 on CUDA: full-attention KV in 512-position pages, each slot's rings and conv states, the scratch.
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
/// Layers of each mixer kind in the admitted schedule.
pub const conv_layers = count(.conv);
pub const full_layers = count(.full_attention);
/// The layers whose keys and values live in pages (liquid's kept-state walk).
pub const paged_layers = full_layers;
pub const sliding_layers = count(.sliding_attention);
/// Elements a KV row holds (4 heads of 128) and a page holds over every full layer's keys and values.
pub const kv_width = c.kv_heads * c.head_dim;
pub const page_elems = full_layers * 2 * kern.page * kv_width;
/// A conv state's rows: the taps before the current row.
pub const state_rows = c.conv_taps - 1;
/// Elements a slot's conv states hold: six bx rows a conv layer.
pub const slot_elems = conv_layers * state_rows * c.hidden;
/// Keys a sliding ring holds: a prompt chunk's rows are written before its first row reads its window back.
pub const ring = (prompt_rows + c.sliding_window + 63) / 64 * 64;
/// Elements a slot's rings hold: every sliding layer's keys and values.
pub const ring_elems = sliding_layers * 2 * ring * kv_width;
/// Key chunks one attention launch walks in parallel at most (nemotron's chunk_blocks).
const chunk_blocks = 64;

fn count(kind: c.LayerType) usize {
    var n: usize = 0;
    for (c.schedule) |t| n += @intFromBool(t == kind);
    return n;
}

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
    xs: u64, // fp32 [R, hidden / 64]: the decode attention merge's sums (read by nothing)
    sums: u64, // fp32 [round_rows, intermediate / 64]: an affine projection's decode rows' 64-value sums
    r: u64, // bf16 [R, hidden]: a block's output
    wide: u64, // bf16 [R, 2 intermediate]: conv in (3 hidden), q|k|v|gate, w1|w3
    y: u64, // bf16 [R, intermediate]: conv out, the gated attention, the SwiGLU
    q: u64, // bf16 [R, hidden]: normed and rotated queries
    att: u64, // bf16 [R, hidden]: attention before its head gate
    kv: u64, // bf16 [R, 2 kv_width]: keys then values a row
    po: u64, // decode attention partials: o, m, l
    pm: u64,
    pl: u64,
    part: u64, // the lane's slice partials (where clusters do not meet them)
    snaps: u64, // bf16 [conv_layers][round_rows][6][hidden]: each window row's conv state after it
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
    arena: u64 = 0, // [pages][full layer][K, V][512][kv_width] bf16
    table: u64 = 0, // int32 [slots][pages a slot]: slot s's page ids in position order
    states: u64 = 0, // bf16 [slots][conv layer][6][hidden]
    rings: u64 = 0, // bf16 [sliding layer][K, V][slots][ring][kv_width]
    ids: []u32 = &.{}, // the tables on the host
    caps: []usize = &.{}, // positions slot s's pages cover
    inv: [2]u64 = .{ 0, 0 }, // fp32 [head_dim / 2]: RoPE's inverse frequencies, full then sliding
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

    /// Bytes a stream's fixed part takes: its conv states and sliding rings.
    pub const fixed_bytes = (slot_elems + ring_elems) * 2;

    /// A slot's regions outside its pages (liquid's walk): conv states, then each sliding layer's K and V rings.
    pub fn fixed(m: *const Model, slot: usize, i: usize) ?[2]u64 {
        const ring_bytes = ring * kv_width * 2;
        if (i == 0) return .{ m.states + slot * slot_elems * 2, slot_elems * 2 };
        if (i > 2 * sliding_layers) return null;
        return .{ m.ringOf((i - 1) / 2, (i - 1) % 2 == 1) + slot * ring_bytes, ring_bytes };
    }

    /// Bytes one stream's caches take at `context` positions: its pages, conv states and rings.
    pub fn streamBytes(context: usize) usize {
        return positions(context) / kern.page * page_elems * 2 + fixed_bytes;
    }

    /// What the host's pool plans: a stream's conv states, rings and table row, a token its rows of the six layers.
    pub fn layout(m: *const Model) kv.Layout {
        return .{ .fixed_bytes = fixed_bytes + m.perSlot() * 4, .token_bytes = page_elems * 2 / kern.page, .page_tokens = kern.page, .snap_bytes = 0 };
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
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const kern.Kernels, context: usize, slots: usize, paged: bool) !Model {
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .context = positions(context), .slots = if (paged) 1 else slots, .paged = paged };
        errdefer m.deinit();
        try m.build(s, k);
        return m;
    }

    /// The device bytes `init` allocates for these sizes, counted by the same path without allocating.
    pub fn openBytes(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const kern.Kernels, context: usize, slots: usize, paged: bool) !usize {
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .context = positions(context), .slots = if (paged) 1 else slots, .paged = paged, .planned = 0 };
        defer m.deinit();
        try m.build(s, k);
        return m.planned.?;
    }

    /// Makes the arena of `pool` pages to lease and the states, rings and tables of `extra + 1` slots (none mapped).
    pub fn attach(m: *Model, s: cuda.Stream, extra: usize, pool: usize) !void {
        if (!m.paged or m.pages != 0 or extra >= round_rows) return error.BadAttach;
        m.forget(m.table);
        m.forget(m.states);
        m.forget(m.rings);
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

    /// The slots' tables, conv states and rings; a window of pages a slot unless paged (their pages come from leases).
    fn makeSlots(m: *Model, s: cuda.Stream) !void {
        const per = m.perSlot();
        m.table = try m.alloc(m.slots * per * 4);
        m.states = try m.alloc(m.slots * slot_elems * 2);
        m.rings = try m.alloc(m.slots * ring_elems * 2);
        if (m.planned != null) return;
        m.ids = try m.gpa.alloc(u32, m.slots * per);
        m.caps = try m.gpa.alloc(usize, m.slots);
        for (m.ids, 0..) |*v, i| v.* = if (m.paged) 0 else @intCast(i);
        @memset(m.caps, if (m.paged) 0 else m.context);
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(m.table, m.ids.ptr, m.ids.len * 4, s.handle), "cuMemcpyHtoDAsync");
        try (DeviceBuffer{ .d = m.d, .ptr = m.states, .len = m.slots * slot_elems * 2, .borrowed = true }).fill8(0, s.handle);
        try (DeviceBuffer{ .d = m.d, .ptr = m.rings, .len = m.slots * ring_elems * 2, .borrowed = true }).fill8(0, s.handle);
        try s.synchronize();
    }

    /// The lane's slice partials any projection of up to a round's rows needs (0 where clusters meet them).
    fn partBytes(m: *const Model, k: *const kern.Kernels) usize {
        var most: usize = 0;
        const w = m.w;
        for (1..round_rows + 1) |rows| {
            most = @max(most, partOf(k, rows, w.head));
            for (w.layers) |l| {
                for ([_]wts.Proj{ l.gate_up, l.down }) |p| most = @max(most, partOf(k, rows, p));
                switch (l.mixer) {
                    .conv => |cv| for ([_]wts.Proj{ cv.in, cv.out }) |p| {
                        most = @max(most, partOf(k, rows, p));
                    },
                    .attention => |a| for ([_]wts.Proj{ a.qkvg, a.gate orelse a.out, a.out }) |p| {
                        most = @max(most, partOf(k, rows, p));
                    },
                }
            }
        }
        return most;
    }

    fn build(m: *Model, s: cuda.Stream, k: *const kern.Kernels) !void {
        if (!m.paged) m.pages = m.slots * m.perSlot();
        if (m.pages > 0) m.arena = try m.alloc(m.pages * page_elems * 2);
        try m.makeSlots(s);
        for (&m.inv) |*p| p.* = try m.alloc(c.head_dim / 2 * 4);
        const r = prompt_rows;
        const h = c.hidden;
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
            .xs = try m.alloc(r * (h / 64) * 4),
            .sums = try m.alloc(round_rows * (c.intermediate / 64) * 4),
            .r = try m.alloc(r * h * 2),
            .wide = try m.alloc(r * 2 * c.intermediate * 2),
            .y = try m.alloc(r * c.intermediate * 2),
            .q = try m.alloc(r * h * 2),
            .att = try m.alloc(r * h * 2),
            .kv = try m.alloc(r * 2 * kv_width * 2),
            .po = try m.alloc(parts * c.head_dim * 4),
            .pm = try m.alloc(parts * 4),
            .pl = try m.alloc(parts * 4),
            .part = try m.alloc(m.partBytes(k)),
            .snaps = try m.alloc(conv_layers * round_rows * state_rows * h * 2),
            .logits = try m.alloc(round_rows * c.vocab * 2),
        };
        if (m.planned != null) return;
        const cfg = &m.w.config;
        for ([_]c.Rotary{ cfg.full_rotary, cfg.sliding_rotary }, m.inv) |rot, p| {
            const inv = inverseFrequencies(rot);
            try (DeviceBuffer{ .d = m.d, .ptr = p, .len = inv.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(&inv));
        }
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.ids);
        m.gpa.free(m.caps);
    }

    /// Full layer a's key (or value) cache as the kernels take it: its part of page 0.
    pub fn cache(m: *const Model, a: usize, value: bool) u64 {
        return m.arena + ((2 * a + @intFromBool(value)) * kern.page * kv_width) * 2;
    }

    /// Sliding layer l's key (or value) rings: [slots][ring][kv_width], slot s's at s * ring rows.
    pub fn ringOf(m: *const Model, l: usize, value: bool) u64 {
        return m.rings + (2 * l + @intFromBool(value)) * m.slots * ring * kv_width * 2;
    }

    /// Slot s's page table row.
    pub fn tableOf(m: *const Model, slot: usize) u64 {
        return m.table + slot * m.perSlot() * 4;
    }

    /// Zero slot s's conv states, queued on `s` (rings and pages are read only where the new stream wrote).
    pub fn clearSlot(m: *const Model, s: cuda.Stream, slot: usize) !void {
        try (DeviceBuffer{ .d = m.d, .ptr = m.states + slot * slot_elems * 2, .len = slot_elems * 2, .borrowed = true }).fill8(0, s.handle);
    }
};

/// A lane weight's slice partials at `rows` (affine-4 projections take none).
fn partOf(k: *const kern.Kernels, rows: usize, p: wts.Proj) usize {
    return switch (p) {
        .nvfp4, .bf16, .fp8g, .fp8, .mxfp8 => |q| k.lane.partBytes(rows, q),
        else => 0,
    };
}

/// RoPE's inverse frequencies as torch's float32 rows: 1 / theta^(2i / 128) rotated, zero past them (cos 1, sin 0).
pub fn inverseFrequencies(rot: c.Rotary) [c.head_dim / 2]f32 {
    var out: [c.head_dim / 2]f32 = undefined;
    const angles: usize = @intFromFloat(rot.partial * @as(f32, c.head_dim) / 2);
    for (&out, 0..) |*v, i| {
        if (i >= angles) {
            v.* = 0;
            continue;
        }
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, c.head_dim);
        const pw: f32 = @floatCast(std.math.pow(f64, rot.theta, e));
        v.* = 1.0 / pw;
    }
    return out;
}

test "the admitted schedule: 13 conv, 6 full and 5 sliding layers; a page 6 MiB, a ring a chunk and a window" {
    try std.testing.expectEqual(@as(usize, 13), conv_layers);
    try std.testing.expectEqual(@as(usize, 6), full_layers);
    try std.testing.expectEqual(@as(usize, 5), sliding_layers);
    try std.testing.expectEqual(@as(usize, 6 << 20), page_elems * 2);
    try std.testing.expectEqual(@as(usize, 1280), ring);
    try std.testing.expectEqual(@as(usize, 2 * (6 << 20) + Model.fixed_bytes), Model.streamBytes(1000));
    try std.testing.expectEqual(@as(usize, 13 * 6 * 2048 * 2), slot_elems * 2);
}

test "RoPE tables: full layers rotate 32 pairs at theta 1e6 over the full head dim, sliding ones all 64 at 1e4" {
    const full = inverseFrequencies(.{ .theta = 1e6, .partial = 0.5 });
    try std.testing.expectEqual(@as(f32, 1), full[0]);
    const want31 = 1.0 / std.math.pow(f64, 1e6, 62.0 / 128.0);
    try std.testing.expect(@abs(full[31] - want31) <= want31 * 0x1p-23); // one float32 rounding
    for (full[32..]) |v| try std.testing.expectEqual(@as(f32, 0), v);
    const sliding = inverseFrequencies(.{ .theta = 1e4, .partial = 1 });
    const want63 = 1.0 / std.math.pow(f64, 1e4, 126.0 / 128.0);
    try std.testing.expect(@abs(sliding[63] - want63) <= want63 * 0x1p-23);
}

test "the pool's layout: 12 KiB a token, a stream's conv states, rings and table row fixed" {
    var m: Model = undefined;
    m.context = Model.positions(32768);
    const l = m.layout();
    try std.testing.expectEqual(@as(u64, 6 << 20), l.pageBytes());
    try std.testing.expectEqual(@as(u64, 12 << 10), l.token_bytes);
    try std.testing.expectEqual(@as(u64, Model.fixed_bytes + 64 * 4), l.fixed_bytes);
    const p = try kv.plan(l, .{ .room = 20 << 30, .engine = 0, .lanes = 8, .cache = 4 << 30, .window = 32768 + 16, .engine_paged = true });
    try std.testing.expectEqual(@as(u32, 8), p.lanes);
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
    _ = try m.note(1, &.{ 9, 3, 4 }, 0);
    try std.testing.expectEqualSlices(u32, &.{ 9, 3, 4 }, ids[4..7]);
    try std.testing.expectEqual(@as(usize, 3 * kern.page), caps[1]);
    try std.testing.expectError(error.BadChain, m.note(0, &.{16}, 0));
    try std.testing.expectError(error.BadChain, m.note(0, &.{ 1, 2, 3, 4, 5 }, 0));
}
