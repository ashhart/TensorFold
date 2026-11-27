//! Host-side hand-off producer: a finished prefill's engine snapshot and pool pages become a protocol-2 export.
const std = @import("std");
const fabric = @import("fabric");
const g = @import("handoff/geometry.zig");
const plan = @import("handoff/plan.zig");
const ex = @import("handoff/export.zig");
const p2 = @import("handoff/p2.zig");
const producer = fabric.producer;

pub const Error = ex.Error || error{ BadSnapshot, NotAHandoffPrompt };

/// One bounded role of the engine's snapshot: its name and this slot's bytes (stale drafter rows are zeroed in place).
pub const Piece = struct { role: []const u8, bytes: []u8 };

/// A KV source's comp and index pages of the slot, logical page 0 first, back to back (a replicated pool's view).
pub const Pages = struct { layer: u32, comp: []const u8, index: []const u8 };

pub const State = struct {
    /// the position the slot stands at: the prompt's tokens before its last
    pos: u64,
    /// the drafter's key rows the snapshot holds valid: positions [pos - ds_rows, pos)
    ds_rows: u64,
    index_format: ex.IndexFormat = .fp8,
    pieces: []const Piece,
    pages: []const Pages,
    /// CED replay: layers from here hold `replay_rows` replayed rows; the window's older ones are another prompt's
    decoder_from: u32 = none,
    replay_rows: u32 = 0,
    /// KV_SPLIT: this producer is shard `shard` of `shards`.
    shard: u32 = 0,
    shards: u32 = 1,

    /// `decoder_from` of a state with no replayed layers
    pub const none: u32 = std.math.maxInt(u32);
};

/// A state read back from the engine, held until `Source.release`.
pub const Captured = struct { ctx: *anyopaque, state: State };

/// Where a state comes from: the engine's active slot at `at` tokens (a device read), or a test's synthetic one.
pub const Source = struct {
    ptr: *anyopaque,
    take: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator, at: u64) anyerror!Captured,
    release: *const fn (ctx: *anyopaque) void,
};

/// An export and everything it reads: the converted parts live in `arena` until `deinit`.
pub const Built = struct {
    arena: std.heap.ArenaAllocator,
    exp: producer.Export,
    /// the parts' bytes: what a table counts against its cap
    bytes: u64,
    /// a streamed export's parts and shape (`skeleton`): converted as the prompt prefills, the rest by `complete`
    parts: []ex.Part = &.{},
    shape: plan.Shape = undefined,

    /// Frees the arena and the frames a responder planned into `exp` with `gpa`; the caller destroys no `Built` itself.
    pub fn deinit(b: *Built, gpa: std.mem.Allocator) void {
        if (b.exp.frames.len > 0) gpa.free(b.exp.frames);
        b.arena.deinit();
        gpa.destroy(b);
    }
};

fn piece(pieces: []const Piece, layer: u32, part: []const u8) ?[]u8 {
    var buf: [48]u8 = undefined;
    const role = std.fmt.bufPrint(&buf, "s.L{d}.{s}", .{ layer, part }) catch return null;
    for (pieces) |p| if (std.mem.eql(u8, p.role, role)) return p.bytes;
    return null;
}

/// A layer's ring (values [rows][576], scales [rows][8]); null when the snapshot holds none for it.
fn ringOf(pieces: []const Piece, layer: u32, kind: []const u8, rows: *u32) Error!?ex.Ring {
    var vb: [8]u8 = undefined;
    var sb: [8]u8 = undefined;
    const v = piece(pieces, layer, std.fmt.bufPrint(&vb, "{s}.v", .{kind}) catch unreachable) orelse return null;
    const s = piece(pieces, layer, std.fmt.bufPrint(&sb, "{s}.s", .{kind}) catch unreachable) orelse return null;
    if (v.len == 0 or v.len % g.cuda_values_bytes != 0 or v.len / g.cuda_values_bytes != s.len / g.cuda_scale_bytes or s.len % g.cuda_scale_bytes != 0) return error.BadSnapshot;
    const n: u32 = @intCast(v.len / g.cuda_values_bytes);
    if (rows.* != 0 and rows.* != n) return error.BadSnapshot; // every ring of a kind has the engine's one row count
    rows.* = n;
    return .{ .values = v, .scales = s };
}

/// Zero the ring slots of positions [from, to): rows the snapshot never held, which would carry another prompt's.
fn clearRows(r: ex.Ring, rows: u32, from: u64, to: u64) void {
    for (from..to) |pos| {
        const slot = pos % rows;
        @memset(@constCast(r.values[slot * g.cuda_values_bytes ..][0..g.cuda_values_bytes]), 0);
        @memset(@constCast(r.scales[slot * g.cuda_scale_bytes ..][0..g.cuda_scale_bytes]), 0);
    }
}

/// A replayed layer's ring holds the last `replay_rows` positions only: the window's older row is zeroed, not stale.
fn clearUnreplayed(st: State, swa: []const ?ex.Ring, rows: u32) Error!void {
    if (st.replay_rows == 0 or rows == 0) return;
    const window: u64 = @min(st.pos, g.release.window);
    const from = st.pos - window;
    const to = st.pos - @min(st.pos, st.replay_rows);
    for (swa[@min(st.decoder_from, swa.len)..]) |maybe| clearRows(maybe orelse return error.MissingRing, rows, from, @max(from, to));
}

/// The sources' pool pages by rank: a split shard knows its own rank's bytes only (the other ranks' stay empty).
fn sourcesOf(arena: std.mem.Allocator, st: State) Error![]ex.SourceState {
    const geo = g.release;
    const split = st.shards > 1;
    if (st.shards == 0 or st.shard >= st.shards or st.shards > plan.max_world) return error.BadShape;
    const sources = try arena.alloc(ex.SourceState, geo.sources.len);
    for (geo.sources, sources) |s, *x| {
        const pg = for (st.pages) |p| {
            if (p.layer == s.layer) break p;
        } else return error.MissingSource;
        const comp = try arena.alloc([]const u8, st.shards);
        const index = try arena.alloc([]const u8, st.shards);
        @memset(comp, &.{});
        @memset(index, &.{});
        comp[st.shard] = pg.comp;
        index[st.shard] = pg.index;
        x.* = .{ .layer = s.layer, .comp = .{ .split = split, .ranks = comp }, .index = .{ .split = split, .ranks = index } };
    }
    return sources;
}

/// The library's view of `st`: slices over the snapshot's bytes, no copy; `ds_rows` rides (0: no drafter rings).
pub fn cudaOf(arena: std.mem.Allocator, st: State, ds_rows: *u32) Error!ex.Cuda {
    const geo = g.release;
    var out: ex.Cuda = .{ .index_format = st.index_format, .sources = &.{}, .swa = &.{}, .carries = &.{} };
    ds_rows.* = 0;
    if (st.shard != 0) { // the other shards carry pool pages only: no ring, carry or drafter row
        out.sources = try sourcesOf(arena, st);
        return out;
    }
    var ring_rows: u32 = 0;
    const swa = try arena.alloc(?ex.Ring, geo.layers);
    for (swa, 0..) |*r, l| r.* = try ringOf(st.pieces, @intCast(l), "swa", &ring_rows);
    if (ring_rows != 0) out.rings.rows = ring_rows;
    try clearUnreplayed(st, swa, ring_rows);
    out.swa = swa;
    const carries = try arena.alloc([]const u8, geo.sources.len);
    for (geo.sources, carries) |s, *c| c.* = if (s.ratio == 2) (piece(st.pieces, s.layer, "carry") orelse return error.MissingCarry) else &.{};
    out.carries = carries;
    out.sources = if (st.pages.len == 0) &.{} else try sourcesOf(arena, st); // none: the pages were streamed, converted already
    // the drafter's three rings ride whole or not at all; a short run of valid rows is zero-filled below it
    var drows: u32 = 0;
    const drafter = try arena.alloc(ex.Ring, geo.mtp_layers);
    var found: u32 = 0;
    for (drafter, 0..) |*r, i| {
        r.* = (try ringOf(st.pieces, geo.layers + @as(u32, @intCast(i)), "ring", &drows)) orelse break;
        found += 1;
    }
    ds_rows.* = 0;
    if (found == geo.mtp_layers) {
        out.drafter = drafter;
        out.rings.drafter_rows = drows;
        const want: u32 = @intCast(@min(st.pos, geo.window));
        if (want > drows) return error.BadSnapshot;
        ds_rows.* = want;
        const valid = @min(st.ds_rows, want);
        for (drafter) |r| clearRows(r, drows, st.pos - want, st.pos - valid);
    }
    return out;
}

/// The export of `st` under handoff `id` for `tokens` (the prompt before its last token): parts, layers, sources.
pub fn build(gpa: std.mem.Allocator, id: [16]u8, tokens: []const u32, st: State) Error!*Built {
    if (tokens.len == 0 or st.pos != tokens.len) return error.NotAHandoffPrompt;
    const b = try gpa.create(Built);
    errdefer gpa.destroy(b);
    b.arena = .init(gpa);
    errdefer b.arena.deinit();
    const a = b.arena.allocator();
    var ds: u32 = 0;
    const cuda = try cudaOf(a, st, &ds);
    const shape: plan.Shape = .{ .upto = st.pos, .ds_rows = ds, .shard = st.shard, .shards = st.shards, .split = st.shards > 1 };
    const parts = try ex.build(a, shape, cuda);
    var bytes: u64 = 0;
    for (parts) |p| bytes += p.bytes.len;
    const laid = try p2.layers(a, parts);
    b.exp = .{ .handoff = id, .tokens = try a.dupe(u32, tokens), .first_token = 0, .tokens_per_row = 1, .layers = laid.layers, .sources = laid.sources };
    b.bytes = bytes;
    b.parts = &.{};
    return b;
}

/// The shape of the export of `pos` tokens with the drafter rows the state will hold (`ds_rows`: 0 for none).
pub fn shapeOf(pos: u64, ds_rows: u32) plan.Shape {
    return .{ .upto = pos, .ds_rows = ds_rows, .shard = 0, .shards = 1, .split = false };
}

/// An export whose parts are allocated, none converted: its manifest is announced now, its frames ready one by one.
pub fn skeleton(gpa: std.mem.Allocator, id: [16]u8, tokens: []const u32, shape: plan.Shape) Error!*Built {
    if (tokens.len == 0 or shape.upto != tokens.len) return error.NotAHandoffPrompt;
    const b = try gpa.create(Built);
    errdefer gpa.destroy(b);
    b.arena = .init(gpa);
    errdefer b.arena.deinit();
    const a = b.arena.allocator();
    const list = try plan.specs(a, shape);
    const parts = try a.alloc(ex.Part, list.len);
    var bytes: u64 = 0;
    for (list, parts) |spec, *p| {
        p.* = .{ .spec = spec, .bytes = try a.alloc(u8, @intCast(spec.bytes())) };
        bytes += p.bytes.len;
    }
    const laid = try p2.layers(a, parts);
    const progress = try a.create(producer.Progress);
    progress.* = .{ .ready = try a.alloc(std.atomic.Value(u8), parts.len) };
    for (progress.ready) |*r| r.* = .init(0);
    b.exp = .{ .handoff = id, .tokens = try a.dupe(u32, tokens), .first_token = 0, .tokens_per_row = 1, .layers = laid.layers, .sources = laid.sources, .progress = progress };
    b.bytes = bytes;
    b.parts = parts;
    b.shape = shape;
    return b;
}

/// The parts a `skeleton` export has not converted yet, converted from `st`; then the export is whole.
pub fn complete(b: *Built, st: State) Error!void {
    const progress = b.exp.progress.?;
    var ds: u32 = 0;
    const cuda = try cudaOf(b.arena.allocator(), st, &ds);
    if (st.pos != b.shape.upto or ds != b.shape.ds_rows) return error.BadSnapshot; // the manifest announced another export
    try ex.checkRings(b.shape, cuda);
    for (b.parts, 0..) |*p, i| {
        if (progress.isReady(i)) continue;
        try ex.convert(p, b.shape, cuda);
        progress.mark(i);
    }
    progress.whole.store(true, .release);
}

/// The bytes of the parts a hand-off at `upto` tokens holds (what `build` allocates), for the table's cap.
pub fn estimate(upto: u64) u64 {
    const geo = g.release;
    const window: u64 = @min(upto, geo.window);
    var n: u64 = (geo.layers + geo.mtp_layers) * window * geo.head_dim * 2;
    for (geo.sources) |s| {
        n += upto / s.ratio * (geo.head_dim + geo.index_dim) * 2;
        if (s.ratio == 2) n += 2 * geo.head_dim * 4;
    }
    return n;
}
