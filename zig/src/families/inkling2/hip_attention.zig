//! Inkling 2's attention on HIP (zig/kernels/hip/inkling2): the launch plan's geometry and the four launches.
const std = @import("std");
const hip = @import("hip");

pub const head_dim = 128;
pub const d_rel = 16;
pub const extent = 1024;
/// Keys a K/V tile holds (tile.hpp kKeys); a ring holds whole tiles.
pub const tile_keys = 64;
/// (row, head) pairs a decode block holds, one wave each (kWaves), and a prompt block's (kPromptWaves).
pub const tile_pairs = 4;
pub const prompt_pairs = 8;
/// Key chunks: a chunk is one block's keys; rows fold their chunks in order, so a row's bits never depend on its call.
pub const sliding_chunk = 256;
pub const global_chunk = 16384;
/// A global chunk's keys fold in sub-chunks of this many: small calls run each in its own block (tile.hpp kSub).
pub const global_sub = 1024;
pub const subs = global_chunk / global_sub;
/// Most rows a call splits its global chunks for.
pub const split_rows = 64;

/// One stream's rows in a call (tile.hpp's Seg): rows row0.. at positions p0.., its caches `off` elements in.
pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
/// A decode block's (row, head) pairs: segment `seg`, pairs m0..m0+3 of it (tile.hpp's Tile).
pub const Tile = extern struct { seg: i32, m0: i32 };

/// A layer's cache shape: `window` 0 is a global layer; `ring` slots a key head (a multiple of 64).
pub const Layer = struct {
    window: u32,
    ring: u32,
    chunk: u32,
    tau: bool,

    pub fn sliding(window: u32, rows: u32) Layer {
        return .{ .window = window, .ring = ringFor(window, rows), .chunk = sliding_chunk, .tau = false };
    }

    pub fn global(capacity: u32) Layer {
        return .{ .window = 0, .ring = @intCast(std.mem.alignForward(usize, capacity, tile_keys)), .chunk = global_chunk, .tau = true };
    }

    /// Bytes of one stream's K (or V) cache with `kv` key heads: [kv][ring][128] bf16, as #4125's and #4387's.
    pub fn cacheBytes(l: Layer, kv: usize) usize {
        return @as(usize, l.ring) * kv * head_dim * 2;
    }
};

/// A layer's caches (addresses): K and V rows.
pub const Cache = struct { k: u64, v: u64 };

/// Sliding ring slots: the window plus a call's rows less one, so a chunk never overwrites keys it still reads.
pub fn ringFor(window: u32, rows: u32) u32 {
    return @intCast(std.mem.alignForward(usize, @as(usize, window) + rows - 1, tile_keys));
}

/// The decode tiles of a call's segments, `group` query heads a key head; returns how many were written.
pub fn tiles(segs: []const Seg, group: u32, out: []Tile) error{TooManyTiles}!usize {
    var n: usize = 0;
    for (segs, 0..) |s, i| {
        var m: usize = 0;
        while (m < @as(usize, @intCast(s.n)) * group) : (m += tile_pairs) {
            if (n == out.len) return error.TooManyTiles;
            out[n] = .{ .seg = @intCast(i), .m0 = @intCast(m) };
            n += 1;
        }
    }
    return n;
}

/// Tiles of `rows` rows in one segment; a call's total is at most the sum over its segments.
pub fn tileCount(rows: usize, group: u32) usize {
    return std.math.divCeil(usize, rows * group, tile_pairs) catch unreachable;
}

/// Chunks a call addresses: up to its last position's.
pub fn chunks(last_pos: usize, chunk: u32) u32 {
    return @intCast(last_pos / chunk + 1);
}

/// Scratch bytes of a call: the bias table (bf16) and the partials (fp32 sums plus each chunk's max and total).
pub fn tableBytes(rows: usize, heads: u32) usize {
    return rows * heads * extent * 2;
}

pub fn partialBytes(rows: usize, heads: u32, n_chunks: u32) usize {
    return rows * heads * n_chunks * (head_dim * 4 + 8);
}

/// The inkling2 image's launches (hip.kernels.Group.inkling2).
pub const Kernels = struct {
    mod: hip.Module,
    prep: hip.Function,
    rel_table: hip.Function,
    attn: hip.Function,
    attn_fold: hip.Function,
    attn_prompt: hip.Function,
    merge: hip.Function,

    pub fn load(r: *const hip.Runtime, image: []align(8) const u8) !Kernels {
        if (image.len == 0) return error.BuiltWithoutKernels;
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .prep = try mod.function("tf_ink2_prep"),
            .rel_table = try mod.function("tf_ink2_rel_table"),
            .attn = try mod.function("tf_ink2_attn"),
            .attn_fold = try mod.function("tf_ink2_attn_fold"),
            .attn_prompt = try mod.function("tf_ink2_attn_prompt"),
            .merge = try mod.function("tf_ink2_merge"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// The constants tf_ink2_prep takes from config.json (Mini: qk 22.63, rel 8, floor 8,192, alpha 0.1).
pub const Prep = struct { qk_clip: f32, rel_clip: f32, floor: f32, alpha: f32, eps: f32 };

/// The rows' device inputs and outputs (addresses); row arrays are `rows` long, the caches the layer's.
pub const Rows = struct {
    rows: u32,
    heads: u32,
    kv_heads: u32,
    pos: u64,
    seg_of: u64,
    segs: u64,
    tiles: u64,
    n_tiles: u32,
    last_pos: u32, // the call's latest row position (a global layer's chunks run to it)
    streams: u32 = 1, // segments in `segs`: one stream's prompt rows take the prompt kernel
};

fn int(x: anytype) c_int {
    return @intCast(x);
}

fn go(f: hip.Function, s: hip.Stream, grid: hip.launch.Dim3, threads: u32, a: *hip.launch.Args) !void {
    try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

fn addAll(a: *hip.launch.Args, values: anytype) !void {
    inline for (values) |v| try a.add(v);
}

/// q, rel (per head) and k, v (per key head, written at slot pos % ring of the row's segment) from the projections.
pub fn prep(k: *const Kernels, s: hip.Stream, r: Rows, l: Layer, c: Prep, raw: [4]u64, scales: [2]u64, q: u64, rel: u64, kv: Cache) !void {
    var a: hip.launch.Args = .{};
    for (raw) |v| try a.add(v);
    for (scales) |v| try a.add(v);
    for ([_]u64{ r.pos, r.seg_of, r.segs, q, rel, kv.k, kv.v }) |v| try a.add(v);
    try addAll(&a, .{ int(r.heads), int(r.kv_heads), int(l.ring), int(@intFromBool(l.tau)) });
    try addAll(&a, .{ c.qk_clip, c.rel_clip, c.floor, c.alpha, c.eps });
    try go(k.prep, s, .{ .x = r.rows, .y = r.heads + 2 * r.kv_heads }, 32, &a);
}

/// The bias table of each (row, head): its rel vector against the layer's rel_logits_proj [16][1024].
pub fn relTable(k: *const Kernels, s: hip.Stream, r: Rows, rel: u64, proj: u64, tab: u64) !void {
    var a: hip.launch.Args = .{};
    try addAll(&a, .{ rel, proj, tab });
    try go(k.rel_table, s, .{ .x = r.rows * r.heads }, 256, &a);
}

/// Attention into chunk partials, then the ordered merge; `sink` and `gate` RAW (no sigmoid in glue: #4387's bug).
pub fn attend(k: *const Kernels, s: hip.Stream, r: Rows, l: Layer, stride: u32, slots: usize, q: u64, kv: Cache, tab: u64, part_o: u64, part_ml: u64, sink: u64, gate: u64, out: u64) !void {
    if (slotsFor(l, r.last_pos + 1) > stride) return error.Inkling2Partials;
    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    const threads = tile_pairs * 32;
    var a: hip.launch.Args = .{};
    for ([_]u64{ q, kv.k, kv.v, tab, r.segs }) |v| try a.add(v);
    var slots_rh = stride; // partials a row and head
    var per: u32 = 1; // partials a chunk: its sub-chunks' when split
    if (l.window > 0) {
        try addAll(&a, .{ r.tiles, part_o, part_ml });
        for ([_]u32{ r.heads, r.kv_heads, l.ring, l.window, l.chunk, 0, stride }) |v| try a.add(int(v));
        try addAll(&a, .{ scale, int(0) });
        try go(k.attn, s, .{ .x = r.n_tiles, .y = r.kv_heads, .z = tileChunks(r.heads / r.kv_heads, l) }, threads, &a);
    } else if (split(r, stride, slots)) {
        // every sub-chunk of `stride` chunks (past the rows return at once), so graphs replay by stride
        per = subs;
        slots_rh = stride * subs;
        try addAll(&a, .{ r.tiles, part_o, part_ml });
        for ([_]u32{ r.heads, r.kv_heads, l.ring, 0, l.chunk, 0, slots_rh }) |v| try a.add(int(v));
        try addAll(&a, .{ scale, int(1) });
        try go(k.attn, s, .{ .x = r.n_tiles, .y = r.kv_heads, .z = slots_rh }, threads, &a);
    } else if (r.streams == 1) { // one stream's rows: 8 pairs a block share each K/V tile, the fold's bits
        try addAll(&a, .{ part_o, part_ml });
        for ([_]u32{ r.heads, r.kv_heads, l.ring, l.chunk, 0, stride }) |v| try a.add(int(v));
        try a.add(scale);
        const blocks = std.math.divCeil(u32, r.rows * (r.heads / r.kv_heads), prompt_pairs) catch unreachable;
        try go(k.attn_prompt, s, .{ .x = blocks, .y = r.kv_heads, .z = stride }, prompt_pairs * 32, &a);
    } else { // a block a chunk, its sub-chunks folded in order
        try addAll(&a, .{ r.tiles, part_o, part_ml });
        for ([_]u32{ r.heads, r.kv_heads, l.ring, l.chunk, 0, stride }) |v| try a.add(int(v));
        try a.add(scale);
        try go(k.attn_fold, s, .{ .x = r.n_tiles, .y = r.kv_heads, .z = stride }, threads, &a);
    }
    var b: hip.launch.Args = .{};
    for ([_]u64{ part_o, part_ml, sink, gate, r.pos, out }) |v| try b.add(v);
    for ([_]u32{ r.heads, l.window, l.chunk, 0, slots_rh, per }) |v| try b.add(int(v));
    try go(k.merge, s, .{ .x = r.rows * r.heads }, head_dim, &b);
}

/// Whether a global call runs a block a sub-chunk: few rows, and partials for every row, head and sub-chunk.
pub fn split(r: Rows, stride: u32, slots: usize) bool {
    return r.rows <= split_rows and @as(usize, r.rows) * r.heads * stride * subs <= slots;
}

/// Chunks a sliding tile's keys span at most: its rows (4 pairs over `group` heads) plus the window before them.
pub fn tileChunks(group: u32, l: Layer) u32 {
    return ((tile_pairs - 1) / group + l.window) / l.chunk + 2;
}

/// Partials a (row, head) needs on a layer for positions below `context`: a sliding row's window, every global chunk.
pub fn slotsFor(l: Layer, context: usize) u32 {
    if (l.window > 0) return (l.window - 1) / l.chunk + 2;
    return chunks(context - 1, l.chunk);
}

/// The key chunk block `z` of a tile reads (tile.hpp): a sliding tile counts from its first row's first key.
pub fn tileChunk(l: Layer, first_pos: usize, z: usize) usize {
    return if (l.window > 0) ((first_pos + 1) -| l.window) / l.chunk + z else z;
}

/// The partial slot chunk `c` of a row at `pos` is stored in; null outside the row's keys (stores nothing).
pub fn partialSlot(l: Layer, c: usize, pos: usize) ?usize {
    const lo = if (l.window > 0) (pos + 1) -| l.window else 0;
    if (c < lo / l.chunk or c > pos / l.chunk) return null;
    return c - lo / l.chunk;
}

test {
    _ = @import("hip_attention_test.zig");
}
