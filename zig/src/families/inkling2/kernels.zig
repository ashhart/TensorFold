//! Inkling 2's attention kernels (inkling2/attention.cu): the launch plan's geometry and the four launches.
const std = @import("std");
const cuda = @import("cuda");

pub const head_dim = 128;
pub const d_rel = 16;
pub const extent = 1024;
/// Keys a tile of tf_ink2_attn reads at once, and (row, head) pairs a block holds.
pub const tile_keys = 64;
pub const tile_pairs = 64;
/// Key chunks: a chunk is one block's keys; rows fold their chunks in order, so a row's bits never depend on its call.
pub const sliding_chunk = 256;
pub const global_chunk = 16384;
/// A global chunk's keys fold in sub-chunks of this many: small calls run each in its own block (attention.cu kSub).
pub const global_sub = 1024;
pub const subs = global_chunk / global_sub;
/// Most rows a call splits its global chunks for (beyond, the tiles fill the GPU without it).
pub const split_rows = 64;
/// A prompt block's (row, head) pairs (8 warps) and its three K/V tile stages.
pub const prompt_pairs = 128;
pub const prompt_threads = 256;
pub const prompt_shared = 3 * 2 * tile_keys * head_dim * 2;

/// One stream's rows in a call (attention.cu's Seg): rows row0.. at positions p0.., its caches `off` elements in.
pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
/// A block's (row, head) pairs: segment `seg`, pairs m0..m0+63 of it (attention.cu's int2).
pub const Tile = extern struct { seg: i32, m0: i32 };

/// A layer's cache shape for the kernels: `window` 0 is a global layer; `ring` slots a key head (a multiple of 64).
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

    /// Bytes of one stream's K (or V) cache on a rank with `kv` key heads (bf16 rows).
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

/// The tiles of a call's segments, `group` query heads a key head; returns how many were written.
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

pub const Kernels = struct {
    mod: cuda.Module,
    prep: cuda.Function,
    rel_table: cuda.Function,
    attn: cuda.Function,
    attn_fold: cuda.Function,
    merge: cuda.Function,
    pmod: cuda.Module,
    attn_prompt: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_attn);
        errdefer mod.unload();
        var pmod = try cuda.Module.load(d, cuda.kernels.inkling2_pattn);
        errdefer pmod.unload();
        const attn_prompt = try pmod.function("tf_ink2_attn_prompt");
        try attn_prompt.allowDynamicShared(prompt_shared);
        return .{
            .mod = mod,
            .pmod = pmod,
            .attn_prompt = attn_prompt,
            .prep = try mod.function("tf_ink2_prep"),
            .rel_table = try mod.function("tf_ink2_rel_table"),
            .attn = try mod.function("tf_ink2_attn"),
            .attn_fold = try mod.function("tf_ink2_attn_fold"),
            .merge = try mod.function("tf_ink2_merge"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.pmod.unload();
        k.mod.unload();
        k.* = undefined;
    }
};

/// The constants tf_ink2_prep takes from config.json.
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

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try goShared(f, s, grid, threads, 0, a);
}

fn goShared(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, shared: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, s, a);
}

/// Whether heads split into whole groups (query head h reads key head h / (heads / kv_heads)): Small 24/4, Mini 24/8.
pub fn headsFit(r: Rows) error{Inkling2Heads}!void {
    if (r.heads == 0 or r.kv_heads == 0 or r.heads % r.kv_heads != 0) return error.Inkling2Heads;
}

/// q, rel (per head) and k, v (per key head, written to the caches at pos % ring) from the projections' rows.
pub fn prep(k: *const Kernels, s: cuda.Stream, r: Rows, l: Layer, c: Prep, raw: [4]u64, scales: [2]u64, q: u64, rel: u64, kv: Cache) !void {
    try headsFit(r);
    var a: cuda.Args = .{};
    for (raw) |v| a.add(v);
    for (scales) |v| a.add(v);
    for ([_]u64{ r.pos, r.seg_of, r.segs, q, rel, kv.k, kv.v }) |v| a.add(v);
    a.add(int(r.heads));
    a.add(int(r.kv_heads));
    a.add(int(l.ring));
    a.add(int(@intFromBool(l.tau)));
    for ([_]f32{ c.qk_clip, c.rel_clip, c.floor, c.alpha, c.eps }) |v| a.add(v);
    try go(k.prep, s, .{ .x = r.rows, .y = r.heads + 2 * r.kv_heads }, 32, &a);
}

/// The bias table of each (row, head): its rel vector against the layer's rel_logits_proj [16][1024].
pub fn relTable(k: *const Kernels, s: cuda.Stream, r: Rows, rel: u64, proj: u64, tab: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ rel, proj, tab }) |v| a.add(v);
    try go(k.rel_table, s, .{ .x = r.rows * r.heads }, 256, &a);
}

/// Attention into chunk partials (`stride` a row and head; a split call's sub-chunks), then the ordered merge.
pub fn attend(k: *const Kernels, s: cuda.Stream, r: Rows, l: Layer, stride: u32, slots: usize, q: u64, kv: Cache, tab: u64, part_o: u64, part_ml: u64, sink: u64, gate: u64, out: u64) !void {
    try headsFit(r);
    if (slotsFor(l, r.last_pos + 1) > stride) return error.Inkling2Partials;
    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    var a: cuda.Args = .{};
    for ([_]u64{ q, kv.k, kv.v, tab, r.segs, r.tiles, part_o, part_ml }) |v| a.add(v);
    var slots_rh = stride; // partials a row and head
    var per: u32 = 1; // partials a chunk: its sub-chunks' when split
    if (l.window > 0) {
        for ([_]u32{ r.heads, r.kv_heads, l.ring, l.window, l.chunk, 0, stride }) |v| a.add(int(v));
        a.add(scale);
        a.add(int(0));
        try go(k.attn, s, .{ .x = r.n_tiles, .y = r.kv_heads, .z = tileChunks(r.heads / r.kv_heads, l) }, 128, &a);
    } else if (split(r, stride, slots)) {
        // every sub-chunk of `stride` chunks (past the rows return at once), so graphs replay by stride
        per = subs;
        slots_rh = stride * subs;
        for ([_]u32{ r.heads, r.kv_heads, l.ring, 0, l.chunk, 0, slots_rh }) |v| a.add(int(v));
        a.add(scale);
        a.add(int(1));
        try go(k.attn, s, .{ .x = r.n_tiles, .y = r.kv_heads, .z = slots_rh }, 128, &a);
    } else if (r.streams == 1) { // one stream's rows: 128 pairs a block share each K/V tile, the fold's bits
        var p: cuda.Args = .{};
        for ([_]u64{ q, kv.k, kv.v, tab, r.segs, part_o, part_ml }) |v| p.add(v);
        for ([_]u32{ r.heads, r.kv_heads, l.ring, l.chunk, 0, stride }) |v| p.add(int(v));
        p.add(scale);
        const blocks = std.math.divCeil(u32, r.rows * (r.heads / r.kv_heads), prompt_pairs) catch unreachable;
        try goShared(k.attn_prompt, s, .{ .x = blocks, .y = r.kv_heads, .z = stride }, prompt_threads, prompt_shared, &p);
    } else { // a block a chunk, its sub-chunks folded in order
        for ([_]u32{ r.heads, r.kv_heads, l.ring, l.chunk, 0, stride }) |v| a.add(int(v));
        a.add(scale);
        try go(k.attn_fold, s, .{ .x = r.n_tiles, .y = r.kv_heads, .z = stride }, 128, &a);
    }
    var b: cuda.Args = .{};
    for ([_]u64{ part_o, part_ml, sink, gate, r.pos, out }) |v| b.add(v);
    for ([_]u32{ r.heads, l.window, l.chunk, 0, slots_rh, per }) |v| b.add(int(v));
    try go(k.merge, s, .{ .x = r.rows * r.heads }, head_dim, &b);
}

/// Whether a global call runs a block a sub-chunk: few rows, and partials for every row, head and sub-chunk.
pub fn split(r: Rows, stride: u32, slots: usize) bool {
    return r.rows <= split_rows and @as(usize, r.rows) * r.heads * stride * subs <= slots;
}

/// Chunks a sliding tile's keys span at most: its rows (64 pairs over `group` heads) plus the window before them.
pub fn tileChunks(group: u32, l: Layer) u32 {
    return (63 / group + l.window) / l.chunk + 2;
}

/// Partials a (row, head) needs on a layer for positions below `context`: a sliding row's window, every global chunk.
pub fn slotsFor(l: Layer, context: usize) u32 {
    if (l.window > 0) return (l.window - 1) / l.chunk + 2;
    return chunks(context - 1, l.chunk);
}

/// The key chunk block `z` of a tile reads (attention.cu): a sliding tile counts from its first row's first key.
pub fn tileChunk(l: Layer, first_pos: usize, z: usize) usize {
    return if (l.window > 0) ((first_pos + 1) -| l.window) / l.chunk + z else z;
}

/// The partial slot tf_ink2_attn stores chunk `c` of a row at `pos` in; null outside the row's keys (stores nothing).
pub fn partialSlot(l: Layer, c: usize, pos: usize) ?usize {
    const lo = if (l.window > 0) (pos + 1) -| l.window else 0;
    if (c < lo / l.chunk or c > pos / l.chunk) return null;
    return c - lo / l.chunk;
}

test {
    _ = @import("kernels_test.zig");
}
