//! Pure host geometry and ABI declarations for Inkling 2 Mini Metal attention.
const std = @import("std");
const config = @import("../config.zig");

pub const head_dim = config.head_dim;
pub const d_rel = config.d_rel;
pub const extent = 1024;
pub const tile_keys = 64;
pub const sliding_chunk = 256;
pub const global_chunk = 16384;
pub const global_sub = 1024;
pub const subs = global_chunk / global_sub;
pub const split_rows = 64;
pub const prompt_pairs = 128;
pub const prompt_threads = 256;
pub const decode_threads = 32;
pub const merge_threads = head_dim;
pub const max_batched_segments = 32;

pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
pub const BatchedSeg = extern struct { k_cache: u64, v_cache: u64, row0: i32, n: i32, p0: i32, ring: u32 };
pub const PrepArgs = extern struct {
    heads: u32,
    kv_heads: u32,
    ring: u32,
    global: u32,
    qk_clip: f32,
    rel_clip: f32,
    floor_pos: f32,
    alpha: f32,
    eps: f32,
};
pub const AttnArgs = extern struct { heads: u32, kv_heads: u32, ring: u32, window: u32, chunk: u32, c0: u32, chunks: u32, scale: f32 };
pub const MergeArgs = extern struct { heads: u32, window: u32, chunk: u32, c0: u32, chunks: u32, subs: u32 = 1 };
pub const PromptArgs = extern struct { heads: u32, kv_heads: u32, ring: u32, chunk: u32, c0: u32, chunks: u32, scale: f32 };
pub const PromptTreeArgs = extern struct { heads: u32, kv_heads: u32, ring: u32, chunk: u32, c0: u32, chunks: u32, scale: f32, p_tile: u32, c_off: u32, max_pos: u32 };
comptime {
    if (@sizeOf(Seg) != 24 or @sizeOf(BatchedSeg) != 32 or @sizeOf(PrepArgs) != 36 or @sizeOf(AttnArgs) != 32) @compileError("Inkling attention ABI drift");
    if (@sizeOf(MergeArgs) != 24 or @sizeOf(PromptArgs) != 28 or @sizeOf(PromptTreeArgs) != 40) @compileError("Inkling prompt ABI drift");
}

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
    pub fn cacheBytes(l: Layer, kv: usize) usize {
        return @as(usize, l.ring) * kv * head_dim * 2;
    }
};
pub const Prep = struct { qk_clip: f32 = 22.627416997969522, rel_clip: f32 = 8, floor: f32 = 8192, alpha: f32 = 0.1, eps: f32 = 1e-6 };
pub const Rows = struct { rows: u32, heads: u32 = 24, kv_heads: u32 = 8, last_pos: u32, streams: u32 = 1 };

pub fn ringFor(window: u32, rows: u32) u32 {
    return @intCast(std.mem.alignForward(usize, @as(usize, window) + rows - 1, tile_keys));
}
pub fn chunks(last_pos: usize, chunk: u32) u32 {
    return @intCast(last_pos / chunk + 1);
}
pub fn slotsFor(l: Layer, context: usize) u32 {
    if (l.window > 0) return (l.window - 1) / l.chunk + 2;
    return chunks(context - 1, l.chunk);
}
pub fn split(r: Rows, stride: u32, slots: usize) bool {
    return r.rows <= split_rows and @as(usize, r.rows) * r.heads * stride * subs <= slots;
}
pub fn usePrompt(r: Rows, l: Layer, stride: u32, slots: usize) bool {
    return l.window == 0 and r.streams == 1 and !split(r, stride, slots);
}
pub fn partialSlot(l: Layer, c: usize, pos: usize) ?usize {
    const lo = if (l.window > 0) (pos + 1) -| l.window else 0;
    if (c < lo / l.chunk or c > pos / l.chunk) return null;
    return c - lo / l.chunk;
}
pub fn partialBytes(rows: usize, heads: u32, n_chunks: u32) usize {
    return rows * heads * n_chunks * (head_dim * 4 + 8);
}

pub fn useGqa3(r: Rows) bool {
    return r.kv_heads > 0 and r.heads == 3 * r.kv_heads;
}
pub fn activeChunks(l: Layer, last_pos: usize) u32 {
    const needed = chunks(last_pos, l.chunk);
    return if (l.window > 0) @min(needed, slotsFor(l, last_pos + 1)) else needed;
}

pub const Path = enum { decode, decode_gqa, prompt };
pub const Launch = struct { path: Path, grid: [3]u32, threads: u32, merge_grid: [3]u32, partials_rh: u32, subs: u32 = 1 };
pub fn attend(r: Rows, l: Layer, stride: u32, slots: usize) error{Inkling2Partials}!Launch {
    if (slotsFor(l, @as(usize, r.last_pos) + 1) > stride) return error.Inkling2Partials;
    const merge_grid = [3]u32{ r.rows * r.heads, 1, 1 };
    if (usePrompt(r, l, stride, slots)) {
        const blocks = std.math.divCeil(u32, r.rows * (r.heads / r.kv_heads), prompt_pairs) catch unreachable;
        return .{ .path = .prompt, .grid = .{ blocks, r.kv_heads, stride }, .threads = prompt_threads, .merge_grid = merge_grid, .partials_rh = stride };
    }
    if (useGqa3(r) and (r.rows >= 2 or activeChunks(l, r.last_pos) >= 16)) {
        return .{ .path = .decode_gqa, .grid = .{ r.rows * r.kv_heads, stride, 1 }, .threads = decode_threads, .merge_grid = merge_grid, .partials_rh = stride };
    }
    return .{ .path = .decode, .grid = .{ r.rows * r.heads, stride, 1 }, .threads = decode_threads, .merge_grid = merge_grid, .partials_rh = stride };
}

pub const BatchedSegSpec = struct {
    rows: u32,
    last_pos: u32,
    layer: Layer,
    is_tree: bool,
};

pub const BatchedLaunch = struct {
    total_rows: u32,
    window: u32,
    chunk: u32,
    tau: bool,
    max_chunks: u32,
    grid: [3]u32,
    threads: u32,
    merge_grid: [3]u32,
};

/// Plans one batched GQA 3:1 decode launch when >= 2 non-tree segments share window/chunk/tau and take `.decode`.
pub fn planBatched(segs: []const BatchedSegSpec, heads: u32, kv_heads: u32, max_stride: u32, slots: usize) ?BatchedLaunch {
    if (segs.len < 2 or segs.len > max_batched_segments) return null;
    if (!useGqa3(.{ .rows = 1, .heads = heads, .kv_heads = kv_heads, .last_pos = 0 })) return null;
    const first = segs[0].layer;
    var total_rows: u32 = 0;
    var max_chunks: u32 = 0;
    for (segs) |seg| {
        if (seg.is_tree or seg.rows == 0 or seg.layer.ring == 0) return null;
        if (seg.layer.window != first.window or seg.layer.chunk != first.chunk or seg.layer.tau != first.tau) return null;
        const need = slotsFor(seg.layer, @as(usize, seg.last_pos) + 1);
        if (need > max_stride) return null;
        const rr = Rows{ .rows = seg.rows, .heads = heads, .kv_heads = kv_heads, .last_pos = seg.last_pos, .streams = 1 };
        if (usePrompt(rr, seg.layer, need, slots)) return null;
        total_rows += seg.rows;
        max_chunks = @max(max_chunks, activeChunks(seg.layer, seg.last_pos));
    }
    if (max_chunks == 0 or max_chunks > max_stride) return null;
    return .{
        .total_rows = total_rows,
        .window = first.window,
        .chunk = first.chunk,
        .tau = first.tau,
        .max_chunks = max_chunks,
        .grid = .{ total_rows * kv_heads, max_chunks, 1 },
        .threads = decode_threads,
        .merge_grid = .{ total_rows * heads, 1, 1 },
    };
}

pub const TreeLaunch = struct {
    global: bool,
    p_tile: u32,
    c_off: u32,
    prefix_grid: [3]u32,
    tail_grid: [3]u32,
    threads: u32,
    merge_grid: [3]u32,
    partials_rh: u32,
};

pub fn attendTree(r: Rows, l: Layer, stride: u32, start: u32) error{Inkling2Partials}!TreeLaunch {
    if (slotsFor(l, @as(usize, r.last_pos) + 1) > stride) return error.Inkling2Partials;
    const merge_grid = [3]u32{ r.rows * r.heads, 1, 1 };
    if (l.window == 0) {
        const p_tile = (start / tile_keys) * tile_keys;
        const prefix_chunks: u32 = if (p_tile > 0) (p_tile - 1) / l.chunk + 1 else 0;
        const c_off = p_tile / l.chunk;
        const c_hi = r.last_pos / l.chunk;
        return .{
            .global = true,
            .p_tile = p_tile,
            .c_off = c_off,
            .prefix_grid = .{ 1, r.kv_heads, prefix_chunks },
            .tail_grid = .{ r.rows, r.kv_heads, c_hi - c_off + 1 },
            .threads = prompt_threads,
            .merge_grid = merge_grid,
            .partials_rh = stride,
        };
    }
    return .{
        .global = false,
        .p_tile = 0,
        .c_off = 0,
        .prefix_grid = .{ 0, 0, 0 },
        .tail_grid = .{ r.rows * r.heads, stride, 1 },
        .threads = decode_threads,
        .merge_grid = merge_grid,
        .partials_rh = stride,
    };
}

test {
    _ = @import("attention_test.zig");
}
