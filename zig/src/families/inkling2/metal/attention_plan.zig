//! Pure host geometry and ABI declarations for Inkling 2 Mini Metal attention.
const std = @import("std");
const config = @import("../config.zig");

pub const head_dim = config.head_dim;
pub const d_rel = config.d_rel;
pub const extent = 1024;
pub const tile_keys = 64;
pub const sliding_chunk = 32;
pub const global_chunk = 16384;
pub const global_sub = 1024;
pub const subs = global_chunk / global_sub;
pub const split_rows = 64;
pub const prompt_pairs = 128;
pub const prompt_threads = 256;
pub const decode_threads = 192;
pub const tree_decode_threads = 32;
pub const decode_row_block = 2;
pub const decode_tile_keys = 32;
pub const merge_threads = head_dim;
pub const max_slots = 8;

pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
pub const SlotMeta = extern struct { ring: u32, chunk: u32, pad0: u32 = 0, pad1: u32 = 0 };
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
pub const PrepSlotsArgs = extern struct {
    heads: u32,
    kv_heads: u32,
    global: u32,
    n_slots: u32,
    rings: [max_slots]u32,
    qk_clip: f32,
    rel_clip: f32,
    floor_pos: f32,
    alpha: f32,
    eps: f32,
};
pub const AttnArgs = extern struct { heads: u32, kv_heads: u32, ring: u32, window: u32, chunk: u32, c0: u32, chunks: u32, scale: f32, rows: u32 = 0, pad: u32 = 0 };
pub const AttnSlotsArgs = extern struct {
    heads: u32,
    kv_heads: u32,
    window: u32,
    c0: u32,
    chunks: u32,
    n_slots: u32,
    scale: f32,
    rows: u32,
    slots: [max_slots]SlotMeta,
};
pub const MergeArgs = extern struct { heads: u32, window: u32, chunk: u32, c0: u32, chunks: u32, subs: u32 = 1 };
pub const PromptArgs = extern struct { heads: u32, kv_heads: u32, ring: u32, chunk: u32, c0: u32, chunks: u32, scale: f32 };
pub const PromptTreeArgs = extern struct { heads: u32, kv_heads: u32, ring: u32, chunk: u32, c0: u32, chunks: u32, scale: f32, p_tile: u32, c_off: u32, max_pos: u32 };
comptime {
    if (@sizeOf(Seg) != 24 or @sizeOf(PrepArgs) != 36 or @sizeOf(AttnArgs) != 40) @compileError("Inkling attention ABI drift");
    if (@sizeOf(MergeArgs) != 24 or @sizeOf(PromptArgs) != 28 or @sizeOf(PromptTreeArgs) != 40) @compileError("Inkling prompt ABI drift");
    if (@sizeOf(SlotMeta) != 16 or @sizeOf(PrepSlotsArgs) != 68 or @sizeOf(AttnSlotsArgs) != 160 or @offsetOf(AttnSlotsArgs, "slots") != 32) @compileError("Inkling slots ABI drift");
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
        return .{ .window = 0, .ring = @intCast(std.mem.alignForward(usize, capacity, tile_keys)), .chunk = globalChunk(capacity), .tau = true };
    }
    pub fn cacheBytes(l: Layer, kv: usize) usize {
        return @as(usize, l.ring) * kv * head_dim * 2;
    }
};
pub const Prep = struct { qk_clip: f32 = 22.627416997969522, rel_clip: f32 = 8, floor: f32 = 8192, alpha: f32 = 0.1, eps: f32 = 1e-6 };
pub const Rows = struct { rows: u32, heads: u32 = 24, kv_heads: u32 = 8, last_pos: u32, streams: u32 = 1 };

/// The fewest keys a slot that fit `capacity` in 64 slots; any whole number of 1,024-key folds gives the same bits.
pub fn globalChunk(capacity: u32) u32 {
    var chunk: u32 = global_sub;
    while (@as(u64, chunk) * 64 < capacity and chunk < global_chunk) chunk *= 2;
    return chunk;
}

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

pub const Path = enum { decode, prompt };
pub const Launch = struct { path: Path, grid: [3]u32, threads: u32, merge_grid: [3]u32, partials_rh: u32, subs: u32 = 1 };
pub fn attend(r: Rows, l: Layer, stride: u32, slots: usize) error{Inkling2Partials}!Launch {
    if (slotsFor(l, @as(usize, r.last_pos) + 1) > stride) return error.Inkling2Partials;
    const merge_grid = [3]u32{ r.rows * r.heads, 1, 1 };
    if (usePrompt(r, l, stride, slots)) {
        const blocks = std.math.divCeil(u32, r.rows * (r.heads / r.kv_heads), prompt_pairs) catch unreachable;
        return .{ .path = .prompt, .grid = .{ blocks, r.kv_heads, stride }, .threads = prompt_threads, .merge_grid = merge_grid, .partials_rh = stride };
    }
    const row_blocks = std.math.divCeil(u32, r.rows, decode_row_block) catch unreachable;
    return .{ .path = .decode, .grid = .{ r.kv_heads, stride, row_blocks }, .threads = decode_threads, .merge_grid = merge_grid, .partials_rh = stride };
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
        .threads = tree_decode_threads,
        .merge_grid = merge_grid,
        .partials_rh = stride,
    };
}

test {
    _ = @import("attention_test.zig");
}
