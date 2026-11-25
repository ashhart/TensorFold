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

pub const Seg = extern struct { row0: i32, n: i32, p0: i32, pad: i32 = 0, off: i64 };
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
comptime {
    if (@sizeOf(Seg) != 24 or @sizeOf(PrepArgs) != 36 or @sizeOf(AttnArgs) != 32) @compileError("Inkling attention ABI drift");
    if (@sizeOf(MergeArgs) != 24 or @sizeOf(PromptArgs) != 28) @compileError("Inkling prompt ABI drift");
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
pub fn tableBytes(rows: usize, heads: u32) usize {
    return rows * heads * extent * 2;
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
    return .{ .path = .decode, .grid = .{ r.rows * r.heads, stride, 1 }, .threads = decode_threads, .merge_grid = merge_grid, .partials_rh = stride };
}

test {
    _ = @import("attention_test.zig");
}
