//! Ripple's DFlash2 draft attention and top-32 on CUDA (ricursive_dflash2_attn.cu) at dflash2.metal's rounding points.

const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;

pub const head_dim = 128;
/// Rows a stream's block holds (DFlash2's block size).
pub const block = 16;
/// Keys a partial covers: rc_df_vec's 128-key blocks.
pub const chunk = 128;
/// The released drafter's heads: 80 query heads over 8 key heads (a group of 10).
pub const heads = 80;
pub const kv_heads = 8;
/// The softmax scale, 1 / sqrt(128) in fp32 (dflash2.zig's).
pub const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
/// The sliding layers' window and the global (NoPE) layers'.
pub const sliding_span = 4096;
pub const global_span = 16384;

/// tf_rcd_attn_part's and tf_rcd_attn_merge's __launch_bounds__: 8 warps.
pub const attn_threads = 256;
const warps = attn_threads / 32;
/// tf_rcd_attn_part's dynamic shared memory: a chunk's K and V of one key head in bf16 (64 KiB, opted into at load).
pub const part_shared = 2 * chunk * head_dim * 2;
/// rc_df_topk: 1,024 threads sort 2,048 values a chunk and keep 32.
pub const top_threads = 1024;
pub const top_chunk = 2048;
pub const top = 32;
/// The drafter's vocabulary: 49 chunks of 2,048.
pub const vocab = 100352;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) usize {
    return (n + d - 1) / d;
}

pub const Kernels = struct {
    mod: cuda.Module,
    attn_part: Function,
    attn_merge: Function,
    top32_part: Function,
    top32_merge: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.ricursive_dflash2_attn);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .attn_part = try partFunction(mod, "tf_rcd_attn_part"),
            .attn_merge = try mod.function("tf_rcd_attn_merge"),
            .top32_part = try mod.function("tf_rcd_top32_part"),
            .top32_merge = try mod.function("tf_rcd_top32_merge"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// An attention partial of the image (the served one or a control), opted into its 64 KiB of dynamic shared memory.
pub fn partFunction(mod: cuda.Module, name: [:0]const u8) !Function {
    const f = try mod.function(name);
    try f.allowDynamicShared(part_shared);
    return f;
}

/// A stream's slice of a layer: ring [rows][8][128] K then V, p at row p % rows; block at anchor; lowest position from.
pub const Segment = extern struct {
    ring: u64,
    anchor: i32,
    from: i32,
    span: i32,
    rows: i32,
};

/// A layer ring's rows for a window: the window and the block, to 128s (4,224 and 16,512 for the released windows).
pub fn ringRows(span: usize) usize {
    return std.mem.alignForward(usize, span + block, chunk);
}

/// The 128-key chunks a (row, head) holds partials for: a window plus the block, span + 15 keys, from any start.
pub fn chunksFor(span: usize) usize {
    return ceil(span + block - 1, chunk) + 1;
}

/// A layer's attention over streams: q, out [streams 16][heads][128] bf16; segs on the device; pm, pl, po partials.
pub const Attn = struct {
    q: u64,
    segs: u64,
    out: u64,
    pm: u64,
    pl: u64,
    po: u64,
    streams: usize,
    span: usize,
    heads: usize = heads,
    kv_heads: usize = kv_heads,
    scale: f32 = scale,

    pub fn chunks(p: Attn) usize {
        return chunksFor(p.span);
    }
};

/// The partials' bytes: pm and pl each `max`, po `values`.
pub const Scratch = struct { max: usize, values: usize };

pub fn scratch(p: Attn) Scratch {
    const slots = p.streams * p.heads * block * p.chunks();
    return .{ .max = slots * 4, .values = slots * head_dim * 4 };
}

/// The kernels' shapes: whole groups, grids in CUDA's limits, int32 indices; the caller keeps segs in span and ring.
pub fn admit(p: Attn) !void {
    if (p.kv_heads == 0 or p.heads == 0 or p.heads % p.kv_heads != 0) return error.AttnShape;
    if (p.streams > 65535 or p.chunks() > 65535 or p.span == 0 or p.span > std.math.maxInt(i32) / 2) return error.AttnShape;
    if (p.streams * block * p.heads > std.math.maxInt(i32)) return error.AttnShape;
}

/// tf_rcd_attn_part: a block a (key head, stream, chunk).
pub fn partGrid(p: Attn) cuda.Dim3 {
    return .{ .x = @intCast(p.kv_heads), .y = @intCast(p.streams), .z = @intCast(p.chunks()) };
}

/// tf_rcd_attn_merge: a warp a (row, head), 8 a block.
pub fn mergeGrid(p: Attn) cuda.Dim3 {
    return .{ .x = @intCast(ceil(p.streams * block * p.heads, warps)), .y = 1, .z = 1 };
}

pub fn attention(k: *const Kernels, s: Stream, p: Attn) !void {
    return attentionVia(k.attn_part, k.attn_merge, s, p);
}

/// The two launches on `part` (opted into part_shared) and `merge`: the served kernels or a control in their place.
pub fn attentionVia(part: Function, merge: Function, s: Stream, p: Attn) !void {
    if (p.streams == 0) return;
    try partVia(part, s, p);
    const cap = int(p.chunks());
    var m: cuda.Args = .{};
    for ([_]u64{ p.pm, p.pl, p.po, p.out }) |v| m.add(v);
    m.add(int(p.streams * block));
    m.add(int(p.heads));
    m.add(cap);
    try cuda.launch.launch(merge, .{ .grid = mergeGrid(p), .block = .{ .x = attn_threads }, .shared = 0 }, s, &m);
}

/// The partial launch alone (the chunks' maxima, sums and weighted values into pm, pl and po), for timing.
pub fn partVia(part: Function, s: Stream, p: Attn) !void {
    if (p.streams == 0) return;
    try admit(p);
    var a: cuda.Args = .{};
    for ([_]u64{ p.q, p.segs, p.pm, p.pl, p.po }) |v| a.add(v);
    a.add(int(p.kv_heads));
    a.add(int(p.heads / p.kv_heads));
    a.add(int(p.chunks()));
    a.add(p.scale);
    try cuda.launch.launch(part, .{ .grid = partGrid(p), .block = .{ .x = attn_threads }, .shared = part_shared }, s, &a);
}

/// Each row's best 32 of n bf16 logits: vals, ids [rows][32] by value, then id; part_vals, part_ids the chunks' bests.
pub const Top32 = struct {
    logits: u64,
    stride: usize,
    n: usize = vocab,
    rows: usize,
    vals: u64,
    ids: u64,
    part_vals: u64,
    part_ids: u64,

    pub fn chunks(p: Top32) usize {
        return ceil(p.n, top_chunk);
    }
};

/// The bytes each of part_vals and part_ids holds.
pub fn top32Scratch(p: Top32) usize {
    return p.rows * p.chunks() * top * 4;
}

/// One merge: the chunks' bests fit one 2,048-value sort (n at most 131,072).
pub fn admitTop32(p: Top32) !void {
    if (p.n == 0 or p.stride < p.n or p.chunks() * top > top_chunk) return error.Top32Shape;
    if (p.rows > 65535 or p.stride > std.math.maxInt(i32)) return error.Top32Shape;
}

pub fn top32(k: *const Kernels, s: Stream, p: Top32) !void {
    return top32Via(k.top32_part, k.top32_merge, s, p);
}

/// The part and merge launches on `part` and `merge`: the served kernels or a control in their place.
pub fn top32Via(part: Function, merge: Function, s: Stream, p: Top32) !void {
    if (p.rows == 0) return;
    try admitTop32(p);
    const chunks = p.chunks();
    var a: cuda.Args = .{};
    a.add(p.logits);
    a.add(int(p.n));
    a.add(int(p.stride));
    a.add(int(chunks));
    a.add(p.part_vals);
    a.add(p.part_ids);
    try cuda.launch.launch(part, .{ .grid = .{ .x = @intCast(chunks), .y = @intCast(p.rows), .z = 1 }, .block = .{ .x = top_threads }, .shared = 0 }, s, &a);
    var m: cuda.Args = .{};
    m.add(p.part_vals);
    m.add(p.part_ids);
    m.add(int(chunks * top));
    m.add(int(chunks * top));
    m.add(@as(c_int, 1));
    m.add(p.vals);
    m.add(p.ids);
    try cuda.launch.launch(merge, .{ .grid = .{ .x = 1, .y = @intCast(p.rows), .z = 1 }, .block = .{ .x = top_threads }, .shared = 0 }, s, &m);
}

test "rings and chunks: 4,224 and 16,512 rows, 34 and 130 chunks a (row, head), the Segment as the kernel's" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 4224), ringRows(sliding_span));
    try t.expectEqual(@as(usize, 16512), ringRows(global_span));
    try t.expectEqual(@as(usize, 34), chunksFor(sliding_span));
    try t.expectEqual(@as(usize, 130), chunksFor(global_span));
    // the widest span + 15 keys from the worst start (one past a chunk's first) still fit: 33 chunks, 129 chunks
    try t.expect((127 + sliding_span + block - 1 + chunk - 1) / chunk <= chunksFor(sliding_span));
    try t.expect((127 + global_span + block - 1 + chunk - 1) / chunk <= chunksFor(global_span));
    try t.expectEqual(@as(usize, 24), @sizeOf(Segment));
    try t.expectEqual(@as(usize, 8), @offsetOf(Segment, "anchor"));
}

test "launch geometry and partials: 8 key heads by streams by chunks, a warp a (row, head)" {
    const t = std.testing;
    const p: Attn = .{ .q = 0, .segs = 0, .out = 0, .pm = 0, .pl = 0, .po = 0, .streams = 8, .span = global_span };
    const g = partGrid(p);
    try t.expectEqual(@as(c_uint, 8), g.x);
    try t.expectEqual(@as(c_uint, 8), g.y);
    try t.expectEqual(@as(c_uint, 130), g.z);
    try t.expectEqual(@as(c_uint, 8 * 16 * 80 / 8), mergeGrid(p).x);
    // the map's 86.5 MB a stream at 128-key chunks: 80 heads x 16 rows x 130 chunks x 128 fp32
    try t.expectEqual(@as(usize, 80 * 16 * 130 * 128 * 4), scratch(.{ .q = 0, .segs = 0, .out = 0, .pm = 0, .pl = 0, .po = 0, .streams = 1, .span = global_span }).values);
    try t.expectEqual(@as(usize, 65536), part_shared);
}

test "admit: whole groups, a grid CUDA launches, one merge of the top-32" {
    const t = std.testing;
    var p: Attn = .{ .q = 0, .segs = 0, .out = 0, .pm = 0, .pl = 0, .po = 0, .streams = 8, .span = sliding_span };
    try admit(p);
    p.heads = 81;
    try t.expectError(error.AttnShape, admit(p));
    p.heads = 80;
    p.streams = 70000;
    try t.expectError(error.AttnShape, admit(p));
    var q: Top32 = .{ .logits = 0, .stride = vocab, .rows = 128, .vals = 0, .ids = 0, .part_vals = 0, .part_ids = 0 };
    try admitTop32(q);
    try t.expectEqual(@as(usize, 49), q.chunks());
    q.n = 131072;
    q.stride = q.n;
    try admitTop32(q);
    q.n = 131073;
    q.stride = q.n;
    try t.expectError(error.Top32Shape, admitTop32(q));
    q.n = vocab;
    q.stride = vocab - 1;
    try t.expectError(error.Top32Shape, admitTop32(q));
}
