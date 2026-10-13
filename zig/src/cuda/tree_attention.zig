//! tree_attention.cu for every CUDA family: drafted rows over committed keys and their paths, groups in order.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Keys a group, a path's most rows, the part kernels' shared memory (four warps' K/V slots; at 256 also the queries).
pub const span = 2048;
pub const max_depth = 128;
const part_smem = [2]u32{ 4 * 2 * 32 * 16 * 16, 4 * 2 * 16 * 32 * 16 + 16 * 32 * 16 };
/// The one-warp blocks' tile stages (tree_attention.cu's TF_TREE_STAGES) and shared memory (at 256 also the queries).
const stages_w1 = 3;
const part_smem_w1 = [2]u32{ stages_w1 * 2 * 32 * 16 * 16, stages_w1 * 2 * 16 * 32 * 16 + 16 * 32 * 16 };
/// A group's slices: its key tiles dealt to 4 warps, combined in order (in the block, or by the merge).
const slices = 4;
/// (row, key head, group) items one-warp blocks may take: their slices' partials sit apart from the combined ones.
pub const slice_items = 256;

/// row_stream [n], streams [s, 4] (committed at [2]), offs [s, 2] (K, V from origin), paths [n, max_depth], depths.
pub const Rows = struct { row_stream: u64, streams: u64, offs: u64, origin: u64, paths: u64, depths: u64 };

pub const Scratch = struct { o: u64, m: u64, l: u64, so: u64, sm: u64, sl: u64 };

pub const TreeAttention = struct {
    mod: Module,
    paths_fn: Function,
    part_fns: [2]Function, // head dim 128, 256
    shared_fns: [2]Function,
    part_w1: [2]Function, // the same slices, one warp a block
    shared_w1: [2]Function,
    merge_fn: Function,
    /// The shared kernel's blocks a key head: a fixed grid walking every (row, group) (bits do not depend on it).
    shared_blocks: usize = 96,
    /// The device's SMs: where the four-warp grid would leave some idle, a slice takes its own block (the same bits).
    sms: usize = 0,

    pub fn load(d: *const Driver) !TreeAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.tree_attention);
        errdefer mod.unload();
        var parts: [2]Function = .{ try mod.function("tf_tree_part"), try mod.function("tf_tree_part_256") };
        for (&parts, part_smem) |*f, b| try f.allowDynamicShared(b);
        var shared: [2]Function = .{ try mod.function("tf_tree_shared"), try mod.function("tf_tree_shared_256") };
        for (&shared, part_smem) |*f, b| try f.allowDynamicShared(b);
        var parts1: [2]Function = .{ try mod.function("tf_tree_part_w1"), try mod.function("tf_tree_part_256_w1") };
        for (&parts1, part_smem_w1) |*f, b| try f.allowDynamicShared(b);
        var shared1: [2]Function = .{ try mod.function("tf_tree_shared_w1"), try mod.function("tf_tree_shared_256_w1") };
        for (&shared1, part_smem_w1) |*f, b| try f.allowDynamicShared(b);
        return .{ .mod = mod, .paths_fn = try mod.function("tf_tree_paths"), .part_fns = parts, .shared_fns = shared, .part_w1 = parts1, .shared_w1 = shared1, .merge_fn = try mod.function("tf_tree_merge") };
    }

    pub fn unload(g: *TreeAttention) void {
        g.mod.unload();
    }

    /// Groups a row with `keys` committed and path keys needs.
    pub fn groups(keys: usize) usize {
        return (keys + span - 1) / span;
    }

    /// fp32 partials of `rows` rows, `hk` key heads, `ng` groups (o [.., 16, dim], max, sum [.., 16]), then slices.
    pub fn scratchBytes(rows: usize, hk: usize, ng: usize, dim: usize) usize {
        return (rows * hk * ng + slice_items * slices) * 16 * (dim + 2) * 4;
    }

    pub fn scratch(base: u64, rows: usize, hk: usize, ng: usize, dim: usize) Scratch {
        const parts = rows * hk * ng * 16;
        const sb = base + parts * (dim + 2) * 4;
        const sp = slice_items * slices * 16;
        return .{ .o = base, .m = base + parts * dim * 4, .l = base + parts * dim * 4 + parts * 4, .so = sb, .sm = sb + sp * dim * 4, .sl = sb + sp * dim * 4 + sp * 4 };
    }

    /// Each row's path (root first) and depth from `parents` int32 [n] (global window rows, -1 at a root).
    pub fn paths(g: *const TreeAttention, s: Stream, parents: u64, out_paths: u64, depths: u64, n: usize) !void {
        if (n == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ parents, out_paths, depths }) |v| a.add(v);
        a.add(@as(i32, @intCast(n)));
        try launch_.launch(g.paths_fn, .{ .grid = .{ .x = @intCast((n + 127) / 128), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }

    /// q [n, h, dim] (128 or 256) over committed keys and paths (kn, vn: window rows) -> out; ng >= a row's groups.
    pub fn run(g: *const TreeAttention, s: Stream, q: u64, kn: u64, vn: u64, rows: Rows, sc: Scratch, out: u64, n: usize, h: usize, hk: usize, dim: usize, ng: usize, scale: f32) !void {
        if (hk == 0 or h % hk != 0 or h / hk > 16 or ng == 0 or (dim != 128 and dim != 256)) return error.TreeShape;
        const which: usize = if (dim == 256) 1 else 0;
        if (n == 0) return;
        // one warp a block where four-warp blocks would leave SMs idle: each slice its own block, its own partial
        const one = @min(n * ng, g.shared_blocks) * hk < g.sms and n * ng * hk <= slice_items;
        const o = if (one) [3]u64{ sc.so, sc.sm, sc.sl } else [3]u64{ sc.o, sc.m, sc.l };
        var a: launch_.Args = .{};
        for ([_]u64{ q, kn, vn, rows.origin, rows.offs, rows.row_stream, rows.streams, rows.paths, rows.depths, o[0], o[1], o[2] }) |v| a.add(v);
        a.add(@as(i32, @intCast(h)));
        a.add(@as(i32, @intCast(hk)));
        a.add(scale);
        a.add(@as(i32, @intCast(ng)));
        var c: launch_.Args = .{};
        for ([_]u64{ q, rows.origin, rows.offs, rows.row_stream, rows.streams, rows.paths, rows.depths, o[0], o[1], o[2] }) |v| c.add(v);
        c.add(@as(i32, @intCast(h)));
        c.add(@as(i32, @intCast(hk)));
        c.add(scale);
        c.add(@as(i32, @intCast(ng)));
        c.add(@as(i32, @intCast(n)));
        const per: usize = if (one) slices else 1;
        const block: u32 = if (one) 32 else 128;
        const smem = if (one) part_smem_w1[which] else part_smem[which];
        try launch_.launch(if (one) g.shared_w1[which] else g.shared_fns[which], .{ .grid = .{ .x = @intCast(@min(n * ng, g.shared_blocks) * per), .y = @intCast(hk), .z = 1 }, .block = .{ .x = block }, .shared = smem }, s, &c);
        // a row's own block: its boundary group and at most the next (a path is at most 128 keys)
        try launch_.launch(if (one) g.part_w1[which] else g.part_fns[which], .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = @intCast(@min(ng, 2) * per) }, .block = .{ .x = block }, .shared = smem }, s, &a);
        var b: launch_.Args = .{};
        for ([_]u64{ o[0], o[1], o[2], rows.row_stream, rows.streams, rows.depths, out }) |v| b.add(v);
        b.add(@as(i32, @intCast(h)));
        b.add(@as(i32, @intCast(hk)));
        b.add(@as(i32, @intCast(ng)));
        b.add(@as(i32, @intCast(dim)));
        b.add(@as(i32, @intFromBool(one)));
        try launch_.launch(g.merge_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = @intCast(h / hk) }, .block = .{ .x = 256 } }, s, &b);
    }
};
