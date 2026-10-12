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

/// row_stream [n], streams [s, 4] (committed at [2]), offs [s, 2] (K, V from origin), paths [n, max_depth], depths.
pub const Rows = struct { row_stream: u64, streams: u64, offs: u64, origin: u64, paths: u64, depths: u64 };

pub const Scratch = struct { o: u64, m: u64, l: u64 };

pub const TreeAttention = struct {
    mod: Module,
    paths_fn: Function,
    part_fns: [2]Function, // head dim 128, 256
    shared_fns: [2]Function,
    merge_fn: Function,
    /// The shared kernel's blocks a key head: a fixed grid walking every (row, group) (bits do not depend on it).
    shared_blocks: usize = 96,

    pub fn load(d: *const Driver) !TreeAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.tree_attention);
        errdefer mod.unload();
        var parts: [2]Function = .{ try mod.function("tf_tree_part"), try mod.function("tf_tree_part_256") };
        for (&parts, part_smem) |*f, b| try f.allowDynamicShared(b);
        var shared: [2]Function = .{ try mod.function("tf_tree_shared"), try mod.function("tf_tree_shared_256") };
        for (&shared, part_smem) |*f, b| try f.allowDynamicShared(b);
        return .{ .mod = mod, .paths_fn = try mod.function("tf_tree_paths"), .part_fns = parts, .shared_fns = shared, .merge_fn = try mod.function("tf_tree_merge") };
    }

    pub fn unload(g: *TreeAttention) void {
        g.mod.unload();
    }

    /// Groups a row with `keys` committed and path keys needs.
    pub fn groups(keys: usize) usize {
        return (keys + span - 1) / span;
    }

    /// The fp32 partials of `rows` rows, `hk` key heads and `ng` groups: o [.., 16, dim], then max and sum [.., 16].
    pub fn scratchBytes(rows: usize, hk: usize, ng: usize, dim: usize) usize {
        return rows * hk * ng * 16 * (dim + 2) * 4;
    }

    pub fn scratch(base: u64, rows: usize, hk: usize, ng: usize, dim: usize) Scratch {
        const parts = rows * hk * ng * 16;
        return .{ .o = base, .m = base + parts * dim * 4, .l = base + parts * dim * 4 + parts * 4 };
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
        var a: launch_.Args = .{};
        for ([_]u64{ q, kn, vn, rows.origin, rows.offs, rows.row_stream, rows.streams, rows.paths, rows.depths, sc.o, sc.m, sc.l }) |v| a.add(v);
        a.add(@as(i32, @intCast(h)));
        a.add(@as(i32, @intCast(hk)));
        a.add(scale);
        a.add(@as(i32, @intCast(ng)));
        var c: launch_.Args = .{};
        for ([_]u64{ q, rows.origin, rows.offs, rows.row_stream, rows.streams, rows.paths, rows.depths, sc.o, sc.m, sc.l }) |v| c.add(v);
        c.add(@as(i32, @intCast(h)));
        c.add(@as(i32, @intCast(hk)));
        c.add(scale);
        c.add(@as(i32, @intCast(ng)));
        c.add(@as(i32, @intCast(n)));
        try launch_.launch(g.shared_fns[which], .{ .grid = .{ .x = @intCast(@min(n * ng, g.shared_blocks)), .y = @intCast(hk), .z = 1 }, .block = .{ .x = 128 }, .shared = part_smem[which] }, s, &c);
        // a row's own block: its boundary group and at most the next (a path is at most 128 keys)
        try launch_.launch(g.part_fns[which], .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = @intCast(@min(ng, 2)) }, .block = .{ .x = 128 }, .shared = part_smem[which] }, s, &a);
        var b: launch_.Args = .{};
        for ([_]u64{ sc.o, sc.m, sc.l, rows.row_stream, rows.streams, rows.depths, out }) |v| b.add(v);
        b.add(@as(i32, @intCast(h)));
        b.add(@as(i32, @intCast(hk)));
        b.add(@as(i32, @intCast(ng)));
        b.add(@as(i32, @intCast(dim)));
        try launch_.launch(g.merge_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = 1 }, .block = .{ .x = 256 } }, s, &b);
    }
};
