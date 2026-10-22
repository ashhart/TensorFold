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
pub const span = 512;
pub const max_depth = 128;
const part_smem = [2]u32{ 4 * 2 * 32 * 16 * 16, 4 * 2 * 16 * 32 * 16 + 16 * 32 * 16 };
/// One-warp blocks' shared memory by tile stages (2: two blocks an SM; 3: one, deeper), at 256 also the queries.
fn smemW1(stages: u32, which: usize) u32 {
    return if (which == 0) stages * 2 * 32 * 16 * 16 else stages * 2 * 16 * 32 * 16 + 16 * 32 * 16;
}
/// A group's slices: its key tiles dealt to 4 warps, combined in order (in the block, or by the merge).
const slices = 4;
/// (row, key head, group) items one-warp blocks may take: their slices' partials sit apart from the combined ones.
pub const slice_items = 1024;

/// row_stream [n], streams [s, 4] (committed at [2]), offs [s, 2] (K, V from origin), paths [n, max_depth], depths.
pub const Rows = struct { row_stream: u64, streams: u64, offs: u64, origin: u64, paths: u64, depths: u64 };

pub const Scratch = struct { o: u64, m: u64, l: u64, so: u64, sm: u64, sl: u64 };

pub const TreeAttention = struct {
    mod: Module,
    paths_fn: Function,
    part_fns: [2]Function, // head dim 128, 256
    shared_fns: [2]Function,
    part_w1: [2][2]Function, // the same slices, one warp a block: [2 stages, 3 stages][head dim]
    shared_w1: [2][2]Function,
    merge_fn: Function,
    /// The shared kernel's blocks a key head: a fixed grid walking every (row, group) (bits do not depend on it).
    shared_blocks: usize = 96,
    /// The device's SMs (0: four-warp blocks only): small grids take pipelined one-warp blocks (the same bits).
    sms: usize = 0,

    pub fn load(d: *const Driver) !TreeAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.tree_attention);
        errdefer mod.unload();
        var parts: [2]Function = .{ try mod.function("tf_tree_part"), try mod.function("tf_tree_part_256") };
        for (&parts, part_smem) |*f, b| try f.allowDynamicShared(b);
        var shared: [2]Function = .{ try mod.function("tf_tree_shared"), try mod.function("tf_tree_shared_256") };
        for (&shared, part_smem) |*f, b| try f.allowDynamicShared(b);
        var parts1: [2][2]Function = .{ .{ try mod.function("tf_tree_part_w1"), try mod.function("tf_tree_part_256_w1") }, .{ try mod.function("tf_tree_part_w1s3"), try mod.function("tf_tree_part_256_w1s3") } };
        var shared1: [2][2]Function = .{ .{ try mod.function("tf_tree_shared_w1"), try mod.function("tf_tree_shared_256_w1") }, .{ try mod.function("tf_tree_shared_w1s3"), try mod.function("tf_tree_shared_256_w1s3") } };
        for (0..2) |st| for (0..2) |which| {
            try parts1[st][which].allowDynamicShared(smemW1(@intCast(st + 2), which));
            try shared1[st][which].allowDynamicShared(smemW1(@intCast(st + 2), which));
        };
        return .{ .mod = mod, .paths_fn = try mod.function("tf_tree_paths"), .part_fns = parts, .shared_fns = shared, .part_w1 = parts1, .shared_w1 = shared1, .merge_fn = try mod.function("tf_tree_merge") };
    }

    pub fn unload(g: *TreeAttention) void {
        g.mod.unload();
    }

    /// Groups a row with `keys` committed and path keys needs.
    pub fn groups(keys: usize) usize {
        return (keys + span - 1) / span;
    }

    /// fp32 partials of `rows` rows' `h` heads (`hk` key heads) in `ng` groups: o [.., dim], max, sum; then slices.
    pub fn scratchBytes(rows: usize, h: usize, hk: usize, ng: usize, dim: usize) usize {
        return (rows * h * ng + slice_items * slices * (h / hk)) * (dim + 2) * 4;
    }

    pub fn scratch(base: u64, rows: usize, h: usize, hk: usize, ng: usize, dim: usize) Scratch {
        const parts = rows * h * ng;
        const sb = base + parts * (dim + 2) * 4;
        const sp = slice_items * slices * (h / hk);
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
        // small grids (a few rows): each slice its own pipelined one-warp block and partial; else four-warp blocks
        const one = g.sms > 0 and n * ng * hk <= slice_items;
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
        c.add(kn); // the replay of a NaN or infinite partial weighs a row's path keys too
        const per: usize = if (one) slices else 1;
        const block: u32 = if (one) 32 else 128;
        // three stages where the one-warp grid fits one wave of the SMs (a block each), else two (two blocks an SM)
        const deep: usize = if (@min(n * ng, g.shared_blocks) * slices * hk <= g.sms) 1 else 0;
        const smem = if (one) smemW1(@intCast(deep + 2), which) else part_smem[which];
        try launch_.launch(if (one) g.shared_w1[deep][which] else g.shared_fns[which], .{ .grid = .{ .x = @intCast(@min(n * ng, g.shared_blocks) * per), .y = @intCast(hk), .z = 1 }, .block = .{ .x = block }, .shared = smem }, s, &c);
        // a row's own block: its boundary group and at most the next (a path is at most 128 keys)
        try launch_.launch(if (one) g.part_w1[deep][which] else g.part_fns[which], .{ .grid = .{ .x = @intCast(n), .y = @intCast(hk), .z = @intCast(@min(ng, 2) * per) }, .block = .{ .x = block }, .shared = smem }, s, &a);
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
