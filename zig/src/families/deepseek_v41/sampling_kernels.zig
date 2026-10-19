//! sampling.cu's launches in their own optional fatbin (`dsv41_sampling`): nucleus statistics and packed rows.

const std = @import("std");
const cuda = @import("cuda");

const Blob = struct {
    pub const bytes align(16) = @embedFile("dsv41_fatbin_dsv41_sampling").*;
};
const image: []const u8 = &Blob.bytes;

/// The build embedded the kernel (-Dnvcc, or -Dfatbins with dsv41_sampling.fatbin).
pub const available = image.len > 0;

pub const Functions = struct {
    module: cuda.Module,
    stats_fn: cuda.Function,
    pack_fn: cuda.Function,
    /// the split statistics (every SM; null in a fatbin built before them: `stats` then runs one CTA a row)
    split: ?Split = null,
    /// why `split` is null: the symbol that did not resolve, and the driver's error
    split_missing: ?struct { symbol: [:0]const u8, err: anyerror } = null,

    pub const Split = struct { max: cuda.Function, sum: cuda.Function, fin: cuda.Function };

    /// Columns a CTA of the split statistics covers (64,640 a rank: 32 chunks a row).
    pub const chunk: usize = 2048;

    pub fn load(d: *const cuda.Driver) !Functions {
        if (!available) return error.SamplingKernelNotBuilt;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var f: Functions = .{
            .module = m,
            .stats_fn = try m.function("_ZN14dsv41_sampling12stats_kernelEPKfxidPd"),
            .pack_fn = try m.function("_ZN14dsv41_sampling11pack_kernelEPKfPKxiiPf"),
        };
        var fs: [3]cuda.Function = undefined;
        for (split_syms, &fs) |sym, *x| x.* = m.function(sym) catch |e| {
            f.split_missing = .{ .symbol = sym, .err = e };
            return f;
        };
        f.split = .{ .max = fs[0], .sum = fs[1], .fin = fs[2] };
        return f;
    }

    const split_syms = [3][:0]const u8{ "dsv41_sampling_stats_max", "dsv41_sampling_stats_sum", "dsv41_sampling_stats_fin" };

    /// Bytes of device scratch `statsSplit` needs for `rows` rows of C columns.
    pub fn splitScratch(rows: usize, C: usize) usize {
        const chunks = std.math.divCeil(usize, C, chunk) catch unreachable;
        return std.mem.alignForward(usize, 4 * rows * chunks, 8) + 8 * rows * chunks;
    }

    /// `stats` over every SM (sum in chunk order); `work` holds `splitScratch(rows, C)` bytes; error.NoSplit without.
    pub fn statsSplit(f: *const Functions, s: cuda.Stream, lg: u64, ld: usize, rows: usize, C: usize, t: f64, work: u64, out: u64) !void {
        const sp = f.split orelse return error.NoSplit;
        if (rows < 1 or C < 1 or !(t > 0)) return error.Shape;
        const chunks = std.math.divCeil(usize, C, chunk) catch unreachable;
        const pmax = work;
        const psum = work + std.mem.alignForward(usize, 4 * rows * chunks, 8);
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(chunks), .y = @intCast(rows), .z = 1 };
        const block: cuda.launch.Dim3 = .{ .x = 256, .y = 1, .z = 1 };
        {
            var args: cuda.Args = .{};
            args.add(lg);
            args.add(@as(i64, @intCast(ld)));
            args.add(@as(c_int, @intCast(C)));
            args.add(@as(c_int, @intCast(chunk)));
            args.add(pmax);
            try cuda.launch.launch(sp.max, .{ .grid = grid, .block = block }, s, &args);
        }
        {
            var args: cuda.Args = .{};
            args.add(lg);
            args.add(@as(i64, @intCast(ld)));
            args.add(@as(c_int, @intCast(C)));
            args.add(@as(c_int, @intCast(chunk)));
            args.add(t);
            args.add(pmax);
            args.add(psum);
            try cuda.launch.launch(sp.sum, .{ .grid = grid, .block = block }, s, &args);
        }
        var args: cuda.Args = .{};
        args.add(pmax);
        args.add(psum);
        args.add(@as(c_int, @intCast(chunks)));
        args.add(out);
        try cuda.launch.launch(sp.fin, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &args);
    }

    pub fn unload(f: *Functions) void {
        f.module.unload();
    }

    /// nucleus.row_stats: float64 [rows, 2] at `out`, each row's (max, sum exp(v / t - max / t)) over its C columns.
    pub fn stats(f: *const Functions, s: cuda.Stream, lg: u64, ld: usize, rows: usize, C: usize, t: f64, out: u64) !void {
        if (rows < 1 or C < 1 or !(t > 0)) return error.Shape;
        var args: cuda.Args = .{};
        args.add(lg);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(c_int, @intCast(C)));
        args.add(t);
        args.add(out);
        try cuda.launch.launch(f.stats_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &args);
    }

    /// cand_gather's packed rows: fp32 [rows, 2k] at `out` from topk_keys' vals fp32 / cols int64 [rows, k].
    pub fn pack(f: *const Functions, s: cuda.Stream, vals: u64, cols: u64, rows: usize, k: usize, id0: u32, out: u64) !void {
        if (rows < 1 or k < 1) return error.Shape;
        var args: cuda.Args = .{};
        args.add(vals);
        args.add(cols);
        args.add(@as(c_int, @intCast(k)));
        args.add(@as(c_int, @intCast(id0)));
        args.add(out);
        try cuda.launch.launch(f.pack_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &args);
    }
};
