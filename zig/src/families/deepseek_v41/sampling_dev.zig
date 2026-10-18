//! The keyed sampler's device steps on one stream: topk_keys.cu candidates, sampling.cu packed rows and statistics.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const pipe = @import("sampling_pipe.zig");

/// The device steps on one stream: topk_keys.cu (`dsv41_kernels`) and sampling.cu, scratch kept between calls.
pub const Gpu = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const dk.Kernels,
    fns: dk.sampling.Functions,
    bufs: [8]?cuda.DeviceBuffer = @splat(null),
    /// nucleus statistics over every SM when the fatbin has them, unless TF_DSV41_SAMP_STATS=row (one CTA a row)
    split: bool,

    pub fn init(d: *const cuda.Driver, stream: cuda.Stream, kernels: *const dk.Kernels) !Gpu {
        const fns = try dk.sampling.Functions.load(d);
        const row = try rowStats();
        const log = std.log.scoped(.dsv41);
        if (fns.split_missing) |m| {
            if (!quiet) log.warn("sampling: nucleus statistics one CTA a row: the sampling fatbin's {s} did not resolve ({t}); rebuild dsv41_sampling.fatbin from this tree's sampling.cu", .{ m.symbol, m.err });
        } else if (row) {
            log.info("sampling: nucleus statistics one CTA a row (TF_DSV41_SAMP_STATS=row)", .{});
        } else log.info("sampling: nucleus statistics split over every SM ({d} columns a CTA)", .{dk.sampling.Functions.chunk});
        return .{ .d = d, .stream = stream, .kernels = kernels, .fns = fns, .split = fns.split != null and !row };
    }

    /// TF_DSV41_SAMP_STATS: split (default) | row.
    pub fn rowStats() !bool {
        const v = std.mem.span(std.c.getenv("TF_DSV41_SAMP_STATS") orelse return false);
        if (v.len == 0 or std.mem.eql(u8, v, "split")) return false;
        if (std.mem.eql(u8, v, "row")) return true;
        return error.BadSampStats;
    }

    pub fn deinit(g: *Gpu) void {
        for (&g.bufs) |*b| if (b.*) |*x| x.free();
        g.fns.unload();
    }

    pub fn device(g: *Gpu) pipe.Device {
        return .{ .ptr = g, .vtable = &.{ .top = top, .stats = stats, .scratch = scratch, .download = download, .stream = streamOf } };
    }

    fn self_(p: *anyopaque) *Gpu {
        return @ptrCast(@alignCast(p));
    }

    fn top(p: *anyopaque, lg: u64, n: usize, C: usize, k: usize, id0: u32, out: u64) anyerror!void {
        const g = self_(p);
        const vals = try scratch(p, 5, 4 * n * k);
        const cols = try scratch(p, 6, 8 * n * k);
        try g.kernels.others(g.stream).topKeys(lg, C, n, C, k, vals, cols);
        try g.fns.pack(g.stream, vals, cols, n, k, id0, out);
    }

    fn stats(p: *anyopaque, lg: u64, n: usize, C: usize, t: f64, out: u64) anyerror!void {
        const g = self_(p);
        if (!g.split) return g.fns.stats(g.stream, lg, C, n, C, t, out);
        const work = try scratch(p, 7, dk.sampling.Functions.splitScratch(n, C));
        try g.fns.statsSplit(g.stream, lg, C, n, C, t, work, out);
    }

    fn scratch(p: *anyopaque, slot: u32, bytes: usize) anyerror!u64 {
        const g = self_(p);
        const b = &g.bufs[slot];
        if (b.* == null or b.*.?.len < bytes) {
            if (b.*) |*x| x.free();
            b.* = null;
            b.* = try cuda.DeviceBuffer.alloc(g.d, bytes);
        }
        return b.*.?.ptr;
    }

    fn download(p: *anyopaque, src: u64, dst: []u8) anyerror!void {
        const g = self_(p);
        try g.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = g.d, .ptr = src, .len = dst.len }, 0, dst);
    }

    fn streamOf(p: *anyopaque) cuda.abi.Stream {
        return self_(p).stream.handle;
    }
};
