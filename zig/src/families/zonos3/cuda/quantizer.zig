//! ZONOS3's codec quantizer on one GPU: a lane's acoustic rows and teacher rows in, ten codes a frame out.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("codec_kernels.zig");
const layers = @import("codec_layers.zig");
const wts = @import("codec_weights.zig");

const Buf = cuda.DeviceBuffer;
const W = wts.width;
const D = wts.latent_dim;

pub const lane_frames = 300; // the frames a teacher lane's rows pool onto
pub const codebooks = 1 + wts.residuals;

/// A capture under the reference's dump names: device fp32 rows [rows, width].
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize) anyerror!void,

    fn at(p: Probe, comptime f: []const u8, index: usize, ptr: u64, rows: usize, width: usize) !void {
        var nb: [48]u8 = undefined;
        try p.f(p.ctx, try std.fmt.bufPrint(&nb, f, .{index}), ptr, rows, width);
    }
};

/// The teacher's rows averaged onto the lane's first `frames` frames: a window two frames wide each (caller frees).
pub fn pool(gpa: std.mem.Allocator, ssl: []const f32, frames: usize) ![]f32 {
    const rows = ssl.len / D;
    if (rows == 0 or ssl.len % D != 0 or frames == 0 or frames > lane_frames) return error.BadTeacherRows;
    const out = try gpa.alloc(f32, frames * D);
    errdefer gpa.free(out);
    const ratio: f32 = @as(f32, @floatFromInt(rows)) / lane_frames;
    const limit: f32 = @floatFromInt(rows);
    const sum = try gpa.alloc(f64, D);
    defer gpa.free(sum);
    for (0..frames) |f| {
        const centre = (@as(f32, @floatFromInt(f)) + 0.5) * ratio;
        const left = @min(@max(centre - ratio, 0), limit);
        const right = @min(@max(centre + ratio, 0), limit);
        @memset(sum, 0);
        var cell: usize = @intFromFloat(@floor(left));
        while (cell < rows and @as(f32, @floatFromInt(cell)) < right) : (cell += 1) {
            const lo = @max(left, @as(f32, @floatFromInt(cell)));
            const hi = @min(right, @as(f32, @floatFromInt(cell + 1)));
            const part: f64 = hi - lo;
            for (sum, ssl[cell * D ..][0..D]) |*s, v| s.* += part * v;
        }
        const span: f64 = right - left;
        for (out[f * D ..][0..D], sum) |*o, s| o.* = @floatCast(s / span);
    }
    return out;
}

const names = .{ "ze_sub", "ze_nearest", "ze_sub_code" };
const F = enum(u8) { sub, nearest, sub_code };

pub const Quantizer = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: kern.Kernels,
    mod: cuda.Module,
    fns: [names.len]cuda.Function,
    dec: *const wts.Weights, // the decoder's code tables and semantic head
    w: *wts.Quantizer,
    bufs: std.ArrayList(Buf),
    hist: u64, // [8][W][6]: the semantic encoder's and head's ConvNeXt histories, zero at a lane's start
    x: u64, // [frames][D]
    y: u64,
    residual: u64,
    e: u64, // [frames][8]
    codes: u64, // i32 [frames][10]
    sc: layers.Scratch,

    /// `dec`: the decoder's weights of the same generator, which outlive this.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, generator: []const u8, dec: *const wts.Weights) !*Quantizer {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const q = try gpa.create(Quantizer);
        errdefer gpa.destroy(q);
        q.* = undefined;
        q.gpa = gpa;
        q.d = d;
        q.stream = stream;
        q.dec = dec;
        q.bufs = .empty;
        q.k = try kern.Kernels.load(d);
        errdefer q.k.unload();
        q.mod = try cuda.Module.load(d, cuda.kernels.zonos3_encoder);
        errdefer q.mod.unload();
        inline for (names, 0..) |name, j| q.fns[j] = try q.mod.function(name);
        q.w = try wts.loadQuantizer(gpa, io, d, generator);
        errdefer {
            q.w.deinit();
            gpa.destroy(q.w);
        }
        errdefer q.freeState();
        const n = lane_frames;
        q.hist = try q.alloc(2 * wts.sem_blocks * W * 6 * 4);
        q.x = try q.alloc(n * D * 4);
        q.y = try q.alloc(n * D * 4);
        q.residual = try q.alloc(n * D * 4);
        q.e = try q.alloc(n * wts.codebook_dim * 4);
        q.codes = try q.alloc(n * 10 * 4);
        q.sc = .{ .n = try q.alloc(n * W * 4), .o = try q.alloc(n * W * 4), .att = 0, .big = try q.alloc(n * wts.hidden * 4) };
        return q;
    }

    pub fn destroy(q: *Quantizer) void {
        q.stream.synchronize() catch {};
        q.freeState();
        q.w.deinit();
        q.gpa.destroy(q.w);
        q.mod.unload();
        q.k.unload();
        q.gpa.destroy(q);
    }

    fn freeState(q: *Quantizer) void {
        for (q.bufs.items) |*b| b.free();
        q.bufs.deinit(q.gpa);
    }

    fn alloc(q: *Quantizer, bytes: usize) !u64 {
        var b = try Buf.alloc(q.d, @max(bytes, 256));
        errdefer b.free();
        try q.bufs.append(q.gpa, b);
        return b.ptr;
    }

    fn buf(q: *Quantizer, ptr: u64) Buf {
        return .{ .d = q.d, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    pub fn download(q: *Quantizer, src: u64, out: []u8) !void {
        try q.buf(src).downloadAsync(0, out, q.stream.handle);
        try q.stream.synchronize();
    }

    fn go(q: *Quantizer, which: F, gx: usize, gy: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(q.fns[@backingInt(which)], .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = 256 } }, q.stream, a);
    }

    /// Codebook `col`'s codes of the rows in `from` [n][width], into the codes buffer's column.
    fn nearest(q: *Quantizer, from: u64, width: usize, col: usize, n: usize) !void {
        try q.k.linear(q.stream, q.w.in_w[col], q.w.in_b[col], from, width, width, wts.codebook_dim, q.e, wts.codebook_dim, n);
        var a: cuda.Args = .{};
        a.add(q.e);
        a.add(q.w.codes[col]);
        a.add(q.codes);
        a.add(@as(i32, @intCast(col)));
        try q.go(.nearest, n, 1, &a);
    }

    /// `acoustic`: device rows [n][D]; `pooled`: the teacher's rows on those frames (pool). The codes [n][10].
    pub fn run(q: *Quantizer, acoustic: u64, pooled: []const f32, out: [][codebooks]i32, probe: ?Probe) !void {
        const n = out.len;
        if (n == 0 or n > lane_frames or pooled.len != n * D) return error.BadLane;
        const k = &q.k;
        const s = q.stream;
        var hist = q.buf(q.hist);
        hist.len = 2 * wts.sem_blocks * W * 6 * 4;
        try hist.fill8(0, s.handle);
        // the semantic code: the pooled teacher rows through the semantic encoder
        try q.buf(q.x).uploadAsync(0, std.mem.sliceAsBytes(pooled), s.handle);
        if (probe) |p| try p.f(p.ctx, "lane.0.pooled", q.x, n, D);
        try k.linear(s, q.w.sem_in_w, q.w.sem_in_b, q.x, D, D, W, q.y, W, n);
        for (q.w.sem, 0..) |b, i| try layers.convnext(k, s, q.sc, b, q.hist + i * W * 6 * 4, q.y, n);
        if (probe) |p| try p.f(p.ctx, "lane.0.semantic", q.y, n, W);
        try q.nearest(q.y, W, 0, n);
        if (probe) |p| try p.f(p.ctx, "lane.0.rvq.0.projected", q.e, n, wts.codebook_dim);
        // its decoded prediction leaves the residual the nine other codebooks take
        try k.gather(s, q.codes, 0, q.dec.sem_table, q.y, W, n);
        if (probe) |p| try p.f(p.ctx, "lane.0.rvq.0.quantized", q.y, n, W);
        for (q.dec.sem, 0..) |b, i| try layers.convnext(k, s, q.sc, b, q.hist + (wts.sem_blocks + i) * W * 6 * 4, q.y, n);
        try k.linear(s, q.dec.sem_out_w, q.dec.sem_out_b, q.y, W, W, D, q.x, D, n);
        if (probe) |p| try p.f(p.ctx, "lane.0.semantic_prediction", q.x, n, D);
        try q.buf(q.residual).copyFrom(0, acoustic, n * D * 4, s.handle);
        {
            var a: cuda.Args = .{};
            a.add(q.residual);
            a.add(q.x);
            a.add(@as(i64, @intCast(n * D)));
            try q.go(.sub, (n * D + 255) / 256, 1, &a);
        }
        if (probe) |p| try p.f(p.ctx, "lane.0.residual", q.residual, n, D);
        for (1..codebooks) |col| {
            try q.nearest(q.residual, D, col, n);
            if (probe) |p| try p.at("lane.0.rvq.{d}.projected", col, q.e, n, wts.codebook_dim);
            var a: cuda.Args = .{};
            a.add(q.residual);
            a.add(q.dec.res_tables + (col - 1) * wts.codebook_size * D * 4);
            a.add(q.codes);
            a.add(@as(i32, @intCast(col)));
            a.add(@as(i32, D));
            try q.go(.sub_code, (D + 255) / 256, n, &a);
        }
        try q.download(q.codes, std.mem.sliceAsBytes(out));
    }
};

test "pool: each frame averages a window two frames wide, clipped at the lane's ends" {
    const t = std.testing;
    // 600 teacher rows on 300 frames: two rows a frame, each window four rows
    const ssl = try t.allocator.alloc(f32, 600 * D);
    defer t.allocator.free(ssl);
    for (0..600) |r| @memset(ssl[r * D ..][0..D], @floatFromInt(r));
    const got = try pool(t.allocator, ssl, 3);
    defer t.allocator.free(got);
    try t.expectApproxEqAbs(@as(f32, 1.0), got[0], 1e-6); // rows 0..2 (the window starts before the lane)
    try t.expectApproxEqAbs(@as(f32, 2.5), got[D], 1e-6); // rows 1..4
    try t.expectApproxEqAbs(@as(f32, 4.5), got[2 * D], 1e-6); // rows 3..6
}
