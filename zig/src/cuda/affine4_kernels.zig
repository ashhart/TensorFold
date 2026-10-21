//! Affine-4 decode, verify and prompt launches over the shared tiled weight layout.
const std = @import("std");
const linear = @import("qlinear.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch_ = @import("launch.zig");
const Affine4 = linear.Affine4;
const Split = linear.Split;
const GemvPart = linear.GemvPart;
const Segs = linear.Segs;
const Parts = linear.Parts;
const symbols = linear.symbols;
const splitK = linear.splitK;
const group_smem = linear.group_smem;
const split_smem = linear.split_smem;
const prefill_smem = linear.prefill_smem;
fn int(v: usize) c_int {
    return @intCast(v);
}
fn u(v: usize) u32 {
    return @intCast(v);
}

/// The affine-4 kernels: qmm_group's cluster tile, lane_gemv and its split-K form, qmm_prefill's prompt GEMM.
pub const Affine4Kernels = struct {
    pub const max_rows = 16; // rows decode() takes: its kernels tile 16 rows
    pub const segs_max = 4; // projections one segmented launch takes

    /// One projection of a shared input and its output.
    pub const Out = struct { q: Affine4, out: u64 };

    group_fn: Function,
    gemv_fn: Function,
    verify_fn: Function, // two m16 row tiles share each weight panel on the GB10 verify path
    verify64_fn: Function, // four, for wider shared frames
    verify128_fn: Function, // eight
    split_fn: Function,
    seg_fn: Function, // the split form over up to four projections of one input
    prefill_fn: Function,
    words_fn: Function, // affine4_pack.cu: MLX words into the tiles
    scales_fn: Function, // affine4_pack.cu: (n, kg) scales or biases into (kg, npad)
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    split: ?Split, // every GPU but GB10: lane_gemv's slices on CTAs of their own (one stream's decode() at a time)
    pdl: bool, // GB10: the decode kernels launch with programmatic dependent launch
    wide_verify: bool, // off GB10: verify windows of 17 to 32 rows share one weight pass (sm_120, when its engine says)

    /// From loaded qmm_group, qmm_prefill, lane_gemv and affine4_pack modules; GB10 decodes without the split form.
    pub fn resolve(d: *const Driver, group_mod: Module, prefill_mod: Module, gemv_mod: Module, pack_mod: Module, sms: usize, gb10: bool) !Affine4Kernels {
        var a: Affine4Kernels = undefined;
        a.group_fn = try group_mod.function(symbols.group);
        a.verify_fn = try group_mod.function(symbols.verify32);
        a.verify64_fn = try group_mod.function(symbols.verify64);
        a.verify128_fn = try group_mod.function(symbols.verify128);
        a.prefill_fn = try prefill_mod.function(symbols.prefill);
        a.gemv_fn = try gemv_mod.function(symbols.gemv);
        a.split_fn = try gemv_mod.function(symbols.gemv_split);
        a.seg_fn = try gemv_mod.function(symbols.gemv_seg);
        a.words_fn = try pack_mod.function(symbols.pack_words);
        a.scales_fn = try pack_mod.function(symbols.pack_scales);
        try a.group_fn.allowDynamicShared(group_smem);
        try a.verify_fn.allowDynamicShared(linear.verify_smem);
        try a.verify64_fn.allowDynamicShared(linear.verify64_smem);
        try a.verify128_fn.allowDynamicShared(linear.verify128_smem);
        try a.gemv_fn.allowDynamicShared(group_smem);
        try a.split_fn.allowDynamicShared(split_smem);
        try a.seg_fn.allowDynamicShared(split_smem);
        try a.prefill_fn.allowDynamicShared(prefill_smem);
        a.gemv_blocks = @max(1, try a.gemv_fn.occupancy(128, group_smem)) * sms;
        a.pdl = gb10;
        a.wide_verify = false;
        a.split = if (gb10) null else try Split.initFor(d, segs_max);
        return a;
    }

    pub fn deinit(a: *Affine4Kernels) void {
        if (a.split) |*sp| sp.deinit();
    }

    /// qmm_fast.tile: MLX words (n, k/8), scales and biases (n, k/64) on the device into q's tiles (Affine4.layout).
    pub fn pack(a: *const Affine4Kernels, s: Stream, words: u64, scales: u64, biases: u64, q: Affine4) !void {
        const kg = q.k / 64;
        const total: u64 = @as(u64, q.npad / 64) * kg * 512;
        var g: launch_.Args = .{};
        g.add(words);
        for ([_]usize{ q.n, q.k / 8, kg }) |v| g.add(int(v));
        g.add(q.w);
        g.add(@as(c_longlong, @intCast(total)));
        try launch_.launch(a.words_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
        for ([_]u64{ scales, biases }, [_]u64{ q.s, q.b }) |src, out| {
            var t: launch_.Args = .{};
            t.add(src);
            for ([_]usize{ q.n, kg, q.npad }) |v| t.add(int(v));
            t.add(out);
            const cells = @as(u64, kg) * q.npad;
            try launch_.launch(a.scales_fn, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, s, &t);
        }
    }

    /// qmm.matmul: x (rows, k) bf16 with group sums xs -> out (rows, n) bf16, qmm_group's tile-2 bits on every path.
    pub fn decode(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize) !void {
        if (rows > max_rows) return error.WindowTooWide;
        const sk = splitK(q.n, q.k);
        if (sk > 1) if (a.split) |sp| return a.gemvSplit(s, x, xs, q, out, rows, sk, sp);
        return if (sk > 1) a.gemv(s, x, xs, q, out, rows, sk) else a.cluster(s, x, xs, q, out, rows, sk);
    }

    /// decode() of each projection of one input; without GB10 one segmented launch, each keeping its split's bits.
    pub fn decodeMany(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (rows > max_rows) return error.WindowTooWide;
        const sp = a.split orelse {
            for (outs) |o| try a.decode(s, x, xs, o.q, o.out, rows);
            return;
        };
        return a.gemvSeg(s, x, xs, outs, rows, sp);
    }

    /// One CTA a tile's K slice over every projection of `outs`, the split form's partials and tickets in `sp`.
    pub fn gemvSeg(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize, sp: Split) !void {
        if (outs.len == 0 or outs.len > segs_max) return error.TooManyParts;
        var segs: Segs = std.mem.zeroes(Segs);
        var blocks: usize = 0;
        var work: usize = 0;
        var ticks: usize = 0;
        for (outs, 0..) |o, i| {
            if (o.q.k != outs[0].q.k) return error.InputMismatch;
            const tiles = (o.q.n + 63) / 64;
            const sk = splitK(o.q.n, o.q.k);
            segs.s[i] = .{ .p = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = int(o.q.n), .npad = int(o.q.npad), .sk = int(sk), .tiles = int(tiles) }, .first = int(blocks), .work_at = int(work), .tick_at = int(ticks) };
            blocks += tiles * sk;
            if (sk > 1) {
                work += tiles * sk;
                ticks += tiles;
            }
        }
        segs.count = int(outs.len);
        if (ticks * 4 > sp.tickets.len or work * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(segs);
        for ([_]usize{ rows, outs[0].q.k, outs[0].q.k }) |v| g.add(int(v));
        g.add(sp.work.ptr);
        g.add(sp.tickets.ptr);
        try launch_.launch(a.seg_fn, .{ .grid = .{ .x = u(blocks) }, .block = .{ .x = 128 }, .shared = split_smem }, s, &g);
    }

    /// 16 rows elsewhere; GB10 verify shares a weight panel across up to eight m16 row tiles, sm_120 across two.
    pub fn verifyRows(a: *const Affine4Kernels) usize {
        return if (a.pdl) 128 else if (a.wide_verify) 32 else max_rows;
    }

    /// decode()'s bits a row, the fewest m16 row tiles that hold `rows` sharing each weight panel.
    pub fn verify(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        if (rows == 0 or rows > a.verifyRows()) return error.WindowTooWide;
        if (rows <= max_rows) return a.decode(s, x, xs, weight, out, rows);
        if (rows <= 32) return a.cluster32(s, x, xs, weight, out, rows);
        if (rows <= 64) return a.wide(a.verify64_fn, linear.verify64_smem, 64, s, x, xs, weight, out, rows);
        return a.wide(a.verify128_fn, linear.verify128_smem, 128, s, x, xs, weight, out, rows);
    }

    /// The same ordered group and K-slice chains as decode(), with both row tiles reusing the loaded weights.
    pub fn cluster32(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        return a.wide(a.verify_fn, linear.verify_smem, 32, s, x, xs, weight, out, rows);
    }

    /// cluster32 with four (64) or eight (128) m16 row tiles a weight panel.
    pub fn clusterRows(a: *const Affine4Kernels, tile: usize, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        return switch (tile) {
            64 => a.wide(a.verify64_fn, linear.verify64_smem, 64, s, x, xs, weight, out, rows),
            128 => a.wide(a.verify128_fn, linear.verify128_smem, 128, s, x, xs, weight, out, rows),
            else => error.WindowTooWide,
        };
    }

    /// One CTA a column tile (a cluster of its K slices), all of `rows` in its m16 row tiles over each loaded panel.
    fn wide(a: *const Affine4Kernels, f: Function, shared: u32, most: usize, s: Stream, x: u64, xs: u64, weight: Affine4, out: u64, rows: usize) !void {
        if (rows == 0 or rows > most) return error.WindowTooWide;
        const tiles = (weight.n + 63) / 64;
        const sk = splitK(weight.n, weight.k);
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = weight.w, .scales = weight.s, .biases = weight.b, .out = out, .n = int(weight.n), .npad = int(weight.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        for ([_]usize{ rows, weight.k, weight.k, 1, sk }) |v| g.add(int(v));
        try launch_.launch(f, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = shared, .cluster = if (sk > 1) .{ .x = u(sk) } else null, .pdl = a.pdl }, s, &g);
    }

    /// Each CTA writes one K slice's partial from zero; the tile's last CTA sums them in slice order.
    pub fn gemvSplit(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize, sp: Split) !void {
        const tiles = (q.n + 63) / 64;
        if (tiles > Split.tiles_max or tiles * sk * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        g.add(sp.work.ptr);
        g.add(sp.tickets.ptr);
        try launch_.launch(a.split_fn, .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = split_smem }, s, &g);
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in slice order, CTAs looping over the tiles.
    pub fn gemv(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        const cfg: launch_.Config = .{ .grid = .{ .x = u(@min(tiles, a.gemv_blocks)) }, .block = .{ .x = 128 }, .shared = group_smem, .pdl = a.pdl };
        try launch_.launch(a.gemv_fn, cfg, s, &g);
    }

    /// qmm_group's tile 2: a cluster of `sk` CTAs a column tile, its K slices summed over distributed shared memory.
    pub fn cluster(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        const rows_t = (rows + 15) / 16;
        const clusters = rows_t * tiles; // one part: the cluster is its sk K slices of one tile
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| g.add(int(v));
        const cfg: launch_.Config = .{
            .grid = .{ .x = u(clusters * sk) },
            .block = .{ .x = 128 },
            .shared = group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = a.pdl,
        };
        try launch_.launch(a.group_fn, cfg, s, &g);
    }

    /// qmm.prefill_matmul (tile 0): weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prompt(a: *const Affine4Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var g: launch_.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| g.add(int(v));
        const grid = rows_t * ((q.n + 127) / 128);
        try launch_.launch(a.prefill_fn, .{ .grid = .{ .x = u(grid), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = prefill_smem }, s, &g);
    }
};
