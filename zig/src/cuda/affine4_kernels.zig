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
const PrefillParts = linear.PrefillParts;
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
    mlp128_fn: Function,
    mlp128: bool = false,
    gemv_fn: Function,
    verify_fn: Function, // two m16 row tiles share each weight panel on the GB10 verify path
    verify64_fn: Function, // four, for wider shared frames
    verify128_fn: Function, // eight
    split_fn: Function,
    seg_fn: Function, // the split form over up to four projections of one input
    many_fn: Function, // GB10 keeps each member's K slices in its CTA
    prefill_fn: Function,
    prefill22_fn: Function, // the prompt tile on 2x2 warps and two stages (`half_prompt` picks it)
    prefill_many_fn: Function, // GB10: one prompt grid over projections sharing the bf16 input
    half_prompt: bool = false, // prompts on prefill22_fn: its four warps a block and two stages are quicker on an RTX 5090
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
        a.mlp128_fn = try group_mod.function("_ZN12tf_qmm_group12group_kernelILi64ELi16ELi128ELi1ELi8ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii");
        try a.mlp128_fn.allowDynamicShared(26880);
        a.mlp128 = false;
        if (sms == 82 or std.c.getenv("TF_CUDA_TEST_MLP128") != null) {
            var device: @import("abi.zig").Device = 0;
            try d.check(d.api.cuCtxGetDevice(&device), "cuCtxGetDevice");
            var major: c_int = 0;
            var minor: c_int = 0;
            try d.check(d.api.cuDeviceGetAttribute(&major, .compute_capability_major, device), "cuDeviceGetAttribute");
            try d.check(d.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, device), "cuDeviceGetAttribute");
            a.mlp128 = try @import("sm120_test_override.zig").select("TF_CUDA_TEST_MLP128", @intCast(10 * major + minor), sms);
        }
        a.verify_fn = try group_mod.function(symbols.verify32);
        a.verify64_fn = try group_mod.function(symbols.verify64);
        a.verify128_fn = try group_mod.function(symbols.verify128);
        a.prefill_fn = try prefill_mod.function(symbols.prefill);
        a.prefill22_fn = try prefill_mod.function(symbols.prefill22);
        a.prefill_many_fn = try prefill_mod.function(symbols.prefill_many);
        a.half_prompt = false;
        a.gemv_fn = try gemv_mod.function(symbols.gemv);
        a.split_fn = try gemv_mod.function(symbols.gemv_split);
        a.many_fn = try gemv_mod.function("tf_lane_gemv_many");
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
        try a.prefill22_fn.allowDynamicShared(linear.prefill22_smem);
        try a.prefill_many_fn.allowDynamicShared(prefill_smem);
        try a.many_fn.allowDynamicShared(group_smem);
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
        if (a.pdl and outs.len > 1) return a.gemvMany(s, x, xs, outs, rows);
        const sp = a.split orelse {
            for (outs) |o| try a.decode(s, x, xs, o.q, o.out, rows);
            return;
        };
        return a.gemvSeg(s, x, xs, outs, rows, sp);
    }

    /// Every member's verify() bits, without changing the GB10 split-buffer policy.
    pub fn verifyMany(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (rows == 0 or rows > a.verifyRows()) return error.WindowTooWide;
        if (rows <= max_rows) return a.decodeMany(s, x, xs, outs, rows);
        if (a.pdl and outs.len > 1) return a.wideMany(s, x, xs, outs, rows);
        for (outs) |o| try a.verify(s, x, xs, o.q, o.out, rows);
    }

    /// The existing wide tile already accepts member-local splits within one common cluster width.
    pub fn wideMany(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (!a.pdl or rows <= max_rows or rows > a.verifyRows()) return error.WindowTooWide;
        if (outs.len < 2 or outs.len > segs_max) return error.TooManyParts;
        var width: usize = 1;
        for (outs, 0..) |o, i| {
            if (o.q.k == 0 or o.q.k % 64 != 0 or o.q.k != outs[0].q.k or o.q.n == 0 or o.q.npad < o.q.n or o.q.npad % 64 != 0) return error.InputMismatch;
            for (outs[0..i]) |prev| if (prev.out == o.out) return error.InputMismatch;
            width = @max(width, splitK(o.q.n, o.q.k));
        }
        var parts: Parts = std.mem.zeroes(Parts);
        var clusters: usize = 0;
        for (outs, 0..) |o, i| {
            const sk = splitK(o.q.n, o.q.k);
            const tiles = (o.q.n + 63) / 64;
            const per = width / sk;
            parts.p[i] = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = int(o.q.n), .npad = int(o.q.npad), .sk = int(sk), .tiles = int(tiles), .first = int(clusters) };
            clusters += (tiles + per - 1) / per;
        }
        parts.count = int(outs.len);
        var args: launch_.Args = .{};
        args.add(x);
        args.add(xs);
        args.add(parts);
        for ([_]usize{ rows, outs[0].q.k, outs[0].q.k, 1, width }) |v| args.add(int(v));
        const f = if (rows <= 32) a.verify_fn else if (rows <= 64) a.verify64_fn else a.verify128_fn;
        const shared = if (rows <= 32) linear.verify_smem else if (rows <= 64) linear.verify64_smem else linear.verify128_smem;
        try launch_.launch(f, .{ .grid = .{ .x = u(clusters * width) }, .block = .{ .x = 128 }, .shared = shared, .cluster = if (width > 1) .{ .x = u(width) } else null, .pdl = true }, s, &args);
    }

    /// Native rows in bounded verify tiles, each output retaining its own contiguous row stride.
    pub fn verifyManyRows(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (outs.len < 2 or outs.len > segs_max) return error.TooManyParts;
        if (rows == 0) return error.WindowTooWide;
        var many: [segs_max]Out = undefined;
        const k = outs[0].q.k;
        const tile = a.verifyRows();
        var at: usize = 0;
        while (at < rows) : (at += tile) {
            for (outs, many[0..outs.len]) |o, *m| m.* = .{ .q = o.q, .out = o.out + at * o.q.n * 2 };
            try a.verifyMany(s, x + at * k * 2, xs + at * (k / 64) * 4, many[0..outs.len], @min(tile, rows - at));
        }
    }

    /// One decode-sized PDL grid over independent persistent member grids.
    pub fn gemvMany(a: *const Affine4Kernels, s: Stream, x: u64, xs: u64, outs: []const Out, rows: usize) !void {
        if (!a.pdl or rows == 0 or rows > max_rows) return error.WindowTooWide;
        if (outs.len < 2 or outs.len > segs_max) return error.TooManyParts;
        var parts: linear.GemvManyParts = std.mem.zeroes(linear.GemvManyParts);
        var blocks: usize = 0;
        for (outs, 0..) |o, i| {
            if (o.q.k == 0 or o.q.k % 64 != 0 or o.q.k != outs[0].q.k or o.q.n == 0 or o.q.npad < o.q.n or o.q.npad % 64 != 0) return error.InputMismatch;
            for (outs[0..i]) |prev| if (prev.out == o.out) return error.InputMismatch;
            const tiles = (o.q.n + 63) / 64;
            const local = @min(tiles, a.gemv_blocks);
            if (local == 0) return error.InputMismatch;
            parts.p[i] = .{ .p = .{ .w = o.q.w, .scales = o.q.s, .biases = o.q.b, .out = o.out, .n = int(o.q.n), .npad = int(o.q.npad), .sk = int(splitK(o.q.n, o.q.k)), .tiles = int(tiles) }, .first = int(blocks), .blocks = int(local) };
            blocks += local;
        }
        parts.count = int(outs.len);
        var args: launch_.Args = .{};
        args.add(x);
        args.add(xs);
        args.add(parts);
        for ([_]usize{ rows, outs[0].q.k, outs[0].q.k }) |v| args.add(int(v));
        try launch_.launch(a.many_fn, .{ .grid = .{ .x = u(blocks) }, .block = .{ .x = 128 }, .shared = group_smem, .pdl = true }, s, &args);
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
        const tuned = a.mlp128 and mlp128Shape(rows, q.n, q.k, sk);
        const width: usize = if (tuned) 128 else 64;
        const tiles = (q.n + width - 1) / width;
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
            .block = .{ .x = if (tuned) 256 else 128 },
            .shared = if (tuned) 26880 else group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = a.pdl or tuned,
        };
        try launch_.launch(if (tuned) a.mlp128_fn else a.group_fn, cfg, s, &g);
    }

    /// qmm.prefill_matmul (tile 0): weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prompt(a: *const Affine4Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
        return a.promptTile(s, x, q, out, rows, a.half_prompt);
    }

    /// The same arithmetic through qmm_prefill's 2x2-warp, two-stage tile.
    pub fn promptHalf(a: *const Affine4Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
        return a.promptTile(s, x, q, out, rows, true);
    }

    fn promptTile(a: *const Affine4Kernels, s: Stream, x: u64, q: Affine4, out: u64, rows: usize, half: bool) !void {
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var g: launch_.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| g.add(int(v));
        const grid = rows_t * ((q.n + 127) / 128);
        const f = if (half) a.prefill22_fn else a.prefill_fn;
        const threads: u32 = if (half) 128 else 256;
        const smem: u32 = if (half) linear.prefill22_smem else prefill_smem;
        try launch_.launch(f, .{ .grid = .{ .x = u(grid), .y = 1, .z = 1 }, .block = .{ .x = threads }, .shared = smem }, s, &g);
    }

    /// One GB10 launch over 2-4 affine-4 prompt projections, each retaining prompt()'s exact tile body.
    pub fn promptMany(a: *const Affine4Kernels, s: Stream, x: u64, outs: []const Out, rows: usize) !void {
        if (!a.pdl or a.half_prompt) return error.UnsupportedPromptMany;
        if (outs.len < 2 or outs.len > segs_max) return error.TooManyParts;
        const k = outs[0].q.k;
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * k * 2);
        const group = @max(1, @min(rows_t, band));
        var parts: PrefillParts = std.mem.zeroes(PrefillParts);
        var blocks: usize = 0;
        for (outs, 0..) |o, i| {
            if (o.q.k != k) return error.InputMismatch;
            parts.p[i] = .{
                .w = o.q.w,
                .scales = o.q.s,
                .biases = o.q.b,
                .out = o.out,
                .n = int(o.q.n),
                .npad = int(o.q.npad),
                .first = int(blocks),
            };
            blocks += rows_t * ((o.q.n + 127) / 128);
        }
        parts.count = int(outs.len);
        var g: launch_.Args = .{};
        g.add(x);
        g.add(parts);
        for ([_]usize{ rows, k, k, group }) |v| g.add(int(v));
        try launch_.launch(a.prefill_many_fn, .{ .grid = .{ .x = u(blocks) }, .block = .{ .x = 256 }, .shared = prefill_smem }, s, &g);
    }
};

test {
    _ = @import("gemv_many_host_test.zig");
}

/// The measured device envelope; other SM120 cards keep the existing tile.
pub fn mlp128Device(sm: u32, sms: usize) bool {
    return @import("sm120_test_override.zig").enabled(sm, sms, null);
}

/// A single K slice and a complete dense 27B MLP output panel.
pub fn mlp128Shape(rows: usize, n: usize, k: usize, sk: usize) bool {
    return rows > 0 and rows <= 16 and n == 17408 and k == 5120 and sk == 1;
}

test "MLP128 keeps unqualified devices and shapes on the existing tile" {
    try std.testing.expect(mlp128Device(120, 82));
    for ([_][2]u32{ .{ 121, 82 }, .{ 120, 170 }, .{ 89, 82 } }) |d|
        try std.testing.expect(!mlp128Device(d[0], d[1]));
    for ([_]usize{ 1, 8, 16 }) |rows| try std.testing.expect(mlp128Shape(rows, 17408, 5120, 1));
    for ([_][4]usize{ .{ 0, 17408, 5120, 1 }, .{ 17, 17408, 5120, 1 }, .{ 16, 17408, 5120, 2 }, .{ 16, 5120, 17408, 1 }, .{ 16, 17408, 2048, 1 } }) |s|
        try std.testing.expect(!mlp128Shape(s[0], s[1], s[2], s[3]));
}
