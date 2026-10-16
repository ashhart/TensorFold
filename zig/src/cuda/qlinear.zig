//! Quantized projections behind one call: a weight view per format, with decode rows and prompt rows chosen per format.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const qmmf = @import("qmmf.zig");
const nvfp4 = @import("nvfp4.zig");
const ck = @import("ck.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// MLX affine 4-bit tiled for qmm: packed words, (kg, npad) scales and biases, n outputs from k inputs.
pub const Affine4 = struct {
    w: u64,
    s: u64,
    b: u64,
    n: usize,
    k: usize,
    npad: usize,

    /// The tiled sizes for n outputs from k inputs: rows padded to 128, then the words', scales' and biases' bytes.
    pub fn layout(n: usize, k: usize) struct { npad: usize, words: usize, scales: usize } {
        const npad = (n + 127) / 128 * 128;
        return .{ .npad = npad, .words = npad * k / 2, .scales = k / 64 * npad * 2 };
    }
};

/// MLX affine codes tiled for affine8.cu at `bits` a weight (2, 3, 5, 6 or 8), with Affine4's padding and tables.
pub const AffineN = struct {
    w: u64,
    s: u64,
    b: u64,
    n: usize,
    k: usize,
    npad: usize,
    bits: u8,

    pub fn layout(n: usize, k: usize, bits: u8) struct { npad: usize, words: usize, scales: usize } {
        const npad = (n + 127) / 128 * 128;
        return .{ .npad = npad, .words = npad * k * bits / 8, .scales = k / 64 * npad * 2 };
    }
};

/// The width an MLX code of `bits` is served at: its own, but 7 on the 8-bit tiles.
pub fn servedBits(bits: u8) u8 {
    return if (bits == 7) 8 else bits;
}

/// The width the embed kernels read a row of `bits` codes at: 4 or 8.
pub fn rowBits(bits: u8) u8 {
    return if (bits <= 4) 4 else 8;
}

/// One projection's weights in any format this module serves; every one but affine4 is a lane matmul mode.
pub const Weight = union(enum) {
    affine4: Affine4,
    fp8g: qmmf.Weight,
    nvfp4: qmmf.Weight,
    fp8: qmmf.Weight, // e4m3 with one fp32 scale (ModelOpt FP8)
    bf16: qmmf.Weight, // unquantized: the checkpoint's bf16 values

    pub fn outputs(w: Weight) usize {
        return switch (w) {
            .affine4 => |q| q.n,
            .fp8g, .nvfp4, .fp8, .bf16 => |q| q.n,
        };
    }

    pub fn inputs(w: Weight) usize {
        return switch (w) {
            .affine4 => |q| q.k,
            .fp8g, .nvfp4, .fp8, .bf16 => |q| q.k,
        };
    }
};

/// Rows in: bf16 `x` rows `ldx` apart (0: packed), their 64-group sums, and the lane's slice partials (partBytes).
pub const Rows = struct { x: u64, ldx: usize = 0, sums: u64 = 0, part: u64 = 0 };

/// The kernels of every format a family loaded; a weight whose format is not loaded is refused.
pub const Linear = struct {
    affine4: ?*const Affine4Kernels = null,
    lane: ?*const qmmf.Lane = null, // decode rows of every lane format; fp8g and bf16 prompt rows too
    nvfp4_prompt: ?*const nvfp4.Prompt = null, // the prompt GEMM: nvfp4 and fp8 prompt rows (linear.py's prefill)
    ck: ?*const ck.Ck = null, // the checkpoint's own math: an fp8 weight with an input scale takes e4m3 rows
    ck_rows: u64 = 0, // its e4m3 rows (at least rows * k bytes), and slice partials where no cluster meets them
    ck_part: u64 = 0,

    /// e4m3 rows of x times an fp8 weight under its input scale, bf16 out: decode and prompt rows alike (row-exact).
    fn checkpoint(l: Linear, s: Stream, c: *const ck.Ck, q: qmmf.Weight, in: Rows, out: u64, rows: usize) !void {
        if (l.ck_rows == 0) return error.NoCheckpointRows;
        try c.quant8(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q.k, q.act, l.ck_rows);
        return c.lane8(s, l.ck_rows, rows, q, q.act, out, if (l.ck_part != 0) l.ck_part else in.part);
    }

    /// Decode rows: out (rows, n) bf16, each row's bits the same at every row count the format takes.
    pub fn decode(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        switch (w) {
            .affine4 => |q| {
                const a = l.affine4 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return a.decode(s, in.x, in.sums, q, out, rows);
            },
            .fp8g => |q| return l.onLane(s, q, .fp8g, in, out, rows),
            .nvfp4 => |q| return l.onLane(s, q, .fp4, in, out, rows),
            .fp8 => |q| return if (q.act != 0 and l.ck != null) l.checkpoint(s, l.ck.?, q, in, out, rows) else l.onLane(s, q, .fp8, in, out, rows),
            .bf16 => |q| return l.onLane(s, q, .bf16, in, out, rows),
        }
    }

    /// Prompt rows: out (rows, n) bf16 by the format's prompt GEMM.
    pub fn prompt(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        switch (w) {
            .affine4 => |q| {
                const a = l.affine4 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return a.prompt(s, in.x, q, out, rows);
            },
            .fp8g => |q| return l.onLane(s, q, .fp8g, in, out, rows),
            .nvfp4 => |q| return l.onPrompt(s, q, .fp4, in, out, rows),
            .fp8 => |q| return if (q.act != 0 and l.ck != null) l.checkpoint(s, l.ck.?, q, in, out, rows) else l.onPrompt(s, q, .fp8, in, out, rows),
            .bf16 => |q| return l.onLane(s, q, .bf16, in, out, rows),
        }
    }

    /// Decode rows as decode(), out (rows, n) fp32 unrounded: a tensor-parallel rank's partial rows (nvfp4 only).
    pub fn decodeF32(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        const q = switch (w) {
            .nvfp4 => |q| q,
            else => return error.EpilogueNotBuilt,
        };
        return l.onLaneAs(s, q, .fp4, in, out, .f32, rows);
    }

    /// Prompt rows as prompt(), out (rows, n) fp32 unrounded (nvfp4 only).
    pub fn promptF32(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        const q = switch (w) {
            .nvfp4 => |q| q,
            else => return error.EpilogueNotBuilt,
        };
        return l.onPromptAs(s, q, .fp4, in, out, .f32, rows);
    }

    fn onPrompt(l: Linear, s: Stream, q: qmmf.Weight, mode: qmmf.Mode, in: Rows, out: u64, rows: usize) !void {
        return l.onPromptAs(s, q, mode, in, out, .bf16, rows);
    }

    fn onPromptAs(
        l: Linear,
        s: Stream,
        q: qmmf.Weight,
        mode: qmmf.Mode,
        in: Rows,
        out: u64,
        kind: qmmf.Out,
        rows: usize,
    ) !void {
        const p = l.nvfp4_prompt orelse return error.FormatNotLoaded;
        if (q.mode != mode) return error.FormatMismatch;
        return p.matmulOut(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q, out, kind);
    }

    fn onLane(l: Linear, s: Stream, q: qmmf.Weight, mode: qmmf.Mode, in: Rows, out: u64, rows: usize) !void {
        return l.onLaneAs(s, q, mode, in, out, .bf16, rows);
    }

    fn onLaneAs(
        l: Linear,
        s: Stream,
        q: qmmf.Weight,
        mode: qmmf.Mode,
        in: Rows,
        out: u64,
        kind: qmmf.Out,
        rows: usize,
    ) !void {
        const lane = l.lane orelse return error.FormatNotLoaded;
        if (q.mode != mode) return error.FormatMismatch;
        return lane.matmulOut(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q, out, kind, in.part);
    }
};

/// The affine-4 instantiations' names (cuobjdump -symbols of the qmm_group, qmm_prefill and lane_gemv fatbins).
pub const symbols = struct {
    pub const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const prefill = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const gemv = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
    pub const gemv_split = "tf_lane_gemv_split";
    pub const pack_words = "tf_pack_dense";
    pub const pack_scales = "tf_transpose_pad16";
};

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs share out.
pub const GemvPart = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232 and @sizeOf(GemvPart) == 48);
}

pub const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
pub const split_smem: u32 = 17664; // LaneTile<64, 16, 64, 1, 4, 4>::SMEM: four stages, so more split CTAs fit an SM
pub const prefill_smem: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM

/// Split-K lane_gemv's fp32 slice partials and per-tile tickets: splitK keeps tiles * sk under 384 and tiles under 192.
pub const Split = struct {
    work: memory.DeviceBuffer,
    tickets: memory.DeviceBuffer,

    pub const tiles_max = 192;
    const work_bytes = 384 * 16 * 64 * 4;

    pub fn init(d: *const Driver) !Split {
        var work = try memory.DeviceBuffer.alloc(d, work_bytes);
        errdefer work.free();
        var tickets = try memory.DeviceBuffer.alloc(d, tiles_max * 4);
        errdefer tickets.free();
        try d.check(d.api.cuMemsetD32_v2(tickets.ptr, 0, tiles_max), "cuMemsetD32");
        return .{ .work = work, .tickets = tickets };
    }

    pub fn deinit(sp: *Split) void {
        sp.work.free();
        sp.tickets.free();
    }
};

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// The affine-4 kernels: qmm_group's cluster tile, lane_gemv and its split-K form, qmm_prefill's prompt GEMM.
pub const Affine4Kernels = struct {
    group_fn: Function,
    gemv_fn: Function,
    split_fn: Function,
    prefill_fn: Function,
    words_fn: Function, // affine4_pack.cu: MLX words into the tiles
    scales_fn: Function, // affine4_pack.cu: (n, kg) scales or biases into (kg, npad)
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    split: ?Split, // every GPU but GB10: lane_gemv's slices on CTAs of their own (one stream's decode() at a time)
    pdl: bool, // GB10: the decode kernels launch with programmatic dependent launch

    /// From loaded qmm_group, qmm_prefill, lane_gemv and affine4_pack modules; GB10 decodes without the split form.
    pub fn resolve(d: *const Driver, group_mod: Module, prefill_mod: Module, gemv_mod: Module, pack_mod: Module, sms: usize, gb10: bool) !Affine4Kernels {
        var a: Affine4Kernels = undefined;
        a.group_fn = try group_mod.function(symbols.group);
        a.prefill_fn = try prefill_mod.function(symbols.prefill);
        a.gemv_fn = try gemv_mod.function(symbols.gemv);
        a.split_fn = try gemv_mod.function(symbols.gemv_split);
        a.words_fn = try pack_mod.function(symbols.pack_words);
        a.scales_fn = try pack_mod.function(symbols.pack_scales);
        try a.group_fn.allowDynamicShared(group_smem);
        try a.gemv_fn.allowDynamicShared(group_smem);
        try a.split_fn.allowDynamicShared(split_smem);
        try a.prefill_fn.allowDynamicShared(prefill_smem);
        a.gemv_blocks = @max(1, try a.gemv_fn.occupancy(128, group_smem)) * sms;
        a.pdl = gb10;
        a.split = if (gb10) null else try Split.init(d);
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
        if (rows > 16) return error.WindowTooWide;
        const sk = splitK(q.n, q.k);
        if (sk > 1) if (a.split) |sp| return a.gemvSplit(s, x, xs, q, out, rows, sk, sp);
        return if (sk > 1) a.gemv(s, x, xs, q, out, rows, sk) else a.cluster(s, x, xs, q, out, rows, sk);
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

/// affine8.cu's shared bytes a width: Tile<64, 64, 6, bits>::SMEM for gemv, Tile<64, 64, 4, bits>::SMEM for split.
pub fn gemvSmem(bits: u8) u32 {
    return if (bits == 8) 38784 else 14208 + 3072 * @as(u32, bits);
}

pub fn splitSmem(bits: u8) u32 {
    return if (bits == 8) 25856 else 9472 + 2048 * @as(u32, bits);
}

/// affine8.cu's Prefill<bits>::SMEM: the prompt GEMM's three stages.
pub fn prefillSmem(bits: u8) u32 {
    return 50688 + 3072 * @as(u32, bits);
}

/// The widths AffineNKernels serves, in the order of its function tables.
pub const tile_widths = [_]u8{ 2, 3, 5, 6, 8 };

fn widthAt(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &tile_widths, bits) orelse error.UnsupportedWidth;
}

/// Every MLX width but 4: codes of 2, 3, 5 and 6 bits on tiles of their width, 7 and 8 on 8-bit tiles, unrounded.
pub const AffineNKernels = struct {
    gemv_fns: [tile_widths.len]Function,
    split_fns: [tile_widths.len]Function,
    prefill_fns: [tile_widths.len]Function,
    pack4_fn: Function, // MLX codes of 2 to 4 bits into the affine-4 tiles
    pack8_fn: Function, // MLX codes of 5 to 8 bits into the 8-bit tiles
    packn_fn: Function, // MLX codes of 2, 3, 5 or 6 bits into tiles of that width
    widen_fn: Function, // MLX rows into MLX rows of 4 or 8-bit codes (token embeddings)
    scales_fn: Function, // affine4_pack.cu's tf_transpose_pad16
    gemv_blocks: [tile_widths.len]usize, // resident gemv CTAs a width: per SM times SMs
    split: ?Split,
    pdl: bool,

    /// From loaded affine8, affine_prefill and affine4_pack modules; GB10 decodes without the split form.
    pub fn resolve(d: *const Driver, mod: Module, prefill_mod: Module, pack_mod: Module, sms: usize, gb10: bool) !AffineNKernels {
        var a: AffineNKernels = undefined;
        for (tile_widths, &a.gemv_fns, &a.split_fns, &a.prefill_fns, &a.gemv_blocks) |bits, *g, *sp, *pf, *blocks| {
            var name: [32]u8 = undefined;
            g.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_affine{d}_gemv", .{bits}, 0));
            sp.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_affine{d}_split", .{bits}, 0));
            pf.* = try prefill_mod.function(try std.fmt.bufPrintSentinel(&name, "tf_affine{d}_prefill", .{bits}, 0));
            try g.allowDynamicShared(gemvSmem(bits));
            try sp.allowDynamicShared(splitSmem(bits));
            try pf.allowDynamicShared(prefillSmem(bits));
            blocks.* = @max(1, try g.occupancy(128, gemvSmem(bits))) * sms;
        }
        a.pack4_fn = try mod.function("tf_affine8_pack4");
        a.pack8_fn = try mod.function("tf_affine8_pack8");
        a.packn_fn = try mod.function("tf_affinen_pack");
        a.widen_fn = try mod.function("tf_affine8_widen");
        a.scales_fn = try pack_mod.function(symbols.pack_scales);
        a.pdl = gb10;
        a.split = if (gb10) null else try Split.init(d);
        return a;
    }

    pub fn deinit(a: *AffineNKernels) void {
        if (a.split) |*sp| sp.deinit();
    }

    /// MLX words (n, k * bits / 32) on the device into the tiles at `w` of the width `servedBits(bits)` serves.
    pub fn packWords(a: *const AffineNKernels, s: Stream, words: u64, bits: u8, n: usize, k: usize, w: u64) !void {
        return a.packAt(s, words, bits, n, k, w, servedBits(bits));
    }

    /// `packWords` onto the tiles of width `wide`: `bits` itself, or 4 (Affine4's tiles) or 8 when the codes fit them.
    pub fn packAt(a: *const AffineNKernels, s: Stream, words: u64, bits: u8, n: usize, k: usize, w: u64, wide: u8) !void {
        if (k % 64 != 0 or k * bits % 32 != 0) return error.UnsupportedShape;
        if (wide < bits or (wide != 4 and wide != 8 and wide != bits)) return error.UnsupportedWidth;
        const kg = k / 64;
        const npad = (n + 127) / 128 * 128;
        const per: u64 = switch (wide) {
            4 => 512,
            8 => 1024,
            else => 128 * @as(u64, wide),
        };
        const total: u64 = @as(u64, npad / 64) * kg * per;
        var g: launch_.Args = .{};
        g.add(words);
        for ([_]usize{ bits, n, k * bits / 32, kg }) |v| g.add(int(v));
        g.add(w);
        g.add(@as(c_longlong, @intCast(total)));
        const f = switch (wide) {
            4 => a.pack4_fn,
            8 => a.pack8_fn,
            else => a.packn_fn,
        };
        try launch_.launch(f, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
    }

    /// (n, k/64) scales or biases into the tiles' (k/64, npad) layout.
    pub fn packTable(a: *const AffineNKernels, s: Stream, src: u64, n: usize, k: usize, npad: usize, out: u64) !void {
        const kg = k / 64;
        var t: launch_.Args = .{};
        t.add(src);
        for ([_]usize{ n, kg, npad }) |v| t.add(int(v));
        t.add(out);
        const cells = @as(u64, kg) * npad;
        try launch_.launch(a.scales_fn, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, s, &t);
    }

    /// MLX rows (n, k) of `bits` codes into MLX rows of `wide`-bit codes at `out`, every code's value kept.
    pub fn widen(a: *const AffineNKernels, s: Stream, words: u64, bits: u8, n: usize, k: usize, out: u64, wide: u8) !void {
        if (wide != 4 and wide != 8) return error.UnsupportedWidth;
        const total: u64 = @as(u64, n) * k * wide / 32;
        var g: launch_.Args = .{};
        g.add(words);
        g.add(int(bits));
        g.add(int(k));
        g.add(out);
        g.add(int(wide));
        g.add(@as(c_longlong, @intCast(total)));
        try launch_.launch(a.widen_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &g);
    }

    /// x (rows, k) bf16 with group sums xs -> out (rows, n) bf16; a row's bits the same at every row count up to 16.
    pub fn decode(a: *const AffineNKernels, s: Stream, x: u64, xs: u64, q: AffineN, out: u64, rows: usize) !void {
        if (rows > 16) return error.WindowTooWide;
        const at = try widthAt(q.bits);
        const sk = splitK(q.n, q.k);
        const tiles = (q.n + 63) / 64;
        const part: GemvPart = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) };
        var g: launch_.Args = .{};
        g.add(x);
        g.add(xs);
        g.add(part);
        for ([_]usize{ rows, q.k, q.k }) |v| g.add(int(v));
        if (sk > 1) if (a.split) |sp| {
            if (tiles > Split.tiles_max or tiles * sk * 16 * 64 * 4 > sp.work.len) return error.SplitTooWide;
            g.add(sp.work.ptr);
            g.add(sp.tickets.ptr);
            return launch_.launch(a.split_fns[at], .{ .grid = .{ .x = u(tiles * sk) }, .block = .{ .x = 128 }, .shared = splitSmem(q.bits) }, s, &g);
        };
        const cfg: launch_.Config = .{ .grid = .{ .x = u(@min(tiles, a.gemv_blocks[at])) }, .block = .{ .x = 128 }, .shared = gemvSmem(q.bits), .pdl = a.pdl };
        try launch_.launch(a.gemv_fns[at], cfg, s, &g);
    }

    /// Prompt rows x (rows, k) -> out (rows, n), bf16: Affine4Kernels.prompt's blocks and bands over AffineN's tiles.
    pub fn prompt(a: *const AffineNKernels, s: Stream, x: u64, q: AffineN, out: u64, rows: usize) !void {
        const at = try widthAt(q.bits);
        const rows_t = (rows + 127) / 128;
        const band = (12 << 20) / (128 * q.k * 2);
        const group = @max(1, @min(rows_t, band));
        var g: launch_.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
        for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| g.add(int(v));
        const grid = rows_t * ((q.n + 127) / 128);
        try launch_.launch(a.prefill_fns[at], .{ .grid = .{ .x = u(grid) }, .block = .{ .x = 256 }, .shared = prefillSmem(q.bits) }, s, &g);
    }
};

test "AffineN's tiles hold `bits` a weight over Affine4's padding" {
    const l = AffineN.layout(10304, 2688, 8);
    try std.testing.expectEqual(@as(usize, 10368), l.npad);
    try std.testing.expectEqual(@as(usize, 10368 * 2688), l.words);
    try std.testing.expectEqual(Affine4.layout(10304, 2688).scales, l.scales);
    try std.testing.expectEqual(@as(usize, 10368 * 2688 * 3 / 8), AffineN.layout(10304, 2688, 3).words);
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |b| try std.testing.expectEqual(b, servedBits(b));
    try std.testing.expectEqual(@as(u8, 8), servedBits(7));
    for ([_]u8{ 2, 3, 4 }) |b| try std.testing.expectEqual(@as(u8, 4), rowBits(b));
    for ([_]u8{ 5, 6, 8 }) |b| try std.testing.expectEqual(@as(u8, 8), rowBits(b));
    try std.testing.expectEqual(@as(u32, 38784), gemvSmem(8));
    try std.testing.expectEqual(@as(u32, 14208 + 3072 * 2), gemvSmem(2));
    try std.testing.expectError(error.UnsupportedWidth, widthAt(4));
}

test "split_k follows the Python shapes" {
    try std.testing.expectEqual(@as(usize, 2), splitK(10304, 2688));
    try std.testing.expectEqual(@as(usize, 8), splitK(2688, 4096));
    try std.testing.expectEqual(@as(usize, 2), splitK(4608, 2688));
    try std.testing.expectEqual(@as(usize, 4), splitK(2688, 5376));
    try std.testing.expectEqual(@as(usize, 1), splitK(131072, 2688));
}

test "a weight view reports its shape in every format" {
    const a: Weight = .{ .affine4 = .{ .w = 0, .s = 0, .b = 0, .n = 10304, .k = 2688, .npad = 10304 } };
    const f: Weight = .{ .fp8g = .{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 7168, .k = 2560, .npad = 7168 } };
    try std.testing.expectEqual(@as(usize, 10304), a.outputs());
    try std.testing.expectEqual(@as(usize, 2560), f.inputs());
}

test "a format that is not loaded is refused before any launch" {
    const l: Linear = .{};
    const w: Weight = .{ .affine4 = .{ .w = 0, .s = 0, .b = 0, .n = 64, .k = 64, .npad = 64 } };
    const s: Stream = undefined; // never reached: the refusal comes first
    try std.testing.expectError(error.FormatNotLoaded, l.decode(s, w, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, w, .{ .x = 0 }, 0, 1));
}

test "the affine-4 tiles' sizes follow qmm_fast.tile" {
    const l = Affine4.layout(10304, 2688); // Nemotron's in_proj: rows to whole 128s, 4 bits a weight, 2 bytes a group
    try std.testing.expectEqual(@as(usize, 10368), l.npad);
    try std.testing.expectEqual(@as(usize, 10368 * 2688 / 2), l.words);
    try std.testing.expectEqual(@as(usize, 42 * 10368 * 2), l.scales);
}

test "a lane weight whose mode is not its format's is refused before any launch" {
    const lane: qmmf.Lane = undefined; // never reached: the refusal comes first
    const l: Linear = .{ .lane = &lane };
    const fp4: qmmf.Weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    const s: Stream = undefined;
    try std.testing.expectError(error.FormatMismatch, l.decode(s, .{ .fp8g = fp4 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.FormatNotLoaded, l.prompt(s, .{ .nvfp4 = fp4 }, .{ .x = 0 }, 0, 1));
    const f8: qmmf.Weight = .{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    try std.testing.expectError(error.EpilogueNotBuilt, l.decodeF32(s, .{ .fp8g = f8 }, .{ .x = 0 }, 0, 1));
    try std.testing.expectError(error.EpilogueNotBuilt, l.promptF32(s, .{ .bf16 = f8 }, .{ .x = 0 }, 0, 1));
}
