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
const exl3 = @import("exl3.zig");
const ggml = @import("ggml.zig");
const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// MLX affine 4-bit tiled for qmm: packed words, (kg, npad) scales and biases, n outputs from k inputs.
pub const proposal = @import("proposal_head.zig");
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
    exl3: exl3.Weight,
    affine4: Affine4,
    fp8g: qmmf.Weight,
    mxfp8: qmmf.Weight, // e4m3 with UE8M0 scale bytes per 32 inputs
    nvfp4: qmmf.Weight,
    fp8: qmmf.Weight, // e4m3 with one fp32 scale (ModelOpt FP8)
    bf16: qmmf.Weight, // unquantized: the checkpoint's bf16 values
    q8_0: ggml.Weight, // ggml Q8_0 packed by ggml.Kernels.pack
    iq2xxs: ggml.Weight, // ggml IQ2_XXS packed by ggml.Kernels.pack

    pub fn outputs(w: Weight) usize {
        return switch (w) {
            .exl3 => |q| q.n,
            .affine4 => |q| q.n,
            .fp8g, .mxfp8, .nvfp4, .fp8, .bf16 => |q| q.n,
            .q8_0, .iq2xxs => |q| q.n,
        };
    }

    pub fn inputs(w: Weight) usize {
        return switch (w) {
            .exl3 => |q| q.k,
            .affine4 => |q| q.k,
            .fp8g, .mxfp8, .nvfp4, .fp8, .bf16 => |q| q.k,
            .q8_0, .iq2xxs => |q| q.k,
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
    exl3: ?*exl3.Dense = null, // the EXL3 kernels and the scratch the dense calls share
    ggml: ?*const ggml.Kernels = null, // q8_0 and iq2xxs rows of both kinds

    /// e4m3 rows of x times an fp8 weight under its input scale, bf16 out: decode and prompt rows alike (row-exact).
    fn checkpoint(l: Linear, s: Stream, c: *const ck.Ck, q: qmmf.Weight, in: Rows, out: u64, rows: usize) !void {
        if (l.ck_rows == 0) return error.NoCheckpointRows;
        try c.quant8(s, in.x, if (in.ldx == 0) q.k else in.ldx, rows, q.k, q.act, l.ck_rows);
        return c.lane8(s, l.ck_rows, rows, q, q.act, out, if (l.ck_part != 0) l.ck_part else in.part);
    }

    /// Decode rows: out (rows, n) bf16, each row's bits the same at every row count the format takes.
    pub fn decode(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        switch (w) {
            .exl3 => |q| {
                const e = l.exl3 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.decode(s, q, in.x, .bf16, rows, out, .bf16);
            },
            .affine4 => |q| {
                const a = l.affine4 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return a.decode(s, in.x, in.sums, q, out, rows);
            },
            .fp8g => |q| return l.onLane(s, q, .fp8g, in, out, rows),
            .mxfp8 => |q| return l.onLane(s, q, .mxfp8, in, out, rows),
            .nvfp4 => |q| return l.onLane(s, q, .fp4, in, out, rows),
            .fp8 => |q| return if (q.act != 0 and l.ck != null) l.checkpoint(s, l.ck.?, q, in, out, rows) else l.onLane(s, q, .fp8, in, out, rows),
            .bf16 => |q| return l.onLane(s, q, .bf16, in, out, rows),
            .q8_0, .iq2xxs => |q| return (l.ggml orelse return error.FormatNotLoaded).decode(s, q, in.x, in.ldx, rows, out, .bf16),
        }
    }

    /// Prompt rows: out (rows, n) bf16 by the format's prompt GEMM.
    pub fn prompt(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        switch (w) {
            .exl3 => |q| {
                const e = l.exl3 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.prompt(s, q, in.x, .bf16, rows, out, .bf16);
            },
            .affine4 => |q| {
                const a = l.affine4 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return a.prompt(s, in.x, q, out, rows);
            },
            .fp8g => |q| return l.onLane(s, q, .fp8g, in, out, rows),
            .mxfp8 => |q| return l.onLane(s, q, .mxfp8, in, out, rows),
            .nvfp4 => |q| return l.onPrompt(s, q, .fp4, in, out, rows),
            .fp8 => |q| return if (q.act != 0 and l.ck != null) l.checkpoint(s, l.ck.?, q, in, out, rows) else l.onPrompt(s, q, .fp8, in, out, rows),
            .bf16 => |q| return l.onLane(s, q, .bf16, in, out, rows),
            // prompt rows always take the prompt GEMM: a row's bits must not depend on where a prompt's chunks are cut
            .q8_0, .iq2xxs => |q| return (l.ggml orelse return error.FormatNotLoaded).promptRows(s, q, in.x, in.ldx, rows, out, .bf16),
        }
    }

    /// Decode rows to FP32 without the output BF16 cast (NVFP4, block FP8, MXFP8 and EXL3), including TP partial rows.
    pub fn decodeF32(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        const q = switch (w) {
            .exl3 => |q| {
                const e = l.exl3 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.decode(s, q, in.x, .bf16, rows, out, .f32);
            },
            .fp8g => |q| return l.onLaneAs(s, q, .fp8g, in, out, .f32, rows),
            .mxfp8 => |q| return l.onLaneAs(s, q, .mxfp8, in, out, .f32, rows),
            .nvfp4 => |q| q,
            else => return error.EpilogueNotBuilt,
        };
        return l.onLaneAs(s, q, .fp4, in, out, .f32, rows);
    }

    /// Prompt rows to FP32 without the output BF16 cast (NVFP4, EXL3; block FP8 and MXFP8 on the lane, as prompt()).
    pub fn promptF32(l: Linear, s: Stream, w: Weight, in: Rows, out: u64, rows: usize) !void {
        const q = switch (w) {
            .exl3 => |q| {
                const e = l.exl3 orelse return error.FormatNotLoaded;
                if (in.ldx != 0 and in.ldx != q.k) return error.RowStride;
                return e.prompt(s, q, in.x, .bf16, rows, out, .f32);
            },
            .fp8g => |q| return l.onLaneAs(s, q, .fp8g, in, out, .f32, rows),
            .mxfp8 => |q| return l.onLaneAs(s, q, .mxfp8, in, out, .f32, rows),
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

/// The affine-4 entry points; verify32 is pending compiler and device qualification.
pub const symbols = struct {
    pub const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const verify32 = "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi5ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const verify64 = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const verify128 = "_ZN12tf_qmm_group12group_kernelILi64ELi128ELi64ELi1ELi4ELi3ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const prefill = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const prefill22 = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi2ELi2ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const gemv = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
    pub const gemv_split = "tf_lane_gemv_split";
    pub const gemv_seg = "tf_lane_gemv_seg";
    pub const pack_words = "tf_pack_dense";
    pub const pack_scales = "tf_transpose_pad16";
};

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs share out.
pub const GemvPart = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int };

/// lane_gemv.cu's Seg and Segs: projections of one input, each at its first block, partials and tickets.
pub const Seg = extern struct { p: GemvPart, first: c_int, work_at: c_int, tick_at: c_int, pad: c_int = 0 };
pub const Segs = extern struct { s: [4]Seg, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232 and @sizeOf(GemvPart) == 48);
    std.debug.assert(@sizeOf(Seg) == 64 and @sizeOf(Segs) == 264);
}

pub const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
pub const verify_smem: u32 = 32640; // LaneTile<64, 32, 64, 1, 4, 5>::SMEM
pub const verify64_smem: u32 = 43008; // LaneTile<64, 64, 64, 1, 4, 4>::SMEM
pub const verify128_smem: u32 = 57600; // LaneTile<64, 128, 64, 1, 4, 3>::SMEM
pub const split_smem: u32 = 17664; // LaneTile<64, 16, 64, 1, 4, 4>::SMEM: four stages, so more split CTAs fit an SM
pub const prefill_smem: u32 = 62976; // Tile<64, 128, 128, 2, 4, 3>::SMEM
pub const prefill22_smem: u32 = 41984; // Tile<64, 128, 128, 2, 2, 2>::SMEM

/// Split-K lane_gemv's fp32 slice partials and per-tile tickets: splitK keeps tiles * sk under 384 and tiles under 192.
pub const Split = struct {
    work: memory.DeviceBuffer,
    tickets: memory.DeviceBuffer,

    pub const tiles_max = 192;
    const work_bytes = 384 * 16 * 64 * 4;

    pub fn init(d: *const Driver) !Split {
        return initFor(d, 1);
    }

    /// Room for `parts` split projections at once (one segmented launch).
    pub fn initFor(d: *const Driver, parts: usize) !Split {
        var work = try memory.DeviceBuffer.alloc(d, parts * work_bytes);
        errdefer work.free();
        var tickets = try memory.DeviceBuffer.alloc(d, parts * tiles_max * 4);
        errdefer tickets.free();
        try d.check(d.api.cuMemsetD32_v2(tickets.ptr, 0, parts * tiles_max), "cuMemsetD32");
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

pub const Affine4Kernels = @import("affine4_kernels.zig").Affine4Kernels;

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

test {
    _ = @import("qlinear_test.zig");
}

pub const block = exl3.block;
test {
    _ = @import("qlinear_exl3_entry_test.zig");
}
