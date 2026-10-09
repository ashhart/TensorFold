//! ggml block-quantized weights (GGUF) on the Intel GPU: type table and kernel set of kernels/xpu/ggml_quant.cl.

const std = @import("std");
const rt = @import("rt.zig");

const spv = @import("kernels.zig").ggml_quant;
const spv_pfgemm = @import("kernels.zig").ggml_pfgemm;

pub const Buf = rt.Buffer;
/// Same names (and order) as the matching tags of qwen_blocks.Format.
pub const Type = enum { q2_k, q4_k, iq4_xs, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_m, q6_k, q5_k, q8_0, iq4_nl };
pub const count = std.enums.values(Type).len;

/// Bytes of one 256-weight block.
pub fn blockBytes(t: Type) u32 {
    return switch (t) {
        .q2_k => 84,
        .q4_k => 144,
        .iq4_xs => 136,
        .iq2_xxs => 66,
        .iq2_xs => 74,
        .iq2_s => 82,
        .iq3_xxs => 98,
        .iq3_s => 110,
        .iq1_m => 56,
        .q6_k => 210,
        .q5_k => 176,
        .q8_0 => 272, // 8 blocks of 32 weights (fp16 d + 32 int8) = 256 weights
        .iq4_nl => 144, // 8 blocks of 32 weights (fp16 d + 16 bytes of nibbles)
    };
}

/// Bytes of one block as the kernels read it: types starting with an fp16 `d` carry it at the end, padded to 4 bytes.
pub fn packedBytes(t: Type) u32 {
    return switch (t) {
        .iq2_xxs => 68,
        .iq2_xs => 76,
        .iq2_s => 84,
        .iq3_xxs => 100,
        .iq3_s => 112,
        .q6_k => 212,
        else => blockBytes(t),
    };
}

/// File blocks -> kernel blocks: `d` (the first two bytes) moves behind the rest, zero padded to packedBytes.
pub fn repack(t: Type, src: []const u8, dst: []u8) void {
    const b = blockBytes(t);
    const p = packedBytes(t);
    const n = src.len / b;
    const rotate = switch (t) {
        .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s => true,
        else => false,
    };
    if (t == .iq4_nl) { // 8 file blocks (d, 16 bytes of nibbles) -> the 128 bytes first (dword aligned), then the 8 fp16 d
        for (0..n) |i| {
            const s = src[i * b ..][0..b];
            const d = dst[i * p ..][0..p];
            for (0..8) |j| {
                @memcpy(d[16 * j ..][0..16], s[18 * j + 2 ..][0..16]);
                @memcpy(d[128 + 2 * j ..][0..2], s[18 * j ..][0..2]);
            }
        }
        return;
    }
    if (t == .q8_0) { // 8 file blocks (d, 32 int8) -> the 256 int8 first (dword aligned), then the 8 fp16 d
        for (0..n) |i| {
            const s = src[i * b ..][0..b];
            const d = dst[i * p ..][0..p];
            for (0..8) |j| {
                @memcpy(d[32 * j ..][0..32], s[34 * j + 2 ..][0..32]);
                @memcpy(d[256 + 2 * j ..][0..2], s[34 * j ..][0..2]);
            }
        }
        return;
    }
    if (b == p) {
        @memcpy(dst[0..src.len], src);
        return;
    }
    for (0..n) |i| {
        const s = src[i * b ..][0..b];
        const d = dst[i * p ..][0..p];
        if (rotate) {
            @memcpy(d[0 .. b - 2], s[2..b]);
            @memcpy(d[b - 2 .. b], s[0..2]);
        } else @memcpy(d[0..b], s);
        @memset(d[b..p], 0);
    }
}

/// Sub-groups a work-group of the matvec kernels (NSGW in ggml_quant.cl) and its work-item count.
pub const nsgw: u32 = 8;
const wg_items: u32 = nsgw * 16;

/// A packed block is PB = 16 * A + 4 * C bytes: A uint4 pieces then C dwords (see ggml_quant.cl).
fn pieces(t: Type) struct { a: usize, c: usize } {
    const pb = packedBytes(t);
    return .{ .a = pb / 16, .c = (pb % 16) / 4 };
}

/// File rows [rows][nb][blockBytes] -> the kernels' row-interleaved layout (groups of 16 rows); rows % 16 == 0.
pub fn packRows(t: Type, src: []const u8, dst: []u8, rows: usize, nb: usize) void {
    const groups = rows / 16;
    const n_threads: usize = @min(8, @max(1, groups / 8));
    var threads: [8]?std.Thread = @splat(null);
    const per = (groups + n_threads - 1) / n_threads;
    for (1..n_threads) |i| {
        const lo = i * per;
        const hi = @min(groups, lo + per);
        if (lo >= hi) break;
        threads[i] = std.Thread.spawn(.{}, packGroups, .{ t, src, dst, nb, lo, hi }) catch null;
        if (threads[i] == null) packGroups(t, src, dst, nb, lo, hi);
    }
    packGroups(t, src, dst, nb, 0, @min(groups, per));
    for (threads) |th| if (th) |x| x.join();
}

fn packGroups(t: Type, src: []const u8, dst: []u8, nb: usize, g0: usize, g1: usize) void {
    const bb = blockBytes(t);
    const pb = packedBytes(t);
    const pc = pieces(t);
    var tmp: [288]u8 = undefined;
    for (g0..g1) |g| {
        for (0..nb) |b| {
            const base = (g * nb + b) * 16 * pb;
            for (0..16) |lane| {
                const blk = src[((g * 16 + lane) * nb + b) * bb ..][0..bb];
                repack(t, blk, tmp[0..pb]);
                for (0..pc.a) |p| dst[base + (p * 16 + lane) * 16 ..][0..16].* = tmp[p * 16 ..][0..16].*;
                for (0..pc.c) |q| dst[base + pc.a * 256 + (q * 16 + lane) * 4 ..][0..4].* = tmp[pc.a * 16 + q * 4 ..][0..4].*;
            }
        }
    }
}

/// Bytes of a [rows][in] tensor of type t in the file (in a multiple of 256).
pub fn tensorBytes(t: Type, rows: u64, in: u64) u64 {
    return rows * (in / 256) * blockBytes(t);
}

/// The GGUF tensor type name (gguf-py spelling) of a type.
pub fn ggufName(t: Type) []const u8 {
    return switch (t) {
        .q2_k => "Q2_K",
        .q4_k => "Q4_K",
        .iq4_xs => "IQ4_XS",
        .iq2_xxs => "IQ2_XXS",
        .iq2_xs => "IQ2_XS",
        .iq2_s => "IQ2_S",
        .iq3_xxs => "IQ3_XXS",
        .iq3_s => "IQ3_S",
        .iq1_m => "IQ1_M",
        .q6_k => "Q6_K",
        .q5_k => "Q5_K",
        .q8_0 => "Q8_0",
        .iq4_nl => "IQ4_NL",
    };
}

pub fn fromGguf(name: []const u8) ?Type {
    inline for (std.enums.values(Type)) |t| {
        if (std.mem.eql(u8, name, ggufName(t))) return t;
    }
    return null;
}

/// Sets every argument (Buffer or integer) in order and queues the kernel.
fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        if (@TypeOf(v) == Buf) try k.setBuffer(i, v) else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

/// A view of b from byte `off`.
pub fn sub(b: Buf, off: usize) Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

pub fn envU32(name: [*:0]const u8, def: u32) u32 {
    const v = std.c.getenv(name) orelse return def;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch def;
}

/// One projection of a fused launch (Set.matvecMulti): weights in kernel layout, bf16 output y[y_off .. y_off + rows).
pub const Seg = struct { t: Type, w: Buf, y: Buf, rows: u32, y_off: u32 };

pub const Set = struct {
    mv: [count]rt.Kernel,
    mvf: [count]rt.Kernel,
    embed: [count]rt.Kernel,
    dq: [count]rt.Kernel,
    bf16: rt.Kernel,
    bf16x2: rt.Kernel,
    bf16r: rt.Kernel,
    quant: rt.Kernel,
    mvm: rt.Kernel,
    mvgu: rt.Kernel,
    /// Multi-row matvec kernels for 2, 4 and 8 activation rows (Set.matvecRows) and their activation quantizer.
    mvr: [3][count]rt.Kernel,
    quant_rows: rt.Kernel,
    mvrf: [2][3]rt.Kernel, // fp32-output 2-, 4- and 8-row, Q4_K and IQ4_XS (the lm_head of a verify window) Q4_K (the lm_head of a verify window)
    mvmr: [3]rt.Kernel, // 2, 4 and 8 rows
    mvgur: [3]rt.Kernel,
    /// Quantized activation rows scratch (8 rows of up to 17408 inputs), made on the first matvecRows.
    xqr: ?Buf = null,
    /// Prefill GEMM (prefillRows): decode kernels (IQ1_M has none), activation prep, GEMM variants, grown scratch.
    pfdec: [count]?rt.Kernel,
    pfprep: rt.Kernel,
    pfgemm: rt.Kernel,
    pf_module: rt.Module,
    pf_wv: ?Buf = null,
    pf_xt: ?Buf = null,
    rms32: rt.Kernel,
    aprep32: rt.Kernel,

    pub fn init(r: *rt.Runtime) !Set {
        @setEvalBranchQuota(100000);
        var m = try r.moduleWith(spv, std.c.getenv("GG_BUILD_FLAGS"));
        var s: Set = undefined;
        inline for (std.enums.values(Type), 0..) |t, i| {
            const name = @tagName(t);
            s.mv[i] = try m.kernel(std.fmt.comptimePrint("mv_{s}", .{name}), .{ wg_items, 1, 1 });
            s.mvf[i] = try m.kernel(std.fmt.comptimePrint("mvf_{s}", .{name}), .{ wg_items, 1, 1 });
            s.embed[i] = try m.kernel(std.fmt.comptimePrint("embed_{s}", .{name}), .{ wg_items, 1, 1 });
            s.dq[i] = try m.kernel(std.fmt.comptimePrint("dq_{s}", .{name}), .{ wg_items, 1, 1 });
            inline for (.{ 2, 4, 8 }, 0..) |rm, j| s.mvr[j][i] = try m.kernel(std.fmt.comptimePrint("mvr{d}_{s}", .{ rm, name }), .{ wg_items, 1, 1 });
        }
        s.bf16 = try m.kernel("mv_bf16", .{ 64, 1, 1 });
        s.bf16x2 = try m.kernel("mv_bf16x2", .{ 64, 1, 1 });
        s.bf16r = try m.kernel("mv_bf16r", .{ 64, 1, 1 });
        s.quant = try m.kernel("quant_x", .{ 64, 1, 1 });
        s.mvm = try m.kernel("mvm", .{ wg_items, 1, 1 });
        s.mvgu = try m.kernel("mvgu", .{ wg_items, 1, 1 });
        s.quant_rows = try m.kernel("quant_rows", .{ 64, 1, 1 });
        s.mvrf[0][0] = try m.kernel("mvrf2_q4_k", .{ wg_items, 1, 1 });
        s.mvrf[1][0] = try m.kernel("mvrf2_iq4_xs", .{ wg_items, 1, 1 });
        s.mvrf[0][1] = try m.kernel("mvrf4_q4_k", .{ wg_items, 1, 1 });
        s.mvrf[1][1] = try m.kernel("mvrf4_iq4_xs", .{ wg_items, 1, 1 });
        s.mvrf[0][2] = try m.kernel("mvrf8_q4_k", .{ wg_items, 1, 1 });
        s.mvrf[1][2] = try m.kernel("mvrf8_iq4_xs", .{ wg_items, 1, 1 });
        s.xqr = null;
        s.pf_wv = null;
        s.pf_xt = null;
        inline for (std.enums.values(Type), 0..) |t, i| {
            s.pfdec[i] = if (t == .iq1_m) null else try m.kernel(std.fmt.comptimePrint("pfdec_{s}", .{@tagName(t)}), .{ wg_items, 1, 1 });
        }
        s.pfprep = try m.kernel("pf_prep", .{ 64, 1, 1 });
        s.pf_module = try r.moduleWith(spv_pfgemm, "-cl-intel-256-GRF-per-thread"); // 128 accumulator registers a lane: the large-GRF build
        s.pfgemm = try s.pf_module.kernel("pfgemm", .{ 16, 1, 1 });
        s.mvmr[0] = try m.kernel("mvmr2", .{ wg_items, 1, 1 });
        s.mvmr[1] = try m.kernel("mvmr4", .{ wg_items, 1, 1 });
        s.mvmr[2] = try m.kernel("mvmr8", .{ wg_items, 1, 1 });
        s.mvgur[0] = try m.kernel("mvgur2", .{ wg_items, 1, 1 });
        s.mvgur[1] = try m.kernel("mvgur4", .{ wg_items, 1, 1 });
        s.mvgur[2] = try m.kernel("mvgur8", .{ wg_items, 1, 1 });
        s.rms32 = try m.kernel("rmsnorm_f32w", .{ 64, 1, 1 });
        s.aprep32 = try m.kernel("attn_prep_f32w", .{ 256, 1, 1 });
        return s;
    }

    /// Test kernel: the in-kernel activation quantization alone; xq holds k int8 bytes then k/32 fp32 scales.
    pub fn quantize(s: *Set, x: Buf, xq: Buf, k: u32) !void {
        try run(&s.quant, .{ (k / 32 + 63) / 64, 1, 1 }, .{ x, xq, sub(xq, k), k });
    }

    /// K ranges a row group is split into: the power of two bringing a launch's sub-group count closest to ~1280.
    pub fn ksplit(rows: u32, nb: u32) u32 {
        if (envU32("GG_KS", 0) != 0) return envU32("GG_KS", 1);
        var best: u32 = 1;
        var best_d: f64 = std.math.inf(f64);
        var ks: u32 = 1;
        while (ks <= nsgw and ks <= nb) : (ks *= 2) {
            const d = @abs(@log(@as(f64, @floatFromInt((rows / 16) * ks)) / @as(f64, @floatFromInt(envU32("GG_SG", 1280)))));
            if (d < best_d) {
                best_d = d;
                best = ks;
            }
        }
        return best;
    }

    /// Work-groups of a launch: a multiple of the 32 Xe cores so every core gets the same work.
    fn launchWgs(groups: u32, ks: u32) u32 {
        const per_wg = nsgw / ks;
        var n: u32 = 32;
        while ((groups + n - 1) / n > per_wg) n += 32;
        const mn = envU32("GG_MINWG", 0);
        return @max(n, mn);
    }

    /// y[y_off + row] = W x (rows % 16 == 0, in % 256 == 0, in <= 17408); w from packRows, x bf16; bf16 or fp32 out.
    pub fn matvec(s: *Set, t: Type, w: Buf, x: Buf, y: Buf, in: u32, y_off: u32, rows: u32, f32out: bool) !void {
        const k = if (f32out) &s.mvf[@intFromEnum(t)] else &s.mv[@intFromEnum(t)];
        const ks = ksplit(rows, in / 256);
        const wgs = launchWgs(rows / 16, ks);
        // arguments 7 and 8 are local memory for the quantized activation (int8 values, one fp32 scale per 32)
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 7, in, null), "setLocal");
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 8, in / 8, null), "setLocal");
        try run(k, .{ wgs, 1, 1 }, .{ w, x, y, in, y_off, rows, ks });
    }

    /// Up to three projections of the same input in one launch, bf16 outputs; IQ1_M has no fused path.
    pub fn matvecMulti(s: *Set, segs: []const Seg, x: Buf, in: u32) !void {
        std.debug.assert(segs.len >= 1 and segs.len <= 3);
        var w: [3]Buf = undefined;
        var y: [3]Buf = undefined;
        var rows: [3]u32 = .{ 0, 0, 0 };
        var offs: [3]u32 = .{ 0, 0, 0 };
        var types: u32 = 0;
        var mask: u32 = 0;
        var total: u32 = 0;
        for (0..3) |i| {
            const sg = if (i < segs.len) segs[i] else segs[0];
            w[i] = sg.w;
            y[i] = sg.y;
            if (i < segs.len) {
                rows[i] = sg.rows;
                offs[i] = sg.y_off;
                types |= @as(u32, @intFromEnum(sg.t)) << @intCast(8 * i);
                mask |= @as(u32, 1) << @intCast(@intFromEnum(sg.t));
                total += sg.rows;
            }
        }
        const ks = ksplit(total, in / 256);
        const wgs = launchWgs(total / 16, ks);
        const k = &s.mvm;
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 17, in, null), "setLocal");
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 18, in / 8, null), "setLocal");
        try run(k, .{ wgs, 1, 1 }, .{ w[0], w[1], w[2], x, y[0], y[1], y[2], in, ks, rows[0], rows[1], rows[2], offs[0], offs[1], offs[2], types, mask });
    }

    /// MLP gate + up + SwiGLU in one launch: act = bf16(silu(bf16(Wg x)) * bf16(Wu x)); in <= 17408, no IQ1_M.
    pub fn gateUp(s: *Set, tg: Type, wg: Buf, tu: Type, wu: Buf, x: Buf, act: Buf, in: u32, rows: u32) !void {
        const k = &s.mvgu;
        const types = @as(u32, @intFromEnum(tg)) | (@as(u32, @intFromEnum(tu)) << 8);
        const mask = (@as(u32, 1) << @intCast(@intFromEnum(tg))) | (@as(u32, 1) << @intCast(@intFromEnum(tu)));
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 8, in, null), "setLocal");
        try k.rt.drv.check(k.rt.drv.api.zeKernelSetArgumentValue(k.handle, 9, in / 8, null), "setLocal");
        try run(k, .{ launchWgs(rows / 16, 1), 1, 1 }, .{ wg, wu, x, act, in, rows, types, mask });
    }

    /// m = 1..8 bf16 rows against one projection into y[r * rows + y_off + row]; row r equals `matvec` alone.
    pub fn matvecRows(s: *Set, t: Type, w: Buf, x: Buf, m: u32, y: Buf, in: u32, y_off: u32, rows: u32) !void {
        if (m == 0 or m > 8) return error.Invalid;
        if (m == 1) return s.matvec(t, w, x, y, in, y_off, rows, false);
        if (s.xqr == null) s.xqr = try s.quant_rows.rt.alloc(8 * (17408 + 17408 / 8));
        const xq = s.xqr.?;
        try run(&s.quant_rows, .{ (m * (in / 32) + 63) / 64, 1, 1 }, .{ x, xq, in, m });
        const ks = ksplit(rows, in / 256);
        const j = rowsIdx(m);
        try run(&s.mvr[j][@intFromEnum(t)], .{ launchWgs(rows / 16, ks), 1, 1 }, .{ w, xq, y, in, y_off, rows, ks, m });
    }

    fn xRows(s: *Set, x: Buf, m: u32, in: u32) !Buf {
        if (s.xqr == null) s.xqr = try s.quant_rows.rt.alloc(8 * (17408 + 17408 / 8));
        try run(&s.quant_rows, .{ (m * (in / 32) + 63) / 64, 1, 1 }, .{ x, s.xqr.?, in, m });
        return s.xqr.?;
    }

    /// matvecMulti for m = 1..4 rows (row invariant, per row as matvecMulti alone): segment y[r * rows + y_off + row].
    pub fn matvecMultiRows(s: *Set, segs: []const Seg, x: Buf, in: u32, m: u32) !void {
        if (m == 0 or m > 8) return error.Invalid;
        if (m == 1) return s.matvecMulti(segs, x, in);
        var w: [3]Buf = undefined;
        var y: [3]Buf = undefined;
        var rows: [3]u32 = .{ 0, 0, 0 };
        var offs: [3]u32 = .{ 0, 0, 0 };
        var types: u32 = 0;
        var mask: u32 = 0;
        var total: u32 = 0;
        for (0..3) |i| {
            const sg = if (i < segs.len) segs[i] else segs[0];
            w[i] = sg.w;
            y[i] = sg.y;
            if (i < segs.len) {
                rows[i] = sg.rows;
                offs[i] = sg.y_off;
                types |= @as(u32, @intFromEnum(sg.t)) << @intCast(8 * i);
                mask |= @as(u32, 1) << @intCast(@intFromEnum(sg.t));
                total += sg.rows;
            }
        }
        const xq = try s.xRows(x, m, in);
        const ks = ksplit(total, in / 256);
        try run(&s.mvmr[rowsIdx(m)], .{ launchWgs(total / 16, ks), 1, 1 }, .{ w[0], w[1], w[2], xq, y[0], y[1], y[2], in, ks, rows[0], rows[1], rows[2], offs[0], offs[1], offs[2], types, mask, m });
    }

    /// gateUp for m = 1..4 rows: act[r * rows + row].
    pub fn gateUpRows(s: *Set, tg: Type, wg: Buf, tu: Type, wu: Buf, x: Buf, act: Buf, in: u32, rows: u32, m: u32) !void {
        if (m == 0 or m > 8) return error.Invalid;
        if (m == 1) return s.gateUp(tg, wg, tu, wu, x, act, in, rows);
        const types = @as(u32, @intFromEnum(tg)) | (@as(u32, @intFromEnum(tu)) << 8);
        const mask = (@as(u32, 1) << @intCast(@intFromEnum(tg))) | (@as(u32, 1) << @intCast(@intFromEnum(tu)));
        const xq = try s.xRows(x, m, in);
        try run(&s.mvgur[rowsIdx(m)], .{ launchWgs(rows / 16, 1), 1, 1 }, .{ wg, wu, xq, act, in, rows, types, mask, m });
    }

    fn grow(r: *rt.Runtime, slot: *?Buf, bytes: usize) !Buf {
        if (slot.*) |b| {
            if (b.len >= bytes) return b;
            var old = b;
            old.free();
        }
        slot.* = try r.alloc(bytes);
        return slot.*.?;
    }

    /// Prefill GEMM for R rows (from ~32): y[r * rows + y_off + col] = bf16(x[r] . W[col]); IQ1_M: error.Invalid
    pub fn prefillRows(s: *Set, t: Type, w: Buf, x: Buf, R: u32, y: Buf, in: u32, y_off: u32, rows: u32) !void {
        const dec = &(s.pfdec[@intFromEnum(t)] orelse return error.Invalid);
        if (rows % 16 != 0 or in % 256 != 0) return error.Invalid;
        const r = dec.rt;
        const ng = rows / 16;
        const nb = in / 256;
        const rp = (R + 63) / 64 * 64; // tokens padded to whole 64-token blocks (zeros), so the GEMM loads need no bounds test
        const wv = try grow(r, &s.pf_wv, @as(usize, in) * rows * 2);
        const xt = try grow(r, &s.pf_xt, @as(usize, rp) * in * 2);
        const skip = envU32("GG_PF_SKIP", 0); // debug: 1 = no prep, 2 = no decode, 4 = no gemm
        if (skip & 1 == 0) try run(&s.pfprep, .{ rp / 64, in / 8, 1 }, .{ x, xt, R, in, rp });
        if (skip & 2 == 0) try run(dec, .{ (ng * nb + nsgw - 1) / nsgw, 1, 1 }, .{ w, wv, nb, ng });
        if (skip & 4 == 0) try run(&s.pfgemm, .{ rp / 64, (rows + 31) / 32, 1 }, .{ wv, xt, y, R, rp, in, rows, y_off });
    }

    /// Verify-window lm_head: m = 1..8 rows against a Q4_K or IQ4_XS head, fp32 logits; other types error.Invalid.
    pub fn headRows(s: *Set, t: Type, w: Buf, x: Buf, m: u32, y: Buf, in: u32, rows: u32) !void {
        if ((t != .q4_k and t != .iq4_xs) or m == 0 or m > 8) return error.Invalid;
        if (m == 1) return s.matvec(t, w, x, y, in, 0, rows, true);
        const xq = try s.xRows(x, m, in);
        const ks = ksplit(rows, in / 256);
        try run(&s.mvrf[if (t == .q4_k) 0 else 1][rowsIdx(m)], .{ launchWgs(rows / 16, ks), 1, 1 }, .{ w, xq, y, in, @as(u32, 0), rows, ks, m });
    }

    /// Kernel index of an m-row launch: 2, 4 or 8 rows.
    fn rowsIdx(m: u32) usize {
        return if (m <= 2) 0 else if (m <= 4) 1 else 2;
    }

    /// m bf16-weight gate projections' worth of tokens: y[t * rows + y_off + row].
    pub fn matvecBf16Rows(s: *Set, w: Buf, x: Buf, y: Buf, in: u32, y_off: u32, rows: u32, m: u32) !void {
        try run(&s.bf16r, .{ rows, m, 1 }, .{ w, x, y, in, y_off, rows });
    }

    /// Embedding rows ids[0 .. m) as bf16 into y[t * in ..].
    pub fn embedRows(s: *Set, t: Type, w: Buf, ids: Buf, y: Buf, in: u32, m: u32) !void {
        try run(&s.embed[@intFromEnum(t)], .{ (in / 256 + wg_items - 1) / wg_items, m, 1 }, .{ w, ids, y, in / 256 });
    }

    /// Two bf16-weight projections (rows each) of the same input in one launch: y0 = W0 x, y1 = W1 x.
    pub fn matvecBf16Pair(s: *Set, w0: Buf, w1: Buf, x: Buf, y0: Buf, y1: Buf, in: u32, rows: u32) !void {
        try run(&s.bf16x2, .{ 2 * rows, 1, 1 }, .{ w0, w1, x, y0, y1, in, rows });
    }

    /// bf16 weights [rows][in] (the small gate projections), bf16 output.
    pub fn matvecBf16(s: *Set, w: Buf, x: Buf, y: Buf, in: u32, y_off: u32, rows: u32) !void {
        try run(&s.bf16, .{ rows, 1, 1 }, .{ w, x, y, in, y_off, rows });
    }

    /// One row of the table (row id in the first word of `ids`) as bf16 into y[0 .. in).
    pub fn embedRow(s: *Set, t: Type, w: Buf, ids: Buf, y: Buf, in: u32) !void {
        try run(&s.embed[@intFromEnum(t)], .{ (in / 256 + wg_items - 1) / wg_items, 1, 1 }, .{ w, ids, y, in / 256 });
    }

    /// Dequantizes `rows` rows of in = 256 * nb weights to fp32 (tests).
    pub fn dequant(s: *Set, t: Type, w: Buf, y: Buf, rows: u32, nb: u32) !void {
        try run(&s.dq[@intFromEnum(t)], .{ (rows * nb + wg_items - 1) / wg_items, 1, 1 }, .{ w, y, rows, nb });
    }
};
