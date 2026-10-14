//! DFlash2 block drafts (opt-in: TF_GLM_DRAFTER=dflash2, TF_GLM_DFLASH2 the checkpoint's folder). IncoAI's drafter
//! (incoai/GLM-5.3-Flash-DFlash2, CC BY-NC-ND 4.0: non-commercial use; read from its own download, never changed on
//! disk or redistributed) after z-lab/dflash's DFlash2 architecture (MIT); this implementation is our own.
//!
//! The drafter is a 5-layer Qwen3-style decoder (hidden 4096, 32/8 heads of 128, SwiGLU 12288) that drafts a whole
//! block at once: rows [anchor, MASK x 7] at positions start .. start + 7 attend, bidirectionally, to each other and
//! to the stream's context: the target's completed residuals after layers 5, 14, 24, 33 and 42 (each the mean of its
//! 4 mHC streams), concatenated, through fc and a norm, then each layer's own k and v projections (a ring of the
//! last 2048 positions). Two-tap dynamic convolutions along the block wrap both sublayers. The block's last 7 rows
//! go through the target's head; a path selector picks each draft among the row's 16 likeliest tokens by the
//! previous draft's codebook row. Drafts never change a reply: the target verifies them.
//!
//! Both ranks run the drafter on their own copy (the head is each rank's vocabulary half, its candidates gathered);
//! rank 0 alone runs the selector, and the verify frame carries its drafts to rank 1.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");
const forward = @import("forward.zig");
const blocks = @import("blocks.zig");
const draw_mod = @import("draw.zig");
const weights_mod = @import("weights.zig");

pub const TAPS = [_]usize{ 5, 14, 24, 33, 42 };
pub const BLOCK: usize = 8;
pub const MASK: u32 = 154856;
pub const WIN: i64 = 2048;
pub const LAYERS: usize = 5;
const D: usize = 4096;
const KVW: usize = 1024;
const QKV: usize = 4096 + 2 * KVW;
const FF: usize = 12288;
const RANK: usize = 256;
const TOPK: usize = 16;
const THETA: f32 = 10000;
const EPS: f32 = 1e-5;
const SCALE: f32 = 0.08838834764831845; // 128^-0.5
const VOCAB: usize = 154880;
const MAX_M: usize = forward.MAX_STREAMS * BLOCK; // block rows at most
const MAX_CTX: usize = @intCast(forward.PREFILL_ROWS_MAX); // context rows one ingest takes at most
const CW: usize = 2 * TOPK; // a row's gathered ints: 16 ids, then 16 values' bits
const CHUNKS: usize = 19; // 4096-column chunks of a rank's head rows at most
const QH: usize = 32;
const HD: usize = 128;
const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// Device bytes a stream's context ring takes (K and V of every layer).
pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;

const B16 = kern.B16;
const LayerW = struct {
    in_norm: u64, post_norm: u64, qn: u64, kn: u64, a_base: u64, m_base: u64,
    qkv: B16, kv: B16, o: B16, gu: B16, down: B16, a_proj: B16, m_proj: B16,
};

/// One stream's draft request: ``n`` rows of the last verify to take into its context first (0: the prompt's,
/// taken at prefill), the anchor (the token the block starts with, at the stream's position); rank 0 gets ``out``.
pub const Req = struct { s: *forward.Seq, n: usize, anchor: u32, out: ?[]u32 = null, probs: ?[]f64 = null, drafts: bool = true };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    weights: cuda.DeviceBuffer,
    bits: u8,
    layers: [LAYERS]LayerW = undefined,
    fc: B16 = undefined,
    hproj: B16 = undefined,
    hnorm: u64 = 0,
    norm: u64 = 0,
    pred: []u16, // the selector's codebooks, bf16 [VOCAB][RANK] (host: it reads 16 rows a draft)
    succ: []u16,
    scratch: cuda.DeviceBuffer,
    // block buffers [MAX_M][..]
    x: u64 = 0, y: u64 = 0, y2: u64 = 0, dyn: u64 = 0, qkv: u64 = 0, q: u64 = 0, kb: u64 = 0, vb: u64 = 0, att: u64 = 0, a: u64 = 0,
    gu: u64 = 0, act: u64 = 0, hn: u64 = 0, g: u64 = 0,
    // context buffers [MAX_CTX][..]
    cin: u64 = 0, c1: u64 = 0, ctx: u64 = 0, ckv: u64 = 0,
    // the target's taps: a verify step's rows and a prompt chunk's, [rows][5 D]
    taps_step: u64 = 0,
    taps_prefill: u64 = 0,
    // tables
    ids: u64 = 0, bpos: u64 = 0, rings: u64 = 0, starts: u64 = 0, froms: u64 = 0, cpos: u64 = 0, crings: u64 = 0, invt: u64 = 0, tk: u64 = 0, gtk: u64 = 0,
    pv: u64 = 0, pi: u64 = 0,
    pm: u64 = 0, pl: u64 = 0, pacc: u64 = 0, // the chunked attention's partials
    attn_chunked: bool = true, // TF_GLM_DFLASH2_ATTN=serial: df_attn's one block a kv head (the chunked kernel otherwise)
    h_ids: []i32, h_bpos: []i32, h_rings: []u64, h_starts: []i64, h_froms: []i64, h_cpos: []i64, h_crings: []u64,
    cand: []i32, // both ranks' gathered candidates, [2][MAX_M][W]
    gh: []f32, // the selector's hidden projections [MAX_M][RANK]
    forward_ms: f64 = 0,
    calls: u64 = 0,
    lost: u64 = 0, // streams whose last verify rows were gone at their draft (their context restarts)
    prof: bool = false, // TF_GLM_DFLASH2_PROF: each phase synchronized and timed (no graphs)
    graphs: [forward.MAX_STREAMS + 1]?cuda.graph.Exec = @splat(null), // the block's pass a stream count
    dump: ?[*:0]const u8 = null, // TF_GLM_DFLASH2_DUMP (development): a one-chunk prompt's taps and its first round, written there
    dump_state: u8 = 0, // 0: waiting for a prompt, 1: its taps written, 2: done
    dump_round: u32 = 0, // the dumped stream's rounds so far, and the one written (TF_GLM_DFLASH2_DUMP_ROUND, 10)
    dump_at: u32 = 10,
    tap_input: bool = false,
    tap_s0: bool = false,
    full_head: bool = false, // TF_GLM_DFLASH2_HEAD=full: the candidates from the verify head (8-bit) instead of the drafts' (4-bit)
    temp: f64 = 1.0, // TF_GLM_DFLASH2_TEMP: the selector's softmax temperature for the drafts' chances (the lane core's depth rule)
    phase_ms: [5]f64 = @splat(0), // context, layers, head and gather, selector, tables
    prof_fine: bool = false, // TF_GLM_DFLASH2_PROF=2: every kernel kind of the block pass synchronized and timed
    split: bool = false, // the MLP linears split across the two ranks (gate and up by columns, down by K; TF_GLM_DFLASH2_SPLIT=0: whole on each)
    ff: usize = FF, // the MLP width this rank holds
    fine_ms: [FINE.len]f64 = @splat(0),
    fine_t: std.Io.Timestamp = undefined,

    pub fn deinit(dr: *Drafter) void {
        for (&dr.graphs) |*g| if (g.*) |*ex| ex.deinit();
        dr.scratch.free();
        dr.weights.free();
        const g = dr.gpa;
        g.free(dr.pred);
        g.free(dr.succ);
        g.free(dr.h_ids);
        g.free(dr.h_bpos);
        g.free(dr.h_rings);
        g.free(dr.h_starts);
        g.free(dr.h_froms);
        g.free(dr.h_cpos);
        g.free(dr.h_crings);
        g.free(dr.cand);
        g.free(dr.gh);
        g.destroy(dr);
    }
};

/// The block pass's kernel kinds (TF_GLM_DFLASH2_PROF=2), in launch order.
pub const FINE = [_][]const u8{ "embed", "rmsnorm", "a_proj", "conv", "qkv", "block_qkv", "attn", "o_proj", "m_proj", "gate_up", "swiglu", "down", "head_mm", "top16", "cand_gather", "hproj", "down_gather", "down_sum" };

fn fine(dr: *Drafter, e: *forward.Engine, which: usize) !void {
    if (!dr.prof_fine) return;
    try e.k.stream.synchronize();
    const now = std.Io.Timestamp.now(e.io, .awake);
    dr.fine_ms[which] += @as(f64, @floatFromInt(dr.fine_t.durationTo(now).toNanoseconds())) / 1e6;
    dr.fine_t = now;
}

/// The kernel kinds' mean ms a round, logged (TF_GLM_DFLASH2_PROF=2) and cleared.
pub fn logFine(dr: *Drafter, rounds: u64) void {
    if (!dr.prof_fine) return;
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const n: f64 = @floatFromInt(@max(rounds, 1));
    for (FINE, dr.fine_ms) |name, ms| w.print(" {s} {d:.3}", .{ name, ms / n }) catch break;
    std.log.info("glm dflash2 kernels (ms a round):{s}", .{w.buffered()});
    dr.fine_ms = @splat(0);
}

/// out[m, n] = x[m, k] . w^T through the engine's own matmul (the drafter's linears packed as the dense layers are).
fn lin(e: *forward.Engine, x: u64, w: B16, out: u64, m: usize, f32_out: bool) !void {
    try kern.mm(&e.k, x, w.k, w, out, f32_out, @intCast(m), e.pbuf.sk); // the prompt buffers' split-K scratch: no prefill runs meanwhile
}

// -- loading -------------------------------------------------------------------------------------------------------

const Ent = struct { off: u64, n: u64 };

/// TF_GLM_DFLASH2_BITS: the linears packed at load to 4 bits (the default: they only propose, and lands as many as
/// 8 bits did on the ruler), 6 or 8 (a scale per 64), 0 keeps bf16.
fn bitsFromEnv() !u8 {
    const v = std.c.getenv("TF_GLM_DFLASH2_BITS") orelse return 4;
    const b = std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadDFlash2Bits;
    return switch (b) {
        0, 4, 6, 8 => b,
        else => error.BadDFlash2Bits,
    };
}

const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    ents: *std.StringHashMap(Ent),
    buf: std.ArrayList(u8) = .empty,
    words: std.ArrayList(u32) = .empty,
    scales: std.ArrayList(u16) = .empty,
    sub: std.ArrayList(u16) = .empty, // a slice of a stored tensor (a rank's share of a split linear)
    w: *cuda.DeviceBuffer,
    tmp: *cuda.DeviceBuffer,
    k: *kern.K,
    at: u64 = 0,
    bits: u8,

    fn take(l: *Loader, n: u64) u64 {
        const p = l.at;
        l.at += std.mem.alignForward(u64, n, 256);
        return l.w.ptr + p;
    }

    fn read(l: *Loader, name: []const u8, want: u64) ![]u8 {
        const e = l.ents.get(name) orelse {
            std.log.err("dflash2: no {s} in the checkpoint", .{name});
            return error.DrafterTensorMissing;
        };
        if (e.n != want) {
            std.log.err("dflash2: {s} holds {d} bytes, not {d}", .{ name, e.n, want });
            return error.DrafterTensorShape;
        }
        try l.buf.resize(l.gpa, e.n);
        _ = try l.file.readPositionalAll(l.io, l.buf.items, e.off);
        return l.buf.items;
    }

    /// A small bf16 tensor as stored.
    fn raw(l: *Loader, name: []const u8, n: u64) !u64 {
        const p = l.take(n);
        try l.w.upload(p - l.w.ptr, try l.read(name, n));
        return p;
    }

    /// The tensors ``names`` (rows ``rows`` each, ``kk`` columns) stacked into one linear, packed and tiled for kern.mm.
    fn linear(l: *Loader, names: []const []const u8, rows: []const usize, kk: usize) !B16 {
        var n: usize = 0;
        for (rows) |r| n += r;
        const bits = l.bits;
        const wb: usize = n * kk * (if (bits == 0) 16 else bits) / 8;
        const sb: usize = if (bits == 0) 0 else n * kk / 64 * 2;
        const dst = l.take(wb);
        const ds: u64 = if (sb > 0) l.take(sb) else 0;
        var r0: usize = 0;
        if (bits == 0) {
            for (names, rows) |name, r| {
                try l.tmp.upload(r0 * kk * 2, try l.read(name, r * kk * 2));
                r0 += r;
            }
        } else {
            try l.words.resize(l.gpa, n * kk * bits / 32);
            try l.scales.resize(l.gpa, n * kk / 64);
            for (names, rows) |name, r| {
                const src = try l.read(name, r * kk * 2);
                weights_mod.quantize(bits, @alignCast(std.mem.bytesAsSlice(u16, src)), r, kk,
                    l.words.items[r0 * kk * bits / 32 ..][0 .. r * kk * bits / 32], l.scales.items[r0 * kk / 64 ..][0 .. r * kk / 64]);
                r0 += r;
            }
            try l.tmp.upload(0, std.mem.sliceAsBytes(l.words.items));
            try l.tmp.upload(std.mem.alignForward(u64, wb, 256), std.mem.sliceAsBytes(l.scales.items));
        }
        try kern.mmTile(l.k, bits, l.tmp.ptr, l.tmp.ptr + std.mem.alignForward(u64, wb, 256), dst, ds, @intCast(n), @intCast(kk));
        try l.k.stream.synchronize();
        return .{ .w = dst, .n = @intCast(n), .k = @intCast(kk), .s = ds, .bits = bits };
    }

    /// Rows ``row0 .. row0 + rn`` and columns ``col0 .. col0 + cn`` of each of the tensors ``names`` (rows ``rows``
    /// each, ``kk`` columns as stored), stacked into one linear of names.len * rn rows and cn columns: a rank's
    /// share of a split linear, packed and tiled as ``linear`` packs a whole one.
    fn linearPart(l: *Loader, names: []const []const u8, rows: []const usize, kk: usize, row0: usize, rn: usize, col0: usize, cn: usize) !B16 {
        const n = names.len * rn;
        const bits = l.bits;
        const wb: usize = n * cn * (if (bits == 0) 16 else bits) / 8;
        const sb: usize = if (bits == 0) 0 else n * cn / 64 * 2;
        const dst = l.take(wb);
        const ds: u64 = if (sb > 0) l.take(sb) else 0;
        try l.sub.resize(l.gpa, rn * cn);
        if (bits != 0) {
            try l.words.resize(l.gpa, n * cn * bits / 32);
            try l.scales.resize(l.gpa, n * cn / 64);
        }
        var r0: usize = 0;
        for (names, rows) |name, r| {
            if (row0 + rn > r or col0 + cn > kk) return error.DrafterTensorShape;
            const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, try l.read(name, r * kk * 2)));
            for (0..rn) |i| @memcpy(l.sub.items[i * cn ..][0..cn], src[(row0 + i) * kk + col0 ..][0..cn]);
            if (bits == 0) {
                try l.tmp.upload(r0 * cn * 2, std.mem.sliceAsBytes(l.sub.items));
            } else {
                weights_mod.quantize(bits, l.sub.items, rn, cn, l.words.items[r0 * cn * bits / 32 ..][0 .. rn * cn * bits / 32], l.scales.items[r0 * cn / 64 ..][0 .. rn * cn / 64]);
            }
            r0 += rn;
        }
        if (bits != 0) {
            try l.tmp.upload(0, std.mem.sliceAsBytes(l.words.items));
            try l.tmp.upload(std.mem.alignForward(u64, wb, 256), std.mem.sliceAsBytes(l.scales.items));
        }
        try kern.mmTile(l.k, bits, l.tmp.ptr, l.tmp.ptr + std.mem.alignForward(u64, wb, 256), dst, ds, @intCast(n), @intCast(cn));
        try l.k.stream.synchronize();
        return .{ .w = dst, .n = @intCast(n), .k = @intCast(cn), .s = ds, .bits = bits };
    }
};

fn linearBytes(n: usize, kk: usize, bits: u8) u64 {
    const wb = std.mem.alignForward(u64, n * kk * (if (bits == 0) 16 else @as(usize, bits)) / 8, 256);
    return wb + if (bits == 0) 0 else std.mem.alignForward(u64, n * kk / 64 * 2, 256);
}

/// The drafter from ``dir`` (config.json and model.safetensors, bf16), its linears packed for the engine's matmul.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *forward.Engine, dir: []const u8) !*Drafter {
    try checkConfig(gpa, io, dir);
    const bits = try bitsFromEnv();
    const d = e.ctx.d;
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(path);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var head: [8]u8 = undefined;
    _ = try file.readPositionalAll(io, &head, 0);
    const hlen = std.mem.readInt(u64, &head, .little);
    if (hlen > 1 << 20) return error.BadDrafterHeader;
    const text = try gpa.alloc(u8, hlen);
    defer gpa.free(text);
    _ = try file.readPositionalAll(io, text, 8);
    const hdr = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer hdr.deinit();
    var ents = std.StringHashMap(Ent).init(gpa);
    defer ents.deinit();
    var it = hdr.value.object.iterator();
    while (it.next()) |x| {
        if (std.mem.eql(u8, x.key_ptr.*, "__metadata__")) continue;
        if (!std.mem.eql(u8, x.value_ptr.object.get("dtype").?.string, "BF16")) return error.DrafterNotBf16;
        const offs = x.value_ptr.object.get("data_offsets").?.array.items;
        try ents.put(x.key_ptr.*, .{ .off = 8 + hlen + @as(u64, @intCast(offs[0].integer)), .n = @intCast(offs[1].integer - offs[0].integer) });
    }
    // the device's share: every linear packed (q|k|v and gate|up stacked, k|v again for the context), the norms and
    // conv kernels as stored; the codebooks stay on the host
    var total: u64 = 0;
    const split = e.ctx.world == 2 and if (std.c.getenv("TF_GLM_DFLASH2_SPLIT")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    const ff: usize = if (split) FF / 2 else FF;
    const small: u64 = 6 * 4096 * 2 + 2 * 128 * 2 + 2 * 2 * 2 * D * 2;
    total += LAYERS * (linearBytes(QKV, D, bits) + linearBytes(2 * KVW, D, bits) + linearBytes(D, D, bits) + linearBytes(2 * ff, D, bits) +
        linearBytes(D, ff, bits) + 2 * linearBytes(KVW, D, bits) + small + 8 * 256);
    total += linearBytes(D, 5 * D, bits) + linearBytes(RANK, D, bits) + 2 * D * 2 + 1024;
    var wbuf = try cuda.DeviceBuffer.alloc(d, total);
    errdefer wbuf.free();
    var tmp = try cuda.DeviceBuffer.alloc(d, linearBytes(2 * FF, D, 0) + 4096); // the largest linear as bf16
    defer tmp.free();
    var l: Loader = .{ .gpa = gpa, .io = io, .file = file, .ents = &ents, .w = &wbuf, .tmp = &tmp, .k = &e.k, .bits = bits };
    defer {
        l.buf.deinit(gpa);
        l.words.deinit(gpa);
        l.scales.deinit(gpa);
        l.sub.deinit(gpa);
    }
    const dr = try gpa.create(Drafter);
    errdefer gpa.destroy(dr);
    dr.* = .{ .gpa = gpa, .d = d, .weights = wbuf, .bits = bits, .split = split, .ff = ff, .pred = &.{}, .succ = &.{}, .scratch = undefined, .h_ids = undefined,
        .h_bpos = undefined, .h_rings = undefined, .h_starts = undefined, .h_froms = undefined, .h_cpos = undefined, .h_crings = undefined,
        .cand = undefined, .gh = undefined };
    var nb: [8][96]u8 = undefined;
    for (&dr.layers, 0..) |*lw, i| {
        const N = struct {
            fn n(b: *[96]u8, li: usize, s: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ li, s });
            }
        }.n;
        lw.in_norm = try l.raw(try N(&nb[0], i, "input_layernorm.weight"), D * 2);
        lw.post_norm = try l.raw(try N(&nb[0], i, "post_attention_layernorm.weight"), D * 2);
        lw.qn = try l.raw(try N(&nb[0], i, "self_attn.q_norm.weight"), 128 * 2);
        lw.kn = try l.raw(try N(&nb[0], i, "self_attn.k_norm.weight"), 128 * 2);
        lw.a_base = try l.raw(try N(&nb[0], i, "attention_conv.base_kernel"), 2 * 2 * D * 2);
        lw.m_base = try l.raw(try N(&nb[0], i, "mlp_conv.base_kernel"), 2 * 2 * D * 2);
        const q = try N(&nb[1], i, "self_attn.q_proj.weight");
        const kq = try N(&nb[2], i, "self_attn.k_proj.weight");
        const v = try N(&nb[3], i, "self_attn.v_proj.weight");
        lw.qkv = try l.linear(&.{ q, kq, v }, &.{ D, KVW, KVW }, D);
        lw.kv = try l.linear(&.{ kq, v }, &.{ KVW, KVW }, D);
        lw.o = try l.linear(&.{try N(&nb[1], i, "self_attn.o_proj.weight")}, &.{D}, D);
        const gate = try N(&nb[1], i, "mlp.gate_proj.weight");
        const up = try N(&nb[2], i, "mlp.up_proj.weight");
        const down = try N(&nb[3], i, "mlp.down_proj.weight");
        if (split) { // this rank's half of the MLP width: gate and up rows, down columns
            lw.gu = try l.linearPart(&.{ gate, up }, &.{ FF, FF }, D, e.ctx.rank * ff, ff, 0, D);
            lw.down = try l.linearPart(&.{down}, &.{D}, FF, 0, D, e.ctx.rank * ff, ff);
        } else {
            lw.gu = try l.linear(&.{ gate, up }, &.{ FF, FF }, D);
            lw.down = try l.linear(&.{down}, &.{D}, FF);
        }
        lw.a_proj = try l.linear(&.{try N(&nb[1], i, "attention_conv.kernel_projection.weight")}, &.{KVW}, D);
        lw.m_proj = try l.linear(&.{try N(&nb[1], i, "mlp_conv.kernel_projection.weight")}, &.{KVW}, D);
    }
    dr.fc = try l.linear(&.{"fc.weight"}, &.{D}, 5 * D);
    dr.hproj = try l.linear(&.{"candidate_selector.hidden_projection.weight"}, &.{RANK}, D);
    dr.hnorm = try l.raw("hidden_norm.weight", D * 2);
    dr.norm = try l.raw("norm.weight", D * 2);
    if (l.at > total) return error.DrafterLayout;
    dr.pred = try gpa.alloc(u16, VOCAB * RANK);
    @memcpy(std.mem.sliceAsBytes(dr.pred), try l.read("candidate_selector.predecessor_codebook", VOCAB * RANK * 2));
    dr.succ = try gpa.alloc(u16, VOCAB * RANK);
    @memcpy(std.mem.sliceAsBytes(dr.succ), try l.read("candidate_selector.successor_codebook", VOCAB * RANK * 2));
    try carveScratch(dr, d);
    dr.h_ids = try gpa.alloc(i32, MAX_M);
    dr.h_bpos = try gpa.alloc(i32, MAX_M);
    dr.h_rings = try gpa.alloc(u64, forward.MAX_STREAMS);
    dr.h_starts = try gpa.alloc(i64, forward.MAX_STREAMS);
    dr.h_froms = try gpa.alloc(i64, forward.MAX_STREAMS);
    dr.h_cpos = try gpa.alloc(i64, MAX_CTX);
    dr.h_crings = try gpa.alloc(u64, MAX_CTX);
    dr.cand = try gpa.alloc(i32, 2 * MAX_M * CW);
    dr.gh = try gpa.alloc(f32, MAX_M * RANK);
    dr.prof = std.c.getenv("TF_GLM_DFLASH2_PROF") != null;
    if (std.c.getenv("TF_GLM_DFLASH2_ATTN")) |v| dr.attn_chunked = !std.mem.eql(u8, std.mem.span(v), "serial");
    dr.prof_fine = if (std.c.getenv("TF_GLM_DFLASH2_PROF")) |v| std.mem.eql(u8, std.mem.span(v), "2") else false;
    dr.dump = std.c.getenv("TF_GLM_DFLASH2_DUMP");
    if (std.c.getenv("TF_GLM_DFLASH2_TAP")) |v| {
        dr.tap_input = std.mem.eql(u8, std.mem.span(v), "input");
        dr.tap_s0 = std.mem.eql(u8, std.mem.span(v), "s0");
    }
    if (std.c.getenv("TF_GLM_DFLASH2_DUMP_ROUND")) |v| dr.dump_at = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 10;
    if (std.c.getenv("TF_GLM_DFLASH2_HEAD")) |v| dr.full_head = std.mem.eql(u8, std.mem.span(v), "full");
    if (std.c.getenv("TF_GLM_DFLASH2_TEMP")) |v| dr.temp = std.fmt.parseFloat(f64, std.mem.span(v)) catch 1.0;
    const inv: [MAX_M]f32 = @splat(1.0);
    try dr.scratch.upload(dr.invt - dr.scratch.ptr, std.mem.sliceAsBytes(&inv));
    std.log.info("dflash2: drafter loaded from {s}, linears at {d} bits ({d:.2} GiB on the device, {d} MiB of codebooks on the host)",
        .{ dir, bits, @as(f64, @floatFromInt(l.at)) / (1 << 30), (2 * VOCAB * RANK * 2) >> 20 });
    return dr;
}

fn carveScratch(dr: *Drafter, d: *const cuda.Driver) !void {
    var at: u64 = 0;
    const M: u64 = MAX_M;
    const C: u64 = MAX_CTX + 16;
    const fields = .{
        .{ "x", M * D * 2 },              .{ "y", M * D * 2 },             .{ "y2", M * D * 2 },            .{ "dyn", M * KVW * 2 },
        .{ "qkv", M * QKV * 2 },          .{ "q", M * D * 2 },             .{ "kb", M * KVW * 2 },          .{ "vb", M * KVW * 2 },
        .{ "att", M * D * 2 },            .{ "a", M * D * 2 },             .{ "gu", M * 2 * FF * 2 },       .{ "act", M * FF * 2 },
        .{ "hn", M * D * 2 },             .{ "g", M * RANK * 4 },          .{ "cin", C * 5 * D * 2 },       .{ "c1", C * D * 2 },
        .{ "ctx", C * D * 2 },            .{ "ckv", C * 2 * KVW * 2 },
        .{ "taps_step", @as(u64, @intCast(forward.STEP_ROWS)) * 5 * D * 2 }, .{ "taps_prefill", @as(u64, @intCast(forward.PREFILL_ROWS_MAX)) * 5 * D * 2 },
        .{ "ids", M * 4 },                .{ "bpos", M * 4 },              .{ "rings", forward.MAX_STREAMS * 8 }, .{ "starts", forward.MAX_STREAMS * 8 },
        .{ "froms", forward.MAX_STREAMS * 8 }, .{ "cpos", C * 8 },         .{ "crings", C * 8 },            .{ "invt", M * 4 },
        .{ "tk", M * CW * 4 },            .{ "gtk", 2 * M * CW * 4 },
        .{ "pv", M * CHUNKS * TOPK * 4 }, .{ "pi", M * CHUNKS * TOPK * 4 },
        .{ "pm", M * QH * CH * 4 },       .{ "pl", M * QH * CH * 4 },      .{ "pacc", M * QH * CH * HD * 4 },
    };
    inline for (fields) |f| {
        @field(dr, f[0]) = at;
        at += std.mem.alignForward(u64, f[1], 256);
    }
    dr.scratch = try cuda.DeviceBuffer.alloc(d, at);
    inline for (fields) |f| @field(dr, f[0]) += dr.scratch.ptr;
    try d.check(d.api.cuMemsetD8_v2(dr.scratch.ptr, 0, at), "zero drafter scratch");
}

/// The checkpoint is the one this was written for (its shape and DFlash2 settings).
fn checkConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer p.deinit();
    const o = p.value.object;
    const df = (o.get("dflash_config") orelse return error.NotDFlash2).object;
    const ok = o.get("hidden_size").?.integer == D and o.get("num_hidden_layers").?.integer == LAYERS and
        o.get("num_attention_heads").?.integer == 32 and o.get("num_key_value_heads").?.integer == 8 and
        o.get("intermediate_size").?.integer == FF and o.get("vocab_size").?.integer == VOCAB and
        df.get("block_size").?.integer == BLOCK and df.get("mask_token_id").?.integer == MASK and
        df.get("selector_top_k").?.integer == TOPK and df.get("selector_rank").?.integer == RANK and
        df.get("conv_kernel_size").?.integer == 2 and df.get("conv_group_size").?.integer == 16;
    if (!ok) return error.DrafterConfigUnexpected;
    const taps = df.get("target_layer_ids").?.array.items;
    if (taps.len != TAPS.len) return error.DrafterConfigUnexpected;
    for (taps, TAPS) |t, w| if (t.integer != w) return error.DrafterConfigUnexpected;
}

// -- the target's side ---------------------------------------------------------------------------------------------

/// After target layer ``li``: its rows' 4-stream means into the taps (a verify step's or a prompt chunk's).
/// TF_GLM_DFLASH2_TAP (development): `mean` (the default), `input` (layer k's input: k - 1's output), `s0` (stream 0).
pub fn tap(dr: *const Drafter, k: *kern.K, x: u64, taps: u64, li: usize, R: i64) !void {
    const at = li + @intFromBool(dr.tap_input);
    const slot: i32 = for (TAPS, 0..) |t, i| {
        if (t == at) break @intCast(i);
    } else return;
    var a: cuda.Args = .{};
    a.add(x); a.add(taps); a.add(slot); a.add(@as(i32, @intFromBool(dr.tap_s0)));
    try k.go(try k.ext(.dflash, "df_tap"), .{ .x = @intCast(R), .y = @intCast(D / 1024) }, 256, 0, &a);
}

pub fn isTap(dr: *const Drafter, li: usize) bool {
    const at = li + @intFromBool(dr.tap_input);
    for (TAPS) |t| if (t == at) return true;
    return false;
}

/// Context runs: ``n`` tap rows from ``taps`` (consecutive) at positions pos0 .. pos0 + n, into ``s``'s ring.
pub const Run = struct { s: *forward.Seq, taps: u64, n: usize, pos0: i64 };

/// The runs' rows through fc, the hidden norm and every layer's k and v, into their rings (and synchronized: the
/// host tables are reused).
/// TF_GLM_DFLASH2_DUMP: a whole one-chunk prompt's taps ([n][5 D] bf16, positions 0 .. n), once.
pub fn dumpPrompt(dr: *Drafter, e: *forward.Engine, n: usize) !void {
    if (dr.dump == null or dr.dump_state == 2 or n > MAX_CTX) return; // the latest prompt until its first round is written
    try e.k.stream.synchronize();
    const buf = try dr.gpa.alloc(u8, n * 5 * D * 2);
    defer dr.gpa.free(buf);
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(buf.ptr), dr.taps_prefill, buf.len), "dump taps");
    try writeFile(dr, "taps.bin", buf);
    try writeFile(dr, "ingest.bin", "");
    dr.dump_state = 1;
    dr.dump_round = 0;
}

/// While dumping: the ingest's rows (their positions, then their taps) appended to ingest.bin.
fn dumpIngest(dr: *Drafter, e: *forward.Engine, N: usize) !void {
    try e.k.stream.synchronize();
    const buf = try dr.gpa.alloc(u8, N * 5 * D * 2);
    defer dr.gpa.free(buf);
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(buf.ptr), dr.cin, buf.len), "dump ingest");
    var pb: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pb, "{s}/ingest.bin", .{std.mem.span(dr.dump.?)}, 0);
    const f = std.c.fopen(path, "ab") orelse return error.DumpOpen;
    defer _ = std.c.fclose(f);
    const hdr = [2]i64{ @intCast(N), 0 };
    _ = std.c.fwrite(std.mem.asBytes(&hdr).ptr, 1, 16, f);
    _ = std.c.fwrite(std.mem.sliceAsBytes(dr.h_cpos[0..N]).ptr, 1, N * 8, f);
    _ = std.c.fwrite(buf.ptr, 1, buf.len, f);
}

fn writeFile(dr: *Drafter, name: []const u8, bytes: []const u8) !void {
    var pb: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ std.mem.span(dr.dump.?), name }, 0);
    const f = std.c.fopen(path, "wb") orelse return error.DumpOpen;
    defer _ = std.c.fclose(f);
    if (std.c.fwrite(bytes.ptr, 1, bytes.len, f) != bytes.len) return error.DumpWrite;
}

pub fn ingest(dr: *Drafter, e: *forward.Engine, runs: []const Run) !void {
    const k = &e.k;
    var N: usize = 0;
    for (runs) |r| N += r.n;
    if (N == 0) return;
    if (N > MAX_CTX) return error.DrafterContextTooWide;
    var row: usize = 0;
    for (runs) |r| {
        if (r.n == 0) continue;
        try k.copy(dr.cin + row * 5 * D * 2, r.taps, r.n * 5 * D * 2);
        for (0..r.n) |j| {
            dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
            dr.h_crings[row + j] = r.s.draft;
        }
        row += r.n;
    }
    if (dr.dump != null and dr.dump_state == 1) try dumpIngest(dr, e, N);
    try k.upload(dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
    try k.upload(dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
    try lin(e, dr.cin, dr.fc, dr.c1, N, false);
    try kern.rmsnorm(k, dr.c1, D, dr.hnorm, dr.ctx, D, null, @intCast(N), D);
    for (dr.layers, 0..) |l, li| {
        try lin(e, dr.ctx, l.kv, dr.ckv, N, false);
        var a: cuda.Args = .{};
        a.add(dr.ckv); a.add(l.kn); a.add(dr.cpos); a.add(dr.crings); a.add(@as(i32, @intCast(li))); a.add(EPS); a.add(THETA);
        try k.go(try k.ext(.dflash, "df_ctx_kv"), .{ .x = @intCast(N), .y = 8 }, 64, 0, &a);
    }
    try k.stream.synchronize();
}

// -- drafting ------------------------------------------------------------------------------------------------------

/// Every request's block drafted on this rank (both ranks, in step: the head's candidates gather); rank 0 (the
/// requests' ``out``) gets 7 drafts and their chances.
/// Every request's kept rows into its context; then the block pass for the ones that draft (``drafts``: the others
/// draft with the MTP head this round and only keep their context current).
pub fn draft(dr: *Drafter, e: *forward.Engine, all: []const Req) !void {
    const k = &e.k;
    if (all.len == 0) return;
    if (all.len > forward.MAX_STREAMS) return error.TooManyStreams;
    const t0 = std.Io.Timestamp.now(e.io, .awake);
    var tp = t0;
    // the last verify's kept rows into each stream's context
    var runs: [forward.MAX_STREAMS]Run = undefined;
    var nr: usize = 0;
    for (all) |q| {
        if (q.n == 0) continue;
        const s = q.s;
        if (s.v_step != e.verifies or q.n > s.v_rows) { // its rows were overwritten (or never verified): restart its context
            s.d_from = s.pos;
            dr.lost += 1;
            continue;
        }
        runs[nr] = .{ .s = s, .taps = dr.taps_step + @as(u64, @intCast(s.v_row0)) * 5 * D * 2, .n = q.n, .pos0 = s.v_pos0 };
        nr += 1;
    }
    try ingest(dr, e, runs[0..nr]);
    try phase(dr, e, &tp, 0);
    var sub: [forward.MAX_STREAMS]Req = undefined;
    var S: usize = 0;
    for (all) |q| if (q.drafts) {
        sub[S] = q;
        S += 1;
    };
    if (S == 0) return;
    const reqs = sub[0..S];
    // the block: [anchor, MASK x 7] a stream, at positions pos .. pos + 7
    const M = S * BLOCK;
    if (dr.split and M > e.buf.rows) return error.DrafterRowsExceedBuffers; // the split's partials use the decode buffers
    for (reqs, 0..) |q, i| {
        dr.h_rings[i] = q.s.draft;
        dr.h_starts[i] = q.s.pos;
        dr.h_froms[i] = q.s.d_from;
        for (0..BLOCK) |t| {
            dr.h_ids[i * BLOCK + t] = @intCast(if (t == 0) q.anchor else MASK);
            dr.h_bpos[i * BLOCK + t] = @intCast(q.s.pos + @as(i64, @intCast(t)));
        }
    }
    try k.upload(dr.ids, std.mem.sliceAsBytes(dr.h_ids[0..M]));
    try k.upload(dr.bpos, std.mem.sliceAsBytes(dr.h_bpos[0..M]));
    try k.upload(dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));
    try k.upload(dr.starts, std.mem.sliceAsBytes(dr.h_starts[0..S]));
    try k.upload(dr.froms, std.mem.sliceAsBytes(dr.h_froms[0..S]));
    try phase(dr, e, &tp, 4);
    const rank0 = reqs[0].out != null;
    const nw = M * CW;
    if (dr.prof or !e.use_graphs or e.sync_each) {
        try blockBody(dr, e, S, rank0, &tp);
    } else if (dr.graphs[S]) |ex| {
        try ex.launchOn(k.stream);
    } else { // the block's pass for S streams, captured once (its tables at fixed places, uploaded above)
        try cuda.graph.beginCapture(k.stream, .thread_local);
        blockBody(dr, e, S, rank0, &tp) catch |err| {
            if (cuda.graph.endCapture(k.stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(k.stream);
        defer g.deinit();
        dr.graphs[S] = try g.instantiate();
        try dr.graphs[S].?.launchOn(k.stream);
    }
    try k.stream.synchronize();
    try phase(dr, e, &tp, 2);
    if (!rank0 and dr.prof_fine and dr.calls % 256 == 255) logFine(dr, 256); // rank 1 logs its own table (rank 0's goes with the trace)
    if (rank0) {
        try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(dr.cand.ptr), dr.gtk, 2 * nw * 4), "dflash2 candidates");
        try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(dr.gh.ptr), dr.g, M * RANK * 4), "dflash2 projections");
        for (reqs, 0..) |q, i| select(dr, i, M, q);
        if (dr.dump != null and dr.dump_state == 1 and S == 1) dr.dump_round += 1;
        if (dr.dump != null and dr.dump_state == 1 and S == 1 and dr.dump_round == dr.dump_at) {
            const hn = try dr.gpa.alloc(u8, M * D * 2);
            defer dr.gpa.free(hn);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(hn.ptr), dr.hn, hn.len), "dump hidden");
            try writeFile(dr, "hidden.bin", hn);
            try writeFile(dr, "cand.bin", std.mem.sliceAsBytes(dr.cand[0 .. 2 * nw]));
            try writeFile(dr, "proj.bin", std.mem.sliceAsBytes(dr.gh[0 .. M * RANK]));
            var meta: [64]u8 = undefined;
            const q = reqs[0];
            const text = try std.fmt.bufPrint(&meta, "{d} {d}", .{ q.anchor, q.s.pos });
            try writeFile(dr, "meta.txt", text);
            try writeFile(dr, "drafts.bin", std.mem.sliceAsBytes(q.out.?[0 .. BLOCK - 1]));
            try writeFile(dr, "probs.bin", std.mem.sliceAsBytes(q.probs.?[0 .. BLOCK - 1]));
            dr.dump_state = 2;
            std.log.info("dflash2: dumped a prompt's taps and its first round to {s}", .{std.mem.span(dr.dump.?)});
        }
    }
    try phase(dr, e, &tp, 3);
    dr.forward_ms += @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(e.io, .awake)).toNanoseconds())) / 1e6;
    dr.calls += 1;
}

/// The block's pass for S streams over the uploaded tables: embed, the 5 layers, the final norm, the draft head's 16
/// best a row gathered from both ranks, and (rank 0) the selector's projections. Capturable.
fn blockBody(dr: *Drafter, e: *forward.Engine, S: usize, rank0: bool, tp: *std.Io.Timestamp) !void {
    const k = &e.k;
    const M = S * BLOCK;
    const nw = M * CW;
    if (dr.prof_fine) dr.fine_t = std.Io.Timestamp.now(e.io, .awake);
    try kern.embed(k, dr.ids, e.w.embed, dr.x, @intCast(M), 1);
    try fine(dr, e, 0);
    const R: i64 = @intCast(M);
    const grid: cuda.Dim3 = .{ .x = @intCast(M), .y = @intCast(D / 1024) };
    for (dr.layers, 0..) |l, li| {
        // attention, wrapped in the attention conv
        try kern.rmsnorm(k, dr.x, D, l.in_norm, dr.y, D, null, R, D);
        try fine(dr, e, 1);
        try lin(e, dr.y, l.a_proj, dr.dyn, M, false);
        try fine(dr, e, 2);
        try conv(k, dr.y, dr.dyn, 0, l.a_base, dr.y2, false, grid);
        try fine(dr, e, 3);
        try lin(e, dr.y2, l.qkv, dr.qkv, M, false);
        try fine(dr, e, 4);
        {
            var a: cuda.Args = .{};
            a.add(dr.qkv); a.add(@as(i32, @intCast(QKV))); a.add(l.qn); a.add(l.kn); a.add(dr.bpos); a.add(EPS); a.add(THETA);
            a.add(dr.q); a.add(dr.kb); a.add(dr.vb);
            try k.go(try k.ext(.dflash, "df_block_qkv"), .{ .x = @intCast(M), .y = 32 + 8 }, 64, 0, &a);
        }
        try fine(dr, e, 5);
        if (dr.attn_chunked) {
            var a: cuda.Args = .{};
            a.add(dr.q); a.add(dr.kb); a.add(dr.vb); a.add(dr.rings); a.add(dr.starts); a.add(dr.froms); a.add(@as(i32, @intCast(li)));
            a.add(SCALE); a.add(dr.pm); a.add(dr.pl); a.add(dr.pacc);
            try k.go(try k.ext(.dflash, "df_attn_part"), .{ .x = @intCast(S), .y = 8, .z = CH }, 256, 0, &a);
            var b: cuda.Args = .{};
            b.add(dr.pm); b.add(dr.pl); b.add(dr.pacc); b.add(dr.att);
            try k.go(try k.ext(.dflash, "df_attn_merge"), .{ .x = @intCast(M), .y = QH }, HD, 0, &b);
        } else {
            var a: cuda.Args = .{};
            a.add(dr.q); a.add(dr.kb); a.add(dr.vb); a.add(dr.rings); a.add(dr.starts); a.add(dr.froms); a.add(@as(i32, @intCast(li)));
            a.add(SCALE); a.add(dr.att);
            try k.go(try k.ext(.dflash, "df_attn"), .{ .x = @intCast(S), .y = 8 }, 256, 0, &a);
        }
        try fine(dr, e, 6);
        try lin(e, dr.att, l.o, dr.a, M, false);
        try fine(dr, e, 7);
        try conv(k, dr.a, dr.dyn, 1, l.a_base + 2 * D * 2, dr.x, true, grid);
        try fine(dr, e, 3);
        // the MLP, wrapped in the MLP conv
        try kern.rmsnorm(k, dr.x, D, l.post_norm, dr.y, D, null, R, D);
        try fine(dr, e, 1);
        try lin(e, dr.y, l.m_proj, dr.dyn, M, false);
        try fine(dr, e, 8);
        try conv(k, dr.y, dr.dyn, 0, l.m_base, dr.y2, false, grid);
        try fine(dr, e, 3);
        try lin(e, dr.y2, l.gu, dr.gu, M, false);
        try fine(dr, e, 9);
        {
            var a: cuda.Args = .{};
            a.add(dr.gu); a.add(dr.act); a.add(@as(i32, @intCast(dr.ff)));
            try k.go(try k.ext(.dflash, "df_swiglu"), .{ .x = @intCast(M), .y = @intCast(dr.ff / 1024) }, 256, 0, &a);
        }
        try fine(dr, e, 10);
        if (dr.split) { // each rank's half of down's K as fp32 partials, gathered and summed
            const b = &e.buf;
            try lin(e, dr.act, l.down, b.part, M, true);
            try fine(dr, e, 11);
            try e.ctx.allGather(b.part, b.gath, M * D, .f32, "dflash2 down");
            try fine(dr, e, 16);
            var a: cuda.Args = .{};
            a.add(b.gath); a.add(dr.a); a.add(@as(i32, @intCast(M)));
            try k.go(try k.ext(.dflash, "df_sum2"), grid, 256, 0, &a);
            try fine(dr, e, 17);
        } else {
            try lin(e, dr.act, l.down, dr.a, M, false);
            try fine(dr, e, 11);
        }
        try conv(k, dr.a, dr.dyn, 1, l.m_base + 2 * D * 2, dr.x, true, grid);
        try fine(dr, e, 3);
    }
    try kern.rmsnorm(k, dr.x, D, dr.norm, dr.hn, D, null, R, D);
    try fine(dr, e, 1);
    try phase(dr, e, tp, 1);
    // every row through the head (a block's first row is the anchor's: the host skips it) and the selector's projection
    // the drafts' head: the MTP drafts' packed copy (they only propose), each rank's 16 best a row
    try blocks.mm(e, &e.mbuf, dr.hn, D, if (dr.full_head) e.w.head else e.w.draft_head, e.mbuf.logits, false, R);
    try fine(dr, e, 12);
    {
        const chunks = cdiv(forward.HEAD_N, 4096);
        var a: cuda.Args = .{};
        a.add(e.mbuf.logits); a.add(@as(i32, @intCast(forward.HEAD_N))); a.add(@as(i32, @intCast(e.w.vocab_offset))); a.add(dr.pv); a.add(dr.pi);
        try k.go(try k.ext(.dflash, "df_top16_part"), .{ .x = @intCast(M), .y = @intCast(chunks) }, 256, 0, &a);
        var b: cuda.Args = .{};
        b.add(dr.pv); b.add(dr.pi); b.add(@as(i32, @intCast(chunks))); b.add(dr.tk);
        try k.go(try k.ext(.dflash, "df_top16_merge"), .{ .x = @intCast(M) }, 256, 0, &b);
    }
    try fine(dr, e, 13);
    try e.ctx.allGather(dr.tk, dr.gtk, nw, .i32, "dflash2 candidates");
    try fine(dr, e, 14);
    if (rank0) try lin(e, dr.hn, dr.hproj, dr.g, M, true);
    try fine(dr, e, 15);
}

fn phase(dr: *Drafter, e: *forward.Engine, t: *std.Io.Timestamp, which: usize) !void {
    if (!dr.prof) return;
    try e.k.stream.synchronize();
    const now = std.Io.Timestamp.now(e.io, .awake);
    dr.phase_ms[which] += @as(f64, @floatFromInt(t.durationTo(now).toNanoseconds())) / 1e6;
    t.* = now;
}

fn conv(k: *kern.K, z: u64, dyn: u64, which: i32, base: u64, out: u64, add: bool, grid: cuda.Dim3) !void {
    var a: cuda.Args = .{};
    a.add(z); a.add(dyn); a.add(which); a.add(base); a.add(out); a.add(@as(i32, @intFromBool(add)));
    try k.go(try k.ext(.dflash, "df_conv"), grid, 256, 0, &a);
}

const Cand = struct { v: f32, id: u32 };

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Stream ``i``'s 7 drafts: at each row, the 16 likeliest of both ranks' candidates, scored by the head's logit plus
/// the previous draft's predecessor row, the row's projection and the candidate's successor row; greedy, its
/// softmax chance kept.
fn select(dr: *Drafter, i: usize, M: usize, q: Req) void {
    const out = q.out.?;
    const probs = q.probs.?;
    const W = CW;
    const T = TOPK;
    var prev: usize = q.anchor;
    for (1..BLOCK) |t| {
        const r = i * BLOCK + t;
        var c: [TOPK]Cand = undefined;
        var n: usize = 0;
        for (0..2) |rk| {
            const blk = dr.cand[rk * M * W + r * W ..][0..W];
            for (0..T) |j| {
                if (blk[j] < 0) continue;
                const cand: Cand = .{ .v = @bitCast(blk[T + j]), .id = @intCast(blk[j]) };
                if (n < TOPK) {
                    c[n] = cand;
                    n += 1;
                    continue;
                }
                var lo: usize = 0;
                for (1..TOPK) |m| if (c[m].v < c[lo].v) {
                    lo = m;
                };
                if (cand.v > c[lo].v) c[lo] = cand;
            }
        }
        const g = dr.gh[r * RANK ..][0..RANK];
        const P = dr.pred[prev * RANK ..][0..RANK];
        var pg: [RANK]f32 = undefined;
        for (&pg, P, g) |*o, p, x| o.* = bf(p) * x;
        var best: usize = 0;
        var sc: [TOPK]f32 = undefined;
        for (0..n) |m| {
            const Sc = dr.succ[@as(usize, c[m].id) * RANK ..][0..RANK];
            var dot: f32 = 0;
            for (pg, Sc) |a, b| dot += a * bf(b);
            sc[m] = c[m].v + dot;
            if (sc[m] > sc[best]) best = m;
        }
        var z: f64 = 0;
        for (sc[0..n]) |x| z += @exp(@as(f64, x - sc[best]) / dr.temp);
        out[t - 1] = c[best].id;
        probs[t - 1] = 1.0 / z;
        prev = c[best].id;
    }
}
