//! DFlash2's checkpoint onto the device: the config checked, linears packed, scratch carved from one arena.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");
const forward = @import("forward.zig");
const weights_mod = @import("weights.zig");
const dflash = @import("dflash.zig");
const D = dflash.D;
const KVW = dflash.KVW;
const QKV = dflash.QKV;
const FF = dflash.FF;
const VOCAB = dflash.VOCAB;
const MAX_M = dflash.MAX_M;
const MAX_CTX = dflash.MAX_CTX;
const CW = dflash.CW;
const CHUNKS = dflash.CHUNKS;
const QH = dflash.QH;
const HD = dflash.HD;
const CH = dflash.CH;
const BLOCK = dflash.BLOCK;
const LAYERS = dflash.LAYERS;
const Drafter = dflash.Drafter;
const RANK = dflash.RANK;
const TOPK = dflash.TOPK;
const TAPS = dflash.TAPS;
const MASK = dflash.MASK;
const B16 = dflash.B16;

const Ent = struct { off: u64, n: u64 };

/// TF_GLM_DFLASH2_BITS: linears packed at load to 4 bits (default; they only propose), 6 or 8; 0 keeps bf16.
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

    /// The tensors ``names`` (``rows`` rows, ``kk`` columns each) stacked into one linear, packed for kern.mm.
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
                weights_mod.quantize(bits, @alignCast(std.mem.bytesAsSlice(u16, src)), r, kk, l.words.items[r0 * kk * bits / 32 ..][0 .. r * kk * bits / 32], l.scales.items[r0 * kk / 64 ..][0 .. r * kk / 64]);
                r0 += r;
            }
            try l.tmp.upload(0, std.mem.sliceAsBytes(l.words.items));
            try l.tmp.upload(std.mem.alignForward(u64, wb, 256), std.mem.sliceAsBytes(l.scales.items));
        }
        try kern.mmTile(l.k, bits, l.tmp.ptr, l.tmp.ptr + std.mem.alignForward(u64, wb, 256), dst, ds, @intCast(n), @intCast(kk));
        try l.k.stream.synchronize();
        return .{ .w = dst, .n = @intCast(n), .k = @intCast(kk), .s = ds, .bits = bits };
    }

    /// A rank's share of a split linear: rows row0.. and columns col0.. of each of ``names``, stacked and packed.
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
    // the device's share: linears packed (q|k|v, gate|up, k|v for context), norms and convs; codebooks stay host
    var total: u64 = 0;
    const split = e.ctx.world == 2 and splitWanted();
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
    dr.* = .{ .gpa = gpa, .d = d, .weights = wbuf, .bits = bits, .split = split, .ff = ff, .pred = &.{}, .succ = &.{}, .scratch = undefined, .h_ids = undefined, .h_bpos = undefined, .h_rings = undefined, .h_starts = undefined, .h_froms = undefined, .h_cpos = undefined, .h_crings = undefined, .cand = undefined, .gh = undefined };
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
    dr.full_head = fullHeadWanted();
    if (std.c.getenv("TF_GLM_DFLASH2_TEMP")) |v| dr.temp = std.fmt.parseFloat(f64, std.mem.span(v)) catch 1.0;
    const inv: [MAX_M]f32 = @splat(1.0);
    try dr.scratch.upload(dr.invt - dr.scratch.ptr, std.mem.sliceAsBytes(&inv));
    std.log.info("dflash2: drafter loaded from {s}, linears at {d} bits ({d:.2} GiB on the device, {d} MiB of codebooks on the host)", .{ dir, bits, @as(f64, @floatFromInt(l.at)) / (1 << 30), (2 * VOCAB * RANK * 2) >> 20 });
    return dr;
}

fn carveScratch(dr: *Drafter, d: *const cuda.Driver) !void {
    var at: u64 = 0;
    const M: u64 = MAX_M;
    const C: u64 = MAX_CTX + 16;
    const fields = .{
        .{ "x", M * D * 2 },                   .{ "y", M * D * 2 },         .{ "y2", M * D * 2 },                                                .{ "dyn", M * KVW * 2 },
        .{ "qkv", M * QKV * 2 },               .{ "q", M * D * 2 },         .{ "kb", M * KVW * 2 },                                              .{ "vb", M * KVW * 2 },
        .{ "att", M * D * 2 },                 .{ "a", M * D * 2 },         .{ "gu", M * 2 * FF * 2 },                                           .{ "act", M * FF * 2 },
        .{ "hn", M * D * 2 },                  .{ "g", M * RANK * 4 },      .{ "cin", C * 5 * D * 2 },                                           .{ "c1", C * D * 2 },
        .{ "ctx", C * D * 2 },                 .{ "ckv", C * 2 * KVW * 2 }, .{ "taps_step", @as(u64, @intCast(forward.STEP_ROWS)) * 5 * D * 2 }, .{ "taps_prefill", @as(u64, @intCast(forward.PREFILL_ROWS_MAX)) * 5 * D * 2 },
        .{ "ids", M * 4 },                     .{ "bpos", M * 4 },          .{ "rings", forward.MAX_STREAMS * 8 },                               .{ "starts", forward.MAX_STREAMS * 8 },
        .{ "froms", forward.MAX_STREAMS * 8 }, .{ "cpos", C * 8 },          .{ "crings", C * 8 },                                                .{ "invt", M * 4 },
        .{ "tk", M * CW * 4 },                 .{ "gtk", 2 * M * CW * 4 },  .{ "pv", M * CHUNKS * TOPK * 4 },                                    .{ "pi", M * CHUNKS * TOPK * 4 },
        .{ "pm", M * QH * CH * 4 },            .{ "pl", M * QH * CH * 4 },  .{ "pacc", M * QH * CH * HD * 4 },
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

/// TF_GLM_DFLASH2_SPLIT=0: each rank holds the drafter's whole MLP (else half each, on two ranks).
pub fn splitWanted() bool {
    const v = std.c.getenv("TF_GLM_DFLASH2_SPLIT") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_GLM_DFLASH2_HEAD=full: the drafter's tokens from the target's full head.
pub fn fullHeadWanted() bool {
    const v = std.c.getenv("TF_GLM_DFLASH2_HEAD") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "full");
}
