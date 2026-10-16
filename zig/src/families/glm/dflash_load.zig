//! DFlash2's checkpoint onto the device: the config checked, linears tiled for glm_mm, scratch carved from one buffer.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_mm.zig");
const taps_mod = @import("cuda_taps.zig");
const df = @import("dflash.zig");
const D = df.D;
const KVW = df.KVW;
const FF = df.FF;
const Drafter = df.Drafter;

const Ent = struct { off: u64, n: u64 };

const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    ents: *const std.StringHashMap(Ent),
    buf: std.ArrayList(u8) = .empty,
    sub: std.ArrayList(u16) = .empty, // a rank's share of a split linear
    w: *cuda.DeviceBuffer,
    tmp: *cuda.DeviceBuffer,
    mk: *const mm.Kernels,
    stream: cuda.Stream,
    at: u64 = 0,

    fn take(l: *Loader, n: u64) u64 {
        const p = l.at;
        l.at += std.mem.alignForward(u64, n, 256);
        return l.w.ptr + p;
    }

    fn read(l: *Loader, name: []const u8, want: u64) ![]u8 {
        const e = l.ents.get(name) orelse {
            std.log.err("dflash2: no {s} in the drafter's checkpoint", .{name});
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

    /// Rows row0.. (rn of each tensor's `rows`) and columns col0.. (cn of `kk`) of `names`, stacked and tiled.
    fn linear(l: *Loader, names: []const []const u8, rows: []const usize, kk: usize, part: ?[4]usize) !mm.Tiled {
        const p: [4]usize = part orelse .{ 0, 0, 0, kk };
        const cn = p[3];
        var n: usize = 0;
        var r0: usize = 0;
        for (names, rows) |name, r| {
            const rn = if (part != null and p[1] > 0) p[1] else r;
            const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, try l.read(name, r * kk * 2)));
            if (part == null) {
                try l.tmp.upload(r0 * kk * 2, std.mem.sliceAsBytes(src));
            } else {
                if (p[0] + rn > r or p[2] + cn > kk) return error.DrafterTensorShape;
                try l.sub.resize(l.gpa, rn * cn);
                for (0..rn) |i| @memcpy(l.sub.items[i * cn ..][0..cn], src[(p[0] + i) * kk + p[2] ..][0..cn]);
                try l.tmp.upload(r0 * cn * 2, std.mem.sliceAsBytes(l.sub.items));
            }
            r0 += rn;
            n += rn;
        }
        const dst = l.take(n * cn * 2);
        try mm.tile(l.mk, l.stream, l.tmp.ptr, dst, n, cn);
        try l.stream.synchronize(); // tmp is reused by the next linear
        return .{ .w = dst, .n = n, .k = cn };
    }
};

fn layerName(b: *[96]u8, li: usize, s: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ li, s });
}

/// The drafter from `dir` (config.json and model.safetensors, bf16) for rank `rank` of `world`, on engine `e`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, rank: usize, world: usize) !*Drafter {
    try checkConfig(gpa, io, dir);
    const d = e.d;
    const head = (e.w.mtp orelse return error.DrafterNeedsDraftHead).draft;
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(path);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var lenb: [8]u8 = undefined;
    _ = try file.readPositionalAll(io, &lenb, 0);
    const hlen = std.mem.readInt(u64, &lenb, .little);
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
    const split = world == 2;
    const ff: usize = if (split) FF / 2 else FF;
    const small: u64 = 6 * 4096 * 2 + 2 * 128 * 2 + 2 * 2 * 2 * D * 2 + 16 * 256;
    const lin_elems: u64 = df.QKV * D + 2 * KVW * D + D * D + 2 * ff * D + D * ff + 2 * KVW * D;
    const total: u64 = df.LAYERS * (lin_elems * 2 + small) + (D * 5 * D + df.RANK * D) * 2 + 2 * D * 2 + 64 * 256;
    var wbuf = try cuda.DeviceBuffer.alloc(d, total);
    errdefer wbuf.free();
    var tmp = try cuda.DeviceBuffer.alloc(d, 2 * FF * D * 2); // the largest linear as stored
    defer tmp.free();
    var module = try cuda.Module.load(d, cuda.kernels.glm_dflash);
    errdefer module.unload();
    var l: Loader = .{ .gpa = gpa, .io = io, .file = file, .ents = &ents, .w = &wbuf, .tmp = &tmp, .mk = &e.m.mm, .stream = e.stream };
    defer {
        l.buf.deinit(gpa);
        l.sub.deinit(gpa);
    }
    const dr = try gpa.create(Drafter);
    errdefer gpa.destroy(dr);
    dr.* = .{ .gpa = gpa, .d = d, .module = module, .k = try df.Kernels.load(module), .weights = wbuf, .split = split, .ff = ff, .rank = rank, .world = world };
    var nb: [4][96]u8 = undefined;
    for (&dr.layers, 0..) |*lw, i| {
        lw.in_norm = try l.raw(try layerName(&nb[0], i, "input_layernorm.weight"), D * 2);
        lw.post_norm = try l.raw(try layerName(&nb[0], i, "post_attention_layernorm.weight"), D * 2);
        lw.qn = try l.raw(try layerName(&nb[0], i, "self_attn.q_norm.weight"), 128 * 2);
        lw.kn = try l.raw(try layerName(&nb[0], i, "self_attn.k_norm.weight"), 128 * 2);
        lw.a_base = try l.raw(try layerName(&nb[0], i, "attention_conv.base_kernel"), 2 * 2 * D * 2);
        lw.m_base = try l.raw(try layerName(&nb[0], i, "mlp_conv.base_kernel"), 2 * 2 * D * 2);
        const q = try layerName(&nb[1], i, "self_attn.q_proj.weight");
        const kq = try layerName(&nb[2], i, "self_attn.k_proj.weight");
        const v = try layerName(&nb[3], i, "self_attn.v_proj.weight");
        lw.qkv = try l.linear(&.{ q, kq, v }, &.{ D, KVW, KVW }, D, null);
        lw.kv = try l.linear(&.{ kq, v }, &.{ KVW, KVW }, D, null);
        lw.o = try l.linear(&.{try layerName(&nb[1], i, "self_attn.o_proj.weight")}, &.{D}, D, null);
        const gate = try layerName(&nb[1], i, "mlp.gate_proj.weight");
        const up = try layerName(&nb[2], i, "mlp.up_proj.weight");
        const down = try layerName(&nb[3], i, "mlp.down_proj.weight");
        if (split) { // this rank's half of the MLP width: gate and up rows, down columns
            lw.gu = try l.linear(&.{ gate, up }, &.{ FF, FF }, D, .{ rank * ff, ff, 0, D });
            lw.down = try l.linear(&.{down}, &.{D}, FF, .{ 0, 0, rank * ff, ff });
        } else {
            lw.gu = try l.linear(&.{ gate, up }, &.{ FF, FF }, D, null);
            lw.down = try l.linear(&.{down}, &.{D}, FF, null);
        }
        lw.a_proj = try l.linear(&.{try layerName(&nb[1], i, "attention_conv.kernel_projection.weight")}, &.{KVW}, D, null);
        lw.m_proj = try l.linear(&.{try layerName(&nb[1], i, "mlp_conv.kernel_projection.weight")}, &.{KVW}, D, null);
    }
    dr.fc = try l.linear(&.{"fc.weight"}, &.{D}, 5 * D, null);
    dr.hproj = try l.linear(&.{"candidate_selector.hidden_projection.weight"}, &.{df.RANK}, D, null);
    dr.hnorm = try l.raw("hidden_norm.weight", D * 2);
    dr.norm = try l.raw("norm.weight", D * 2);
    if (l.at > total) return error.DrafterLayout;
    dr.pred = try gpa.alloc(u16, df.VOCAB * df.RANK);
    @memcpy(std.mem.sliceAsBytes(dr.pred), try l.read("candidate_selector.predecessor_codebook", df.VOCAB * df.RANK * 2));
    dr.succ = try gpa.alloc(u16, df.VOCAB * df.RANK);
    @memcpy(std.mem.sliceAsBytes(dr.succ), try l.read("candidate_selector.successor_codebook", df.VOCAB * df.RANK * 2));
    try carve(dr, d, head.n);
    dr.h_cpos = try gpa.alloc(i64, df.MAX_CTX);
    dr.h_crings = try gpa.alloc(u64, df.MAX_CTX);
    dr.cand = try gpa.alloc(i32, world * df.MAX_M * df.CW);
    dr.gh = try gpa.alloc(f32, df.MAX_M * df.RANK);
    std.log.info("dflash2: drafter from {s} (CC BY-NC-ND 4.0: non-commercial use only), {d:.2} GiB of linears on the device", .{ dir, @as(f64, @floatFromInt(l.at)) / (1 << 30) });
    return dr;
}

/// The drafter's scratch, every field one carved piece of one buffer.
fn carve(dr: *Drafter, d: *const cuda.Driver, head_n: usize) !void {
    const M: u64 = df.MAX_M;
    const C: u64 = df.MAX_CTX;
    const chunks: u64 = (head_n + 4095) / 4096;
    var part: u64 = 16;
    for (dr.layers) |l| for ([_]mm.Tiled{ l.qkv, l.kv, l.o, l.gu, l.down, l.a_proj, l.m_proj }) |w| {
        part = @max(part, mm.partBytes(64, w));
    };
    part = @max(part, @max(mm.partBytes(64, dr.fc), mm.partBytes(64, dr.hproj)));
    const fields = .{
        .{ "x", M * D * 2 },                        .{ "y", M * D * 2 },                 .{ "y2", M * D * 2 },                        .{ "dyn", M * KVW * 2 },
        .{ "qkv", M * df.QKV * 2 },                 .{ "q", M * D * 2 },                 .{ "kb", M * KVW * 2 },                      .{ "vb", M * KVW * 2 },
        .{ "att", M * D * 2 },                      .{ "a", M * D * 2 },                 .{ "gu", M * 2 * FF * 2 },                   .{ "act", M * FF * 2 },
        .{ "hn", M * D * 2 },                       .{ "xs", M * (D / 64) * 4 },         .{ "g", M * df.RANK * 4 },                   .{ "logits", M * head_n * 2 },
        .{ "part", part },                          .{ "sums", 2 * M * D * 4 },          .{ "cin", C * df.TAP_ROW },                  .{ "c1", C * D * 2 },
        .{ "ctx", C * D * 2 },                      .{ "ckv", C * 2 * KVW * 2 },         .{ "taps_step", df.STEP_ROWS * df.TAP_ROW }, .{ "taps_prefill", C * df.TAP_ROW },
        .{ "ids", M * 4 },                          .{ "bpos", M * 4 },                  .{ "rings", df.MAX_S * 8 },                  .{ "starts", df.MAX_S * 8 },
        .{ "froms", df.MAX_S * 8 },                 .{ "cpos", C * 8 },                  .{ "crings", C * 8 },                        .{ "gtk", 2 * M * df.CW * 4 },
        .{ "pv", M * chunks * df.TOPK * 4 },        .{ "pi", M * chunks * df.TOPK * 4 }, .{ "pm", M * df.QH * df.CH * 4 },            .{ "pl", M * df.QH * df.CH * 4 },
        .{ "pacc", M * df.QH * df.CH * df.HD * 4 },
    };
    var at: u64 = 0;
    inline for (fields) |f| {
        @field(dr, f[0]) = at;
        at += std.mem.alignForward(u64, f[1], 256);
    }
    dr.scratch = try cuda.DeviceBuffer.alloc(d, at);
    inline for (fields) |f| @field(dr, f[0]) += dr.scratch.ptr;
    try d.check(d.api.cuMemsetD8_v2(dr.scratch.ptr, 0, at), "cuMemsetD8");
}

/// Device bytes the drafter's scratch takes beside its weights (the memory budget's share, before loading).
pub fn checkConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer p.deinit();
    const o = p.value.object;
    const dc = (o.get("dflash_config") orelse return error.NotDFlash2).object;
    const ok = o.get("hidden_size").?.integer == D and o.get("num_hidden_layers").?.integer == df.LAYERS and
        o.get("num_attention_heads").?.integer == 32 and o.get("num_key_value_heads").?.integer == 8 and
        o.get("intermediate_size").?.integer == FF and o.get("vocab_size").?.integer == df.VOCAB and
        dc.get("block_size").?.integer == df.BLOCK and dc.get("mask_token_id").?.integer == df.MASK and
        dc.get("selector_top_k").?.integer == df.TOPK and dc.get("selector_rank").?.integer == df.RANK and
        dc.get("conv_kernel_size").?.integer == 2 and dc.get("conv_group_size").?.integer == 16;
    if (!ok) return error.DrafterConfigUnexpected;
    const taps = dc.get("target_layer_ids").?.array.items;
    if (taps.len != taps_mod.layers.len) return error.DrafterConfigUnexpected;
    for (taps, taps_mod.layers) |t, w| if (t.integer != w) return error.DrafterConfigUnexpected;
}
