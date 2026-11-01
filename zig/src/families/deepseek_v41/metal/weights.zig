//! A backbone layer's tensors and the model's embed, norm and head from the release's pack into Metal buffers.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const glue = @import("glue.zig");
const experts = @import("experts.zig");
const mapped = @import("mapped.zig");

const Config = host.config.Config;
const Pack = host.pack.Pack;
const Info = host.pack.Info;
const names = host.names;
const projection = host.projection;

pub const Ref = glue.Ref;
pub const Map = mapped.Map;

/// Kernel load widths: dense FP8 weights are read as uint4, every other raw tensor as uint2 at most.
const align_fp8: u32 = 16;
const align_raw: u32 = 8;
/// Distinct buffers one layer's addresses and Refs may span (its arena and the mapped shards it touches).
pub const max_backing = 8;

/// A dense FP8 projection: E4M3 codes [n][k] and E8M0 scales [n/32][k/32].
pub const Fp8 = struct { w: Ref, s: Ref, k: u32, n: u32 };

/// An Engram layer's GPU half: wkv [(hc + 1) * hidden][cols * 256] FP8, q_weight and k_weight [hc][hidden] bf16.
pub const Engram = struct { wkv: Fp8, q: Ref, k: Ref };

/// How a tensor lands on the device: bytes as stored, widened to fp32, or rounded to bf16 once.
pub const Into = enum { raw, f32, bf16 };
const Item = struct { info: Info, into: Into, at: usize };

/// Where a tensor lands: in a mapped shard's own buffer (zero-copy), or at `off` in the layer's arena (buf null).
pub const Place = struct {
    buf: ?mtl.Buffer = null,
    off: usize,

    pub fn ref(p: Place, arena: mtl.Buffer) Ref {
        return .{ .buf = p.buf orelse arena, .off = p.off };
    }

    fn addr(p: Place, arena: mtl.Buffer) u64 {
        return (p.buf orelse arena).gpuAddress() + p.off;
    }
};
pub const Fp8Place = struct { w: Place, s: Place, k: u32, n: u32 };

pub fn bf16Round(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// The forward's names for a layer's tensors, in the order weights.Layer holds them.
pub const Layer = struct {
    arena: mtl.Buffer,
    table: mtl.Buffer,
    layer: u32,
    attn_norm: Ref,
    ffn_norm: Ref,
    q_norm: Ref,
    kv_norm: Ref,
    hc_attn: [3]Ref,
    hc_ffn: [3]Ref,
    sink: Ref,
    wq_a: Fp8,
    wkv: Fp8,
    wq_b: Fp8,
    wo_a: Fp8,
    wo_b: Fp8,
    /// the compressor's wkv and wgate and the indexer's wk: bf16 [n][k] in the release
    comp_wkv: ?Ref = null,
    comp_wgate: ?Ref = null,
    comp_norm: ?Ref = null,
    ix_wk: ?Ref = null,
    ix_k_norm: ?Ref = null,
    ix_wq_b: ?Fp8 = null,
    ix_wp: ?Ref = null,
    gate_w: Ref,
    gate_b: Ref,
    /// the image rows' routing bias (`ffn.gate.bias_vl`), set when the vision tower is loaded; null: text only
    gate_bvl: ?Ref = null,
    shared: [3]Fp8,
    engram: ?Engram = null,

    /// the arena, then each mapped shard buffer a tensor of this layer lives in
    back: [max_backing]mtl.Buffer = undefined,
    nback: u8 = 0,

    /// The arena and table are the layer's own; mapped shards belong to their `Map`.
    pub fn deinit(l: *Layer) void {
        l.arena.deinit();
        l.table.deinit();
    }

    /// The buffers a launch reads through addresses (the expert table points into them).
    pub fn backing(l: *const Layer) []const mtl.Buffer {
        return l.back[0..l.nback];
    }

    /// The bytes behind a GPU address of one of this layer's backing buffers.
    pub fn bytesAt(l: *const Layer, addr: u64) [*]const u8 {
        for (l.backing()) |b| {
            const base = b.gpuAddress();
            if (addr >= base and addr - base < b.length()) return b.contents() + (addr - base);
        }
        unreachable;
    }
};

/// Tensors queued by name, then read into one arena in a single pass (`fill`).
pub const Loader = struct {
    a: std.mem.Allocator,
    pack: *const Pack,
    map: ?*Map = null,
    items: std.ArrayList(Item) = .empty,
    size: usize = 0,
    buf: [256]u8 = undefined,
    back: [max_backing]mtl.Buffer = undefined,
    nback: u8 = 0,

    /// A mapped shard buffer this layer reads, once.
    pub fn note(l: *Loader, b: mtl.Buffer) !void {
        for (l.back[0..l.nback]) |x| if (x.id == b.id) return;
        if (l.nback == max_backing) return error.TooManyBackingBuffers;
        l.back[l.nback] = b;
        l.nback += 1;
    }

    /// Bytes as stored go to a mapped shard when the map is on and `al` divides the tensor's file offset.
    pub fn add(l: *Loader, info: Info, into: Into, al: u32) !Place {
        if (into == .raw) if (l.map) |m| if (try m.place(info, al)) |p| {
            try l.note(p.buf);
            return .{ .buf = p.buf, .off = p.off };
        };
        const n: usize = @intCast(info.nbytes);
        const bytes = switch (into) {
            .raw => n,
            .f32 => n / info.dtype.size() * 4,
            .bf16 => n / info.dtype.size() * 2,
        };
        const off = std.mem.alignForward(usize, l.size, 256);
        l.size = off + bytes;
        try l.items.append(l.a, .{ .info = info, .into = into, .at = off });
        return .{ .off = off };
    }

    pub fn native(l: *Loader, block: []const u8, spec: names.NativeSpec) !Place {
        const name = try std.fmt.bufPrint(&l.buf, "{s}.{s}", .{ block, spec.suffix });
        const info = try l.pack.need(name);
        if (!spec.store.accepts(info.dtype) or info.rank != spec.rank) return error.BadNativeTensor;
        for (spec.dims(), info.dims()) |want, got| if (want != got) return error.BadNativeTensor;
        const into: Into = switch (spec.store) {
            .norm, .f32, .bias => .f32,
            .half => .bf16,
            .param => if (info.dtype == .bf16) .raw else return error.UnsupportedNative,
        };
        return l.add(info, into, align_raw);
    }

    pub fn fp8(l: *Loader, block: []const u8, suffix: []const u8, k: u32, n: u32) !Fp8Place {
        const prefix = try std.fmt.bufPrint(&l.buf, "{s}.{s}", .{ block, suffix });
        const p = try projection.read(l.pack, prefix);
        if (p != .fp8_block) return error.UnsupportedProjection;
        const f = p.fp8_block;
        if (f.k != k or f.n != n) return error.BadProjectionShape;
        return .{ .w = try l.add(f.weight, .raw, align_fp8), .s = try l.add(f.scale, .raw, align_raw), .k = k, .n = n };
    }

    /// A bf16 projection kept as stored: `{block}.{suffix}.weight` [n][k].
    pub fn half(l: *Loader, block: []const u8, suffix: []const u8, k: u32, n: u32) !Place {
        const name = try std.fmt.bufPrint(&l.buf, "{s}.{s}.weight", .{ block, suffix });
        const info = try l.pack.need(name);
        if (!info.is(.bf16, &.{ n, k })) return error.BadProjectionShape;
        return l.add(info, .raw, align_fp8);
    }

    pub fn mxfp4(l: *Loader, block: []const u8, e: u32, p: names.ExpertProj, c: *const Config) ![2]Place {
        const prefix = try std.fmt.bufPrint(&l.buf, "{s}.ffn.experts.{d}.{t}", .{ block, e, p });
        const pr = try projection.read(l.pack, prefix);
        if (pr != .mxfp4) return error.UnsupportedProjection;
        const kn = p.kn(c);
        if (pr.mxfp4.k != kn[0] or pr.mxfp4.n != kn[1]) return error.BadProjectionShape;
        return .{ try l.add(pr.mxfp4.weight, .raw, align_raw), try l.add(pr.mxfp4.scale, .raw, align_raw) };
    }

    pub fn fill(l: *Loader, io: std.Io, dst: []u8) !void {
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(l.a);
        for (l.items.items) |it| {
            const n: usize = @intCast(it.info.nbytes);
            if (it.into == .raw) {
                try l.pack.read(io, it.info, .{ .offset = 0, .len = it.info.nbytes }, dst[it.at..][0..n]);
                continue;
            }
            try tmp.resize(l.a, n);
            try l.pack.read(io, it.info, .{ .offset = 0, .len = it.info.nbytes }, tmp.items);
            const count = n / it.info.dtype.size();
            for (0..count) |i| {
                const v: f32 = switch (it.info.dtype) {
                    .f32 => @bitCast(std.mem.readInt(u32, tmp.items[4 * i ..][0..4], .little)),
                    .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, tmp.items[2 * i ..][0..2], .little)) << 16),
                    .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, tmp.items[2 * i ..][0..2], .little)))),
                    else => return error.BadNativeTensor,
                };
                if (it.into == .f32) std.mem.writeInt(u32, dst[it.at + 4 * i ..][0..4], @bitCast(v), .little) else std.mem.writeInt(u16, dst[it.at + 2 * i ..][0..2], bf16Round(v), .little);
            }
        }
    }
};

pub fn fix(arena: mtl.Buffer, f: Fp8Place) Fp8 {
    return .{ .w = f.w.ref(arena), .s = f.s.ref(arena), .k = f.k, .n = f.n };
}

fn fixOpt(arena: mtl.Buffer, f: ?Fp8Place) ?Fp8 {
    return if (f) |v| fix(arena, v) else null;
}

fn refOpt(arena: mtl.Buffer, p: ?Place) ?Ref {
    return if (p) |v| v.ref(arena) else null;
}

/// Layer `layer`'s tensors from `pack` (release format; a layer past the backbone is a DSpark block, its heads apart).
pub fn load(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, c: *const Config, layer: u32) !Layer {
    return loadFrom(a, io, device, pack, c, layer, null);
}

/// As `load`, with raw tensors wrapped in `map`'s shard buffers where their file offsets allow, the rest copied.
pub fn loadFrom(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, c: *const Config, layer: u32, map: ?*Map) !Layer {
    if (c.format != .release_fp8) return error.UnsupportedFormat;
    const ex = c.expertsOf(layer);
    var l: Loader = .{ .a = a, .pack = pack, .map = map };
    defer l.items.deinit(a);
    var block_buf: [32]u8 = undefined;
    const block = try names.blockPrefix(c, layer, &block_buf);
    const nat = try names.blockNatives(c, layer);
    var off: [24]Place = undefined;
    for (nat.items(), 0..) |spec, i| off[i] = try l.native(block, spec);
    const per_group_k = c.heads / c.o_groups * c.head_dim;
    var w: Layer = undefined;
    w.layer = layer;
    w.gate_bvl = null;
    const wq_a = try l.fp8(block, "attn.wq_a", c.hidden, c.q_lora);
    const wkv = try l.fp8(block, "attn.wkv", c.hidden, c.head_dim);
    const wq_b = try l.fp8(block, "attn.wq_b", c.q_lora, c.heads * c.head_dim);
    const wo_a = try l.fp8(block, "attn.wo_a", per_group_k, c.o_groups * c.o_lora);
    const wo_b = try l.fp8(block, "attn.wo_b", c.o_groups * c.o_lora, c.hidden);
    var comp_wkv: ?Place = null;
    var comp_wgate: ?Place = null;
    var ix_wk: ?Place = null;
    var ix_wq_b: ?Fp8Place = null;
    if (c.isKvSource(layer)) {
        comp_wkv = try l.half(block, "attn.compressor.wkv", c.hidden, c.head_dim);
        if (c.compressRatio(layer) > 1) comp_wgate = try l.half(block, "attn.compressor.wgate", c.hidden, c.head_dim);
        ix_wk = try l.half(block, "attn.indexer.wk", c.head_dim, c.index_dim);
    }
    if (c.isIndexSource(layer)) ix_wq_b = try l.fp8(block, "attn.indexer.wq_b", c.q_lora, c.index_heads * c.index_dim);
    const shared = [3]Fp8Place{
        try l.fp8(block, "ffn.shared_experts.w1", c.hidden, c.expert_width),
        try l.fp8(block, "ffn.shared_experts.w2", c.expert_width, c.hidden),
        try l.fp8(block, "ffn.shared_experts.w3", c.hidden, c.expert_width),
    };
    var eng_wkv: ?Fp8Place = null;
    if (c.isEngram(layer)) eng_wkv = try l.fp8(block, "engram.wkv", c.hashCols() * c.engram_head_dim, (c.hc_mult + 1) * c.hidden);
    const ptrs = try a.alloc([6]Place, ex.count);
    defer a.free(ptrs);
    for (ptrs, 0..) |*p, e| {
        const w1 = try l.mxfp4(block, @intCast(e), .w1, c);
        const w3 = try l.mxfp4(block, @intCast(e), .w3, c);
        const w2 = try l.mxfp4(block, @intCast(e), .w2, c);
        p.* = .{ w1[0], w1[1], w3[0], w3[1], w2[0], w2[1] };
    }
    const arena = try device.buffer(@max(l.size, 256), mtl.ResourceOptions.shared);
    errdefer arena.deinit();
    try l.fill(io, arena.contents()[0..l.size]);
    const table = try device.buffer(ex.count * @sizeOf(experts.ExpertPtrs), mtl.ResourceOptions.shared);
    errdefer table.deinit();
    for (table.slice(experts.ExpertPtrs, ex.count), ptrs) |*t, p| {
        t.* = .{ .w1p = p[0].addr(arena), .w1s = p[1].addr(arena), .w3p = p[2].addr(arena), .w3s = p[3].addr(arena), .w2p = p[4].addr(arena), .w2s = p[5].addr(arena) };
    }
    w.arena = arena;
    w.table = table;
    w.back[0] = arena;
    @memcpy(w.back[1 .. 1 + l.nback], l.back[0..l.nback]);
    w.nback = l.nback + 1;
    w.wq_a = fix(arena, wq_a);
    w.wkv = fix(arena, wkv);
    w.wq_b = fix(arena, wq_b);
    w.wo_a = fix(arena, wo_a);
    w.wo_b = fix(arena, wo_b);
    w.ix_wq_b = fixOpt(arena, ix_wq_b);
    w.comp_wkv = refOpt(arena, comp_wkv);
    w.comp_wgate = refOpt(arena, comp_wgate);
    w.ix_wk = refOpt(arena, ix_wk);
    for (&w.shared, shared) |*d, v| d.* = fix(arena, v);
    w.comp_norm = null;
    w.ix_k_norm = null;
    w.ix_wp = null;
    w.engram = if (eng_wkv) |v| .{ .wkv = fix(arena, v), .q = undefined, .k = undefined } else null;
    for (nat.items(), 0..) |spec, i| {
        const r = off[i].ref(arena);
        const s = spec.suffix;
        if (std.mem.eql(u8, s, "attn_norm.weight")) w.attn_norm = r;
        if (std.mem.eql(u8, s, "ffn_norm.weight")) w.ffn_norm = r;
        if (std.mem.eql(u8, s, "hc_attn_fn")) w.hc_attn[0] = r;
        if (std.mem.eql(u8, s, "hc_attn_scale")) w.hc_attn[1] = r;
        if (std.mem.eql(u8, s, "hc_attn_base")) w.hc_attn[2] = r;
        if (std.mem.eql(u8, s, "hc_ffn_fn")) w.hc_ffn[0] = r;
        if (std.mem.eql(u8, s, "hc_ffn_scale")) w.hc_ffn[1] = r;
        if (std.mem.eql(u8, s, "hc_ffn_base")) w.hc_ffn[2] = r;
        if (std.mem.eql(u8, s, "attn.q_norm.weight")) w.q_norm = r;
        if (std.mem.eql(u8, s, "attn.kv_norm.weight")) w.kv_norm = r;
        if (std.mem.eql(u8, s, "attn.attn_sink")) w.sink = r;
        if (std.mem.eql(u8, s, "attn.compressor.norm.weight")) w.comp_norm = r;
        if (std.mem.eql(u8, s, "attn.indexer.k_norm.weight")) w.ix_k_norm = r;
        if (std.mem.eql(u8, s, "attn.indexer.weights_proj.weight")) w.ix_wp = r;
        if (std.mem.eql(u8, s, "ffn.gate.weight")) w.gate_w = r;
        if (std.mem.eql(u8, s, "ffn.gate.bias")) w.gate_b = r;
        if (std.mem.eql(u8, s, "engram.q_weight")) w.engram.?.q = r;
        if (std.mem.eql(u8, s, "engram.k_weight")) w.engram.?.k = r;
    }
    return w;
}

/// The model's tensors outside the blocks: embed and head [vocab][hidden] bf16, the final norm in fp32.
pub const Top = struct {
    arena: mtl.Buffer,
    embed: Ref,
    norm: Ref,
    head: Ref,

    pub fn deinit(t: *Top) void {
        t.arena.deinit();
    }
};

/// The release keeps embed and head in bf16 and the norm in bf16 (widened to fp32 here).
pub fn loadTop(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, c: *const Config) !Top {
    return loadTopFrom(a, io, device, pack, c, null);
}

/// As `loadTop`, embed and head in `map`'s shard buffers when their file offsets allow.
pub fn loadTopFrom(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, c: *const Config, map: ?*Map) !Top {
    var l: Loader = .{ .a = a, .pack = pack, .map = map };
    defer l.items.deinit(a);
    var off: [3]Place = undefined;
    for ([_][]const u8{ "embed.weight", "head.weight" }, 0..) |name, i| {
        const info = try pack.need(name);
        if (!info.is(.bf16, &.{ c.vocab, c.hidden })) return error.BadNativeTensor;
        off[i] = try l.add(info, .raw, align_fp8);
    }
    const norm = try pack.need("norm.weight");
    if (!(norm.is(.bf16, &.{c.hidden}) or norm.is(.f32, &.{c.hidden}))) return error.BadNativeTensor;
    off[2] = try l.add(norm, .f32, align_raw);
    const arena = try device.buffer(@max(l.size, 256), mtl.ResourceOptions.shared);
    errdefer arena.deinit();
    try l.fill(io, arena.contents()[0..l.size]);
    return .{ .arena = arena, .embed = off[0].ref(arena), .head = off[1].ref(arena), .norm = off[2].ref(arena) };
}
