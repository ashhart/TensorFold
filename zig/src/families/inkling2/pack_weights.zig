//! One TP rank's Inkling 2 weights from an EXL3 pack: EXL3 projections and experts, the rest as stored.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const cfg = @import("shape.zig");
const shard = @import("shard.zig");
const exl3 = @import("exl3.zig");
const experts = @import("experts.zig");

const Tensor = core.checkpoint.Tensor;
pub const Linear = cuda.exl3.Weight;

/// The text decoder's tensors sit under this prefix; the vision tower and the drafter ship beside the pack.
pub const prefix = "model.llm.";

/// q, r, gate by the rank's query heads, k, v by its key heads, o by its rows; the raw tensors cut the same way.
pub const Attention = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    gate: Linear,
    o: Linear,
    sink: u64, // bf16 [heads][hidden]
    q_scale: u64, // bf16 [heads][128]
    rel_scale: u64, // bf16 [heads][16]
    rel_proj: u64, // bf16 [16][1024], whole
    k_conv: u64, // sliding: bf16 [kv * 128][4]; global: static taps bf16 [4][kv * 128]
    v_conv: u64,
    k_proj: u64 = 0, // global: the dynamic taps' projection, bf16 [kv * 4][hidden]
    v_proj: u64 = 0,
};

pub const Layer = struct {
    kind: cfg.Kind,
    norms: [3]u64, // attn_norm (sliding) or q/k/v input norms (global), bf16 [hidden]
    attn_query: u64, // fp32 [nq][hidden]; 0 on layer 0 (no mix)
    nq: u32,
    mlp_query: u64, // fp32 [hidden]
    mlp_norm: u64,
    attn_conv: u64, // bf16 [hidden][4]
    mlp_conv: u64,
    attn: Attention,
    router: u64, // bf16 [experts][hidden]
    bias: u64, // fp32 [experts]
    experts: cuda.exl3_experts.Layer,
    bits: experts.Bits, // the experts' least and most half-bit widths
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    config: cfg.Config,
    share: shard.Share,
    padded_vocab: usize = 0,
    embed: u64 = 0, // bf16 [vocab][hidden], whole on every rank
    embed_norm: u64 = 0,
    norm: u64 = 0,
    out_query: u64 = 0, // fp32 [hidden]
    head: Linear = undefined, // the rank's vocabulary rows of the 8-bit head
    layers: []Layer = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    fn put(ctx: *anyopaque, bytes: []const u8) anyerror!u64 {
        const w: *Weights = @ptrCast(@alignCast(ctx));
        const b = try cuda.DeviceBuffer.fromHost(w.d, bytes);
        errdefer {
            var owned = b;
            owned.free();
        }
        try w.buffers.append(w.gpa, b);
        w.bytes += b.len;
        return b.ptr;
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    w: *Weights,
    ck: *core.Checkpoint,
    sink: exl3.Sink,
    buf: [192]u8 = undefined,
    name: []const u8 = "",

    fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        L.name = try std.fmt.bufPrint(&L.buf, prefix ++ fmt, args);
        return L.ck.get(L.name);
    }

    fn expect(L: *Loader, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, dims: []const usize) !Tensor {
        const t = try L.get(fmt, args);
        if (!t.is(dtype, dims)) {
            std.log.err("inkling2: {s} is not {s}{any}", .{ L.name, @tagName(dtype), dims });
            return error.UnexpectedInkling2Tensor;
        }
        return t;
    }

    fn put(L: *Loader, bytes: []const u8) !u64 {
        return L.sink.put(L.sink.ctx, bytes);
    }

    /// Rows [lo, hi) of a stored row-major tensor whose rows are `row_bytes` long.
    fn rows(L: *Loader, t: Tensor, lo: usize, hi: usize) !u64 {
        const rb = t.bytes.len / t.dim(0);
        return L.put(t.bytes[lo * rb .. hi * rb]);
    }

    /// Columns [lo, hi) of every row of a stored 2-D bf16 tensor.
    fn cols(L: *Loader, t: Tensor, lo: usize, hi: usize) !u64 {
        const n = t.dim(1);
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(L.gpa);
        for (0..t.dim(0)) |r| try out.appendSlice(L.gpa, t.bytes[2 * (r * n + lo) .. 2 * (r * n + hi)]);
        return L.put(out.items);
    }

    fn lin(L: *Loader, i: u32, name: []const u8, n: usize, k: usize, c: exl3.Cut) !Linear {
        var b: [128]u8 = undefined;
        return exl3.linear(L.gpa, L.ck, L.sink, try std.fmt.bufPrint(&b, prefix ++ "layers.{d}.attn.{s}", .{ i, name }), n, k, c);
    }

    fn attention(L: *Loader, i: u32) !Attention {
        const c = &L.w.config;
        const s = L.w.share;
        const h: usize = c.hidden;
        const D: usize = c.head_dim;
        const H: usize = c.heads;
        const KV: usize = c.kv_heads;
        const q: shard.Span = .{ .lo = s.heads.lo * D, .hi = s.heads.hi * D };
        const kv: shard.Span = .{ .lo = s.kv.lo * D, .hi = s.kv.hi * D };
        const rel: shard.Span = .{ .lo = s.heads.lo * c.d_rel, .hi = s.heads.hi * c.d_rel };
        var a: Attention = .{
            .q = try L.lin(i, "wq_du", H * D, h, .{ .cols = q }),
            .k = try L.lin(i, "wk_dv", KV * D, h, .{ .cols = kv }),
            .v = try L.lin(i, "wv_dv", KV * D, h, .{ .cols = kv }),
            .r = try L.lin(i, "wr_du", H * c.d_rel, h, if (s.whole()) .whole else .{ .cols = rel }),
            .gate = try L.lin(i, "gate_attn", H * D, h, .{ .cols = q }),
            .o = try L.lin(i, "wo_ud", h, H * D, .{ .rows = q }),
            .sink = try L.rows(try L.expect("layers.{d}.attn.attn_sink_proj.weight", .{i}, .bf16, &.{ H, h }), s.heads.lo, s.heads.hi),
            .q_scale = try L.rows(try L.expect("layers.{d}.attn.q_l2_clip.scale", .{i}, .bf16, &.{ H, D }), s.heads.lo, s.heads.hi),
            .rel_scale = try L.rows(try L.expect("layers.{d}.attn.rel_l2_clip.scale", .{i}, .bf16, &.{ H, c.d_rel }), s.heads.lo, s.heads.hi),
            .rel_proj = try L.rows(try L.expect("layers.{d}.attn.rel_logits_proj.proj", .{i}, .bf16, &.{ c.d_rel, c.rel_extent }), 0, c.d_rel),
            .k_conv = 0,
            .v_conv = 0,
        };
        if (c.layer[i] == .sliding) {
            a.k_conv = try L.rows(try L.expect("layers.{d}.attn.k_sconv.weight", .{i}, .bf16, &.{ KV * D, 1, c.conv }), kv.lo, kv.hi);
            a.v_conv = try L.rows(try L.expect("layers.{d}.attn.v_sconv.weight", .{i}, .bf16, &.{ KV * D, 1, c.conv }), kv.lo, kv.hi);
        } else {
            a.k_conv = try L.cols(try L.expect("layers.{d}.attn.k_sconv.static_w", .{i}, .bf16, &.{ c.conv, KV * D }), kv.lo, kv.hi);
            a.v_conv = try L.cols(try L.expect("layers.{d}.attn.v_sconv.static_w", .{i}, .bf16, &.{ c.conv, KV * D }), kv.lo, kv.hi);
            const t0 = s.kv.lo * c.conv;
            const t1 = s.kv.hi * c.conv;
            a.k_proj = try L.rows(try L.expect("layers.{d}.attn.k_sconv.weight_proj", .{i}, .bf16, &.{ KV * c.conv, h }), t0, t1);
            a.v_proj = try L.rows(try L.expect("layers.{d}.attn.v_sconv.weight_proj", .{i}, .bf16, &.{ KV * c.conv, h }), t0, t1);
        }
        return a;
    }

    fn vector(L: *Loader, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, n: usize) !u64 {
        return L.put((try L.expect(fmt, args, dtype, &.{n})).bytes);
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = &L.w.config;
        const h: usize = c.hidden;
        const global = c.layer[i] == .global;
        var l: Layer = undefined;
        l.kind = c.layer[i];
        l.norms = @splat(0);
        if (global) {
            for ([_][]const u8{ "q", "k", "v" }, 0..) |n, j| l.norms[j] = try L.vector("layers.{d}.{s}_input_norm.weight", .{ i, n }, .bf16, h);
        } else l.norms[0] = try L.vector("layers.{d}.attn_norm.weight", .{i}, .bf16, h);
        l.nq = if (global) 3 else 1;
        l.attn_query = 0;
        if (i > 0) {
            const t = try L.get("attn_res.attn_queries.{d}", .{i - 1});
            if (t.dtype != .f32 or t.numel() != l.nq * h) return error.UnexpectedInkling2Tensor;
            l.attn_query = try L.put(t.bytes);
        }
        l.mlp_query = try L.vector("attn_res.mlp_queries.{d}", .{i}, .f32, h);
        l.mlp_norm = try L.vector("layers.{d}.mlp_norm.weight", .{i}, .bf16, h);
        l.attn_conv = try L.put((try L.expect("layers.{d}.attn_sconv.weight", .{i}, .bf16, &.{ h, 1, c.conv })).bytes);
        l.mlp_conv = try L.put((try L.expect("layers.{d}.mlp_sconv.weight", .{i}, .bf16, &.{ h, 1, c.conv })).bytes);
        l.attn = try L.attention(i);
        l.router = try L.put((try L.expect("layers.{d}.mlp.gate.weight", .{i}, .bf16, &.{ c.experts, h })).bytes);
        l.bias = try L.vector("layers.{d}.mlp.gate.bias", .{i}, .f32, c.experts);
        const ex = try exl3.experts(L.gpa, L.ck, L.sink, prefix, i, c.experts, h, c.intermediate, L.w.share.inter);
        l.experts = ex.layer;
        l.bits = ex.bits;
        return l;
    }
};

/// The rank's weights of `layers` (a prefix for a truncated pack), the embedding whole, its vocabulary's head rows.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dense: *cuda.exl3.Dense, stream: cuda.Stream, dir: []const u8, c: cfg.Config, rank: usize, world: usize) !Weights {
    {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        try exl3.admit(gpa, bytes); // exllamav3 1.x on mul1, before a tensor is read
    }
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    const head_svh = try ck.get(prefix ++ "unembed.svh");
    const padded = head_svh.dim(0);
    if (padded < c.vocab or padded % 128 != 0) return error.UnexpectedInkling2Tensor;
    var w: Weights = .{ .gpa = gpa, .d = d, .config = c, .share = try shard.Share.of(&c, padded, rank, world), .padded_vocab = padded };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = gpa, .w = &w, .ck = &ck, .sink = .{ .ctx = &w, .put = Weights.put, .dense = dense, .stream = stream } };
    w.layers = try gpa.alloc(Layer, c.layers);
    for (w.layers, 0..) |*l, i| {
        l.* = try L.layer(@intCast(i));
        ck.release(); // the pages read so far, before they crowd the device's allocations
    }
    w.embed = try L.put((try L.expect("embed.weight", .{}, .bf16, &.{ c.vocab, c.hidden })).bytes);
    w.embed_norm = try L.vector("embed_norm.weight", .{}, .bf16, c.hidden);
    w.norm = try L.vector("norm.weight", .{}, .bf16, c.hidden);
    w.out_query = try L.vector("attn_res.output_query", .{}, .f32, c.hidden);
    const v = w.share.vocab;
    w.head = try exl3.linear(gpa, &ck, L.sink, prefix ++ "unembed", padded, c.hidden, if (w.share.whole()) .whole else .{ .cols = v });
    try stream.synchronize();
    return w;
}

pub const sidecars: []const []const u8 = &.{ "model-mtp.safetensors", "model-vision.safetensors" };

pub const Projections = struct { q: Linear, k: Linear, v: Linear, o: Linear, gate: Linear };

pub fn readBits(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fs.path.join(gpa, &.{ dir, "quantization_config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    return std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .allocate = .alloc_always });
}

pub fn bitsMap(json: std.json.Value) !std.json.ObjectMap {
    if (json != .object) return error.BadExl3Bits;
    const bits = json.object.get("bits") orelse return error.BadExl3Bits;
    if (bits != .object) return error.BadExl3Bits;
    return bits.object;
}

/// Exact Mini dimensions shared by the EXL3 projections and the grouped expert tiles.
pub fn validateMini(c: config.Text) !void {
    const heads: config.Heads = .{ .q = 24, .kv = 8, .dim = 128 };
    if (c.hidden != 2304 or c.expert_inter != 1152 or c.experts != 128 or c.topk != 8 or
        !std.meta.eql(c.local, heads) or !std.meta.eql(c.global, heads) or c.gate_cap <= 0 or c.up_cap <= 0)
        return error.Inkling2MiniExl3Shape;
}

pub fn miniAttention(ck: *core.Checkpoint, sink: exl3.MiniSink, bits: std.json.ObjectMap, i: u32, h: usize, q: usize, kv: usize) !Projections {
    const l: MiniLoader = .{ .ck = ck, .sink = sink, .bits = bits };
    return .{
        .q = try l.linear(i, "q_proj.linear", q, h),
        .k = try l.linear(i, "k_proj.linear", kv, h),
        .v = try l.linear(i, "v_proj.linear", kv, h),
        .o = try l.linear(i, "o_proj.linear", h, q),
        .gate = try l.linear(i, "o_proj.gate_proj", q, h),
    };
}

const MiniLoader = struct {
    ck: *core.Checkpoint,
    sink: exl3.MiniSink,
    bits: std.json.ObjectMap,

    fn linear(l: MiniLoader, i: u32, name: []const u8, n: usize, k: usize) !Linear {
        var buf: [96]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, "{d}.self_attn.{s}", .{ i, name });
        return exl3.miniLinear(l.ck, l.sink, l.bits, base, n, k);
    }
};

test "Mini EXL3 bit table uses projection-specific halves" {
    const a = std.testing.allocator;
    const doc = try std.json.parseFromSlice(std.json.Value, a,
        \\{"bits":{"gate_up":3.5,"down":3,"other":4,"dense":6,"bad":3.2}}
    , .{});
    defer doc.deinit();
    const bits = try bitsMap(doc.value);
    try std.testing.expectEqual(7, try exl3.sidecarHalves(bits, "gate_up"));
    try std.testing.expectEqual(6, try exl3.sidecarHalves(bits, "down"));
    try std.testing.expectEqual(8, try exl3.sidecarHalves(bits, "other"));
    try std.testing.expectEqual(12, try exl3.sidecarHalves(bits, "dense"));
    try std.testing.expectError(error.BadExl3Bits, exl3.sidecarHalves(bits, "bad"));
}

test "Mini EXL3 attention dimensions resolve built mul1 plans" {
    for ([_][2]usize{ .{ 2304, 3072 }, .{ 2304, 1024 }, .{ 3072, 2304 } }) |kn| {
        for ([_]u32{ 12, 16 }) |width| {
            const p = cuda.exl3.plan(kn[0], kn[1], 192, 8);
            try std.testing.expect(cuda.exl3.variantIndex(width, .mul1, p.wk) != null);
            try std.testing.expectEqual(0, kn[0] % (16 * p.sk * p.wk));
        }
    }
}
