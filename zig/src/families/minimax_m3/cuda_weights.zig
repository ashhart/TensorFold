//! A TP rank's MiniMax-M3.1-Flash weights from a published EXL3 pack: EXL3 projections and experts, the rest as stored.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const exl3 = @import("cuda_exl3.zig");

const Tensor = core.checkpoint.Tensor;
/// An EXL3 projection (cuda.exl3.Dense).
pub const Linear = cuda.exl3.Weight;
/// A lane weight (cuda.qmmf): the bf16 indexer here, the release's MXFP8 in dspark/.
pub const Mx = cuda.qmmf.Weight;

/// The checkpoint's text tensors sit under this prefix; the vision tower beside it.
pub const prefix = "language_model.";

/// q, k, v by the rank's heads, o by its rows; its index q heads | index k as one bf16 lane weight; the head norms.
pub const Attention = struct { q: Linear, k: Linear, v: Linear, o: Linear, index: Mx, q_norm: u64, k_norm: u64, iq_norm: u64, ik_norm: u64 };
/// gate and up by the rank's intermediate columns, down by its rows.
pub const Mlp = struct { gate: Linear, up: Linear, down: Linear };
/// dspark/'s MLP: gate | up of the rank's rows stacked, down by its columns (MXFP8 lane weights).
pub const MxMlp = struct { gu: Mx, down: Mx };
/// The fp32 router [experts][hidden] and its score bias, the rank's share of the routed experts, the shared expert.
pub const Moe = struct { router: u64, bias: u64, routed: cuda.experts.Layer, shared: Mlp };
pub const Layer = struct { in_norm: u64, post_norm: u64, attn: Attention, ffn: union(enum) { dense: Mlp, moe: Moe } };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    embed: u64 = 0, // bf16 [vocab][hidden], whole on every rank
    norm: u64 = 0, // the final norm, bf16 [hidden]
    head: Linear = undefined, // this rank's vocabulary columns of the 8-bit head
    layers: []Layer = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0, // device bytes the weights hold

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

/// A load's device side: the uploads' stream, the dense EXL3 kernels whose scratch each weight sizes.
pub const Tools = struct { d: *const cuda.Driver, stream: cuda.Stream, dense: *cuda.exl3.Dense };

/// Rows and columns ([lo, hi), null: all) of a stored MXFP8 projection, its whole shape if given: a stacked part.
pub const MxPart = struct { name: []const u8, rows: ?[2]usize = null, cols: ?[2]usize = null, dims: ?[2]usize = null };

pub const Loader = struct {
    gpa: std.mem.Allocator,
    t: Tools,
    w: *Weights,
    ck: *core.Checkpoint,
    name_buf: [256]u8 = undefined,

    /// A device copy of `bytes`, owned by the weights.
    pub fn put(L: *Loader, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(L.t.d, bytes);
        errdefer {
            var owned = b;
            owned.free();
        }
        try L.w.buffers.append(L.gpa, b);
        L.w.bytes += b.len;
        return b.ptr;
    }

    pub fn name(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&L.name_buf, fmt, args);
    }

    pub fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        return L.ck.get(try L.name(fmt, args));
    }

    /// A tensor of `dtype` with exactly `dims`.
    pub fn expect(L: *Loader, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, dims: []const usize) !Tensor {
        const name_ = try L.name(fmt, args);
        const t = try L.ck.get(name_);
        if (t.dtype != dtype or t.rank != dims.len or !std.mem.eql(usize, t.shape[0..t.rank], dims)) {
            std.log.err("minimax_m3: {s} is not {s}{any}", .{ name_, @tagName(dtype), dims });
            return error.UnexpectedMiniMaxTensor;
        }
        return t;
    }

    /// A bf16 vector of `n` as stored, on the device.
    fn vector(L: *Loader, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
        const t = try L.expect(fmt, args, .bf16, &.{n});
        return L.put(t.bytes);
    }

    /// MXFP8 parts (e4m3 `.weight` [n, k], e8m0 `.weight_scale_inv` [n, k/32]) cut and stacked as one lane weight.
    pub fn mx(L: *Loader, parts: []const MxPart) !Mx {
        var total: usize = 0;
        var k: usize = 0;
        for (parts, 0..) |p, i| {
            const w = try L.get("{s}.weight", .{p.name});
            if (w.dtype != .f8_e4m3 or w.rank != 2) return error.UnexpectedMiniMaxTensor;
            if (p.dims) |d| if (w.dim(0) != d[0] or w.dim(1) != d[1]) {
                std.log.err("minimax_m3: {s} is [{d}, {d}], not [{d}, {d}]", .{ p.name, w.dim(0), w.dim(1), d[0], d[1] });
                return error.UnexpectedMiniMaxTensor;
            };
            const rows = p.rows orelse [2]usize{ 0, w.dim(0) };
            const cols = p.cols orelse [2]usize{ 0, w.dim(1) };
            if (rows[1] > w.dim(0) or cols[1] > w.dim(1) or rows[0] >= rows[1] or cols[0] >= cols[1]) return error.UnexpectedMiniMaxTensor;
            if (cols[0] % 64 != 0 or (cols[1] - cols[0]) % 64 != 0 or (rows[1] - rows[0]) % 64 != 0) return error.UnexpectedMiniMaxTensor;
            if (i == 0) k = cols[1] - cols[0] else if (k != cols[1] - cols[0]) return error.UnexpectedMiniMaxTensor;
            total += rows[1] - rows[0];
        }
        const codes = try L.gpa.alloc(u8, total * k);
        defer L.gpa.free(codes);
        const scales = try L.gpa.alloc(u8, total * k / 32);
        defer L.gpa.free(scales);
        var at: usize = 0;
        for (parts) |p| {
            const w = try L.get("{s}.weight", .{p.name});
            const s = try L.get("{s}.weight_scale_inv", .{p.name});
            if (s.dtype != .u8 or s.rank != 2 or s.dim(0) != w.dim(0) or s.dim(1) != w.dim(1) / 32) return error.UnexpectedMiniMaxTensor;
            const kk = w.dim(1);
            const rows = p.rows orelse [2]usize{ 0, w.dim(0) };
            const cols = p.cols orelse [2]usize{ 0, kk };
            for (rows[0]..rows[1]) |r| {
                @memcpy(codes[at * k ..][0..k], w.bytes[r * kk + cols[0] ..][0..k]);
                @memcpy(scales[at * (k / 32) ..][0 .. k / 32], s.bytes[r * (kk / 32) + cols[0] / 32 ..][0 .. k / 32]);
                at += 1;
            }
        }
        const npad = cuda.mxfp8.padded(total);
        const pc = try L.gpa.alloc(u8, npad * k);
        defer L.gpa.free(pc);
        cuda.mxfp8.packCodes(pc, codes, total, k);
        const ps = try L.gpa.alloc(u8, npad * k / 32);
        defer L.gpa.free(ps);
        cuda.mxfp8.packScales(ps, scales, total, k);
        const dc = try L.put(pc);
        const ds = try L.put(ps);
        return cuda.mxfp8.weight(dc, ds, total, k);
    }

    fn attention(L: *Loader, i: u32) !Attention {
        const c = &L.w.config;
        const s = L.w.share;
        const D: usize = c.head_dim;
        const h: usize = c.hidden;
        const q = s.queries(c);
        const kv: shard.Span = .{ .lo = s.kv.lo * D, .hi = s.kv.hi * D };
        var names: [2][96]u8 = undefined;
        const iq = try L.expect(prefix ++ "model.layers.{d}.self_attn.index_q_proj.weight", .{i}, .f16, &.{ c.index_heads * c.index_dim, h });
        const ik = try L.expect(prefix ++ "model.layers.{d}.self_attn.index_k_proj.weight", .{i}, .f16, &.{ c.index_dim, h });
        var rows: std.ArrayList(u16) = .empty;
        defer rows.deinit(L.gpa);
        try exl3.f16Rows(L.gpa, iq, s.kv.lo * c.index_dim, s.kv.hi * c.index_dim, &rows); // index head h goes with key head h
        try exl3.f16Rows(L.gpa, ik, 0, c.index_dim, &rows);
        const n = (s.kv.len() + 1) * c.index_dim;
        const tiles = try L.gpa.alloc(u16, cuda.mxfp8.padded(n) * h);
        defer L.gpa.free(tiles);
        cuda.qmmf.packBf16(tiles, rows.items, n, h);
        return .{
            .q = try exl3.linear(L, try std.fmt.bufPrint(&names[0], prefix ++ "model.layers.{d}.self_attn.q_proj", .{i}), c.heads * D, h, .{ .cols = .{ .lo = q.lo * D, .hi = q.hi * D } }),
            .k = try exl3.linear(L, try std.fmt.bufPrint(&names[0], prefix ++ "model.layers.{d}.self_attn.k_proj", .{i}), c.kv_heads * D, h, .{ .cols = kv }),
            .v = try exl3.linear(L, try std.fmt.bufPrint(&names[0], prefix ++ "model.layers.{d}.self_attn.v_proj", .{i}), c.kv_heads * D, h, .{ .cols = kv }),
            .o = try exl3.linear(L, try std.fmt.bufPrint(&names[1], prefix ++ "model.layers.{d}.self_attn.o_proj", .{i}), h, c.heads * D, .{ .rows = .{ .lo = q.lo * D, .hi = q.hi * D } }),
            .index = cuda.modelopt.bf16Weight(try L.put(std.mem.sliceAsBytes(tiles)), n, h),
            .q_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.q_norm.weight", .{i}, D),
            .k_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.k_norm.weight", .{i}, D),
            .iq_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.index_q_norm.weight", .{i}, c.index_dim),
            .ik_norm = try L.vector(prefix ++ "model.layers.{d}.self_attn.index_k_norm.weight", .{i}, c.index_dim),
        };
    }

    /// A gate/up/down MLP of `inter` under `module` cut to `span` of its intermediate.
    fn mlp(L: *Loader, module: []const u8, inter: usize, span: shard.Span) !Mlp {
        const h: usize = L.w.config.hidden;
        var name_buf: [160]u8 = undefined;
        return .{
            .gate = try exl3.linear(L, try std.fmt.bufPrint(&name_buf, "{s}.gate_proj", .{module}), inter, h, .{ .cols = span }),
            .up = try exl3.linear(L, try std.fmt.bufPrint(&name_buf, "{s}.up_proj", .{module}), inter, h, .{ .cols = span }),
            .down = try exl3.linear(L, try std.fmt.bufPrint(&name_buf, "{s}.down_proj", .{module}), h, inter, .{ .rows = span }),
        };
    }

    fn moe(L: *Loader, i: u32) !Moe {
        const c = &L.w.config;
        const router = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.gate.weight", .{i}, .f32, &.{ c.experts, c.hidden });
        const rp = try L.put(router.bytes);
        const bias = try L.expect(prefix ++ "model.layers.{d}.block_sparse_moe.e_score_correction_bias", .{i}, .f32, &.{c.experts});
        const bp = try L.put(bias.bytes);
        var mb: [128]u8 = undefined;
        const shared = try L.mlp(try std.fmt.bufPrint(&mb, prefix ++ "model.layers.{d}.block_sparse_moe.shared_experts", .{i}), c.shared_intermediate, L.w.share.shared);
        return .{ .router = rp, .bias = bp, .routed = try exl3.experts(L, i), .shared = shared };
    }

    fn layer(L: *Loader, i: u32) !Layer {
        const c = &L.w.config;
        var mb: [96]u8 = undefined;
        return .{
            .in_norm = try L.vector(prefix ++ "model.layers.{d}.input_layernorm.weight", .{i}, c.hidden),
            .post_norm = try L.vector(prefix ++ "model.layers.{d}.post_attention_layernorm.weight", .{i}, c.hidden),
            .attn = try L.attention(i),
            .ffn = if (c.layer[i].moe) .{ .moe = try L.moe(i) } else .{ .dense = try L.mlp(try std.fmt.bufPrint(&mb, prefix ++ "model.layers.{d}.mlp", .{i}), c.dense_intermediate, L.w.share.dense) },
        };
    }
};

/// The rank's weights of `layers` (a prefix for a thin check), the embeddings whole, its vocabulary's head rows.
pub fn load(gpa: std.mem.Allocator, io: std.Io, tools: Tools, dir: []const u8, c: cfg.Config, share: shard.Share, layers: []const u32) !Weights {
    var w: Weights = .{ .gpa = gpa, .config = c, .share = share };
    errdefer w.deinit();
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    var L: Loader = .{ .gpa = gpa, .t = tools, .w = &w, .ck = &ck };
    w.layers = try gpa.alloc(Layer, layers.len);
    for (layers, w.layers) |i, *l| {
        if (i >= c.layers) return error.BadLayer;
        l.* = try L.layer(i);
        ck.release(); // the pages read so far, before they crowd the device's allocations
    }
    const embed = try L.expect(prefix ++ "model.embed_tokens.weight", .{}, .bf16, &.{ c.vocab, c.hidden });
    w.embed = try L.put(embed.bytes);
    w.norm = try L.vector(prefix ++ "model.norm.weight", .{}, c.hidden);
    w.head = try exl3.linear(&L, prefix ++ "lm_head", c.vocab, c.hidden, .{ .cols = .{ .lo = share.vocab.lo, .hi = share.vocab.hi } });
    try tools.stream.synchronize();
    return w;
}
