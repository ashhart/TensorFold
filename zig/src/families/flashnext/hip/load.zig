//! Flash Next's checkpoint onto a HIP rank: each tensor read from the mapped shards, cut to the rank, uploaded.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const model = @import("model.zig");

const quant = core.quant;
const Tensor = core.checkpoint.Tensor;
const Host = quant.Host;
const Rank = quant.Rank;
const Span = quant.Span;
const Allocator = std.mem.Allocator;

/// The table core/quant reads projections through: the family's checkpoint tables, over core's checkpoint.
pub const Tables = @import("checkpoint_tables.zig").Tables(core.checkpoint.Checkpoint, core.quant.mlx.Width);

/// The device copy of one rank's weights: every buffer it uploaded, freed together.
pub const Weights = struct {
    gpa: Allocator,
    buffers: std.ArrayList(hip.DeviceBuffer) = .empty,
    layers: []model.Layer = &.{},
    model: model.Model = undefined,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

const Loader = struct {
    a: Allocator,
    t: *const Tables,
    up: hip.Upload,
    rank: ?Rank,
    /// The checkpoint stores centred norms as gamma (true) or as gamma - 1 (false).
    around_one: bool,

    fn key(l: *const Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(l.a, fmt, args);
    }

    /// A projection, its rows and inputs checked against the model's description where it gives them.
    fn proj(l: *const Loader, k: []const u8) !Host {
        const h = try quant.read(l.a, l.t, k);
        const want = (try tensors.shape(l.t.cfg, k)) orelse return h;
        const n, const width = switch (h) {
            .mlx => |m| .{ m.words.shape[0], m.scales.shape[1] * m.group },
            else => return h,
        };
        if (n != want.n or width != want.k) return error.UnexpectedShape;
        return h;
    }

    fn put(l: *const Loader, h: Host, partial: bool) !model.Projection {
        var p = try quant.view(try quant.upload(l.up.uploader(), h), false);
        p.partial = partial;
        return p;
    }

    fn tensor(l: *const Loader, t: Tensor) !u64 {
        return (try l.up.uploader().tensor(t)).ptr;
    }

    /// A float tensor widened to fp32, rows `spans` of it when given.
    fn float(l: *const Loader, k: []const u8, cut: ?[]const Span) !Tensor {
        var t = try quant.convert.float32(l.a, try l.t.get(k));
        if (cut) |s| t = try quant.slice.takeRows(l.a, t, s);
        return t;
    }

    /// A centred RMSNorm's fp32 scale from its stored value (tensors.centred).
    fn scale(l: *const Loader, k: []const u8) !u64 {
        var t = try l.float(k, null);
        const v = try floats(l.a, t);
        for (v) |*x| x.* = tensors.centred(x.*, l.around_one);
        t.bytes = std.mem.sliceAsBytes(v);
        return l.tensor(t);
    }

    fn world(l: *const Loader) usize {
        return if (l.rank) |r| r.world else 1;
    }

    /// Rows `[from + n r, from + n (r + 1))` of each segment of `sizes` for this rank, every segment split evenly.
    fn spans(l: *const Loader, sizes: []const usize, out: []Span) []const Span {
        var base: usize = 0;
        const r = if (l.rank) |x| x.rank else 0;
        for (sizes, out[0..sizes.len]) |size, *s| {
            const part = size / l.world();
            s.* = .{ .from = base + r * part, .to = base + (r + 1) * part };
            base += size;
        }
        return out[0..sizes.len];
    }

    fn rows(l: *const Loader, h: Host, sizes: []const usize) !Host {
        if (l.rank == null) return h;
        var buf: [4]Span = undefined;
        return quant.sliceRows(l.a, h, l.spans(sizes, &buf));
    }

    fn cols(l: *const Loader, h: Host, what: []const u8) !Host {
        const r = l.rank orelse return h;
        return quant.sliceCols(l.a, h, r, what);
    }

    /// The down projection with the inject rows stacked under it: one product gives the low rank and the gates.
    fn hc(l: *const Loader, base: []const u8, inject: bool) !model.Hc {
        var down = try l.proj(try l.named(base, tensors.hc.down));
        if (inject) down = try stackRows(l.a, down, try l.proj(try l.named(base, tensors.hc.inject)));
        return .{
            .scale = try l.scale(try l.named(base, tensors.hc.norm)),
            .down = try l.put(down, false),
            .up = try l.put(try l.proj(try l.named(base, tensors.hc.up)), false),
        };
    }

    /// `name` under `stem`.
    fn named(l: *const Loader, stem: []const u8, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(l.a, "{s}.{s}", .{ stem, name });
    }

    fn linear(l: *const Loader, base: []const u8, c: *const config.Config) !model.Linear {
        const kw = c.key_heads * c.key_dim;
        const vw = c.value_heads * c.value_dim;
        const qkv = [_]usize{ kw, kw, vw };
        var buf: [4]Span = undefined;
        const conv = if (l.rank != null) l.spans(&qkv, &buf) else null;
        var hbuf: [4]Span = undefined;
        const heads = if (l.rank != null) l.spans(&.{c.value_heads}, &hbuf) else null;
        const t = tensors.linear;
        var conv_t = try l.float(try l.named(base, t.conv), conv);
        conv_t.rank = 2;
        return .{
            .qkv = try l.put(try l.rows(try l.proj(try l.named(base, t.qkv)), &qkv), false),
            .z = try l.put(try l.rows(try l.proj(try l.named(base, t.z)), &.{vw}), false),
            .a = try l.put(try l.rows(try l.proj(try l.named(base, t.a)), &.{c.value_heads}), false),
            .b = try l.put(try l.rows(try l.proj(try l.named(base, t.b)), &.{c.value_heads}), false),
            .out = try l.put(try l.cols(try l.proj(try l.named(base, t.out)), "out_proj groups"), l.rank != null),
            .conv = try l.tensor(conv_t),
            .a_log = try l.tensor(try l.float(try l.named(base, t.a_log), heads)),
            .dt_bias = try l.tensor(try l.float(try l.named(base, t.dt_bias), heads)),
            .norm = try l.tensor(try l.float(try l.named(base, t.norm), null)),
        };
    }

    /// A sparse layer; under tp a rank's query heads, the kv head they read, its share of o's inputs.
    fn sparse(l: *const Loader, base: []const u8, c: *const config.Config, whole: bool) !model.Sparse {
        const hd = c.head_dim;
        const w = if (whole) 1 else l.world();
        const r = if (whole or l.rank == null) 0 else l.rank.?.rank;
        const heads = c.heads / w;
        const q_span = [_]Span{.{ .from = r * heads * 2 * hd, .to = (r + 1) * heads * 2 * hd }};
        const kv = r * c.kv_heads / @max(w, c.kv_heads);
        const kv_n = @max(1, c.kv_heads / w);
        const kv_span = [_]Span{.{ .from = kv * kv_n * hd, .to = (kv + 1) * kv_n * hd }};
        const t = tensors.sparse;
        const q = try l.proj(try l.named(base, t.q));
        const k = try l.proj(try l.named(base, t.k));
        const v = try l.proj(try l.named(base, t.v));
        const o = try l.proj(try l.named(base, t.o));
        return .{
            .q = try l.put(if (w > 1) try quant.sliceRows(l.a, q, &q_span) else q, false),
            .k = try l.put(if (w > 1) try quant.sliceRows(l.a, k, &kv_span) else k, false),
            .v = try l.put(if (w > 1) try quant.sliceRows(l.a, v, &kv_span) else v, false),
            .o = try l.put(if (w > 1) try l.cols(o, "o_proj groups") else o, w > 1),
            .index = try l.put(try l.proj(try l.named(base, t.index)), false),
            .q_norm = try l.scale(try l.named(base, t.q_norm)),
            .k_norm = try l.scale(try l.named(base, t.k_norm)),
            .iq_norm = try l.scale(try l.named(base, t.iq_norm)),
            .ik_norm = try l.scale(try l.named(base, t.ik_norm)),
        };
    }

    fn moe(l: *const Loader, base: []const u8, c: *const config.Config, whole: bool) !model.Routed {
        var r = try quant.experts.routed(l.a, l.t, try l.key("{s}.{s}", .{ base, tensors.moe.prefix }), .{ .experts = c.experts, .hidden = c.hidden, .top_k = c.topk });
        if (!whole) if (l.rank) |rk| try quant.experts.sliceRank(l.a, &r, rk);
        const u = l.up.uploader();
        return .{
            .rows32 = (try u.tensor(r.rows32)).ptr,
            .top_k = r.top_k,
            .rows = r.router.shape[0],
            .remap = if (r.remap) |t| (try u.tensor(t)).ptr else 0,
            .exact = true,
            .experts = .{
                .fused = try quant.view(try quant.upload(u, r.experts.fused), true),
                .down = try quant.view(try quant.upload(u, r.experts.down), true),
                .count = r.experts.count,
                .width = r.experts.width,
                .dims = r.experts.dims,
            },
        };
    }

    fn layer(l: *const Loader, base: []const u8, c: *const config.Config, kind: config.Kind, whole: bool) !model.Layer {
        return .{
            .ahc = try l.hc(try l.named(base, tensors.hc.attn), true),
            .mhc = try l.hc(try l.named(base, tensors.hc.mlp), true),
            .mixer = switch (kind) {
                .linear_attention => .{ .linear = try l.linear(base, c) },
                .sparse_attention => .{ .sparse = try l.sparse(base, c, whole) },
            },
            .moe = try l.moe(base, c, whole),
        };
    }

    fn ple(l: *const Loader, index: usize) !model.Ple {
        var buf: [128]u8 = undefined;
        const base = try tensors.layer(&buf, index);
        const t = tensors.ple;
        const kv = try stackRows(l.a, try l.proj(try l.named(base, t.key)), try l.proj(try l.named(base, t.value)));
        var conv = try l.float(try l.named(base, t.conv), null);
        conv.rank = 2;
        return .{
            .layer = index,
            .kv = try l.put(kv, false),
            .key_scale = try l.scale(try l.named(base, t.norm_key)),
            .query_scale = try l.scale(try l.named(base, t.norm_query)),
            .conv_scale = try l.scale(try l.named(base, t.norm_conv)),
            .conv = try l.tensor(conv),
        };
    }
};

/// An fp32 tensor's values copied out (its bytes may sit at any alignment).
fn floats(a: Allocator, t: Tensor) ![]f32 {
    const v = try a.alloc(f32, t.bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(v), t.bytes);
    return v;
}

/// Rows of `under` stacked below `top` (one product for both): same K, bits and group.
fn stackRows(a: Allocator, top: Host, under: Host) !Host {
    const t = top.mlx;
    const u = under.mlx;
    if (t.bits != u.bits or t.group != u.group or t.words.shape[1] != u.words.shape[1]) return error.UnexpectedTensor;
    return .{ .mlx = .{ .words = try join(a, t.words, u.words), .scales = try join(a, t.scales, u.scales), .biases = try join(a, t.biases, u.biases), .bits = t.bits, .group = t.group } };
}

fn join(a: Allocator, x: Tensor, y: Tensor) !Tensor {
    if (x.dtype != y.dtype or x.rank != 2 or y.rank != 2 or x.shape[1] != y.shape[1]) return error.UnexpectedTensor;
    const bytes = try a.alloc(u8, x.bytes.len + y.bytes.len);
    @memcpy(bytes[0..x.bytes.len], x.bytes);
    @memcpy(bytes[x.bytes.len..], y.bytes);
    var out = x;
    out.shape[0] = x.shape[0] + y.shape[0];
    out.bytes = bytes;
    return out;
}

/// Whether the centred norms are stored as gamma, from every layer's attention hc norm mean (tensors.normsAroundOne).
fn aroundOne(a: Allocator, t: *const Tables, layers: usize) !bool {
    const means = try a.alloc(f64, layers);
    for (means, 0..) |*m, i| {
        var buf: [128]u8 = undefined;
        const k = try std.fmt.allocPrint(a, "{s}.{s}.{s}", .{ try tensors.layer(&buf, i), tensors.hc.attn, tensors.hc.norm });
        const v = try floats(a, try quant.convert.float32(a, try t.get(k)));
        var sum: f64 = 0;
        for (v) |x| sum += x;
        m.* = sum / @as(f64, @floatFromInt(v.len));
    }
    return tensors.normsAroundOne(means);
}

/// The draft head's rows: the family's draft vocabulary (ascending ids) of the output head.
pub const draft_ids = @embedFile("../draft_vocab_default.txt");

fn draftSpans(a: Allocator) ![]Span {
    var list: std.ArrayList(Span) = .empty;
    var it = std.mem.tokenizeAny(u8, draft_ids, "\n\r ");
    while (it.next()) |word| {
        const id = try std.fmt.parseInt(usize, word, 10);
        if (list.items.len > 0 and list.items[list.items.len - 1].to == id) {
            list.items[list.items.len - 1].to += 1;
        } else try list.append(a, .{ .from = id, .to = id + 1 });
    }
    return list.items;
}

/// The draft vocabulary as token ids.
pub fn draftIds(a: Allocator) ![]u32 {
    var list: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, draft_ids, "\n\r ");
    while (it.next()) |word| try list.append(a, try std.fmt.parseInt(u32, word, 10));
    return list.toOwnedSlice(a);
}

test "the draft ids are one allocation of the vocabulary's length" {
    const ids = try draftIds(std.testing.allocator);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 79591), ids.len);
}

/// One rank's weights (`rank` null: the whole model) from the mapped checkpoint `ck`; rank 0 also holds the MTP head.
pub fn load(gpa: Allocator, d: *const hip.Runtime, ck: *core.checkpoint.Checkpoint, c: *const config.Config, rank: ?Rank, act: model.Kind) !Weights {
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    const tables: Tables = .{ .ck = ck, .cfg = c };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const l: Loader = .{ .a = arena.allocator(), .t = &tables, .up = .{ .r = d, .buffers = &w.buffers, .gpa = gpa }, .rank = rank, .around_one = try aroundOne(arena.allocator(), &tables, c.layers) };
    const world_n: usize = if (rank) |r| r.world else 1;
    const dims = try model.Dims.of(c, world_n);
    const head = try l.proj(tensors.lm_head);
    w.layers = try gpa.alloc(model.Layer, c.layers);
    var ple: ?model.Ple = null;
    for (0..c.layers) |i| {
        _ = arena.reset(.retain_capacity);
        var buf: [128]u8 = undefined;
        w.layers[i] = try l.layer(try tensors.layer(&buf, i), c, try c.kind(i), false);
        if (c.ple[i]) ple = try l.ple(i);
    }
    _ = arena.reset(.retain_capacity);
    w.model = .{
        .dims = dims,
        .act = act,
        .embed = try l.put(try l.proj(tensors.embed), false),
        .layers = w.layers,
        .mix = try l.hc(tensors.mixer, false),
        .head = try l.put(try l.rows(head, &.{c.vocab}), false),
        .ple = ple orelse return error.NoPleLayer,
    };
    const drafts = if (rank) |r| r.rank == 0 else true;
    if (drafts and c.mtp.layers > 0) {
        _ = arena.reset(.retain_capacity);
        const draft_rows = try quant.sliceRows(l.a, head, try draftSpans(l.a));
        w.model.mtp = .{
            .e_norm = try l.scale(tensors.mtp.norm_embedding),
            .h_norm = try l.scale(tensors.mtp.norm_hidden),
            .fc_e = try l.put(try l.proj(tensors.mtp.fc_embedding), false),
            .fc_h = try l.put(try l.proj(tensors.mtp.fc_hidden), false),
            .layer = try l.layer(tensors.mtp.layer0, c, c.mtp.kinds[0], true),
            .mix = try l.hc(tensors.mtp.mixer, false),
            .dims = try model.Dims.of(c, 1),
            .draft = try l.put(draft_rows, false),
        };
    }
    return w;
}
