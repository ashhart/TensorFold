//! Flash Next's checkpoint onto a HIP device: each tensor read from the mapped shards and uploaded.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const model = @import("model.zig");

const quant = core.quant;
const Tensor = core.checkpoint.Tensor;
const Host = quant.Host;
const Span = quant.Span;
const Allocator = std.mem.Allocator;

/// The table core/quant reads projections through: the family's checkpoint tables, over core's checkpoint.
pub const Tables = @import("checkpoint_tables.zig").Tables(core.checkpoint.Checkpoint, core.quant.mlx.Width);

/// The device copy of the weights: every buffer it uploaded, freed together.
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

    fn put(l: *const Loader, h: Host) !model.Projection {
        return quant.view(try quant.upload(l.up.uploader(), h), false);
    }

    fn tensor(l: *const Loader, t: Tensor) !u64 {
        return (try l.up.uploader().tensor(t)).ptr;
    }

    /// A float tensor widened to fp32.
    fn float(l: *const Loader, k: []const u8) !Tensor {
        return quant.convert.float32(l.a, try l.t.get(k));
    }

    /// A centred RMSNorm's fp32 scale from its stored value (tensors.centred).
    fn scale(l: *const Loader, k: []const u8) !u64 {
        var t = try l.float(k);
        const v = try floats(l.a, t);
        for (v) |*x| x.* = tensors.centred(x.*, l.around_one);
        t.bytes = std.mem.sliceAsBytes(v);
        return l.tensor(t);
    }

    /// The down projection with the inject rows stacked under it: one product gives the low rank and the gates.
    fn hc(l: *const Loader, base: []const u8, inject: bool) !model.Hc {
        var down = try l.proj(try l.named(base, tensors.hc.down));
        if (inject) down = try stackRows(l.a, down, try l.proj(try l.named(base, tensors.hc.inject)));
        return .{
            .scale = try l.scale(try l.named(base, tensors.hc.norm)),
            .down = try l.put(down),
            .up = try l.put(try l.proj(try l.named(base, tensors.hc.up))),
        };
    }

    /// `name` under `stem`.
    fn named(l: *const Loader, stem: []const u8, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(l.a, "{s}.{s}", .{ stem, name });
    }

    fn linear(l: *const Loader, base: []const u8) !model.Linear {
        const t = tensors.linear;
        var conv_t = try l.float(try l.named(base, t.conv));
        conv_t.rank = 2;
        return .{
            .qkv = try l.put(try l.proj(try l.named(base, t.qkv))),
            .z = try l.put(try l.proj(try l.named(base, t.z))),
            .a = try l.put(try l.proj(try l.named(base, t.a))),
            .b = try l.put(try l.proj(try l.named(base, t.b))),
            .out = try l.put(try l.proj(try l.named(base, t.out))),
            .conv = try l.tensor(conv_t),
            .a_log = try l.tensor(try l.float(try l.named(base, t.a_log))),
            .dt_bias = try l.tensor(try l.float(try l.named(base, t.dt_bias))),
            .norm = try l.tensor(try l.float(try l.named(base, t.norm))),
        };
    }

    fn sparse(l: *const Loader, base: []const u8) !model.Sparse {
        const t = tensors.sparse;
        return .{
            .q = try l.put(try l.proj(try l.named(base, t.q))),
            .k = try l.put(try l.proj(try l.named(base, t.k))),
            .v = try l.put(try l.proj(try l.named(base, t.v))),
            .o = try l.put(try l.proj(try l.named(base, t.o))),
            .index = try l.put(try l.proj(try l.named(base, t.index))),
            .q_norm = try l.scale(try l.named(base, t.q_norm)),
            .k_norm = try l.scale(try l.named(base, t.k_norm)),
            .iq_norm = try l.scale(try l.named(base, t.iq_norm)),
            .ik_norm = try l.scale(try l.named(base, t.ik_norm)),
        };
    }

    fn moe(l: *const Loader, base: []const u8, c: *const config.Config) !model.Routed {
        const r = try quant.experts.routed(l.a, l.t, try l.key("{s}.{s}", .{ base, tensors.moe.prefix }), .{ .experts = c.experts, .hidden = c.hidden, .top_k = c.topk });
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

    fn layer(l: *const Loader, base: []const u8, c: *const config.Config, kind: config.Kind) !model.Layer {
        return .{
            .ahc = try l.hc(try l.named(base, tensors.hc.attn), true),
            .mhc = try l.hc(try l.named(base, tensors.hc.mlp), true),
            .mixer = switch (kind) {
                .linear_attention => .{ .linear = try l.linear(base) },
                .sparse_attention => .{ .sparse = try l.sparse(base) },
            },
            .moe = try l.moe(base, c),
        };
    }

    fn ple(l: *const Loader, index: usize) !model.Ple {
        var buf: [128]u8 = undefined;
        const base = try tensors.layer(&buf, index);
        const t = tensors.ple;
        const kv = try stackRows(l.a, try l.proj(try l.named(base, t.key)), try l.proj(try l.named(base, t.value)));
        var conv = try l.float(try l.named(base, t.conv));
        conv.rank = 2;
        return .{
            .layer = index,
            .kv = try l.put(kv),
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

/// The whole model's weights from the mapped checkpoint `ck`, the MTP head's included.
pub fn load(gpa: Allocator, d: *const hip.Runtime, ck: *core.checkpoint.Checkpoint, c: *const config.Config, act: model.Kind) !Weights {
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    const tables: Tables = .{ .ck = ck, .cfg = c };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const l: Loader = .{ .a = arena.allocator(), .t = &tables, .up = .{ .r = d, .buffers = &w.buffers, .gpa = gpa }, .around_one = try aroundOne(arena.allocator(), &tables, c.layers) };
    const dims = try model.Dims.of(c, 1);
    const head = try l.proj(tensors.lm_head);
    w.layers = try gpa.alloc(model.Layer, c.layers);
    var ple: ?model.Ple = null;
    for (0..c.layers) |i| {
        _ = arena.reset(.retain_capacity);
        var buf: [128]u8 = undefined;
        w.layers[i] = try l.layer(try tensors.layer(&buf, i), c, try c.kind(i));
        if (c.ple[i]) ple = try l.ple(i);
    }
    _ = arena.reset(.retain_capacity);
    w.model = .{
        .dims = dims,
        .act = act,
        .embed = try l.put(try l.proj(tensors.embed)),
        .layers = w.layers,
        .mix = try l.hc(tensors.mixer, false),
        .head = try l.put(head),
        .ple = ple orelse return error.NoPleLayer,
    };
    if (c.mtp.layers > 0) {
        _ = arena.reset(.retain_capacity);
        const draft_rows = try quant.sliceRows(l.a, head, try draftSpans(l.a));
        w.model.mtp = .{
            .e_norm = try l.scale(tensors.mtp.norm_embedding),
            .h_norm = try l.scale(tensors.mtp.norm_hidden),
            .fc_e = try l.put(try l.proj(tensors.mtp.fc_embedding)),
            .fc_h = try l.put(try l.proj(tensors.mtp.fc_hidden)),
            .layer = try l.layer(tensors.mtp.layer0, c, c.mtp.kinds[0]),
            .mix = try l.hc(tensors.mtp.mixer, false),
            .dims = try model.Dims.of(c, 1),
            .draft = try l.put(draft_rows),
        };
    }
    return w;
}
