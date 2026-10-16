//! Nemotron-H's checkpoint onto a HIP rank: each tensor read from the mapped shards, cut to the rank, uploaded.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("../config.zig");
const draft_ids = @import("../draft_ids.zig");
const tensors = @import("tensors.zig");
const model = @import("model.zig");

const quant = core.quant;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;
const Host = quant.Host;
const Span = quant.Span;
const Allocator = std.mem.Allocator;

/// The table core/quant reads projections through: one width for every matrix, as admission checked.
const Tables = struct {
    ck: *Checkpoint,
    w: quant.mlx.Width,

    pub fn has(t: *const Tables, key: []const u8) bool {
        return t.ck.has(key);
    }

    pub fn get(t: *const Tables, key: []const u8) !Tensor {
        return t.ck.get(key);
    }

    pub fn width(t: *const Tables, _: []const u8) !quant.mlx.Width {
        return t.w;
    }
};

/// The device copy of one rank's weights: every buffer it uploaded, freed together.
pub const Weights = struct {
    gpa: Allocator,
    buffers: std.ArrayList(hip.DeviceBuffer) = .empty,
    blocks: []model.Block = &.{},
    model: model.Model = undefined,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.blocks);
        w.* = undefined;
    }

    /// Device bytes of every buffer the load made.
    pub fn bytes(w: *const Weights) u64 {
        var n: u64 = 0;
        for (w.buffers.items) |b| n += b.len;
        return n;
    }
};

const Loader = struct {
    a: Allocator,
    t: *const Tables,
    up: hip.Upload,
    d: model.Dims,

    fn key(l: *const Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(l.a, fmt, args);
    }

    fn proj(l: *const Loader, k: []const u8) !Host {
        return quant.read(l.a, l.t, k);
    }

    fn put(l: *const Loader, h: Host, partial: bool) !model.Projection {
        var p = try quant.view(try quant.upload(l.up.uploader(), h), false);
        p.partial = partial;
        return p;
    }

    fn tensor(l: *const Loader, t: Tensor) !u64 {
        return (try l.up.uploader().tensor(t)).ptr;
    }

    /// A float vector or matrix widened to fp32 (rows `cut` of it when given), on the device.
    fn float(l: *const Loader, k: []const u8, cut: ?[]const Span) !u64 {
        var t = try quant.convert.float32(l.a, try l.t.get(k));
        if (cut) |s| t = try quant.slice.takeRows(l.a, t, s);
        return l.tensor(t);
    }

    fn cutting(l: *const Loader) bool {
        return l.d.world > 1;
    }

    /// Rows `[base + n r, base + n (r + 1))` of each segment of `sizes` for this rank, every segment split evenly.
    fn spans(l: *const Loader, sizes: []const usize, out: []Span) []const Span {
        var base: usize = 0;
        for (sizes, out[0..sizes.len]) |size, *s| {
            const part = size / l.d.world;
            s.* = .{ .from = base + l.d.rank * part, .to = base + (l.d.rank + 1) * part };
            base += size;
        }
        return out[0..sizes.len];
    }

    fn rows(l: *const Loader, h: Host, sizes: []const usize) !Host {
        if (!l.cutting()) return h;
        var buf: [8]Span = undefined;
        return quant.sliceRows(l.a, h, l.spans(sizes, &buf));
    }

    fn cols(l: *const Loader, h: Host, what: []const u8) !Host {
        if (!l.cutting()) return h;
        return quant.sliceCols(l.a, h, .{ .rank = l.d.rank, .world = l.d.world }, what);
    }

    /// in_proj by segment (z, x, B, C, dt), out_proj by input groups, the conv's channels (x, B, C) and heads.
    fn mamba(l: *const Loader, stem: []const u8, c: *const config.Config) !model.Mamba {
        const t = tensors.mamba;
        const inner = c.mamba_heads * c.mamba_head_dim;
        const bc = c.groups * c.state;
        var cb: [8]Span = undefined;
        const conv = if (l.cutting()) l.spans(&.{ inner, bc, bc }, &cb) else null;
        var hb: [1]Span = undefined;
        const heads = if (l.cutting()) l.spans(&.{c.mamba_heads}, &hb) else null;
        var ib: [1]Span = undefined;
        const gnorm = if (l.cutting()) l.spans(&.{inner}, &ib) else null;
        var taps = try quant.convert.float32(l.a, try l.t.get(try l.key("{s}.mixer.{s}", .{ stem, t.conv_w })));
        // MLX stores (C, K, 1), torch (C, 1, K): the same bytes, channel-major
        if (taps.rank != 3 or taps.shape[1] * taps.shape[2] != c.conv_kernel) return error.UnexpectedTensor;
        taps.rank = 2;
        taps.shape[1] = c.conv_kernel;
        if (conv) |s| taps = try quant.slice.takeRows(l.a, taps, s);
        return .{
            .in_proj = try l.put(
                try l.rows(
                    try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.in_proj })),
                    &.{ inner, inner, bc, bc, c.mamba_heads },
                ),
                false,
            ),
            .out_proj = try l.put(
                try l.cols(try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.out_proj })), "out_proj groups"),
                l.cutting(),
            ),
            .conv_w = try l.tensor(taps),
            .conv_b = try l.float(try l.key("{s}.mixer.{s}", .{ stem, t.conv_b }), conv),
            .a_log = try l.float(try l.key("{s}.mixer.{s}", .{ stem, t.a_log }), heads),
            .d = try l.float(try l.key("{s}.mixer.{s}", .{ stem, t.d }), heads),
            .dt_bias = try l.float(try l.key("{s}.mixer.{s}", .{ stem, t.dt_bias }), heads),
            .gnorm = try l.float(try l.key("{s}.mixer.{s}", .{ stem, t.norm }), gnorm),
        };
    }

    /// A rank's query heads, the kv head they read and its share of o's inputs; `whole` for the MTP head's.
    fn attention(l: *const Loader, stem: []const u8, c: *const config.Config, whole: bool) !model.Attention {
        const t = tensors.attention;
        const hd = c.head_dim;
        const q = try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.q }));
        const k = try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.k }));
        const v = try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.v }));
        const o = try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.o }));
        if (whole or !l.cutting()) return .{
            .q = try l.put(q, false),
            .k = try l.put(k, false),
            .v = try l.put(v, false),
            .o = try l.put(o, false),
        };
        const w = l.d.world;
        const r = l.d.rank;
        const heads = c.heads / w;
        const q_span = [_]Span{.{ .from = r * heads * hd, .to = (r + 1) * heads * hd }};
        const kv = r * c.kv_heads / @max(w, c.kv_heads);
        const kv_n = l.d.kv_heads;
        const kv_span = [_]Span{.{ .from = kv * kv_n * hd, .to = (kv + 1) * kv_n * hd }};
        return .{
            .q = try l.put(try quant.sliceRows(l.a, q, &q_span), false),
            .k = try l.put(try quant.sliceRows(l.a, k, &kv_span), false),
            .v = try l.put(try quant.sliceRows(l.a, v, &kv_span), false),
            .o = try l.put(try l.cols(o, "o_proj groups"), true),
        };
    }

    /// A routed stack (experts, n, k) as stored, this rank's run of experts when cut.
    fn stack(l: *const Loader, k: []const u8, cut: ?Span) !model.Projection {
        var parts: [3]Tensor = undefined;
        for (&parts, tensors.parts) |*p, s| {
            p.* = try l.t.get(try l.key("{s}.{s}", .{ k, s }));
            if (p.rank != 3) return error.UnexpectedTensor;
            if (cut) |span| p.* = try quant.slice.takeRows(l.a, p.*, &.{span});
        }
        if (parts[0].dtype != .u32) return error.UnsupportedQuantization;
        parts[0].dtype = .i32;
        const pair = try quant.mlx.tables(l.a, parts[1], parts[2]);
        const h: Host = .{ .mlx = .{
            .words = parts[0],
            .scales = pair[0],
            .biases = pair[1],
            .bits = l.t.w.bits,
            .group = l.t.w.group,
        } };
        return quant.view(try quant.upload(l.up.uploader(), h), true);
    }

    /// The router (fp32) on every rank, this rank's experts, the shared expert on rank 0; `whole` for the MTP head's.
    fn moe(l: *const Loader, ck: *Checkpoint, stem: []const u8, whole: bool) !model.Moe {
        const t = tensors.moe;
        const cut: ?Span = if (whole or !l.cutting()) null else .{
            .from = l.d.first_expert,
            .to = l.d.first_expert + l.d.rank_experts,
        };
        const shared = whole or l.d.rank == 0;
        if (!shared) try ck.skip(try l.key("{s}.mixer.shared_experts.", .{stem}));
        return .{
            .router = try l.float(try l.key("{s}.mixer.{s}", .{ stem, t.router }), null),
            .bias = try l.tensor(try l.t.get(try l.key("{s}.mixer.{s}", .{ stem, t.bias }))),
            .fc1 = try l.stack(try l.key("{s}.mixer.{s}", .{ stem, t.fc1 }), cut),
            .fc2 = try l.stack(try l.key("{s}.mixer.{s}", .{ stem, t.fc2 }), cut),
            .shared_up = if (shared) try l.put(
                try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.shared_up })),
                false,
            ) else null,
            .shared_down = if (shared) try l.put(
                try l.proj(try l.key("{s}.mixer.{s}", .{ stem, t.shared_down })),
                false,
            ) else null,
        };
    }
};

/// The draft ids as runs of the vocabulary head's rows, in the ids' order.
fn draftSpans(a: Allocator, ids: []const u32) ![]Span {
    var list: std.ArrayList(Span) = .empty;
    for (ids) |id| {
        if (list.items.len > 0 and list.items[list.items.len - 1].to == id) {
            list.items[list.items.len - 1].to += 1;
        } else try list.append(a, .{ .from = id, .to = id + 1 });
    }
    return list.toOwnedSlice(a);
}

/// The MTP head from mtp-4bit.safetensors, whole, and the vocabulary head's draft rows.
fn loadMtp(gpa: Allocator, io: std.Io, l: *Loader, dir: []const u8, c: *const config.Config, head: Host) !model.Mtp {
    var ck: Checkpoint = .{ .gpa = gpa, .io = io };
    defer ck.close();
    try ck.add(dir, tensors.mtp_file);
    const t: Tables = .{ .ck = &ck, .w = l.t.w };
    const main_t = l.t;
    l.t = &t;
    defer l.t = main_t;
    const m = tensors.mtp;
    const eh = try quant.halves(l.a, try l.proj(m.eh_proj));
    const ids = try draft_ids.parse(l.a, @embedFile("../draft_ids.txt"), c.vocab);
    const ids32 = try l.a.alloc(i32, ids.len);
    for (ids32, ids) |*x, id| x.* = @intCast(id);
    const out: model.Mtp = .{
        .enorm = try l.float(m.enorm, null),
        .hnorm = try l.float(m.hnorm, null),
        .fc_e = try l.put(eh[0], false),
        .fc_h = try l.put(eh[1], false),
        .attn_norm = try l.float(m.attn_norm, null),
        .attn = try l.attention("layers.0", c, true),
        .moe_norm = try l.float(m.moe_norm, null),
        .moe = try l.moe(&ck, "layers.1", true),
        .final_norm = try l.float(m.final_norm, null),
        .draft = try l.put(try quant.sliceRows(l.a, head, try draftSpans(l.a, ids)), false),
        .draft_ids = try l.tensor(.{
            .dtype = .i32,
            .rank = 1,
            .shape = .{ ids.len, 1, 1, 1, 1 },
            .bytes = std.mem.sliceAsBytes(ids32),
        }),
        .draft_count = ids.len,
        .dims = l.d.whole(c),
    };
    if (ck.unused() != 0) return error.UnusedMtpTensors;
    return out;
}

/// What a load takes: the rank, and whether rank 0 reads the MTP head.
pub const Options = struct { world: usize = 1, rank: usize = 0, mtp: bool = true, act: model.Kind = .bf16 };

/// One rank's weights from the model folder `dir` (admitted before): blocks by kind, the head, the MTP head on rank 0.
pub fn load(
    gpa: Allocator,
    io: std.Io,
    r: *const hip.Runtime,
    dir: []const u8,
    c: *const config.Config,
    o: Options,
) !Weights {
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const tables: Tables = .{ .ck = &ck, .w = .{ .bits = @intCast(c.bits), .group = @intCast(c.group_size) } };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const dims = try model.Dims.of(c, o.world, o.rank);
    var l: Loader = .{
        .a = arena.allocator(),
        .t = &tables,
        .up = .{ .r = r, .buffers = &w.buffers, .gpa = gpa },
        .d = dims,
    };
    w.blocks = try gpa.alloc(model.Block, c.layers);
    for (c.kinds[0..c.layers], 0..) |kind, i| {
        _ = arena.reset(.retain_capacity);
        var buf: [64]u8 = undefined;
        const stem = try tensors.layer(&buf, i);
        w.blocks[i] = .{
            .norm = try l.float(try l.key("{s}.norm.weight", .{stem}), null),
            .mixer = switch (kind) {
                .mamba => .{ .mamba = try l.mamba(stem, c) },
                .attention => .{ .attention = try l.attention(stem, c, false) },
                .moe => .{ .moe = try l.moe(&ck, stem, false) },
            },
        };
    }
    _ = arena.reset(.retain_capacity);
    const head = try l.proj(tensors.lm_head);
    w.model = .{
        .dims = dims,
        .act = o.act,
        .embed = try l.put(try l.proj(tensors.embed), false),
        .blocks = w.blocks,
        .norm_f = try l.float(tensors.norm_f, null),
        .head = try l.put(try l.rows(head, &.{c.vocab}), false),
    };
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    if (o.mtp and o.rank == 0) w.model.mtp = try loadMtp(gpa, io, &l, dir, c, head);
    return w;
}

test "draft ids become runs of the head's rows in their order" {
    const spans = try draftSpans(std.testing.allocator, &.{ 4, 5, 6, 9, 2, 3 });
    defer std.testing.allocator.free(spans);
    try std.testing.expectEqual(@as(usize, 3), spans.len);
    try std.testing.expectEqual(Span{ .from = 4, .to = 7 }, spans[0]);
    try std.testing.expectEqual(Span{ .from = 2, .to = 4 }, spans[2]);
}
