//! ZONOS3 Mini's weights as published (model.pth, bf16) in Metal buffers; speaker and condition rows stay on the host.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;
const pth = @import("core").torch_pth;
const cfg = @import("../config.zig");

const D = cfg.dim;

pub const Gdn = struct { in_proj: u64, conv: u64, out_weight: u64, a_log: u64, dt_bias: u64, wo: u64 };
pub const Attn = struct { wq: u64, wkv: u64, wo: u64, temp: u64, gater: u64 };
pub const Dense = struct { w_in: u64, w_out: u64 };
pub const Moe = struct { gate_up: u64, down: u64, latent_down: u64, latent_up: u64, router: u64, eda: u64, bias: u64 };

pub const Layer = struct {
    attention_norm: u64,
    ffn_norm: u64,
    mix: union(enum) { gdn: Gdn, attn: Attn },
    ffn: union(enum) { dense: Dense, moe: Moe },
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bytes: usize,
    tables: u64, // bf16 [10 * audio_vocab + text_rows][D]: the ten audio tables, then the text table
    text_rows: usize,
    queries: u64, // bf16 [sources][D]
    layers: [cfg.layers]Layer,
    out_norm: u64,
    head: u64, // bf16 [head_rows][D]
    // host rows that replace a prompt row's summed embedding
    speaker_w: []f32, // [D][speaker_dim]
    speaker_b: []f32, // [D]
    conditions: [cfg.conditions][]f32, // [buckets + 1][D]
    /// The release's extra text rows (phoneme, start and pause tokens) in fp32, ids from `text_rows` on; none: empty.
    extra: []f32 = &.{},
    /// The ten audio tables' padding rows summed in codebook order: a text row's embedding beside its text row.
    pad_sum: [D]f32 = @splat(0),
    /// The first edit-marker text id and how many follow; none without the speech-edit adapter.
    markers: usize = 0,
    marker_count: usize = 0,

    /// The host rows; the buffers go with the device.
    pub fn deinit(w: *Weights) void {
        w.gpa.free(w.speaker_w);
        w.gpa.free(w.speaker_b);
        for (w.conditions) |c| w.gpa.free(c);
        w.gpa.free(w.extra);
    }

    /// A text row whose id is an extra one: the padding rows' sum plus its row, the embedding sum's last add.
    pub fn extraRow(w: *const Weights, id: usize, out: *[D]f32) void {
        for (out, w.pad_sum, w.extra[(id - w.text_rows) * D ..][0..D]) |*o, pad, x| o.* = pad + x;
    }

    /// The edit adapter's marker rows, after the extra ones: a prompt's j-th masked span is id markers + j.
    pub fn addMarkers(w: *Weights, rows: []const f32) !void {
        const all = try w.gpa.alloc(f32, w.extra.len + rows.len);
        @memcpy(all[0..w.extra.len], w.extra);
        @memcpy(all[w.extra.len..], rows);
        w.markers = w.text_rows + w.extra.len / D;
        w.marker_count = rows.len / D;
        w.gpa.free(w.extra);
        w.extra = all;
    }

    /// Text ids this model takes: its table's rows, then the extra ones.
    pub fn textIds(w: *const Weights) usize {
        return w.text_rows + w.extra.len / D;
    }

    /// The speaker row: speaker_projection(vec) in fp32.
    pub fn speakerRow(w: *const Weights, vec: *const [cfg.speaker_dim]f32, out: *[D]f32) void {
        for (out, 0..) |*o, n| {
            var acc: f32 = 0;
            for (vec, w.speaker_w[n * cfg.speaker_dim ..][0..cfg.speaker_dim]) |x, m| acc = @mulAdd(f32, m, x, acc);
            o.* = acc + w.speaker_b[n];
        }
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
    a: *const pth.Archive,
    w: *Weights,

    fn device(l: *Loader, bytes: []const u8) !u64 {
        const at = try l.g.upload(bytes);
        l.w.bytes += bytes.len;
        return at;
    }

    fn one(l: *Loader, name: []const u8, shape: []const u64) !u64 {
        const raw = try l.a.readRaw(l.gpa, name, .bf16, shape);
        defer l.gpa.free(raw);
        return l.device(raw);
    }

    fn host(l: *Loader, name: []const u8, shape: []const u64) ![]f32 {
        const raw = try l.a.readRaw(l.gpa, name, .bf16, shape);
        defer l.gpa.free(raw);
        const out = try l.gpa.alloc(f32, raw.len / 2);
        for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16);
        return out;
    }
};

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch unreachable;
}

/// The release's zonos-text-extra/1 rows beside the model, fp32, which follow the text table's; none: empty.
fn extraRows(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, text_rows: usize) ![]f32 {
    const path = try std.fs.path.join(gpa, &.{ dir, extra_rows_file });
    defer gpa.free(path);
    var archive = pth.Archive.openNested(gpa, io, path) catch |e| switch (e) {
        error.FileNotFound => return &.{},
        else => return e,
    };
    defer archive.close();
    const rows = try archive.tensor("rows");
    if (rows.shape.len != 2 or rows.shape[1] != D or rows.shape[0] == 0 or text_rows != cfg.text_vocab + 1) return error.ShapeMismatch;
    const out = try archive.readF32(gpa, "rows", rows.shape);
    for (out) |x| if (!std.math.isFinite(x)) return error.BadExtraTextRows;
    return out;
}

/// Where a release keeps its phoneme, start and pause tokens, under the model's folder.
pub const extra_rows_file = "text_rows/pause_tokens_s5000.pt";

pub fn load(gpa: std.mem.Allocator, io: std.Io, g: *Gpu, dir: []const u8, c: cfg.Config) !*Weights {
    const path = try std.fs.path.join(gpa, &.{ dir, "model.pth" });
    defer gpa.free(path);
    var archive = try pth.Archive.open(gpa, io, path);
    defer archive.close();
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = undefined;
    w.gpa = gpa;
    w.bytes = 0;
    w.speaker_w = &.{};
    w.speaker_b = &.{};
    w.conditions = @splat(&.{});
    w.extra = &.{};
    w.pad_sum = @splat(0);
    w.markers = 0;
    w.marker_count = 0;
    errdefer w.deinit();
    var l: Loader = .{ .gpa = gpa, .g = g, .a = &archive, .w = w };
    var nb: [160]u8 = undefined;

    // the eleven embedding tables in one buffer
    const text = try archive.tensor("multi_embedder.embedders.10.weight");
    if (text.shape.len != 2 or text.shape[1] != D or text.shape[0] < cfg.text_vocab + 1) return error.ShapeMismatch;
    w.text_rows = @intCast(text.shape[0]);
    {
        const audio = cfg.audio_vocab * D * 2;
        const all = try gpa.alloc(u8, cfg.codebooks * audio + w.text_rows * D * 2);
        defer gpa.free(all);
        for (0..cfg.codebooks) |cb| {
            const raw = try archive.readRaw(gpa, fmt(&nb, "multi_embedder.embedders.{d}.weight", .{cb}), .bf16, &.{ cfg.audio_vocab, D });
            defer gpa.free(raw);
            @memcpy(all[cb * audio ..][0..audio], raw);
            // the sum starts from the first table's row, as the reference's does
            for (&w.pad_sum, 0..) |*o, i| {
                const bits = std.mem.readInt(u16, raw[(cfg.audio_pad_id * D + i) * 2 ..][0..2], .little);
                const x: f32 = @bitCast(@as(u32, bits) << 16);
                o.* = if (cb == 0) x else o.* + x;
            }
        }
        const raw = try archive.readRaw(gpa, "multi_embedder.embedders.10.weight", .bf16, &.{ w.text_rows, D });
        defer gpa.free(raw);
        @memcpy(all[cfg.codebooks * audio ..], raw);
        w.tables = try l.device(all);
    }
    w.extra = try extraRows(gpa, io, dir, w.text_rows);
    w.queries = try l.one("block_attnres_queries", &.{ cfg.sources, D });
    w.out_norm = try l.one("out_norm.weight", &.{D});
    w.head = try l.one("multi_output.weight", &.{ cfg.head_rows, D });
    w.speaker_w = try l.host("speaker_projection.weight", &.{ D, cfg.speaker_dim });
    w.speaker_b = try l.host("speaker_projection.bias", &.{D});
    for (&w.conditions, 0..) |*t, i| t.* = try l.host(fmt(&nb, "condition_embeddings.{d}.weight", .{i}), &.{ c.buckets[i] + 1, D });

    const pre = "layers.{d}.";
    for (&w.layers, 0..) |*L, li| {
        L.attention_norm = try l.one(fmt(&nb, pre ++ "attention_norm.weight", .{li}), &.{D});
        L.ffn_norm = try l.one(fmt(&nb, pre ++ "ffn_norm.weight", .{li}), &.{D});
        const at = pre ++ "attention.";
        if (cfg.usesGdn(li)) {
            L.mix = .{ .gdn = .{
                .in_proj = try l.one(fmt(&nb, at ++ "in_proj.weight", .{li}), &.{ cfg.gdn_proj, D }),
                .conv = try l.one(fmt(&nb, at ++ "conv.weight", .{li}), &.{ cfg.gdn_conv, 1, cfg.gdn_kernel }),
                .out_weight = try l.one(fmt(&nb, at ++ "out_weight", .{li}), &.{cfg.gdn_dim}),
                .a_log = try l.one(fmt(&nb, at ++ "A_log", .{li}), &.{cfg.gdn_heads}),
                .dt_bias = try l.one(fmt(&nb, at ++ "dt_bias", .{li}), &.{cfg.gdn_heads}),
                .wo = try l.one(fmt(&nb, at ++ "wo.weight", .{li}), &.{ D, cfg.gdn_inner }),
            } };
        } else {
            L.mix = .{ .attn = .{
                .wq = try l.one(fmt(&nb, at ++ "wq.weight", .{li}), &.{ cfg.attn_inner, D }),
                .wkv = try l.one(fmt(&nb, at ++ "wkv.weight", .{li}), &.{ 2, cfg.kv_width, D }),
                .wo = try l.one(fmt(&nb, at ++ "wo.weight", .{li}), &.{ D, cfg.attn_inner }),
                .temp = try l.one(fmt(&nb, at ++ "temp", .{li}), &.{ 1, cfg.heads, 1 }),
                .gater = try l.one(fmt(&nb, at ++ "gater.weight", .{li}), &.{ cfg.heads, D }),
            } };
        }
        const ff = pre ++ "feed_forward.";
        if (li == 0) {
            L.ffn = .{ .dense = .{
                .w_in = try l.one(fmt(&nb, ff ++ "w_in.weight", .{li}), &.{ 2, cfg.ffn_hidden, D }),
                .w_out = try l.one(fmt(&nb, ff ++ "w_out.weight", .{li}), &.{ D, cfg.ffn_hidden }),
            } };
        } else {
            const bias = try archive.readRaw(gpa, fmt(&nb, ff ++ "router.balancing_biases", .{li}), .f32, &.{cfg.experts});
            defer gpa.free(bias);
            L.ffn = .{ .moe = .{
                .gate_up = try l.one(fmt(&nb, ff ++ "experts.gate_up_proj", .{li}), &.{ cfg.experts, 2 * cfg.moe_hidden, cfg.moe_latent }),
                .down = try l.one(fmt(&nb, ff ++ "experts.down_proj", .{li}), &.{ cfg.experts, cfg.moe_latent, cfg.moe_hidden }),
                .latent_down = try l.one(fmt(&nb, ff ++ "latent_down.weight", .{li}), &.{ cfg.moe_latent, D }),
                .latent_up = try l.one(fmt(&nb, ff ++ "latent_up.weight", .{li}), &.{ D, cfg.moe_latent }),
                .router = try l.one(fmt(&nb, ff ++ "router.router_linear.weight", .{li}), &.{ cfg.experts, D }),
                .eda = try l.one(fmt(&nb, ff ++ "router.rmsnorm_eda.weight", .{li}), &.{D}),
                .bias = try l.device(bias),
            } };
        }
    }
    return w;
}
