//! ZONOS3 Mini's weights as published (model.pth, bf16) on the device; the speaker and condition rows stay on the host.
const std = @import("std");
const cuda = @import("cuda");
const pth = @import("torch_pth");
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
    bufs: std.ArrayList(cuda.DeviceBuffer),
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

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        w.gpa.free(w.speaker_w);
        w.gpa.free(w.speaker_b);
        for (w.conditions) |c| w.gpa.free(c);
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
    d: *const cuda.Driver,
    a: *const pth.Archive,
    w: *Weights,

    fn device(l: *Loader, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(l.d, @max(bytes.len, 256));
        errdefer b.free();
        try b.upload(0, bytes);
        try l.w.bufs.append(l.gpa, b);
        l.w.bytes += bytes.len;
        return b.ptr;
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

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: cfg.Config) !*Weights {
    const path = try std.fs.path.join(gpa, &.{ dir, "model.pth" });
    defer gpa.free(path);
    var archive = try pth.Archive.open(gpa, io, path);
    defer archive.close();
    const w = try gpa.create(Weights);
    errdefer gpa.destroy(w);
    w.* = undefined;
    w.gpa = gpa;
    w.bufs = .empty;
    w.bytes = 0;
    w.speaker_w = &.{};
    w.speaker_b = &.{};
    w.conditions = @splat(&.{});
    errdefer w.deinit();
    var l: Loader = .{ .gpa = gpa, .d = d, .a = &archive, .w = w };
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
        }
        const raw = try archive.readRaw(gpa, "multi_embedder.embedders.10.weight", .bf16, &.{ w.text_rows, D });
        defer gpa.free(raw);
        @memcpy(all[cfg.codebooks * audio ..], raw);
        w.tables = try l.device(all);
    }
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
