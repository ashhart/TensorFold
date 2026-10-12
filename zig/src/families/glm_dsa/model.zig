//! GLM-5.3 on a pair of Macs: each layer's share of the weights, the vocabulary, a window's forward and the MTP head's.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("../../core/exlr8.zig");
const dg = @import("../../core/exlr8_dense_gpu.zig");
const pr = @import("../../core/pair.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const ly = @import("layer.zig");
const wt = @import("weights.zig");
const st = ex.safetensors;
const pack = ex.pack;
const Ref = kn.Ref;

pub const Vocab = struct {
    k: gpu.Packed,
    embed_fp8: u64,
    embed_scale: u64,
    groups: [1]gpu.Groups,
    norm: u64,
    /// This Mac's head rows [lo, hi).
    part: [2]u32,
};

/// The MTP head's input (enorm(embedding) | hnorm(hidden)) and the hidden rows it carries to the next draft.
pub const Head = struct {
    cat: mtl.Buffer,
    prev: mtl.Buffer,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    c: cfg.Config,
    K: ly.Kernels,
    layers: []ly.Layer,
    mtp: ?ly.Layer,
    vocab: Vocab,
    pair: *pr.Pair,
    /// This Mac's logits for a window, and the MTP head's buffers.
    logits: mtl.Buffer,
    head: Head,
    /// Bytes this Mac holds on the GPU (weights and caches), and the most it may.
    bytes: u64 = 0,
    limit: u64,
    device: mtl.Device,

    /// This rank's share of `dir`'s first `layers` layers (and the MTP layer), refused before passing `limit` bytes.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: cfg.Config, layers: u32, mtp: bool, context: u32, slots: u32, pair: *pr.Pair, limit: u64) !Model {
        if (layers == 0 or layers > c.layers) return error.BadGlmLayers;
        var m: Model = .{ .gpa = gpa, .c = c, .K = try ly.Kernels.load(gpa, device, &c), .layers = try gpa.alloc(ly.Layer, layers), .mtp = null, .vocab = undefined, .pair = pair, .logits = undefined, .head = undefined, .limit = limit, .device = device };
        var manifest = try ex.manifest.Manifest.load(gpa, io, try std.fmt.allocPrint(gpa, "{s}/manifests/exlr8-manifest-home.json", .{dir}));
        defer manifest.deinit();
        for (m.layers, 0..) |*l, n| l.* = try m.loadLayer(io, device, dir, &manifest, @intCast(n), context, slots);
        if (mtp and c.mtp > 0) m.mtp = try m.loadLayer(io, device, dir, &manifest, c.layers, context, slots);
        m.vocab = try loadVocab(gpa, io, device, dir, &m.c);
        const R = gpu.max_rows;
        const n = m.vocab.part[1] - m.vocab.part[0];
        m.logits = try device.buffer(R * @as(usize, n) * 4, mtl.ResourceOptions.shared);
        const P = gpu.prompt_rows;
        m.head = .{ .cat = try device.buffer(P * 2 * @as(usize, c.hidden) * 4, mtl.ResourceOptions.shared), .prev = try device.buffer(P * @as(usize, c.hidden) * 4, mtl.ResourceOptions.shared) };
        m.bytes += m.vocab.k.bytes.length() + m.logits.length();
        return m;
    }

    fn loadLayer(m: *Model, io: std.Io, device: mtl.Device, dir: []const u8, manifest: *const ex.manifest.Manifest, n: u32, context: u32, slots: u32) !ly.Layer {
        var p = pack.Pack.init(m.gpa);
        defer p.deinit();
        var xl: ?ex.experts.Layer = if (m.c.isMoe(n)) try ex.experts.Layer.open(m.gpa, io, dir, @intCast(n)) else null;
        defer if (xl) |*x| x.close(io);
        const w = try wt.load(m.gpa, io, dir, &m.c, manifest, n, &p, if (xl) |*x| x else null);
        const caches = @as(u64, slots) * context * (576 + if (w.full) @as(u64, 128) else 0) * 2;
        if (m.device.allocated() + p.bytes.items.len + caches > m.limit) {
            std.log.err("glm_dsa: layer {d} would take this Mac past its load limit ({d:.1} of {d:.1} GB)", .{ n, @as(f64, @floatFromInt(m.device.allocated() + p.bytes.items.len + caches)) / 1e9, @as(f64, @floatFromInt(m.limit)) / 1e9 });
            return error.OverLoadLimit;
        }
        const l = try ly.Layer.init(m.gpa, device, w, &p, slots, context, mtl.ResourceOptions.private);
        m.bytes += l.k.bytes.length() + l.cache.length() + if (l.index_cache) |b| b.length() else 0;
        return l;
    }

    fn loadVocab(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: *const cfg.Config) !Vocab {
        var f = try st.File.open(gpa, io, try std.fmt.allocPrint(gpa, "{s}/dense/exlr8-vocab.safetensors", .{dir}));
        defer f.close(io);
        var p = pack.Pack.init(gpa);
        defer p.deinit();
        const emb = try ex.dense.Matrix.get(&f, "embed_tokens");
        if (emb.record.format != .fp8_row or emb.record.in != c.hidden) return error.UnsupportedGlmEmbedding;
        const part = c.vocabPart();
        var v: Vocab = .{ .k = undefined, .embed_fp8 = try p.vector(emb.data[0].bytes), .embed_scale = try p.vector(emb.data[1].bytes), .groups = undefined, .norm = try p.vector((f.get("norm.weight") orelse return error.MissingGlmTensor).bytes), .part = part };
        const head = try p.linear(try ex.dense.Matrix.get(&f, "lm_head"), .{ part[0], part[1] }, .{ 0, c.hidden });
        if (head != .rows) return error.UnsupportedGlmHead;
        v.k = try gpu.Packed.init(gpa, device, &p, &.{.{ .lo = head.rows.job, .hi = head.rows.job + 1 }}, &v.groups);
        return v;
    }

    pub fn glue(m: *const Model, x_stride: u32, y_stride: u32) kn.GlueArgs {
        return .{ .width = m.c.hidden, .x_stride = x_stride, .y_stride = y_stride, .eps = m.c.eps };
    }

    /// One decoder layer: attention, the exchange, the MLP or the experts, the exchange into `next`'s norm.
    fn layer(m: *Model, L: *const ly.Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, next: Ref) !void {
        const p = m.pair;
        const hid = m.c.hidden;
        try ly.attention(&m.K, L, s, e, &m.c, rows, last_pos, null, .{ .buf = s.x }, p.part());
        e.barrier();
        p.addRms(e, .{ .buf = s.h }, L.k.at(L.w.post_norm), .{ .buf = s.x }, rows, hid, m.c.eps);
        e.barrier();
        const o = p.part();
        if (L.w.mlp == .moe) {
            try ly.route(&m.K, L, s, e, &m.c, rows, .{ .buf = s.x }, null, null);
            try ly.expertsGpu(&m.K, L, s, e, &m.c, rows, o);
        } else try ly.denseMlp(&m.K, L, s, e, &m.c, rows, .{ .buf = s.x }, o);
        e.barrier();
        p.addRms(e, .{ .buf = s.h }, next, .{ .buf = s.x }, rows, hid, m.c.eps);
        e.barrier();
    }

    /// The window's rows through every backbone layer: s.h the last hidden state, s.x it after the final norm.
    pub fn backbone(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref) !void {
        const hid = m.c.hidden;
        const v = &m.vocab;
        kn.embed(&m.K.k, e, ids, v.k.at(v.embed_fp8), v.k.at(v.embed_scale), m.glue(hid, hid), .{ .buf = s.h }, rows);
        e.barrier();
        const L0 = &m.layers[0];
        kn.addRms(&m.K.k, e, .{ .buf = s.h }, .{ .buf = s.y }, L0.k.at(L0.w.in_norm), m.glue(hid, hid), .{ .buf = s.x }, rows);
        e.barrier();
        for (m.layers, 0..) |*L, i| {
            const next = if (i + 1 < m.layers.len) m.layers[i + 1].k.at(m.layers[i + 1].w.in_norm) else v.k.at(v.norm);
            try m.layer(L, s, e, rows, last_pos, next);
        }
    }

    /// Each row's next token over the whole vocabulary from normed rows `x` (1-16; the two Macs' halves) into `picks`.
    pub fn pick(m: *Model, e: mtl.ComputeEncoder, x: Ref, rows: u32, picks: Ref) !void {
        const v = &m.vocab;
        const n = v.part[1] - v.part[0];
        try dg.encodeProject(&m.K.dg, e, x, v.k.dense(v.groups[0]), .{ .buf = m.logits }, rows, m.c.hidden, n, true);
        e.barrier();
        m.pair.argmax(e, .{ .buf = m.logits }, n, v.part[0], n, rows, picks);
        e.barrier();
    }

    /// The MTP layer on token ids[r] and row r of `hidden`: s.h its output (the next draft's), s.x that normed.
    pub fn draft(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref, hidden: Ref) !void {
        const L = &(m.mtp orelse return error.NoMtpHead);
        const t = L.w.mtp.?;
        const hid = m.c.hidden;
        const v = &m.vocab;
        kn.embed(&m.K.k, e, ids, v.k.at(v.embed_fp8), v.k.at(v.embed_scale), m.glue(hid, hid), .{ .buf = s.h }, rows);
        e.barrier();
        kn.addRms(&m.K.k, e, .{ .buf = s.h }, .{ .buf = s.y }, L.k.at(t.enorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat }, rows);
        kn.addRms(&m.K.k, e, hidden, .{ .buf = s.y }, L.k.at(t.hnorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat, .off = hid * 4 }, rows);
        e.barrier();
        try ly.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], .{ .buf = m.head.cat }, 2 * hid, .{ .buf = s.rot }, .{ .buf = s.h }, hid, rows);
        e.barrier();
        kn.addRms(&m.K.k, e, .{ .buf = s.h }, .{ .buf = s.y }, L.k.at(L.w.in_norm), m.glue(hid, hid), .{ .buf = s.x }, rows);
        e.barrier();
        try m.layer(L, s, e, rows, last_pos, L.k.at(t.head_norm));
    }

    pub fn deinit(m: *Model) void {
        for (m.layers) |*l| l.deinit(m.gpa);
        if (m.mtp) |*l| l.deinit(m.gpa);
        m.vocab.k.deinit(m.gpa);
        m.logits.deinit();
        m.head.cat.deinit();
        m.head.prev.deinit();
        m.K.deinit();
    }
};
