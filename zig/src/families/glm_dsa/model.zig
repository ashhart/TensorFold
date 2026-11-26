//! GLM-5.3 on 1..8 Macs (TP, PP, hybrid, heterogeneous): EXLR8 or native HF FP8 block-128 weights.
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
const hfw = @import("hf_weights.zig");
const st = ex.safetensors;
const pack = ex.pack;
const Ref = kn.Ref;

pub const Vocab = struct {
    k: gpu.Packed,
    embed_fp8: u64 = 0,
    embed_scale: u64 = 0,
    embed_bf16: u64 = 0,
    lm_bf16_off: u64 = 0,
    lm_q4_w: ?mtl.Buffer = null,
    lm_q4_s: ?mtl.Buffer = null,
    lm_is_q8: bool = false,
    is_bf16: bool = false,
    groups: [1]gpu.Groups,
    norm: u64 = 0,
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
    is_fp8: bool = false,
    /// This Mac's logits for a window, and the MTP head's buffers.
    logits: mtl.Buffer,
    head: Head,
    /// Bytes this Mac holds on the GPU (weights and caches), and the most it may.
    bytes: u64 = 0,
    limit: u64,
    device: mtl.Device,
    mtp_gate_thresh: f32 = 0.26,

    /// Load this rank's slice of `dir` (auto-detecting native HF FP8 vs EXLR8), refused past `limit` bytes.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: cfg.Config, mtp: bool, context: u32, slots: u32, pair: *pr.Pair, limit: u64) !Model {
        const n_local = c.localLayers();
        if (n_local == 0 or c.layer1 > c.layers) return error.BadGlmLayers;
        const is_hf = hfw.isHfCheckpoint(io, gpa, dir);
        var m: Model = .{
            .gpa = gpa,
            .c = c,
            .K = try ly.Kernels.load(gpa, device, &c),
            .layers = try gpa.alloc(ly.Layer, n_local),
            .mtp = null,
            .vocab = undefined,
            .pair = pair,
            .is_fp8 = is_hf,
            .logits = undefined,
            .head = undefined,
            .limit = limit,
            .device = device,
        };
        const want_mtp = mtp and c.mtp > 0 and c.pp == 1;
        if (is_hf) {
            const bq = try device.queue();
            defer bq.deinit();
            var store = try hfw.Store.init(gpa, io, dir);
            defer store.deinit();
            for (m.layers, 0..) |*l, idx| {
                const n = c.layer0 + @as(u32, @intCast(idx));
                l.* = try m.loadLayerHf(device, bq, &store, n, context, slots);
                if ((idx + 1) % 10 == 0 or idx + 1 == m.layers.len) {
                    std.log.info("glm_dsa rank {d}/{d}: loaded HF FP8 layer {d}/{d} ({d:.1} GiB)", .{
                        c.rank,
                        c.ranks,
                        idx + 1,
                        m.layers.len,
                        @as(f64, @floatFromInt(m.bytes)) / (1024.0 * 1024.0 * 1024.0),
                    });
                }
            }
            if (want_mtp) m.mtp = try m.loadLayerHf(device, bq, &store, c.layers, context, slots);
            const v = try hfw.loadVocab(gpa, device, bq, &store, &m.c, c.isFirstStage() or want_mtp, c.isLastStage(), want_mtp);
            m.vocab = .{ .k = v.k, .embed_bf16 = v.embed_bf16, .lm_bf16_off = v.lm_bf16_off, .lm_q4_w = v.lm_q4_w, .lm_q4_s = v.lm_q4_s, .lm_is_q8 = v.lm_is_q8, .is_bf16 = true, .groups = v.groups, .norm = v.norm, .part = v.part };
        } else {
            var manifest = try ex.manifest.Manifest.load(gpa, io, try std.fmt.allocPrint(gpa, "{s}/manifests/exlr8-manifest-home.json", .{dir}));
            defer manifest.deinit();
            for (m.layers, 0..) |*l, idx| {
                const n = c.layer0 + @as(u32, @intCast(idx));
                l.* = try m.loadLayerExlr8(io, device, dir, &manifest, n, context, slots);
            }
            if (want_mtp) m.mtp = try m.loadLayerExlr8(io, device, dir, &manifest, c.layers, context, slots);
            m.vocab = try loadVocabExlr8(gpa, io, device, dir, &m.c);
        }
        const R = gpu.max_rows;
        const n = m.vocab.part[1] - m.vocab.part[0];
        m.logits = try device.buffer(@max(@as(usize, 16), R * @as(usize, n) * 4), mtl.ResourceOptions.shared);
        const P = gpu.prompt_rows;
        m.head = .{
            .cat = try device.buffer(P * 2 * @as(usize, c.hidden) * 4, mtl.ResourceOptions.shared),
            .prev = try device.buffer(P * @as(usize, c.hidden) * 4, mtl.ResourceOptions.shared),
        };
        m.bytes += m.vocab.k.bytes.length() + m.logits.length() + (if (m.vocab.lm_q4_w) |b| b.length() else 0) + (if (m.vocab.lm_q4_s) |b| b.length() else 0);
        return m;
    }

    fn loadLayerHf(m: *Model, device: mtl.Device, bq: mtl.Queue, store: *hfw.Store, n: u32, context: u32, slots: u32) !ly.Layer {
        var p = pack.Pack.init(m.gpa);
        defer p.deinit();
        const loaded = try hfw.loadLayer(m.gpa, device, store, &m.c, n, &p);
        const caches = @as(u64, slots) * context * (576 + if (loaded.layer.full) @as(u64, 128) else 0) * 2;
        const moe_bytes: u64 = if (loaded.fp8_moe) |*fm| fm.bytes() else 0;
        if (m.device.allocated() + p.bytes.items.len + caches > m.limit) {
            std.log.err("glm_dsa: layer {d} would exceed load limit ({d:.1} of {d:.1} GiB)", .{
                n,
                @as(f64, @floatFromInt(m.device.allocated() + p.bytes.items.len + caches)) / (1024.0 * 1024.0 * 1024.0),
                @as(f64, @floatFromInt(m.limit)) / (1024.0 * 1024.0 * 1024.0),
            });
            if (loaded.fp8_moe) |fm| {
                var fm_mut = fm;
                fm_mut.deinit();
            }
            return error.OverLoadLimit;
        }
        const l = try ly.Layer.initWithFp8Queue(m.gpa, device, bq, loaded.layer, &p, loaded.fp8_moe, slots, context, mtl.ResourceOptions.private);
        m.bytes += l.k.bytes.length() + moe_bytes + l.cache.length() + if (l.index_cache) |b| b.length() else 0;
        return l;
    }

    fn loadLayerExlr8(m: *Model, io: std.Io, device: mtl.Device, dir: []const u8, manifest: *const ex.manifest.Manifest, n: u32, context: u32, slots: u32) !ly.Layer {
        var p = pack.Pack.init(m.gpa);
        defer p.deinit();
        var xl: ?ex.experts.Layer = if (m.c.isMoe(n)) try ex.experts.Layer.open(m.gpa, io, dir, @intCast(n)) else null;
        defer if (xl) |*x| x.closeCopied(io);
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

    fn loadVocabExlr8(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: *const cfg.Config) !Vocab {
        var f = try st.File.open(gpa, io, try std.fmt.allocPrint(gpa, "{s}/dense/exlr8-vocab.safetensors", .{dir}));
        defer f.closeCopied(io);
        var p = pack.Pack.init(gpa);
        defer p.deinit();
        const emb = try ex.dense.Matrix.get(&f, "embed_tokens");
        if (emb.record.format != .fp8_row or emb.record.in != c.hidden) return error.UnsupportedGlmEmbedding;
        const part = c.vocabPart();
        var v: Vocab = .{ .k = undefined, .embed_fp8 = try p.vector(emb.data[0].bytes), .embed_scale = try p.vector(emb.data[1].bytes), .is_bf16 = false, .groups = undefined, .norm = try p.vector((f.get("norm.weight") orelse return error.MissingGlmTensor).bytes), .part = part };
        const head = try p.linear(try ex.dense.Matrix.get(&f, "lm_head"), .{ part[0], part[1] }, .{ 0, c.hidden });
        if (head != .rows) return error.UnsupportedGlmHead;
        v.k = try gpu.Packed.init(gpa, device, &p, &.{.{ .lo = head.rows.job, .hi = head.rows.job + 1 }}, &v.groups);
        return v;
    }

    pub fn glue(m: *const Model, x_stride: u32, y_stride: u32) kn.GlueArgs {
        return .{ .width = m.c.hidden, .x_stride = x_stride, .y_stride = y_stride, .eps = m.c.eps };
    }

    pub fn embedTokens(m: *const Model, e: mtl.ComputeEncoder, ids: Ref, h: Ref, rows: u32) void {
        m.embedTokensGated(e, ids, h, rows, null);
    }

    pub fn embedTokensGated(m: *const Model, e: mtl.ComputeEncoder, ids: Ref, h: Ref, rows: u32, gate: ?mtl.Buffer) void {
        const hid = m.c.hidden;
        const v = &m.vocab;
        if (v.is_bf16) {
            kn.embedBf16(&m.K.k, e, ids, v.k.at(v.embed_bf16), m.glue(hid, hid), h, rows, gate);
        } else {
            kn.embed(&m.K.k, e, ids, v.k.at(v.embed_fp8), v.k.at(v.embed_scale), m.glue(hid, hid), h, rows, gate);
        }
    }

    /// One decoder layer: attention, TP all-reduce + post-norm, MLP or experts, TP all-reduce + next-norm.
    fn layer(m: *Model, L: *const ly.Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, next: Ref, tree_args: kn.TreeArgs) !void {
        return m.layerGated(L, s, e, rows, last_pos, next, null, null, tree_args, null);
    }

    fn layerGated(m: *Model, L: *const ly.Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, next: Ref, gate: ?mtl.Buffer, gate_meta: ?mtl.Buffer, tree_args: kn.TreeArgs, keep_meta: ?mtl.Buffer) !void {
        const p = m.pair;
        const hid = m.c.hidden;
        const xh_off4: u32 = if (rows >= 2 and rows <= 8) @intCast(gpu.max_rows * (hid / 4)) else 0;
        const pk1 = p.packTarget();
        const attn_packed = try ly.attentionGated(&m.K, L, s, e, &m.c, rows, last_pos, null, .{ .buf = s.x }, p.part(), pk1.info, pk1.base, gate, tree_args);
        e.barrier();
        const xh1 = if (attn_packed)
            p.addRmsPrepackedXhGated(e, .{ .buf = s.h }, L.k.at(L.w.post_norm), .{ .buf = s.x }, rows, hid, m.c.eps, xh_off4, gate_meta)
        else
            p.addRmsXh(e, .{ .buf = s.h }, L.k.at(L.w.post_norm), .{ .buf = s.x }, rows, hid, m.c.eps, xh_off4);
        e.barrier();
        if (xh_off4 != 0 and !xh1) {
            kn.fast.castXh(&m.K.k.fast_p, e, .{ .buf = s.x }, s.xhRef(hid), (rows * hid) / 4);
            e.barrier();
        }
        const o = p.part();
        const pk2 = p.packTarget();
        var mlp_packed = false;
        if (L.w.mlp == .moe) {
            mlp_packed = try ly.moeGpuGated(&m.K, L, s, e, &m.c, rows, o, pk2.info, pk2.base, gate, keep_meta);
        } else {
            mlp_packed = try ly.denseMlpPack(&m.K, L, s, e, &m.c, rows, .{ .buf = s.x }, o, pk2.info, pk2.base);
        }
        e.barrier();
        const xh2 = if (mlp_packed)
            p.addRmsPrepackedXhGated(e, .{ .buf = s.h }, next, .{ .buf = s.x }, rows, hid, m.c.eps, xh_off4, gate_meta)
        else
            p.addRmsXh(e, .{ .buf = s.h }, next, .{ .buf = s.x }, rows, hid, m.c.eps, xh_off4);
        e.barrier();
        if (xh_off4 != 0 and !xh2) {
            kn.fast.castXh(&m.K.k.fast_p, e, .{ .buf = s.x }, s.xhRef(hid), (rows * hid) / 4);
            e.barrier();
        }
    }

    /// The window's rows through this stage's layers (with PP recv/send at stage boundaries).
    pub fn backbone(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref) !void {
        return m.backboneTapTree(s, e, rows, last_pos, ids, null, .{});
    }

    pub fn backboneTap(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref, df2: anytype) !void {
        return m.backboneTapTree(s, e, rows, last_pos, ids, df2, .{});
    }

    pub fn backboneTapTree(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref, df2: anytype, tree_args: kn.TreeArgs) !void {
        const hid = m.c.hidden;
        const v = &m.vocab;
        if (m.c.isFirstStage()) {
            m.embedTokens(e, ids, .{ .buf = s.h }, rows);
        } else {
            m.pair.pipeRecv(e, .{ .buf = s.h }, rows, hid);
        }
        e.barrier();
        const L0 = &m.layers[0];
        var g0 = m.glue(hid, hid);
        if (rows >= 2 and rows <= 8) g0.pad0 = @intCast(gpu.max_rows * (hid / 4));
        kn.addRms(&m.K.k, e, .{ .buf = s.h }, .{ .buf = s.y }, L0.k.at(L0.w.in_norm), g0, .{ .buf = s.x }, rows);
        e.barrier();
        for (m.layers, 0..) |*L, i| {
            const next = if (i + 1 < m.layers.len)
                m.layers[i + 1].k.at(m.layers[i + 1].w.in_norm)
            else if (m.c.isLastStage())
                v.k.at(v.norm)
            else
                L.k.at(L.w.in_norm);
            try m.layer(L, s, e, rows, last_pos, next, tree_args);
            if (@TypeOf(df2) != @TypeOf(null)) if (df2) |d| d.tap(e, m.c.layer0 + @as(u32, @intCast(i)), .{ .buf = s.h }, rows);
        }
        if (!m.c.isLastStage()) {
            m.pair.pipeSend(e, .{ .buf = s.h }, rows, hid);
            e.barrier();
        }
    }

    fn pickImpl(m: *Model, e: mtl.ComputeEncoder, x: Ref, rows: u32, picks: Ref, draft_q4: bool, gate: ?mtl.Buffer, gate_meta: ?mtl.Buffer) !bool {
        var wrote_prob = false;
        if (m.c.isLastStage()) {
            const v = &m.vocab;
            const n = v.part[1] - v.part[0];
            const fused_q4 = draft_q4 and (rows == 1 or rows == 2) and v.lm_q4_w != null and v.lm_q4_s != null and (m.c.hidden % 1024 == 0);
            const fused_bf16 = !fused_q4 and v.is_bf16 and v.lm_bf16_off != 0 and rows >= 1 and rows <= 16 and (m.c.hidden % 512 == 0);
            if (fused_q4 or fused_bf16) {
                const pk = m.pair.reservePick(rows, fused_q4);
                const n_tgs: u32 = (n + 15) / 16;
                if (fused_q4) {
                    kn.fast.lmDraftArgmax(&m.K.k.fast_p, e, x, .{ .buf = v.lm_q4_w.? }, .{ .buf = v.lm_q4_s.? }, .{ n, m.c.hidden, v.part[0], rows }, .{ .buf = m.logits }, n_tgs, v.lm_is_q8, gate);
                } else {
                    kn.fast.lmBf16Argmax(&m.K.k.fast_p, e, x, v.k.at(v.lm_bf16_off), .{ n, m.c.hidden, v.part[0], rows }, .{ .buf = m.logits }, n_tgs);
                }
                e.barrier();
                if (fused_q4) {
                    kn.fast.lmArgmaxProbReduce(&m.K.k.fast_p, e, .{ .buf = m.logits }, n_tgs, pk.send, rows, gate);
                } else {
                    kn.fast.lmArgmaxReduce(&m.K.k.fast_p, e, .{ .buf = m.logits }, n_tgs, pk.send, rows);
                }
                e.barrier();
                m.pair.finishPickGated(e, pk.x, rows, picks, fused_q4, gate_meta);
                e.barrier();
                wrote_prob = fused_q4;
            } else {
                try dg.encodeProject(&m.K.dg, e, x, v.k.denseFor(v.groups[0], rows), .{ .buf = m.logits }, rows, m.c.hidden, n, true);
                e.barrier();
                m.pair.argmax(e, .{ .buf = m.logits }, n, v.part[0], n, rows, picks);
                e.barrier();
            }
            if (m.c.pp > 1) {
                m.pair.returnPicks(e, picks, rows);
                e.barrier();
            }
        } else if (m.c.rank == 0) {
            m.pair.waitPicks(e, picks, rows);
            e.barrier();
        }
        return wrote_prob;
    }

    /// Each row's next token over the whole vocabulary (exact BF16 lm_head + TP argmax; relays to rank 0 if PP>1).
    pub fn pick(m: *Model, e: mtl.ComputeEncoder, x: Ref, rows: u32, picks: Ref) !void {
        _ = try m.pickImpl(e, x, rows, picks, false, null, null);
    }

    /// Draft-only next token using INT4 g64 lm_head; returns true if exact softmax prob was written at `picks[1]`.
    pub fn pickDraft(m: *Model, e: mtl.ComputeEncoder, x: Ref, rows: u32, picks: Ref) !bool {
        return m.pickImpl(e, x, rows, picks, true, null, null);
    }

    pub fn pickDraftGated(m: *Model, e: mtl.ComputeEncoder, x: Ref, rows: u32, picks: Ref, gate: ?mtl.Buffer, gate_meta: ?mtl.Buffer) !bool {
        return m.pickImpl(e, x, rows, picks, true, gate, gate_meta);
    }

    /// Build exact 15 indirect dispatch grids for r=1 MTP step at `last_pos` (used by GPU early-stop `glm_mtp_gate`).
    pub fn mtpGateArgs(m: *const Model, step: u32, last_pos: u32, fill_grids: bool) kn.fast.MtpGateArgs {
        var a: kn.fast.MtpGateArgs = .{ .step = step, .n_grids = if (fill_grids) 15 else 0, .thresh = m.mtp_gate_thresh, .grids = @splat(.{ 0, 0, 0, 0 }) };
        if (!fill_grids) return a;
        const L = &(m.mtp orelse return a);
        const c = &m.c;
        const nh = c.localHeads();
        const hid = c.hidden;
        const loc = c.localHalves() * 128;
        const n_voc = m.vocab.part[1] - m.vocab.part[0];
        var gx_x: u32 = 1;
        if (L.fp8_x_n >= 2) {
            gx_x = (L.fp8_x[0].?.outs + 7) / 8 + (L.fp8_x[1].?.outs + 7) / 8 + (if (L.fp8_x_n >= 3) (L.fp8_x[2].?.outs + 7) / 8 else 0);
        } else if (L.fp8_x_n == 1) gx_x = (L.fp8_x[0].?.outs + 7) / 8;
        const fuse_uv = L.k.fast_dense and c.nope == 192 and c.v_dim == 256 and c.kv_lora == 512;
        const att_grid: [4]u32 = if (last_pos < 256 and nh >= 8 and fuse_uv)
            .{ nh * 4, 1, 1, 0 }
        else blk: {
            const use_flash = last_pos >= 32 and nh >= 8 and (nh % 8 == 0);
            const ns: u32 = if (last_pos < 32) 1 else if (use_flash) (if (last_pos < 128) 8 else 16) else if (last_pos < 512) 8 else kn.splits;
            break :blk .{ if (use_flash) nh / 8 else (nh + 7) / 8, ns, 1, 0 };
        };
        a.grids = .{
            .{ (hid + 15) / 16, 1, 1, 0 },
            .{ gx_x, 1, 1, 0 },
            .{ if (L.fp8_q_b) |q| (q.outs + 7) / 8 else 1, 1, 1, 0 },
            .{ if (L.fp8_wq_b) |q| (q.outs + 7) / 8 else 1, 1, 1, 0 },
            .{ c.i_heads, 1, 1, 0 },
            .{ (hid + 15) / 16, 1, 1, 0 },
            .{ 128, 1, 1, 0 },
            .{ (loc + 7) / 8, gpu.slots, 1, 0 },
            .{ (hid + 15) / 16, 1, 1, 0 },
            .{ (n_voc + 15) / 16, 1, 1, 0 },
            .{ if (last_pos >= 4096) last_pos / 128 + 1 else last_pos / 32 + 1, 1, 1, 0 },
            att_grid,
            .{ nh * 16, 1, 1, 0 },
            .{ 8, nh, 1, 0 },
            .{ 1, 1, 1, 0 },
        };
        return a;
    }

    /// The MTP layer on token ids[r] and row r of `hidden`: s.h its output (the next draft's), s.x that normed.
    pub fn draft(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref, hidden: Ref) !void {
        return m.draftGated(s, e, rows, last_pos, ids, hidden, null, null, null, .{});
    }

    pub fn draftGated(m: *Model, s: *const gpu.Scratch, e: mtl.ComputeEncoder, rows: u32, last_pos: u32, ids: Ref, hidden: Ref, gate: ?mtl.Buffer, gate_meta: ?mtl.Buffer, keep_meta: ?mtl.Buffer, tree_args: kn.TreeArgs) !void {
        const L = &(m.mtp orelse return error.NoMtpHead);
        const t = L.w.mtp.?;
        const hid = m.c.hidden;
        m.embedTokensGated(e, ids, .{ .buf = s.h }, rows, gate);
        e.barrier();
        if ((hid % 4 == 0) and hid >= 4096 and m.K.k.fast_p.mtp_cat_rms.maxThreads() >= 1024) {
            kn.fast.mtpCatRms(&m.K.k.fast_p, e, .{ .buf = s.h }, hidden, L.k.at(t.enorm), L.k.at(t.hnorm), .{ .width = hid, .eps = m.c.eps }, .{ .buf = m.head.cat }, rows, gate);
        } else {
            kn.addRms(&m.K.k, e, .{ .buf = s.h }, .{ .buf = s.y }, L.k.at(t.enorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat }, rows);
            kn.addRms(&m.K.k, e, hidden, .{ .buf = s.y }, L.k.at(t.hnorm), m.glue(hid, 2 * hid), .{ .buf = m.head.cat, .off = hid * 4 }, rows);
        }
        e.barrier();
        if (t.eh_sharded) {
            const p = m.pair;
            const pk = p.packTarget();
            var prepacked = false;
            if (rows == 1 and t.eh_q8_w != 0 and (t.eh_ins % 512 == 0) and (hid % 16 == 0)) {
                kn.fast.ehQ8Pack1(&m.K.k.fast_p, e, .{ .buf = m.head.cat }, L.k.at(t.eh_q8_w), L.k.at(t.eh_q8_s), .{
                    .outs = hid,
                    .ins = t.eh_ins,
                    .x_col = t.eh_x_col,
                    .x_stride = 2 * hid,
                    .rank = pk.info[0],
                    .tp_size = pk.info[1],
                    .slot_u2 = pk.info[2],
                }, p.part(), pk.base, gate);
                prepacked = true;
            } else if (rows >= 2 and rows <= 8 and t.eh_q8_w != 0 and (t.eh_ins % 512 == 0) and (hid % 16 == 0)) {
                kn.fast.ehQ8PackN(&m.K.k.fast_p, e, .{ .buf = m.head.cat }, L.k.at(t.eh_q8_w), L.k.at(t.eh_q8_s), .{
                    .outs = hid,
                    .ins = t.eh_ins,
                    .x_col = t.eh_x_col,
                    .x_stride = 2 * hid,
                    .rank = pk.info[0],
                    .tp_size = pk.info[1],
                    .slot_u2 = pk.info[2],
                    .pad0 = rows,
                }, p.part(), pk.base, keep_meta);
                prepacked = true;
            } else if (rows <= 6 and pk.info[1] > 1 and L.k.fast_dense and t.eh == .rows and (t.eh_ins % 512 == 0) and (hid % 4 == 0)) {
                try dg.encodeProjectPackBf16(&m.K.dg, e, .{ .buf = m.head.cat }, L.k.dense16(L.groups[ly.Layer.eh_g]), p.part(), rows, 2 * hid, hid, pk.info, pk.base);
                prepacked = true;
            } else {
                try ly.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], .{ .buf = m.head.cat }, 2 * hid, .{ .buf = s.rot }, p.part(), hid, rows);
            }
            e.barrier();
            const xh_off4: u32 = if (rows >= 2 and rows <= 8) @intCast(gpu.max_rows * (hid / 4)) else 0;
            const xh_ok = p.addRmsReplaceXhGated(e, .{ .buf = s.h }, L.k.at(L.w.in_norm), .{ .buf = s.x }, rows, hid, m.c.eps, prepacked, xh_off4, gate_meta);
            e.barrier();
            if (xh_off4 != 0 and !xh_ok) {
                kn.fast.castXh(&m.K.k.fast_p, e, .{ .buf = s.x }, s.xhRef(hid), (rows * hid) / 4);
                e.barrier();
            }
        } else {
            try ly.linear(&m.K, L, e, t.eh, L.groups[ly.Layer.eh_g], .{ .buf = m.head.cat }, 2 * hid, .{ .buf = s.rot }, .{ .buf = s.h }, hid, rows);
            e.barrier();
            var g0 = m.glue(hid, hid);
            if (rows >= 2 and rows <= 8) g0.pad0 = @intCast(gpu.max_rows * (hid / 4));
            kn.addRms(&m.K.k, e, .{ .buf = s.h }, .{ .buf = s.y }, L.k.at(L.w.in_norm), g0, .{ .buf = s.x }, rows);
            e.barrier();
        }
        try m.layerGated(L, s, e, rows, last_pos, L.k.at(t.head_norm), gate, gate_meta, tree_args, keep_meta);
    }

    pub fn deinit(m: *Model) void {
        for (m.layers) |*l| l.deinit(m.gpa);
        m.gpa.free(m.layers);
        if (m.mtp) |*l| l.deinit(m.gpa);
        m.vocab.k.deinit(m.gpa);
        if (m.vocab.lm_q4_w) |b| b.deinit();
        if (m.vocab.lm_q4_s) |b| b.deinit();
        m.logits.deinit();
        m.head.cat.deinit();
        m.head.prev.deinit();
        m.K.deinit();
    }
};
