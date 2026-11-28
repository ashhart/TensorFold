//! TP-sharded incoai/GLM-5.3-DFlash2 parallel block-diffusion drafter with optional 32-core ANE `fc` acceleration.
const std = @import("std");
const Io = std.Io;
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const pr = @import("../../core/pair.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const md = @import("model.zig");
const ld = @import("dflash2_load.zig");
const ane_mod = @import("dflash2_ane.zig");
const hybrid = @import("dflash2_hybrid.zig");
const Ref = kn.Ref;

pub const max_layers = ld.max_layers;
pub const block_size = ld.block_size;
pub const draft_tokens = ld.draft_tokens;
pub const num_taps = ld.num_taps;
pub const Config = ld.Config;

const TapArgs = extern struct { rows: u32, hidden: u32, k_lo: u32, k_loc: u32, tap_idx: u32, p0: u32 = 0, p1: u32 = 0, p2: u32 = 0 };
const GemvArgs = extern struct { rows: u32, K: u32, N: u32, out_stride: u32, in_bf16: u32, pack_rank: u32, pack_tp: u32, pack_slot_u2: u32 };
const NormArgs = extern struct { rows: u32, width: u32, add: u32, eps: f32 };
const ConvArgs = extern struct { rows: u32, width: u32, block: u32, groups: u32, side: u32, p0: u32 = 0, p1: u32 = 0, p2: u32 = 0 };
const ConvNormArgs = extern struct { rows: u32, width: u32, block: u32, groups: u32, eps: f32, p0: u32 = 0, p1: u32 = 0, p2: u32 = 0 };
const PrepArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos0: u32, ring: u32, dyn_meta: u32 = 0, eps: f32, theta: f32 };
const AttnArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, gqa: u32, lo: u32, hi: u32, ring: u32, n_splits: u32, scale: f32, dyn_meta: u32 = 0, window: u32 = 0, sl_from: u32 = 0, split_sz: u32 = 128, p0: u32 = 0, p1: u32 = 0, p2: u32 = 0 };
pub const AcceptArgs = extern struct { rows: u32, base_pos: u32, mask_tok: u32, pad0: u32 = 0 };
pub const TreeAcceptArgs = extern struct { rows: u32, base_pos: u32, mask_tok: u32, k_loc: u32, parents: [8]i32 = @splat(-1) };
const LmArgs = extern struct { vocab_loc: u32, K: u32, vocab_lo: u32, rows: u32, n_tgs: u32, p0: u32 = 0, p1: u32 = 0, p2: u32 = 0 };
const WalkArgs = extern struct { slots: u32, rank: u32, top: u32, vocab: u32, tp_rank: u32, tp_size: u32, slot_u32: u32, anchor: u32, edge_scale: f32 = 0.6, p0: u32 = 0, p1: u32 = 0, p2: u32 = 0 };
const AneArgs = extern struct { rows: u32, channels: u32, s_ane: u32, row0: u32, src_stride: u32, src_off: u32, add: u32, p0: u32 = 0 };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    c: Config,
    tp: u32,
    tp_rank: u32,
    heads_loc: u32,
    kv_loc: u32,
    ffn_loc: u32,
    k_lo: u32,
    k_loc: u32,
    pipes: ld.Pipes,
    fc_w: mtl.Buffer,
    fc_q8: ld.Q8,
    hidden_norm: mtl.Buffer,
    norm: mtl.Buffer,
    hproj_w: mtl.Buffer,
    pred_cb: mtl.Buffer,
    succ_cb: mtl.Buffer,
    layers: [max_layers]ld.LayerW,
    slots: []ld.SlotCache,
    taps_loc: mtl.Buffer,
    hctx_tmp: mtl.Buffer,
    hctx: mtl.Buffer,
    ctx_k: mtl.Buffer,
    ctx_v: mtl.Buffer,
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    x: mtl.Buffer,
    xc: mtl.Buffer,
    dyn: mtl.Buffer,
    q: mtl.Buffer,
    q_bf16: mtl.Buffer,
    k_tmp: mtl.Buffer,
    v_tmp: mtl.Buffer,
    attn: mtl.Buffer,
    y: mtl.Buffer,
    yc: mtl.Buffer,
    act: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    hid: mtl.Buffer,
    tv: mtl.Buffer,
    ti: mtl.Buffer,
    seq: mtl.Buffer,
    meta: mtl.Buffer,
    mtp_hist: mtl.Buffer,
    bytes: u64 = 0,
    ane: ?*ane_mod.AneFc = null,
    log_left: u64 = 0,

    pub fn tryLoad(gpa: std.mem.Allocator, io: Io, device: mtl.Device, mc: *const cfg.Config, model_dir: []const u8, n_slots: u32) !?*Drafter {
        const res = (try ld.resolveDir(gpa, io, mc, model_dir)) orelse return null;
        defer gpa.free(res.dir);
        return try load(gpa, io, device, mc, res.c, res.dir, n_slots);
    }

    fn load(gpa: std.mem.Allocator, io: Io, device: mtl.Device, mc: *const cfg.Config, c: Config, dir: []const u8, n_slots: u32) !*Drafter {
        const t0 = std.Io.Timestamp.now(io, .awake);
        const st_path = try std.fmt.allocPrint(gpa, "{s}/model.safetensors", .{dir});
        defer gpa.free(st_path);
        var meta = try st.Metadata.read(gpa, io, st_path);
        defer meta.deinit();
        const file = try Io.Dir.cwd().openFile(io, st_path, .{});
        defer file.close(io);

        const tp = mc.tp;
        const tp_rank = mc.tp_rank;
        const heads_loc = c.heads / tp;
        const kv_loc = c.kv_heads / tp;
        const ffn_loc = c.ffn / tp;
        const k_total = num_taps * c.hidden;
        const k_loc = k_total / tp;
        const k_lo = tp_rank * k_loc;
        const q_loc = heads_loc * 128;
        const kv_dim_loc = kv_loc * 128;
        const max_lm_tgs = ((mc.vocab_hi - mc.vocab_lo) + 31) / 32;

        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        const sh = mtl.ResourceOptions.shared;
        d.* = .{
            .gpa = gpa,
            .device = device,
            .c = c,
            .tp = tp,
            .tp_rank = tp_rank,
            .heads_loc = heads_loc,
            .kv_loc = kv_loc,
            .ffn_loc = ffn_loc,
            .k_lo = k_lo,
            .k_loc = k_loc,
            .pipes = try ld.Pipes.load(device),
            .fc_w = try ld.loadColSlice(gpa, device, io, file, &meta, "fc.weight", c.hidden, k_lo, k_loc, k_total),
            .fc_q8 = undefined,
            .hidden_norm = try ld.loadFull(device, io, file, &meta, "hidden_norm.weight"),
            .norm = try ld.loadFull(device, io, file, &meta, "norm.weight"),
            .hproj_w = try ld.loadFull(device, io, file, &meta, "candidate_selector.hidden_projection.weight"),
            .pred_cb = try ld.loadFull(device, io, file, &meta, "candidate_selector.predecessor_codebook"),
            .succ_cb = try ld.loadFull(device, io, file, &meta, "candidate_selector.successor_codebook"),
            .layers = undefined,
            .slots = try gpa.alloc(ld.SlotCache, n_slots),
            .taps_loc = try device.buffer(@as(usize, gpu.prompt_rows) * k_loc * 2, sh),
            .hctx_tmp = try device.buffer(@as(usize, gpu.prompt_rows) * c.hidden * 4, sh),
            .hctx = try device.buffer(@as(usize, gpu.prompt_rows) * c.hidden * 4, sh),
            .ctx_k = try device.buffer(@as(usize, gpu.prompt_rows) * kv_dim_loc * 4, sh),
            .ctx_v = try device.buffer(@as(usize, gpu.prompt_rows) * kv_dim_loc * 4, sh),
            .ids = try device.buffer(block_size * 4, sh),
            .resid = try device.buffer(block_size * c.hidden * 4, sh),
            .x = try device.buffer(block_size * c.hidden * 4, sh),
            .xc = try device.buffer(block_size * c.hidden * 4, sh),
            .dyn = try device.buffer(block_size * 4 * (c.hidden / c.group) * 4, sh),
            .q = try device.buffer(block_size * q_loc * 4, sh),
            .q_bf16 = try device.buffer(block_size * q_loc * 2, sh),
            .k_tmp = try device.buffer(block_size * kv_dim_loc * 4, sh),
            .v_tmp = try device.buffer(block_size * kv_dim_loc * 4, sh),
            .attn = try device.buffer(block_size * q_loc * 4, sh),
            .y = try device.buffer(block_size * c.hidden * 4, sh),
            .yc = try device.buffer(block_size * c.hidden * 4, sh),
            .act = try device.buffer(block_size * ffn_loc * 4, sh),
            .pm = try device.buffer(block_size * heads_loc * 18 * 4, sh),
            .pl = try device.buffer(block_size * heads_loc * 18 * 4, sh),
            .po = try device.buffer(block_size * heads_loc * 18 * 128 * 4, sh),
            .hid = try device.buffer(draft_tokens * c.rank * 4, sh),
            .tv = try device.buffer(draft_tokens * max_lm_tgs * 16 * 4, sh),
            .ti = try device.buffer(draft_tokens * max_lm_tgs * 16 * 4, sh),
            .seq = try device.buffer(256 * 4, sh),
            .meta = try device.buffer(128, sh),
            .mtp_hist = try device.buffer(128 * c.hidden * 4, sh),
            .log_left = if (tp_rank == 0) (@import("sys").environment.read(u64, "TF_GLM_DFLASH2_LOG", 0) catch 0) else 0,
        };
        d.fc_q8 = try ld.loadLayers(gpa, device, io, file, &meta, c, tp, tp_rank, d.pipes.quant_q8, d.fc_w, k_loc, &d.layers);
        const kv_bytes = @as(usize, c.ring()) * kv_dim_loc * 2;
        for (d.slots) |*sl| {
            sl.from = 0;
            sl.end = 0;
            for (0..c.layers) |i| {
                sl.k[i] = try device.buffer(kv_bytes, mtl.ResourceOptions.private);
                sl.v[i] = try device.buffer(kv_bytes, mtl.ResourceOptions.private);
            }
        }
        d.ane = ld.initAne(gpa, device, d.fc_w, c.hidden, k_loc);
        const l0 = &d.layers[0];
        d.bytes = d.fc_w.length() + d.fc_q8.bytes() + d.pred_cb.length() + d.succ_cb.length() + @as(u64, c.layers) * (l0.q_w.bytes() + l0.k_w.length() + l0.v_w.length() + l0.o_w.bytes() + l0.gate_w.bytes() + l0.up_w.bytes() + l0.down_w.bytes() + l0.aconv_proj.bytes() + l0.mconv_proj.bytes());
        const dt = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        std.log.info("glm_dsa DFlash2 loaded in {d:.2}s ({d:.2} GiB/rank Q8, TP={d}, layers={d}, block={d}, ANE={})", .{ dt, @as(f64, @floatFromInt(d.bytes)) / (1024.0 * 1024.0 * 1024.0), tp, c.layers, c.block, d.ane != null });
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        if (d.ane) |a| a.deinit();
        for (d.slots) |sl| for (0..d.c.layers) |i| {
            sl.k[i].deinit();
            sl.v[i].deinit();
        };
        d.gpa.free(d.slots);
        for (0..d.c.layers) |i| d.layers[i].deinit();
        d.fc_q8.deinit();
        inline for (.{ d.fc_w, d.hidden_norm, d.norm, d.hproj_w, d.pred_cb, d.succ_cb, d.taps_loc, d.hctx_tmp, d.hctx, d.ctx_k, d.ctx_v, d.ids, d.resid, d.x, d.xc, d.dyn, d.q, d.q_bf16, d.k_tmp, d.v_tmp, d.attn, d.y, d.yc, d.act, d.pm, d.pl, d.po, d.hid, d.tv, d.ti, d.seq, d.meta, d.mtp_hist }) |b| b.deinit();
        d.pipes.deinit();
        d.gpa.destroy(d);
    }

    pub fn resetSlot(d: *Drafter, slot: u32) void {
        const sl = &d.slots[slot];
        sl.from = 0;
        sl.end = 0;
        sl.use_mtp = false;
        sl.df2_rounds = 0;
        sl.df2_toks = 0;
        sl.last6 = @splat(0);
        sl.hist_n = 0;
        sl.hist_pos0 = 0;
    }

    pub fn tapIndex(d: *const Drafter, layer: u32) ?u32 {
        for (d.c.taps, 0..) |l, i| if (l == layer) return @intCast(i);
        return null;
    }

    /// Capture target layer post-layer residual `s_h` ([rows, 6144] f32) into this rank's K-slice of `taps_loc`.
    pub fn tap(d: *const Drafter, e: mtl.ComputeEncoder, layer: u32, s_h: Ref, rows: u32) void {
        const ti = d.tapIndex(layer) orelse return;
        const tap_lo = ti * d.c.hidden;
        const ov_lo = @max(tap_lo, d.k_lo);
        const ov_hi = @min(tap_lo + d.c.hidden, d.k_lo + d.k_loc);
        if (ov_lo >= ov_hi) return;
        e.setPipeline(d.pipes.tap);
        e.setBuffer(s_h.buf, s_h.off, 0);
        e.setBuffer(d.taps_loc, 0, 1);
        e.setValue(TapArgs{ .rows = rows, .hidden = d.c.hidden, .k_lo = d.k_lo, .k_loc = d.k_loc, .tap_idx = ti }, 2);
        e.dispatchThreads(.{ .width = ov_hi - ov_lo, .height = rows }, .{ .width = 256 });
    }

    fn gemv(d: *const Drafter, e: mtl.ComputeEncoder, x_ref: Ref, w: mtl.Buffer, out: Ref, rows: u32, K: u32, N: u32, in_bf16: bool, pack_info: [4]u32, pack_base: ?Ref) void {
        const args: GemvArgs = .{ .rows = rows, .K = K, .N = N, .out_stride = N, .in_bf16 = @intFromBool(in_bf16), .pack_rank = pack_info[0], .pack_tp = if (pack_base != null) pack_info[1] else 1, .pack_slot_u2 = pack_info[2] };
        if (rows <= 8) {
            e.setPipeline(d.pipes.gemv);
            e.setBuffer(x_ref.buf, x_ref.off, 0);
            e.setBuffer(w, 0, 1);
            e.setValue(args, 2);
            e.setBuffer(out.buf, out.off, 3);
            const pb = pack_base orelse out;
            e.setBuffer(pb.buf, pb.off, 4);
            e.dispatchGroups(.{ .width = (N + 15) / 16 }, .{ .width = 256 });
        } else {
            e.setPipeline(d.pipes.gemm);
            e.setBuffer(x_ref.buf, x_ref.off, 0);
            e.setBuffer(w, 0, 1);
            e.setValue(args, 2);
            e.setBuffer(out.buf, out.off, 3);
            e.dispatchGroups(.{ .width = (N + 63) / 64, .height = (rows + 31) / 32 }, .{ .width = 128 });
        }
    }

    fn gemvQ8(d: *const Drafter, e: mtl.ComputeEncoder, x_ref: Ref, q: ld.Q8, out: Ref, rows: u32, K: u32, N: u32, in_mode: u32, pack_info: [4]u32, pack_base: ?Ref) void {
        const args: GemvArgs = .{ .rows = rows, .K = K, .N = N, .out_stride = N, .in_bf16 = in_mode, .pack_rank = pack_info[0], .pack_tp = if (pack_base != null) pack_info[1] else 1, .pack_slot_u2 = pack_info[2] };
        if (N <= 512 and (N % 2 == 0) and (K % 512 == 0) and pack_base == null and in_mode != 1) {
            e.setPipeline(d.pipes.q8_gemv_q1);
            e.setBuffer(x_ref.buf, x_ref.off, 0);
            e.setBuffer(q.w, 0, 1);
            e.setBuffer(q.s, 0, 2);
            e.setValue(args, 3);
            e.setBuffer(out.buf, out.off, 4);
            e.dispatchGroups(.{ .width = N / 2 }, .{ .width = 256 });
            return;
        }
        if (in_mode == 2 and pack_base == null and (N % 16 == 0) and (K % 512 == 0)) {
            e.setPipeline(d.pipes.q8_gemv_q2_xh);
            e.setBuffer(x_ref.buf, x_ref.off, 0);
            e.setBuffer(q.w, 0, 1);
            e.setBuffer(q.s, 0, 2);
            e.setValue(args, 3);
            e.setBuffer(out.buf, out.off, 4);
            e.dispatchGroups(.{ .width = N / 16 }, .{ .width = 256 });
            return;
        }
        e.setPipeline(d.pipes.q8_gemv);
        e.setBuffer(x_ref.buf, x_ref.off, 0);
        e.setBuffer(q.w, 0, 1);
        e.setBuffer(q.s, 0, 2);
        e.setValue(args, 3);
        e.setBuffer(out.buf, out.off, 4);
        const pb = pack_base orelse out;
        e.setBuffer(pb.buf, pb.off, 5);
        e.dispatchGroups(.{ .width = (N + 15) / 16 }, .{ .width = 256 });
    }

    /// Launch async 32-core ANE `fc` right after target layer 75 during prompt prefill; returns wait val.
    pub fn startAneFc(d: *Drafter, queue: mtl.Queue, cb: *mtl.CommandBuffer, enc: *mtl.ComputeEncoder, dispatch: mtl.DispatchType, rows: u32) ?u64 {
        const a = d.ane orelse return null;
        if (a.failed.load(.acquire) or rows < 64 or rows > ane_mod.max_slices * ane_mod.slice_rows) return null;
        enc.barrier();
        const used = (rows + ane_mod.slice_rows - 1) / ane_mod.slice_rows;
        enc.setPipeline(d.pipes.ane_pack);
        for (0..a.chunks) |c| {
            for (0..used) |sl| {
                enc.setBuffer(d.taps_loc, 0, 0);
                enc.setBuffer(a.inputs[c][sl].buffer, 0, 1);
                enc.setValue(AneArgs{ .rows = rows, .channels = ane_mod.chunk_k, .s_ane = ane_mod.slice_rows, .row0 = @intCast(sl * ane_mod.slice_rows), .src_stride = d.k_loc, .src_off = @intCast(c * ane_mod.chunk_k), .add = 0 }, 2);
                enc.dispatchThreads(.{ .width = ane_mod.slice_rows, .height = ane_mod.chunk_k }, .{ .width = 16, .height = 16 });
            }
        }
        const vals = a.queue(used);
        enc.end();
        cb.signal(a.event, vals[0]);
        cb.commit();
        cb.* = queue.commandBuffer();
        enc.* = cb.compute(dispatch);
        return vals[1];
    }

    /// Absorb `rows` tapped target rows (starting at `taps_loc[row_off..]`) at positions `[pos0, pos0 + rows)`.
    pub fn absorb(d: *Drafter, pair: *pr.Pair, queue: mtl.Queue, cb: *mtl.CommandBuffer, enc: *mtl.ComputeEncoder, dispatch: mtl.DispatchType, slot: u32, pos0: u32, row_off: u32, rows: u32, ane_wait: ?u64, fused: bool) void {
        if (rows == 0) return;
        const sl = &d.slots[slot];
        if (pos0 > sl.end) sl.from = pos0;
        const part = pair.part();
        if (ane_wait) |wv| {
            if (d.ane) |a| {
                enc.end();
                cb.commit();
                cb.* = queue.commandBuffer();
                cb.waitFor(a.event, wv);
                enc.* = cb.compute(dispatch);
                const used = (rows + ane_mod.slice_rows - 1) / ane_mod.slice_rows;
                enc.setPipeline(d.pipes.ane_unpack);
                for (0..a.chunks) |c| {
                    for (0..used) |s_idx| {
                        enc.setBuffer(a.outputs[c][s_idx].buffer, 0, 0);
                        enc.setBuffer(part.buf, part.off, 1);
                        enc.setValue(AneArgs{ .rows = rows, .channels = d.c.hidden, .s_ane = ane_mod.slice_rows, .row0 = @intCast(s_idx * ane_mod.slice_rows), .src_stride = 0, .src_off = 0, .add = @intFromBool(c > 0) }, 2);
                        enc.dispatchThreads(.{ .width = d.c.hidden, .height = ane_mod.slice_rows }, .{ .width = 32, .height = 8 });
                    }
                    if (c + 1 < a.chunks) enc.barrier();
                }
                enc.barrier();
                pair.addRmsReplace(enc.*, .{ .buf = d.hctx_tmp }, .{ .buf = d.hidden_norm }, .{ .buf = d.hctx }, rows, d.c.hidden, d.c.eps, false);
                enc.barrier();
            }
        } else {
            const pk = pair.packTarget();
            const prepack = d.tp > 1 and rows <= 8;
            const taps_ref: Ref = .{ .buf = d.taps_loc, .off = @as(usize, row_off) * d.k_loc * 2 };
            if (rows <= 8) {
                d.gemvQ8(enc.*, taps_ref, d.fc_q8, part, rows, d.k_loc, d.c.hidden, 1, pk.info, if (prepack) pk.base else null);
            } else {
                d.gemv(enc.*, taps_ref, d.fc_w, part, rows, d.k_loc, d.c.hidden, true, pk.info, null);
            }
            enc.barrier();
            pair.addRmsReplace(enc.*, .{ .buf = d.hctx_tmp }, .{ .buf = d.hidden_norm }, .{ .buf = d.hctx }, rows, d.c.hidden, d.c.eps, prepack);
            enc.barrier();
        }
        const kv_dim = d.kv_loc * 128;
        const dummy = [4]u32{ 0, 1, 0, 0 };
        for (0..d.c.layers) |i| {
            const L = &d.layers[i];
            if (rows <= 8) {
                d.gemvQ8(enc.*, .{ .buf = d.hctx }, L.k_q8, .{ .buf = d.ctx_k }, rows, d.c.hidden, kv_dim, 0, dummy, null);
                d.gemvQ8(enc.*, .{ .buf = d.hctx }, L.v_q8, .{ .buf = d.ctx_v }, rows, d.c.hidden, kv_dim, 0, dummy, null);
            } else {
                d.gemv(enc.*, .{ .buf = d.hctx }, L.k_w, .{ .buf = d.ctx_k }, rows, d.c.hidden, kv_dim, false, dummy, null);
                d.gemv(enc.*, .{ .buf = d.hctx }, L.v_w, .{ .buf = d.ctx_v }, rows, d.c.hidden, kv_dim, false, dummy, null);
            }
            enc.barrier();
            enc.setPipeline(d.pipes.prep);
            inline for (.{ d.q, d.ctx_k, d.ctx_v, L.q_norm, L.k_norm }, 0..) |b, k| enc.setBuffer(b, 0, k);
            enc.setValue(PrepArgs{ .rows = rows, .heads = 0, .kv_heads = d.kv_loc, .pos0 = pos0, .ring = d.c.ring(), .dyn_meta = @intFromBool(fused), .eps = d.c.eps, .theta = d.c.theta }, 5);
            inline for (.{ d.q_bf16, sl.k[i], sl.v[i], d.meta }, 6..) |b, k| enc.setBuffer(b, 0, k);
            enc.dispatchGroups(.{ .width = d.kv_loc, .height = rows }, .{ .width = 32 });
            enc.barrier();
        }
        if (!fused) sl.end = pos0 + rows;
    }

    fn encNorm(d: *const Drafter, e: mtl.ComputeEncoder, r_buf: mtl.Buffer, x_buf: mtl.Buffer, w_buf: mtl.Buffer, out_buf: mtl.Buffer, add: bool) void {
        e.setPipeline(d.pipes.norm);
        e.setBuffer(r_buf, 0, 0);
        e.setBuffer(x_buf, 0, 1);
        e.setBuffer(w_buf, 0, 2);
        e.setValue(NormArgs{ .rows = block_size, .width = d.c.hidden, .add = @intFromBool(add), .eps = d.c.eps }, 3);
        e.setBuffer(out_buf, 0, 4);
        e.dispatchGroups(.{ .width = block_size }, .{ .width = 256 });
        e.barrier();
    }

    fn encConv(d: *const Drafter, e: mtl.ComputeEncoder, in_buf: mtl.Buffer, base_buf: mtl.Buffer, out_buf: mtl.Buffer, side: u32) void {
        e.setPipeline(d.pipes.conv);
        e.setBuffer(in_buf, 0, 0);
        e.setBuffer(d.dyn, 0, 1);
        e.setBuffer(base_buf, 0, 2);
        e.setValue(ConvArgs{ .rows = block_size, .width = d.c.hidden, .block = block_size, .groups = d.c.hidden / d.c.group, .side = side }, 3);
        e.setBuffer(out_buf, 0, 4);
        e.dispatchThreads(.{ .width = d.c.hidden, .height = block_size }, .{ .width = 256 });
        e.barrier();
    }

    fn encConvAddNorm(d: *const Drafter, e: mtl.ComputeEncoder, r_buf: mtl.Buffer, y_buf: mtl.Buffer, base_buf: mtl.Buffer, w_buf: mtl.Buffer, out_buf: mtl.Buffer) void {
        e.setPipeline(d.pipes.conv_add_rms);
        inline for (.{ r_buf, y_buf, d.dyn, base_buf, w_buf }, 0..) |b, k| e.setBuffer(b, 0, k);
        e.setValue(ConvNormArgs{ .rows = block_size, .width = d.c.hidden, .block = block_size, .groups = d.c.hidden / d.c.group, .eps = d.c.eps }, 5);
        e.setBuffer(out_buf, 0, 6);
        e.dispatchGroups(.{ .width = block_size }, .{ .width = 256 });
        e.barrier();
    }

    /// Parallel 8-row DFlash2 block forward + Q4-MMA/BF16 lm_head top-16 + GPU best-first tree walk -> `d.seq[1..30]`.
    pub fn draftBlock(d: *Drafter, m: *md.Model, e: mtl.ComputeEncoder, slot: u32, anchor_tok: u32, anchor_pos: u32, fused: bool, base_pos: u32, win_rows: u32, use_tree: bool) void {
        const sl = &d.slots[slot];
        const c = &d.c;
        const n = block_size;
        const h = c.hidden;
        const q_loc = d.heads_loc * 128;
        const kv_dim = d.kv_loc * 128;
        const dyn_w = 4 * (h / c.group);
        const dummy = [4]u32{ 0, 1, 0, 0 };

        if (!fused) {
            const ids = d.ids.slice(u32, n);
            ids[0] = anchor_tok;
            for (1..n) |j| ids[j] = c.mask;
        }
        m.embedTokens(e, .{ .buf = d.ids }, .{ .buf = d.resid }, n);
        e.barrier();

        const lo: u32 = if (fused) @max((base_pos + 2) -| c.window, sl.from) else @max((anchor_pos + 1) -| c.window, sl.from);
        const hi: u32 = if (fused) (base_pos + win_rows + n) else (anchor_pos + n);
        const split_sz: u32 = if (hi - lo <= 512) 32 else 128;
        const n_splits: u32 = ((hi - lo) + split_sz - 1) / split_sz;
        const gqa: u32 = d.heads_loc / d.kv_loc;

        d.encNorm(e, d.resid, d.yc, d.layers[0].in_norm, d.x, false);
        for (0..c.layers) |i| {
            const L = &d.layers[i];
            d.gemvQ8(e, .{ .buf = d.x }, L.aconv_proj, .{ .buf = d.dyn }, n, h, dyn_w, 0, dummy, null);
            e.barrier();
            d.encConv(e, d.x, L.aconv_base, d.xc, 0);

            d.gemvQ8(e, .{ .buf = d.xc }, L.q_w, .{ .buf = d.q }, n, h, q_loc, 2, dummy, null);
            d.gemvQ8(e, .{ .buf = d.xc }, L.k_q8, .{ .buf = d.k_tmp }, n, h, kv_dim, 2, dummy, null);
            d.gemvQ8(e, .{ .buf = d.xc }, L.v_q8, .{ .buf = d.v_tmp }, n, h, kv_dim, 2, dummy, null);
            e.barrier();
            e.setPipeline(d.pipes.prep);
            inline for (.{ d.q, d.k_tmp, d.v_tmp, L.q_norm, L.k_norm }, 0..) |b, k| e.setBuffer(b, 0, k);
            e.setValue(PrepArgs{ .rows = n, .heads = d.heads_loc, .kv_heads = d.kv_loc, .pos0 = anchor_pos, .ring = c.ring(), .dyn_meta = if (fused) 2 else 0, .eps = c.eps, .theta = c.theta }, 5);
            inline for (.{ d.q_bf16, sl.k[i], sl.v[i], d.meta }, 6..) |b, k| e.setBuffer(b, 0, k);
            e.dispatchGroups(.{ .width = d.heads_loc + d.kv_loc, .height = n }, .{ .width = 32 });
            e.barrier();

            const aa: AttnArgs = .{ .rows = n, .heads = d.heads_loc, .kv_heads = d.kv_loc, .gqa = gqa, .lo = lo, .hi = hi, .ring = c.ring(), .n_splits = n_splits, .scale = 1.0 / @sqrt(128.0), .dyn_meta = @intFromBool(fused), .window = c.window, .sl_from = sl.from, .split_sz = split_sz };
            e.setPipeline(d.pipes.attend);
            inline for (.{ d.q_bf16, sl.k[i], sl.v[i] }, 0..) |b, k| e.setBuffer(b, 0, k);
            e.setValue(aa, 3);
            inline for (.{ d.pm, d.pl, d.po, d.meta }, 4..) |b, k| e.setBuffer(b, 0, k);
            e.dispatchGroups(.{ .width = d.kv_loc, .height = n, .depth = n_splits }, .{ .width = 32 * gqa });
            e.barrier();
            e.setPipeline(d.pipes.merge);
            inline for (.{ d.pm, d.pl, d.po }, 0..) |b, k| e.setBuffer(b, 0, k);
            e.setValue(aa, 3);
            inline for (.{ d.attn, d.meta }, 4..) |b, k| e.setBuffer(b, 0, k);
            e.dispatchGroups(.{ .width = d.heads_loc, .height = n }, .{ .width = 32 });
            e.barrier();

            const pk1 = m.pair.packTarget();
            d.gemvQ8(e, .{ .buf = d.attn }, L.o_w, if (d.tp > 1) m.pair.part() else .{ .buf = d.y }, n, q_loc, h, 2, pk1.info, if (d.tp > 1) pk1.base else null);
            e.barrier();
            if (d.tp > 1) {
                m.pair.sumOnly(e, .{ .buf = d.y }, n, h, true);
                e.barrier();
            }
            d.encConvAddNorm(e, d.resid, d.y, L.aconv_base, L.post_norm, d.x);

            d.gemvQ8(e, .{ .buf = d.x }, L.mconv_proj, .{ .buf = d.dyn }, n, h, dyn_w, 0, dummy, null);
            e.barrier();
            d.encConv(e, d.x, L.mconv_base, d.xc, 0);

            e.setPipeline(d.pipes.q8_gu_silu);
            inline for (.{ d.xc, L.gate_w.w, L.gate_w.s, L.up_w.w, L.up_w.s }, 0..) |b, k| e.setBuffer(b, 0, k);
            e.setValue(GemvArgs{ .rows = n, .K = h, .N = d.ffn_loc, .out_stride = d.ffn_loc, .in_bf16 = 2, .pack_rank = 0, .pack_tp = 1, .pack_slot_u2 = 0 }, 5);
            e.setBuffer(d.act, 0, 6);
            e.dispatchGroups(.{ .width = (d.ffn_loc + 7) / 8 }, .{ .width = 256 });
            e.barrier();

            const pk2 = m.pair.packTarget();
            d.gemvQ8(e, .{ .buf = d.act }, L.down_w, if (d.tp > 1) m.pair.part() else .{ .buf = d.y }, n, d.ffn_loc, h, 2, pk2.info, if (d.tp > 1) pk2.base else null);
            e.barrier();
            if (d.tp > 1) {
                m.pair.sumOnly(e, .{ .buf = d.y }, n, h, true);
                e.barrier();
            }
            d.encConvAddNorm(e, d.resid, d.y, L.mconv_base, if (i + 1 < c.layers) d.layers[i + 1].in_norm else d.norm, d.x);
        }

        const x1: Ref = .{ .buf = d.x, .off = @as(usize, h) * 4 };
        const v_loc = m.vocab.part[1] - m.vocab.part[0];
        const use_q4 = m.vocab.lm_q4_w != null and m.vocab.lm_q4_s != null and (h % 64 == 0);
        const n_tgs = if (use_q4) (v_loc + 63) / 64 else (v_loc + 31) / 32;
        const la: LmArgs = .{ .vocab_loc = v_loc, .K = h, .vocab_lo = m.vocab.part[0], .rows = draft_tokens, .n_tgs = n_tgs };
        if (use_q4) {
            e.setPipeline(if (m.vocab.lm_is_q8) d.pipes.lm_q8_topk else d.pipes.lm_q4_topk);
            e.setBuffer(x1.buf, x1.off, 0);
            e.setBuffer(m.vocab.lm_q4_w.?, 0, 1);
            e.setBuffer(m.vocab.lm_q4_s.?, 0, 2);
            e.setValue(la, 3);
            e.setBuffer(d.tv, 0, 4);
            e.setBuffer(d.ti, 0, 5);
            e.dispatchGroups(.{ .width = n_tgs }, .{ .width = 128 });
        } else {
            const lm_ref = m.vocab.k.at(m.vocab.lm_bf16_off);
            e.setPipeline(d.pipes.lm_topk);
            e.setBuffer(x1.buf, x1.off, 0);
            e.setBuffer(lm_ref.buf, lm_ref.off, 1);
            e.setValue(la, 2);
            e.setBuffer(d.tv, 0, 3);
            e.setBuffer(d.ti, 0, 4);
            e.dispatchGroups(.{ .width = n_tgs }, .{ .width = 256 });
        }
        d.gemv(e, x1, d.hproj_w, .{ .buf = d.hid }, draft_tokens, h, c.rank, false, dummy, null);
        e.barrier();

        const ex = m.pair.reserveRaw(draft_tokens * 32 * 4);
        e.setPipeline(d.pipes.topk_red);
        e.setBuffer(d.tv, 0, 0);
        e.setBuffer(d.ti, 0, 1);
        e.setValue(la, 2);
        e.setBuffer(ex.send.buf, ex.send.off, 3);
        e.dispatchGroups(.{ .width = draft_tokens }, .{ .width = 256 });
        e.barrier();
        m.pair.waitRaw(e, ex.x);
        e.barrier();

        e.setPipeline(d.pipes.walk);
        e.setBuffer(ex.send.buf, ex.send.off, 0);
        e.setBuffer(ex.recv.buf, ex.recv.off, 1);
        inline for (.{ d.hid, d.pred_cb, d.succ_cb }, 2..) |b, k| e.setBuffer(b, 0, k);
        e.setValue(WalkArgs{ .slots = draft_tokens, .rank = c.rank, .top = c.top_k, .vocab = c.vocab, .tp_rank = d.tp_rank, .tp_size = d.tp, .slot_u32 = ex.slot_u32, .anchor = if (fused) 0xFFFF_FFFF else anchor_tok, .p0 = @intFromBool(use_tree) }, 5);
        inline for (.{ d.seq, d.meta }, 6..) |b, k| e.setBuffer(b, 0, k);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = c.rank });
    }

    /// Execute optional absorb + optional block draft in a single command buffer, returning GPU ms.
    pub fn absorbAndDraft(d: *Drafter, m: *md.Model, queue: mtl.Queue, dispatch: mtl.DispatchType, slot: u32, pos0: u32, row_off: u32, to_absorb: u32, anchor_tok: u32, anchor_pos: u32, depth: u32, use_tree: bool, held_out: []u32, probs_out: []f32) !f64 {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var cb = queue.commandBuffer();
        var enc = cb.compute(dispatch);
        if (to_absorb > 0) d.absorb(m.pair, queue, &cb, &enc, dispatch, slot, pos0, row_off, to_absorb, null, false);
        if (depth > 0) d.draftBlock(m, enc, slot, anchor_tok, anchor_pos, false, anchor_pos, 0, use_tree);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("glm_dsa DFlash2 GPU failed: {s}", .{msg});
            return error.GpuFailure;
        }
        if (m.pair.gaveUp() != 0 or m.pair.failed.load(.acquire)) return error.PairLinkFailed;
        const gpu_ms = (cb.gpuEnd() - cb.gpuStart()) * 1000.0;
        if (depth > 0) d.readSeq(anchor_pos, to_absorb, depth, gpu_ms, held_out, probs_out);
        return gpu_ms;
    }

    pub fn readSeq(d: *Drafter, anchor_pos: u32, abs_n: u32, depth: u32, gpu_ms: f64, held_out: []u32, probs_out: []f32) void {
        const seq_u = d.seq.slice(u32, 32);
        const seq_f = d.seq.slice(f32, 32);
        @memcpy(held_out[0..depth], seq_u[1 .. 1 + depth]);
        @memcpy(probs_out[0..depth], seq_f[24 .. 24 + depth]);
        if (d.log_left > 0) {
            d.log_left -= 1;
            std.log.info("df2 pos={d} abs={d} d={d} gpu={d:.2}ms tok={any} p={d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}", .{
                anchor_pos, abs_n, depth, gpu_ms, seq_u[1..8], seq_f[8], seq_f[9], seq_f[10], seq_f[11], seq_f[12], seq_f[13], seq_f[14],
            });
        }
    }

    /// Single-CB fused target verify + GPU greedy accept + optional MTP d=0 + DFlash2 absorb + 7-tok block draft!
    pub fn windowFused(d: *Drafter, e: anytype, slot: u32, tokens: []const u32, pred_depth: u32, out: []u32) !void {
        return hybrid.windowFused(d, e, slot, tokens, pred_depth, out);
    }
};
