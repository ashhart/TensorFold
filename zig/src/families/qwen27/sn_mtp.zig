//! Native Metal single-layer MTP drafter for SenseNova-Loop-9B (BF16 and AWQ checkpoints).
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const attn = @import("attention.zig");
const wts = @import("gpu_weights.zig");
const loop = @import("loop.zig");
const projection = @import("projection.zig");
const scratch_mod = @import("sn_mtp_scratch.zig");
const Config = @import("config.zig").Config;
const Model = @import("model.zig").Model;
const mtp_weights = core.sensenova_loop.mtp_weights;
const round_contract = core.tree_round;
const round_gpu = core.tree_round_gpu;
const lref = scratch_mod.lref;
const buf_opts = scratch_mod.buf_opts;
const scratch_count = scratch_mod.scratch_count;
const ScratchField = scratch_mod.ScratchField;
const batch = @import("sn_mtp_batch.zig");

pub const max_depth = scratch_mod.max_depth;
pub const batch_cap = scratch_mod.batch_cap;
pub const batch_streams = scratch_mod.batch_streams;
pub const Features = scratch_mod.Features;
pub const AbsorbItem = scratch_mod.AbsorbItem;
pub const HoldItem = scratch_mod.HoldItem;
pub const StreamCtx = scratch_mod.StreamCtx;

pub const Mtp = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    weights: mtp_weights.Weights,
    attention: attn.Attention,
    ops: round_gpu.Ops,
    geometry: attn.planning.Geometry,
    abs_scratch: attn.Scratch,
    abs_binding: attn.Binding,
    chain_scratch: [max_depth - 1]attn.Scratch,
    chain_binding: [max_depth - 1]attn.Binding,
    bufs: [scratch_count]mtl.Buffer,
    parents_storage: [batch_cap]i32,
    kept_storage: [batch_cap]u32,
    ctxs: []?StreamCtx,
    capacity: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    vocab: usize,
    abs_rows: u32,
    bits: u32 = 16,
    draft_head: ?core.sensenova_loop.linear_gpu.AwqMatrix = null,
    depth: u32,
    plain_guard: bool,
    step_ms: f64 = 0.5,
    tap_layer: [1]u32,
    residency: ?mtl.ResidencySet.Follow = null,

    /// True when this SenseNova-Loop checkpoint has a companion `mtp.safetensors`.
    pub fn available(io: std.Io, dir: []const u8, cfg: Config) bool {
        return cfg.looped and cfg.mhc_iterations > 0 and mtp_weights.Weights.hasFile(io, dir);
    }

    /// Per-stream device KV and speculative chain bytes (capacity tokens).
    pub fn kvSlotBytes(cfg: Config, capacity: usize) usize {
        return 2 * cfg.kvDim() * capacity * 2 + max_depth * 16 + (max_depth + 1) * cfg.hidden * 2;
    }

    pub fn slotBytes(m: *const Mtp) usize {
        return 2 * m.kv_dim * m.capacity * 2 + max_depth * 16 + (max_depth + 1) * m.hidden * 2;
    }

    pub fn loadWeights(gpa: std.mem.Allocator, io: std.Io, model: *Model, dir: []const u8) !mtp_weights.Weights {
        const c = model.config;
        return mtp_weights.Weights.load(gpa, io, model.device, model.queue, dir, c.hidden, c.heads, c.kv_heads, c.head_dim, c.intermediate);
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model: *Model, dir: []const u8, slots: u32, capacity: usize) !*Mtp {
        var weights = try loadWeights(gpa, io, model, dir);
        errdefer weights.deinit();
        return initFromWeights(gpa, model, weights, slots, capacity);
    }

    pub fn initFromWeights(gpa: std.mem.Allocator, model: *Model, weights: mtp_weights.Weights, slots: u32, capacity: usize) !*Mtp {
        if (slots == 0 or capacity < 16 or model.kernels.sn == null or (model.weights.embed_bf16 == null and model.weights.embed.weight.shape[0] == 0)) {
            return error.MissingSenseNovaMtpPrereqs;
        }
        var w_owned = weights;
        errdefer w_owned.deinit();
        const cfg = model.config;
        const dev = model.device;
        const units = dev.tensorUnits();
        const geometry = try attn.planning.Geometry.fromConfig(cfg, units);
        var attention = try scratch_mod.buildAttention(gpa, dev, geometry, units);
        errdefer attention.deinit();
        const ops = try round_gpu.Ops.init(dev);
        errdefer ops.deinit();
        const kv_dim = cfg.kvDim();
        const eff_batch = geometry.promptRows(@intCast(@min(batch_cap, capacity)));
        const abs_limits = attn.Limits{ .rows = eff_batch, .streams = batch_streams, .keys = @intCast(capacity), .shared_prefix = geometry.shared_prefix, .geometry = geometry };
        var abs_scratch = try attn.Scratch.init(dev, abs_limits);
        errdefer abs_scratch.deinit();
        var abs_binding = try attn.Binding.init(dev, abs_limits);
        errdefer abs_binding.deinit();
        const chain_limits = attn.Limits{ .rows = batch_streams, .streams = batch_streams, .keys = @intCast(capacity), .shared_prefix = geometry.shared_prefix, .geometry = geometry };
        var chain_scratch: [max_depth - 1]attn.Scratch = undefined;
        var chain_binding: [max_depth - 1]attn.Binding = undefined;
        var made_chain: usize = 0;
        errdefer for (0..made_chain) |i| {
            chain_scratch[i].deinit();
            chain_binding[i].deinit();
        };
        for (0..max_depth - 1) |i| {
            chain_scratch[i] = try attn.Scratch.init(dev, chain_limits);
            errdefer chain_scratch[i].deinit();
            chain_binding[i] = try attn.Binding.init(dev, chain_limits);
            made_chain += 1;
        }
        const sizes = scratch_mod.scratchSizes(cfg, eff_batch);
        var bufs: [scratch_count]mtl.Buffer = undefined;
        var made_bufs: usize = 0;
        errdefer for (bufs[0..made_bufs]) |b| b.deinit();
        for (sizes, &bufs) |bytes, *b| {
            b.* = try dev.buffer(@max(bytes, 16), buf_opts);
            made_bufs += 1;
        }
        var draft_head: ?core.sensenova_loop.linear_gpu.AwqMatrix = null;
        errdefer if (draft_head) |*dh| dh.deinit();
        if (w_owned.bits == 4 and model.weights.head.bits == 16 and model.kernels.sn != null and model.weights.head.sn != null) {
            const runs = model.weights.head.sn.?.runs;
            if (runs.len == 1 and runs[0].mat == .bf16) {
                draft_head = try mtp_weights.quantizeMat(&model.kernels.sn.?, model.queue, runs[0].mat.bf16);
            }
        }
        if (w_owned.bits == 4 or draft_head != null) {
            try model.kernels.sn.?.reserve(@import("gpu_frame.zig").max_rows, true);
        }
        const ctxs = try gpa.alloc(?StreamCtx, slots);
        @memset(ctxs, null);
        errdefer {
            for (ctxs) |*slot_opt| if (slot_opt.*) |*sc| sc.deinit();
            gpa.free(ctxs);
        }
        ctxs[0] = try StreamCtx.init(dev, capacity, kv_dim, cfg.hidden);
        const depth = std.math.clamp(try @import("sys").environment.read(u32, "TF_MTP_DRAFTS", 3), 1, max_depth);
        const plain_guard = try @import("sys").environment.read(bool, "TF_MTP_PLAIN_GUARD", true);
        const m = try gpa.create(Mtp);
        m.* = .{
            .gpa = gpa,
            .model = model,
            .weights = w_owned,
            .attention = attention,
            .ops = ops,
            .geometry = geometry,
            .abs_scratch = abs_scratch,
            .abs_binding = abs_binding,
            .chain_scratch = chain_scratch,
            .chain_binding = chain_binding,
            .bufs = bufs,
            .parents_storage = undefined,
            .kept_storage = undefined,
            .ctxs = ctxs,
            .capacity = capacity,
            .hidden = cfg.hidden,
            .heads = cfg.heads,
            .kv_heads = cfg.kv_heads,
            .head_dim = cfg.head_dim,
            .kv_dim = kv_dim,
            .intermediate = cfg.intermediate,
            .vocab = cfg.vocab,
            .abs_rows = eff_batch,
            .bits = w_owned.bits,
            .draft_head = draft_head,
            .depth = depth,
            .plain_guard = plain_guard,
            .tap_layer = .{@intCast(cfg.layers - 1)},
        };
        for (&m.parents_storage, 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1;
        for (&m.kept_storage, 0..) |*k, r| k.* = @intCast(r);
        @memcpy(m.buf(.kept).slice(u32, eff_batch), m.kept_storage[0..eff_batch]);
        return m;
    }

    pub fn deinit(m: *Mtp) void {
        m.setResident(null);
        for (m.ctxs) |*slot_opt| if (slot_opt.*) |*sc| sc.deinit();
        m.gpa.free(m.ctxs);
        for (m.bufs) |b| b.deinit();
        for (&m.chain_scratch, &m.chain_binding) |*s, *b| {
            s.deinit();
            b.deinit();
        }
        m.abs_scratch.deinit();
        m.abs_binding.deinit();
        m.ops.deinit();
        m.attention.deinit();
        if (m.draft_head) |*dh| dh.deinit();
        m.weights.deinit();
        m.gpa.destroy(m);
    }

    pub fn appendBuffers(m: *const Mtp, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        return scratch_mod.appendMtpBuffers(m, a, list);
    }

    pub fn setResident(m: *Mtp, hook: ?mtl.ResidencySet.Follow) void {
        if (m.residency) |old| m.applyResident(old, false);
        m.residency = hook;
        if (hook) |f| m.applyResident(f, true);
    }

    fn applyResident(m: *Mtp, f: mtl.ResidencySet.Follow, joining: bool) void {
        scratch_mod.applyMtpResident(m, f, joining);
    }

    pub fn openStream(m: *Mtp, s: usize) !void {
        if (s == 0) return error.BadMtpStream;
        for (m.ctxs) |slot_opt| if (slot_opt) |sc| if (sc.in_use and sc.owner == s) return error.DuplicateMtpStream;
        for (m.ctxs) |*slot_opt| if (slot_opt.*) |*sc| if (!sc.in_use) {
            sc.reset(s);
            return;
        };
        for (m.ctxs) |*slot_opt| if (slot_opt.* == null) {
            var sc = try StreamCtx.init(m.model.device, m.capacity, m.kv_dim, m.hidden);
            sc.reset(s);
            slot_opt.* = sc;
            if (m.residency) |f| {
                for ([_]mtl.Buffer{ sc.keys, sc.values, sc.picks_buf, sc.chain_out_buf }) |b| mtl.resident.fields.one(b, f, true);
            }
            return;
        };
        return error.NoFreeMtpSlot;
    }

    pub fn releaseStream(m: *Mtp, s: usize) void {
        for (m.ctxs) |*slot_opt| if (slot_opt.*) |*sc| if (sc.in_use and sc.owner == s) {
            sc.reset(0);
            return;
        };
    }

    pub fn ctxOf(m: *const Mtp, s: usize) !*StreamCtx {
        for (m.ctxs) |*slot_opt| if (slot_opt.*) |*sc| if (sc.in_use and sc.owner == s) return sc;
        return error.UnknownMtpStream;
    }

    pub fn absorb(m: *Mtp, items: []const AbsorbItem) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        for (items) |it| {
            const ctx = try m.ctxOf(it.stream);
            const total = it.follow.len;
            if (total == 0) continue;
            if (it.rows) |sel| if (sel.len != total) return error.BadMtpAbsorb;
            if (it.features.row_bytes != m.hidden * 2) return error.BadMtpFeatures;
            if (ctx.len == 0 and it.start > 0) {
                ctx.begin = it.start;
                ctx.len = it.start;
            }
            if (it.start != ctx.len or (it.start - ctx.begin) + total > m.capacity) return error.BadMtpPosition;
            var done: usize = 0;
            while (done < total) {
                const take: u32 = @intCast(@min(m.abs_rows, total - done));
                const last_chunk = done + take == total;
                const pos = it.start + done;
                const raw_ref = try scratch_mod.stageRawFeatures(m, it.features, it.rows, done, take);
                @memcpy(m.buf(.ids).slice(u32, take), it.follow[done..][0..take]);
                const pos_dst = m.buf(.pos).slice(u32, take);
                for (pos_dst, 0..) |*p, r| p.* = @intCast(pos + r);
                const cache_view = m.cacheView(ctx);
                var plan = try attn.Plan.init(m.gpa, &.{.{ .parents = m.parents_storage[0..take], .start = @intCast(pos - ctx.begin), .capacity = cache_view.capacity }}, m.geometry);
                defer plan.deinit();
                try m.abs_scratch.upload(plan);
                try m.abs_binding.upload(plan, &.{cache_view});
                const cb = m.model.queue.commandBuffer();
                const e = cb.compute(.serial);
                var ended = false;
                errdefer if (!ended) e.end();
                try m.encodeStep(e, &m.abs_scratch, &m.abs_binding, plan, cache_view, take, raw_ref, m.ref(.ids), m.ref(.pos), true, if (last_chunk and !it.prompt_only) ctx else null, it.propose and last_chunk and !it.prompt_only, 0);
                e.end();
                ended = true;
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.log.err("sensenova mtp absorb GPU failure: {s}", .{msg});
                    return error.GpuFailed;
                }
                ctx.len += take;
                done += take;
            }
            ctx.has_out = !it.prompt_only;
            ctx.has_pick0 = it.propose and !it.prompt_only;
        }
    }

    pub fn hold(m: *Mtp, items: []const HoldItem) !void {
        if (items.len > 1 and batch.canBatch(m, &.{}, items)) return batch.run(m, &.{}, items);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        for (items) |it| {
            const ctx = try m.ctxOf(it.stream);
            const depth = @min(it.depth, max_depth);
            if (depth == 0) {
                ctx.held_count = 0;
                continue;
            }
            if (!ctx.has_out) {
                for (ctx.held_tokens[0..depth]) |*t| t.* = it.pending;
                ctx.held_count = depth;
                continue;
            }
            const chain_steps: usize = if (depth > 1) depth - 1 else 0;
            const need_gpu = !ctx.has_pick0 or chain_steps > 0;
            if (need_gpu) {
                if (ctx.len < ctx.begin or (ctx.len - ctx.begin) + chain_steps > m.capacity) return error.BadMtpPosition;
                const cache_view = m.cacheView(ctx);
                var plans: [max_depth - 1]attn.Plan = undefined;
                var built: usize = 0;
                defer for (plans[0..built]) |*p| p.deinit();
                try m.stageChain(cache_view, ctx.begin, ctx.len, chain_steps, &plans, &built);
                const cb = m.model.queue.commandBuffer();
                const e = cb.compute(.serial);
                var ended = false;
                errdefer if (!ended) e.end();
                if (!ctx.has_pick0) {
                    try m.encodeHeadPick(e, ctx, 0);
                    ctx.has_pick0 = true;
                }
                try m.encodeChain(e, ctx, cache_view, &plans, chain_steps);
                e.end();
                ended = true;
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.log.err("sensenova mtp hold GPU failure: {s}", .{msg});
                    return error.GpuFailed;
                }
            }
            scratch_mod.populateHeld(ctx, it.pending, depth, m.vocab);
        }
    }

    pub fn draftRound(m: *Mtp, absorbs: []const AbsorbItem, holds: []const HoldItem) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const depth = if (holds.len == 1) @min(holds[0].depth, max_depth) else 0;
        if (absorbs.len == 1 and holds.len == 1 and absorbs[0].stream == holds[0].stream and
            !absorbs[0].prompt_only and absorbs[0].follow.len > 0 and absorbs[0].follow.len <= m.abs_rows and depth > 0)
        {
            const it = absorbs[0];
            const hit = holds[0];
            const ctx = try m.ctxOf(it.stream);
            const take: u32 = @intCast(it.follow.len);
            if (it.rows) |sel| if (sel.len != take) return error.BadMtpAbsorb;
            if (it.features.row_bytes != m.hidden * 2) return error.BadMtpFeatures;
            const start_begin = if (ctx.len == 0 and it.start > 0) it.start else ctx.begin;
            const start_len = if (ctx.len == 0 and it.start > 0) it.start else ctx.len;
            const chain_steps: usize = depth - 1;
            if (it.start != start_len or (it.start - start_begin) + take + chain_steps > m.capacity) {
                return error.BadMtpPosition;
            }
            const raw_ref = try scratch_mod.stageRawFeatures(m, it.features, it.rows, 0, take);
            @memcpy(m.buf(.ids).slice(u32, take), it.follow);
            const pos_dst = m.buf(.pos).slice(u32, take);
            for (pos_dst, 0..) |*p, r| p.* = @intCast(it.start + r);
            const cache_view = m.cacheViewAt(ctx, start_begin);
            var abs_plan = try attn.Plan.init(m.gpa, &.{.{ .parents = m.parents_storage[0..take], .start = @intCast(it.start - start_begin), .capacity = cache_view.capacity }}, m.geometry);
            defer abs_plan.deinit();
            try m.abs_scratch.upload(abs_plan);
            try m.abs_binding.upload(abs_plan, &.{cache_view});
            var plans: [max_depth - 1]attn.Plan = undefined;
            var built: usize = 0;
            defer for (plans[0..built]) |*p| p.deinit();
            try m.stageChain(cache_view, start_begin, it.start + take, chain_steps, &plans, &built);
            const cb = m.model.queue.commandBuffer();
            const e = cb.compute(.serial);
            var ended = false;
            errdefer if (!ended) e.end();
            try m.encodeStep(e, &m.abs_scratch, &m.abs_binding, abs_plan, cache_view, take, raw_ref, m.ref(.ids), m.ref(.pos), true, ctx, it.propose, 0);
            e.barrier();
            if (!it.propose) try m.encodeHeadPick(e, ctx, 0);
            try m.encodeChain(e, ctx, cache_view, &plans, chain_steps);
            e.end();
            ended = true;
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("sensenova mtp draftRound GPU failure: {s}", .{msg});
                return error.GpuFailed;
            }
            ctx.begin = start_begin;
            ctx.len = it.start + take;
            ctx.has_out = true;
            ctx.has_pick0 = true;
            scratch_mod.populateHeld(ctx, hit.pending, depth, m.vocab);
            return;
        }
        if ((absorbs.len > 1 or holds.len > 1) and batch.canBatch(m, absorbs, holds)) {
            return batch.run(m, absorbs, holds);
        }
        if (absorbs.len > 0) try m.absorb(absorbs);
        if (holds.len > 0) try m.hold(holds);
    }

    pub fn held(m: *const Mtp, streams: []const usize, outs: []const []u32) !void {
        for (streams, outs) |s, out| {
            const ctx = try m.ctxOf(s);
            if (out.len > ctx.held_count) return error.TooFewMtpDrafts;
            @memcpy(out, ctx.held_tokens[0..out.len]);
        }
    }

    /// Warm GEMM/splitK/attention pipelines before residency lock and time single-step `hold`.
    pub fn warmup(m: *Mtp, io: std.Io) !void {
        return scratch_mod.warmup(m, io);
    }

    fn stageChain(m: *Mtp, cache_view: attn.Cache, start_begin: u64, base_pos: u64, chain_steps: usize, plans: *[max_depth - 1]attn.Plan, built: *usize) !void {
        const cpos_ptr: [*]u32 = @ptrCast(@alignCast(m.buf(.chain_pos).contents()));
        for (0..chain_steps) |s| {
            const pos_k = base_pos + s;
            cpos_ptr[s * batch_streams] = @intCast(pos_k);
            plans[s] = try attn.Plan.init(m.gpa, &.{.{ .parents = m.parents_storage[0..1], .start = @intCast(pos_k - start_begin), .capacity = cache_view.capacity }}, m.geometry);
            built.* += 1;
            try m.chain_scratch[s].upload(plans[s]);
            try m.chain_binding[s].upload(plans[s], &.{cache_view});
        }
    }

    fn encodeChain(m: *Mtp, e: mtl.ComputeEncoder, ctx: *StreamCtx, cache_view: attn.Cache, plans: *const [max_depth - 1]attn.Plan, chain_steps: usize) !void {
        for (0..chain_steps) |s| {
            const hin = ctx.chainOutRef(s, m.hidden);
            const ids_ref = projection.Ref{ .buffer = ctx.picks_buf, .offset = s * 16 };
            const pos_ref = projection.Ref{ .buffer = m.buf(.chain_pos), .offset = s * batch_streams * 4 };
            try m.encodeStep(e, &m.chain_scratch[s], &m.chain_binding[s], plans[s], cache_view, 1, hin, ids_ref, pos_ref, false, ctx, true, s + 1);
        }
    }

    pub fn encodeHeadBatch(m: *Mtp, e: mtl.ComputeEncoder, in_ref: projection.Ref, ctxs: []const *StreamCtx, slot_idx: usize) !void {
        const rows: u32 = @intCast(ctxs.len);
        try scratch_mod.encodeHead(m, e, in_ref, rows);
        e.barrier();
        for (ctxs, 0..) |ctx, i| {
            try m.ops.argmax(e, .{ .buf = m.buf(.logits), .off = i * m.vocab * 2 }, ctx.pickRef(slot_idx), .{ .rows = 1, .vocab = @intCast(m.vocab), .stride = @intCast(m.vocab) });
        }
        e.barrier();
    }

    fn encodeHeadPick(m: *Mtp, e: mtl.ComputeEncoder, ctx: *StreamCtx, slot_idx: usize) !void {
        return m.encodeHeadBatch(e, ctx.chainOutRef(slot_idx, m.hidden), &.{ctx}, slot_idx);
    }

    pub fn encodeTransformer(
        m: *Mtp,
        e: mtl.ComputeEncoder,
        scratch: *const attn.Scratch,
        binding: *const attn.Binding,
        plan: attn.Plan,
        caches: []const attn.Cache,
        rows: u32,
        raw_hin: projection.Ref,
        embed_ids: ?projection.Ref,
        pos_ref: projection.Ref,
        norm_target_hin: bool,
        keys_only: bool,
        dims_slot: usize,
    ) !void {
        const cfg = m.model.config;
        const glue = m.model.glue;
        const sn = &m.model.kernels.sn.?;
        const w = &m.weights;
        const hidden_u32: u32 = @intCast(m.hidden);
        const kv_dim_u32: u32 = @intCast(m.kv_dim);
        const prompt_lin = rows >= 16 and caches.len == 1;
        const hin_norm_src = if (norm_target_hin) blk: {
            try glue.headNormWith(e, raw_hin, wts.ref(m.model.weights.norm), m.ref(.hin), hidden_u32, rows, cfg.eps, cfg.shift_norms);
            break :blk m.ref(.hin);
        } else raw_hin;
        if (embed_ids) |ids_ref| {
            try scratch_mod.encodeEmbed(m, e, ids_ref, m.ref(.h), rows);
        }
        e.barrier();
        const en_plane = m.ref(.planes);
        const hn_plane = projection.Ref{ .buffer = m.buf(.planes), .offset = @as(usize, m.abs_rows) * m.hidden * 2 };
        try glue.headNormWith(e, m.ref(.h), wts.ref(w.norm_e), en_plane, hidden_u32, rows, cfg.eps, true);
        try glue.headNormWith(e, hin_norm_src, wts.ref(w.norm_h), hn_plane, hidden_u32, rows, cfg.eps, true);
        e.barrier();
        try glue.tapRows(e, m.ref(.planes), m.ref(.kept), m.ref(.cat), .{ .rows = rows, .width = hidden_u32, .capacity = m.abs_rows, .planes = 2 });
        e.barrier();
        try scratch_mod.encodeMat(sn, e, w.fc, m.ref(.cat), m.ref(.h), rows, hidden_u32, prompt_lin, null);
        e.barrier();
        try glue.headNormWith(e, m.ref(.h), wts.ref(w.input_norm), m.ref(.x), hidden_u32, rows, cfg.eps, true);
        e.barrier();
        const q_dim_u32: u32 = @intCast(2 * m.heads * m.head_dim);
        try scratch_mod.encodeMatSplit(sn, e, w.qkv, m.ref(.x), m.ref(.qg), rows, q_dim_u32, prompt_lin, null, .{
            .split1 = q_dim_u32,
            .y1 = .{ .buf = m.buf(.k) },
            .ldy1 = kv_dim_u32,
            .split2 = q_dim_u32 + kv_dim_u32,
            .y2 = .{ .buf = m.buf(.v) },
            .ldy2 = kv_dim_u32,
        });
        e.barrier();
        const prep: attn.Preprocess = .{ .qg = m.ref(.qg), .key = m.ref(.k), .q_gain = wts.ref(w.q_norm), .k_gain = wts.ref(w.k_norm), .norm_q = m.ref(.norm_q), .norm_k = m.ref(.norm_k), .query = m.ref(.query), .rotated_key = m.ref(.rot_k), .positions = pos_ref, .rows = rows, .eps = cfg.eps, .theta = cfg.rope_theta, .geometry = m.geometry, .shift = true, .rope = cfg.rope };
        if (keys_only) {
            try m.attention.keysOnly(e, scratch, binding, plan, prep, m.ref(.v), caches);
            return;
        }
        try m.attention.preprocess(e, prep);
        try m.attention.encode(e, scratch, binding, plan, m.ref(.query), m.ref(.rot_k), m.ref(.v), m.ref(.attn_out), caches, m.ref(.qg), prompt_lin, false);
        e.barrier();
        try scratch_mod.encodeMat(sn, e, w.o, m.ref(.attn_out), m.ref(.r), rows, hidden_u32, prompt_lin, null);
        e.barrier();
        glue.residual(e, m.ref(.h), m.ref(.r), m.ref(.hin), hidden_u32, rows);
        e.barrier();
        try glue.headNormWith(e, m.ref(.hin), wts.ref(w.post_norm), m.ref(.x), hidden_u32, rows, cfg.eps, true);
        e.barrier();
        try scratch_mod.encodeMat(sn, e, w.gate_up, m.ref(.x), m.ref(.gu), rows, @intCast(2 * m.intermediate), prompt_lin, null);
        e.barrier();
        const gu_ref = m.ref(.gu);
        const up_ref = projection.Ref{ .buffer = gu_ref.buffer, .offset = gu_ref.offset + m.intermediate * 2 };
        std.debug.assert(dims_slot <= max_depth);
        const dims_slice = m.buf(.dummy_dims).slice(i32, (max_depth + 1) * 4);
        dims_slice[dims_slot * 4] = @intCast(rows);
        dims_slice[dims_slot * 4 + 1] = @intCast(16 * ((rows + 15) / 16));
        const dims_ref = projection.Ref{ .buffer = m.buf(.dummy_dims), .offset = dims_slot * 16 };
        try m.model.kernels.mlp(e, gu_ref, up_ref, dims_ref, m.ref(.act), m.ref(.dummy_sums), @intCast(m.intermediate), rows);
        const down_xs = if (comptime @hasDecl(@import("core"), "sensenova_loop")) blk: {
            const v = if (m.model.kernels.sn_xs128) |xs128| xs128.sums else null;
            m.model.kernels.sn_xs128 = null;
            break :blk v;
        } else null;
        e.barrier();
        try scratch_mod.encodeMat(sn, e, w.down, m.ref(.act), m.ref(.r), rows, hidden_u32, prompt_lin, down_xs);
        e.barrier();
        glue.residual(e, m.ref(.hin), m.ref(.r), m.ref(.h), hidden_u32, rows);
        e.barrier();
        try glue.headNormWith(e, m.ref(.h), wts.ref(w.norm), m.ref(.x), hidden_u32, rows, cfg.eps, true);
        e.barrier();
    }

    pub fn encodeStep(
        m: *Mtp,
        e: mtl.ComputeEncoder,
        scratch: *const attn.Scratch,
        binding: *const attn.Binding,
        plan: attn.Plan,
        cache_view: attn.Cache,
        rows: u32,
        raw_hin: projection.Ref,
        ids_ref: projection.Ref,
        pos_ref: projection.Ref,
        norm_target_hin: bool,
        ctx_out: ?*StreamCtx,
        predict_pick: bool,
        slot_idx: usize,
    ) !void {
        const dims_slot = if (ctx_out != null) slot_idx else 0;
        try m.encodeTransformer(e, scratch, binding, plan, &.{cache_view}, rows, raw_hin, ids_ref, pos_ref, norm_target_hin, ctx_out == null, dims_slot);
        if (ctx_out) |ctx| {
            const hidden_u32: u32 = @intCast(m.hidden);
            const src = projection.Ref{ .buffer = m.buf(.x), .offset = @as(usize, rows - 1) * m.hidden * 2 };
            try m.model.glue.unstack(e, src, ctx.chainOutRef(slot_idx, m.hidden), .{ .rows = 1, .width = hidden_u32, .stride = hidden_u32, .offset = 0 });
            e.barrier();
            if (predict_pick) try m.encodeHeadPick(e, ctx, slot_idx);
        }
    }

    pub fn cacheView(m: *const Mtp, ctx: *const StreamCtx) attn.Cache {
        return m.cacheViewAt(ctx, ctx.begin);
    }

    pub fn cacheViewAt(m: *const Mtp, ctx: *const StreamCtx, begin: u64) attn.Cache {
        return .{
            .keys = .{ .buffer = ctx.keys, .offset = begin * m.head_dim * 2 },
            .values = .{ .buffer = ctx.values, .offset = begin * m.head_dim * 2 },
            .capacity = @intCast(m.capacity - begin),
            .key_stride = @intCast(m.capacity * m.head_dim),
            .value_stride = @intCast(m.capacity * m.head_dim),
        };
    }

    pub fn buf(m: *const Mtp, f: ScratchField) mtl.Buffer {
        return m.bufs[@backingInt(f)];
    }

    pub fn ref(m: *const Mtp, f: ScratchField) projection.Ref {
        return .{ .buffer = m.buf(f) };
    }
};
