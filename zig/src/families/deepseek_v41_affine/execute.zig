//! Run prefill and decode passes over the affine checkpoint's cached state.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const c = @import("config.zig");
const dspark = @import("draft.zig");
const layer = @import("layer.zig");
const ahead = @import("ahead.zig");
const api = @import("forward.zig");
const Enc = api.Enc;
const Head = api.Head;
const Dump = api.Dump;
const Vis = api.Vis;
const Next = api.Next;
const Buffer = mtl.Buffer;
const nax_rows = api.nax_rows;
const early_commit = 2;
var early_event: ?mtl.Event = null;
var early_value: u64 = 0;

/// The groups this call completes at `ratio`: [first, end) (a group's last position is inside the call).
pub fn completed(base: usize, rows: usize, ratio: usize) [2]usize {
    return .{ base / ratio, (base + rows) / ratio };
}

pub fn run(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, dump: ?Dump) !void {
    return runWith(m, s, cache, tokens, head, dump, null);
}

/// `run` over prompt rows that may lie in image spans (`vis`): their rows are the tower's, not the token embedding.
pub fn runWith(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, dump: ?Dump, vis: ?*const Vis) !void {
    return runSegs(m, s, &.{.{ .cache = cache, .tokens = tokens }}, head, dump, vis, null, false, false, 0);
}

/// `runWith` for a prompt pass's chunk: `next`, the chunk after it, has its engram rows gathered while this one runs.
pub fn runChunk(m: *Model, s: *st.Scratch, cache: *st.Cache, tokens: []const u32, head: Head, vis: ?*const Vis, next: ?Next, decoder_skip: bool, export_at: usize) !void {
    return runSegs(m, s, &.{.{ .cache = cache, .tokens = tokens }}, head, null, vis, next, false, decoder_skip, export_at);
}

/// A stream's rows in a forward: its cache, its tokens, from scratch row `r0` (the rows of the streams before it).
pub const Seg = struct { cache: *st.Cache, tokens: []const u32, r0: usize = 0 };

/// Several streams' rows in one forward, each stream's attention on its own rows (row kernels: each matches alone).
pub fn many(m: *Model, s: *st.Scratch, segs: []const Seg, head: Head) !void {
    const short_index = m.short_index and !m.projection_audit and segs.len == 1 and
        segs[0].tokens.len <= 1 + c.draft_block and segs[0].cache.len + segs[0].tokens.len <= st.picks * 2 + 1 and
        (head == .all or head == .greedy);
    runSegs(m, s, segs, head, null, null, null, short_index, false, 0) catch |err| {
        if (!short_index or err != error.DeepseekRetryShortIndex) return err;
        // The failed pass wrote only tentative rows, inside the unwrapped

        // ring. Cache lengths and the last committed logits are untouched.
        segs[0].cache.short_index_replays += 1;
        try runSegs(m, s, segs, head, null, null, null, false, false, 0);
    };
}

fn runSegs(m: *Model, s: *st.Scratch, segs: []const Seg, head: Head, dump: ?Dump, vis: ?*const Vis, next: ?Next, short_index: bool, decoder_skip: bool, export_at: usize) !void {
    var rows: usize = 0;
    for (segs) |g| {
        if (g.r0 != rows or g.tokens.len == 0 or g.tokens.len > st.max_rows) return error.DeepseekRowsExceeded;
        if (g.cache.len + g.tokens.len > g.cache.capacity) return error.DeepseekCacheFull;
        rows += g.tokens.len;
    }
    if (rows == 0 or rows > s.rows) return error.DeepseekRowsExceeded;
    if (segs.len > 1 and (vis != null or dump != null or head == .last)) return error.DeepseekSharedForward;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const audit_attention = (m.audit_attention_clip or m.attention_eight_audit or m.attention_stream_audit) and segs.len == 1 and rows <= 1 + c.draft_block;
    if (audit_attention) s.attention_differs.slice(u32, 1)[0] = 0;

    // host side: ids, positions, the engram history and lookups, the first collapse (copy 0)
    var images = false;
    var dsts: [c.engram_layers.len][]u16 = undefined;
    // a lone chunk gathered ahead (runChunk's `next` the call before) keeps its ids and rows; the rest is gathered now
    const ready = segs.len == 1 and ahead.preparedFor(s, segs[0]);
    if (ready) s.engram_set = s.prepared.?.set;
    s.prepared = null;
    for (segs) |g| {
        const base = g.cache.len;
        const n = g.tokens.len;
        for (s.ids.slice(u32, g.r0 + n)[g.r0..], g.tokens) |*d, t| d.* = t;
        for (s.pos.slice(i32, g.r0 + n)[g.r0..], 0..) |*p, i| p.* = @intCast(base + i);
        if (!ready) g.cache.engram_ids.shrinkRetainingCapacity(base);
        for (g.tokens, s.img.slice(i32, g.r0 + n)[g.r0..], s.img_code.slice(i32, g.r0 + n)[g.r0..], 0..) |t, *flag, *code, i| {
            code.* = Vis.code(vis, base + i);
            flag.* = @intFromBool(code.* != -1);
            images = images or code.* != -1;
            if (!ready) g.cache.engram_ids.appendAssumeCapacity(m.engram.compress(t, code.* == -1)); // image rows end every n-gram
        }
        if (ready) continue;

        const width = c.engram_cols * c.engram_dim;
        for (&dsts, s.engram_rows[s.engram_set]) |*d, b| d.* = b.slice(u16, (g.r0 + n) * width)[g.r0 * width ..];
        try m.engram.gatherRows(g.cache.engram_ids.items, base, n, dsts);
    }
    for (s.pre[0].slice(f32, rows * c.hc), 0..) |*v, i| v.* = if (i % c.hc == 0) 1 else 0;

    if (m.projection_audit) s.projection_error.slice(u32, 1)[0] = 0;
    if (short_index) s.short_index_status.slice(u32, 1)[0] = 0;
    const wo_decode = m.woa_batch != null and segs.len == 1 and (head == .all or head == .greedy);
    const audit_wo = wo_decode and m.woa_batch.?.audit != null;
    if (audit_wo) s.wo_mismatch.slice(u32, 1)[0] = 0;
    const overlap_hc = m.hc_overlap != null and segs.len == 1 and rows <= m.hc_overlap_cap and
        (head == .all or head == .greedy);
    const hc_audit = overlap_hc and m.hc_overlap.?.audit != null;
    if (hc_audit) s.hc_overlap_error.slice(u32, 1)[0] = 0;
    const router_audit = if (m.router_variant) |p| p.audit != null else false;
    if (router_audit) s.router_variant_error.slice(u32, 1)[0] = 0;
    var first: ?mtl.CommandBuffer = null;
    defer if (first) |f| {
        f.wait();
        f.release();
    };
    var cb = m.queue.commandBufferUnretained(); // every buffer it binds is the model's or the scratch's; cb.wait() below
    const w = &m.weights;
    var e = cb.compute(.concurrent);
    api.pending_expand = false;
    api.pending_topk = 0;
    var x: Enc = .{ .ep = &e, .cbp = &cb, .m = m, .s = s, .rows = rows, .base = segs[0].cache.len, .units = segs.len == 1 and rows >= nax_rows, .segs = segs, .short_index = short_index, .wo_decode = wo_decode, .overlap_hc = overlap_hc };
    x.pipe("ds_embed");
    x.buf(s.ids, 0, 0);
    x.tensor(w.embed, 1);
    x.buf(s.x, 0, 2);
    x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    // the call's RoPE (cos, sin) at s.pos, beside the embedding: the kernels that rotate by s.pos read it (trig)
    x.with();
    x.pipe("ds_rope_table");
    x.buf(s.pos, 0, 0);
    x.buf(m.freq_plain, 0, 1);
    x.buf(m.freq_compressed, 0, 2);
    x.buf(s.trig, 0, 3);
    x.args(@as(i32, @intCast(rows)), 4);
    x.threads(.{ 32, rows, 2 }, .{ 32, 1, 1 });
    s.trig_rows = rows;
    defer s.trig_rows = 0;
    if (images) {
        const v = &(m.vision orelse return error.NoVisionTower);
        x.pipe("ds_put_image");
        x.buf(s.x, 0, 0);
        x.buf(s.img_code, 0, 1);
        x.buf(vis.?.rows, 0, 2);
        x.tensor(v.start, 3);
        x.tensor(v.end, 4);
        x.tensor(v.newline, 5);
        x.threads(.{ c.hidden, rows, 1 }, .{ 256, 1, 1 });
    }
    const drafting = w.stages != null and for (segs) |g| {
        if (g.cache.stage_rings == null) break false;
    } else true;
    const can_skip = decoder_skip and x.units and m.kernels.attend_nax != null and vis == null;
    const planner = @import("decoder_prefill.zig");
    const deferred = can_skip and export_at != 0;
    const chunk_end = x.base + rows;
    const command_split = if (deferred)
        if (planner.inputStart(export_at, planner.global_kv_layer) < chunk_end) planner.global_kv_layer else 0
    else
        early_commit;
    std.debug.assert(!deferred or export_at >= chunk_end);
    var decoder_complete = true;
    for (&w.layers, 0..) |*l, i| {
        if (deferred and i >= planner.global_kv_layer) {
            const floor = planner.inputStart(export_at, i);
            if (floor >= chunk_end) {
                if (i == planner.global_kv_layer) try layer.decoderKeys(x, i, l);
                decoder_complete = false;
                break;
            }
            x.window_floor = floor;
        }
        if (can_skip) {
            // Keep the original tensor path even at a seam with only a few

            // useful rows. Padding here only affects discarded coordinates.
            const keep = if (deferred)
                if (i <= planner.global_kv_layer) rows else @min(rows, @max(nax_rows, chunk_end - x.window_floor))
            else
                planner.inputRows(rows, i);
            const discard = x.rows - keep;
            std.debug.assert(discard == 0 or api.pending_expand);
            x.rows = keep;
            x.base += discard;
            x.token_skip += discard;
            x.expand_skip = discard;
            if (!deferred and x.token_skip != 0) x.window_floor = x.base;
        }
        if (drafting) if (std.mem.indexOfScalar(usize, &c.draft_targets, i)) |t| {
            // the draft head reads its target layers' attention inputs: the mean of the four streams
            x.flush();
            x.pipe("ds_stream_mean");
            x.buf(s.x, 0, 0);
            x.buf(s.mh, 0, 1);
            x.args([2]i32{ @intCast(t * c.hidden), @intCast(c.draft_targets.len * c.hidden) }, 2);
            x.threads(.{ c.hidden, x.rows, 1 }, .{ 256, 1, 1 });
        };
        try layer.block(x, i, l);
        x.expand_skip = 0;
        // the first layers run while the host encodes the rest (a GPU event orders the two command buffers)
        if (i + 1 == command_split and dump == null) {
            const ev = early_event orelse blk: {
                early_event = try m.device.event();
                break :blk early_event.?;
            };
            early_value += 1;
            e.end();
            cb.signalEvent(ev, early_value);
            first = cb.retain(); // the call's first layers: checked, timed and released after its wait
            cb.commit();
            cb = m.queue.commandBufferUnretained();
            cb.waitForEvent(ev, early_value);
            e = cb.compute(.concurrent);
        }
        if (dump) |d| { // the streams after each block: the buffers are untracked, so each block completes first
            x.flush();
            e.end();
            cb.commit();
            cb.wait();
            if (cb.status() != .completed) return error.DeepseekGpuFailed;
            @memcpy(d.layers[i].contents()[0 .. rows * c.hc * c.hidden * 2], s.x.contents()[0 .. rows * c.hc * c.hidden * 2]);
            cb = m.queue.commandBufferUnretained();
            e = cb.compute(.concurrent);
        }
    }
    std.debug.assert(decoder_complete or head == .none);
    const overlap = m.condition_overlap and drafting and segs.len == 1 and rows <= 1 + c.draft_block and !x.units and
        (head == .all or head == .greedy);
    const condition_audit = overlap and m.condition_overlap_audit;
    if (condition_audit) {
        s.condition_error.slice(u32, 1)[0] = 0;
        dspark.condition(x);
        checkConditionRings(x, false);
    }
    if (overlap) {
        conditionTail(x, head);
        if (condition_audit) checkConditionRings(x, true);
    } else if (decoder_complete) {
        if (drafting) dspark.condition(x);
        // the last block's FFN pre collapses the streams for the head
        x.boundary(null, s.pre[0], s.pre[1], w.norm);
        switch (head) {
            .none => {},
            .last => x.linear(w.head, s.normed, (x.rows - 1) * c.hidden * 2, s.logits, 0, 1),
            .all, .greedy => {
                x.linear(w.head, s.normed, 0, s.logits, 0, rows);
                if (head == .greedy) {
                    const p = m.kernels.greedy.?;
                    x.fence();
                    p.encodeTiles(e, s.logits, s.greedy_parts, rows);
                    x.fence();
                    p.encodeFinish(e, s.greedy_parts, s.greedy_tokens, rows);
                }
            },
        }
    }
    e.end();

    cb.commit();
    if (next) |nx| if (segs.len == 1) { // the next chunk's engram rows, on the host while the GPU runs this one

        ahead.prepare(m, s, segs[0].cache, segs[0].cache.len + rows, nx);
    };
    cb.wait();
    const early_ok = if (first) |f| f.status() == .completed else true;
    if (first) |f| if (!early_ok) {
        if (f.failure()) |why| std.log.err("deepseek_v41: the forward's first layers failed: {s}", .{why});
    };
    if (!early_ok) return error.DeepseekGpuFailed;
    if (cb.status() != .completed) {
        if (cb.failure()) |why| std.log.err("deepseek_v41: the forward failed: {s}", .{why});
        return error.DeepseekGpuFailed;
    }
    if (router_audit and s.router_variant_error.slice(u32, 1)[0] != 0) {
        std.log.err("fast router differs mask={d}", .{s.router_variant_error.slice(u32, 1)[0]});
        return error.RouterVariantLiveDiffers;
    }
    if (overlap_hc) segs[0].cache.hc_wide_rounds[rows - 1] += 1;
    if (hc_audit and s.hc_overlap_error.slice(u32, 1)[0] != 0) {
        std.log.err("hyperconnection live audit differs mask={d}", .{s.hc_overlap_error.slice(u32, 1)[0]});
        return error.HcOverlapLiveDiffers;
    }
    if (audit_wo and s.wo_mismatch.slice(u32, 1)[0] != 0) return error.DeepseekWoABatchDiffers;
    if (short_index) {
        const status = s.short_index_status.slice(u32, 1)[0];
        if (status & 1 != 0) return error.DeepseekRetryShortIndex;
        if (status & 2 != 0) return error.DeepseekShortIndexDiffers;
    }
    if (audit_attention and s.attention_differs.slice(u32, 1)[0] != 0)
        return error.DeepseekClippedAttentionDiffers;
    if (m.projection_audit and s.projection_error.slice(u32, 1)[0] != 0)
        return error.DeepseekAttentionProjectionDiffers;
    if (condition_audit) {
        if (s.condition_error.slice(u32, 1)[0] != 0) return error.DeepseekConditionDiffers;
        segs[0].cache.condition_audited += rows * c.stages * c.head_dim;
    }
    for (segs) |g| {
        const n = g.tokens.len;
        g.cache.len += n;
        if (head != .none) {
            const last = if (head == .last) 0 else g.r0 + n - 1;
            @memcpy(g.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + last * c.vocab * 2)[0 .. c.vocab * 2]);
        }
        g.cache.last_start = g.cache.len - n;
        g.cache.last_rows = n;
        g.cache.last_r0 = g.r0;
    }
}

/// Overlap the target head with DSpark ring updates, which read disjoint buffers.
fn conditionTail(x: Enc, head: Head) void {
    const s = x.s;
    const w = &x.m.weights;
    const stages = &w.stages.?;
    const view = x.of(x.segs[0]);
    const rings = x.segs[0].cache.stage_rings.?;

    x.boundary(null, s.pre[0], s.pre[1], w.norm);
    x.with();
    x.linear(stages[0].main_proj.?, s.mh, 0, s.mainx, 0, x.rows);

    x.rms(s.mainx, 0, stages[0].main_norm.?, s.mainx, 0, x.rows, c.hidden);
    x.with();
    x.linear(w.head, s.normed, 0, s.logits, 0, x.rows);

    const greedy = if (head == .greedy) x.m.kernels.greedy else null;
    if (greedy) |p| {
        x.fence();
        p.encodeTiles(x.ep.*, s.logits, s.greedy_parts, x.rows);
    }
    for (stages, s.condition_kv, 0..) |*stage, kv, k| {
        if (k != 0 or greedy != null) x.with();
        x.linear(stage.block.attn.wkv, s.mainx, 0, kv, 0, x.rows);
    }

    if (greedy) |p| {
        x.fence();
        p.encodeFinish(x.ep.*, s.greedy_parts, s.greedy_tokens, x.rows);
    }
    for (stages, s.condition_kv, rings, 0..) |*stage, kv, ring, k| {
        if (k != 0 or greedy != null) x.with();
        view.kvFinish(kv, stage.block.attn.kv_norm, x.m.freq(c.layers + k), ring);
    }
}

fn checkConditionRings(x: Enc, compare: bool) void {
    const s = x.s;
    const rings = x.segs[0].cache.stage_rings.?;
    for (rings, 0..) |ring, k| {
        x.pipe("tf_condition_ring_audit");
        x.buf(ring, 0, 0);
        x.buf(s.pos, 0, 1);
        x.buf(s.condition_ref, k * (1 + c.draft_block) * c.head_dim * 2, 2);
        x.buf(s.condition_error, 0, 3);
        x.args([2]u32{ st.ring, @intFromBool(compare) }, 4);
        x.threads(.{ c.head_dim, x.rows, 1 }, .{ 256, 1, 1 });
    }
}
