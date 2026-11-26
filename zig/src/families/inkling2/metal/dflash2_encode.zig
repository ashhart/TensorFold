//! Metal command-buffer encoding and absorb/block scheduling helpers for Inkling 2 Mini DFlash 2.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ck = @import("../../../core/checkpoint_metal.zig");
const w = @import("dflash2_weights.zig");
const k = @import("dflash2_kernels.zig");
const q4 = @import("draft_head_q4.zig");
const st = @import("dflash2_state.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = k.Ref;

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}

fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

pub fn effectiveFrom(wi: st.AbsorbWork) u64 {
    var from = wi.lane.from;
    if (wi.start > wi.prev_end) from = wi.start;
    if (wi.keep_from > from and wi.keep_from > wi.prev_end) from = wi.keep_from;
    return from;
}

pub fn postAbsorb(items: []const dr.Absorb, work: []const st.AbsorbWork, stream: *Stream, lane: *const st.Lane) struct { end: u64, from: u64 } {
    var end = lane.end;
    var from = lane.from;
    for (items, work) |it, wi| if (it.stream == stream) {
        end = wi.end;
        from = effectiveFrom(wi);
    };
    return .{ .end = end, .from = from };
}

pub fn project(d: anytype, e: mtl.ComputeEncoder, linear: w.Linear, input: Ref, output: Ref, rows: usize) !void {
    if (d.loaded.copies) |*copies| {
        return copies.encodeProject(e, linear.key, linear.n, linear.k, input, output, rows);
    }
    d.fast.dispatch(d.target, e, linear.bf16 orelse return error.DraftPreparationUnqualified, input, output, rows, false);
}

pub fn absorbBatch(d: anytype, e: mtl.ComputeEncoder, taps: Ref, segs: []const st.AbsorbSeg, total_rows: usize) !void {
    const c = &d.loaded.config;
    const s = &d.scratch;
    try project(d, e, d.loaded.weights.fc, taps, at(s.fcx, 0), total_rows);
    e.barrier();
    k.norm16(&d.pipelines, e, at(s.fcx, 0), r(d.loaded.weights.hidden_norm), at(s.hctx, 0), .{ .rows = @intCast(total_rows), .width = c.hidden, .eps = c.eps, .add = 0 });
    e.barrier();
    var ctx_segs: [st.max_batch_streams]k.CtxSeg = undefined;
    var row_off: u32 = 0;
    for (segs, 0..) |seg, bi| {
        ctx_segs[bi] = .{ .row0 = row_off, .rows = @intCast(seg.rows), .pos0 = @intCast(seg.pos0) };
        row_off += @intCast(seg.rows);
    }
    const ctx_args: k.CtxArgs = .{ .rows = @intCast(total_rows), .segs = @intCast(segs.len), .ring = @intCast(w.ring_size), .eps = c.eps, .theta = c.theta };
    for (d.loaded.weights.layers, 0..) |layer, i| {
        try project(d, e, layer.k_proj, at(s.hctx, 0), at(s.ck, 0), total_rows);
        try project(d, e, layer.v_proj, at(s.hctx, 0), at(s.cv, 0), total_rows);
        e.barrier();
        var rings: [st.max_batch_streams]k.Ring = undefined;
        for (segs, 0..) |seg, bi| rings[bi] = .{ .k = at(seg.lane.k[i], 0), .v = at(seg.lane.v[i], 0) };
        k.ctxKv(&d.pipelines, e, at(s.ck, 0), at(s.cv, 0), r(layer.k_norm), rings[0..segs.len], ctx_args, ctx_segs[0..segs.len]);
        if (i + 1 < d.loaded.weights.layers.len) e.barrier();
    }
}

pub fn prepareAbsorbs(d: anytype, items: []const dr.Absorb, work: []st.AbsorbWork) !bool {
    const row_bytes = w.taps_n * w.hidden * 2;
    const keep_window: u64 = w.window_size + w.block_size;
    var has_rows = false;
    for (items, work, 0..) |item, *wi, ii| {
        const lane = d.contexts.getPtr(item.stream) orelse return error.DraftStreamUnknown;
        const features = item.features;
        if (features.space != .device or features.dtype != .bf16 or features.row_bytes != row_bytes) return error.DraftFeaturesUnsupported;
        const n: usize = if (item.rows) |rs| blk: {
            if (rs.len == 0 or rs.len > st.max_window_rows or item.prompt) return error.DraftFeaturesUnsupported;
            for (rs, 0..) |rv, ri| {
                if (rv >= features.rows) return error.DraftFeaturesUnsupported;
                if (ri > 0 and rv <= rs[ri - 1]) return error.DraftFeaturesUnsupported;
            }
            break :blk rs.len;
        } else blk: {
            if (features.rows == 0 or features.rows > 1048576) return error.DraftFeaturesUnsupported;
            break :blk features.rows;
        };
        var prev_end = lane.end;
        for (items[0..ii], work[0..ii]) |prev_it, prev_w| if (prev_it.stream == item.stream) {
            prev_end = prev_w.end;
        };
        if (item.start < prev_end) return error.DraftContextGap;
        const end = item.start + n;
        const keep_from = if (item.prompt) @max(item.start, item.stream.prompt_len -| keep_window) else item.start;
        if (keep_from < end) has_rows = true;
        wi.* = .{ .lane = lane, .prev_end = prev_end, .start = item.start, .keep_from = keep_from, .end = end, .src = .{ .id = @ptrFromInt(features.buffer) }, .offset = features.offset, .rows_map = item.rows };
    }
    return has_rows;
}

pub fn encodeAbsorbs(d: anytype, cb: mtl.CommandBuffer, work: []const st.AbsorbWork) !void {
    const row_bytes = w.taps_n * w.hidden * 2;
    var active_blit: ?mtl.BlitEncoder = null;
    var active_compute: ?mtl.ComputeEncoder = null;
    errdefer {
        if (active_blit) |b| b.end();
        if (active_compute) |e| e.end();
    }
    const seg_limit = @min(d.max_batch, st.max_batch_streams);
    var idx: usize = 0;
    var pos: u64 = work[0].keep_from;
    while (idx < work.len) {
        var segs: [st.max_batch_streams]st.AbsorbSeg = undefined;
        var seg_count: usize = 0;
        var total_rows: usize = 0;
        var contiguous = true;
        while (idx < work.len and seg_count < seg_limit and total_rows < st.absorb_rows) {
            const wi = work[idx];
            if (pos >= wi.end) {
                idx += 1;
                if (idx < work.len) pos = work[idx].keep_from;
                continue;
            }
            var same_lane = false;
            for (segs[0..seg_count]) |prev_s| if (prev_s.lane == wi.lane) {
                same_lane = true;
                break;
            };
            if (same_lane) break;
            const take: usize = @intCast(@min(st.absorb_rows - total_rows, wi.end - pos));
            const rel: usize = @intCast(pos - wi.start);
            const off = if (wi.rows_map != null) wi.offset else wi.offset + rel * row_bytes;
            if (wi.rows_map != null) contiguous = false else if (seg_count > 0) {
                const prev = segs[seg_count - 1];
                if (prev.rows_map != null or wi.src.id != prev.src.id or off != prev.src_off + prev.rows * row_bytes) contiguous = false;
            }
            const sub_map = if (wi.rows_map) |rs| rs[rel..][0..take] else null;
            segs[seg_count] = .{ .lane = wi.lane, .src = wi.src, .src_off = off, .pos0 = pos, .rows = take, .rows_map = sub_map };
            seg_count += 1;
            total_rows += take;
            pos += take;
            if (pos >= wi.end) {
                idx += 1;
                if (idx < work.len) pos = work[idx].keep_from;
            }
        }
        if (seg_count == 0) break;
        const taps_ref = if (contiguous) at(segs[0].src, segs[0].src_off) else blk: {
            const b = cb.blit();
            active_blit = b;
            var dst_row: usize = 0;
            for (segs[0..seg_count]) |seg| {
                if (seg.rows_map) |rs| {
                    var ri: usize = 0;
                    while (ri < rs.len) {
                        var run: usize = 1;
                        while (ri + run < rs.len and rs[ri + run] == rs[ri] + run) run += 1;
                        b.copy(seg.src, seg.src_off + @as(usize, rs[ri]) * row_bytes, d.scratch.taps, dst_row * row_bytes, run * row_bytes);
                        dst_row += run;
                        ri += run;
                    }
                } else {
                    b.copy(seg.src, seg.src_off, d.scratch.taps, dst_row * row_bytes, seg.rows * row_bytes);
                    dst_row += seg.rows;
                }
            }
            b.end();
            active_blit = null;
            break :blk at(d.scratch.taps, 0);
        };
        const e = cb.compute(.concurrent);
        active_compute = e;
        try absorbBatch(d, e, taps_ref, segs[0..seg_count], total_rows);
        e.end();
        active_compute = null;
    }
}

pub fn commitAbsorbs(d: anytype, items: []const dr.Absorb, work: []const st.AbsorbWork) void {
    var prefix: [st.max_window_rows]u32 = undefined;
    for (&prefix, 0..) |*p, i| p.* = @intCast(i);
    for (items, work) |item, wi| {
        if (item.offered > 0 and wi.lane.grown > 0) {
            const kept = item.rows orelse prefix[0..@min(item.follow.len, prefix.len)];
            const sampled = if (item.stream.sampling) |s| s.temperature > 0 else false;
            d.landing.observeTree(sampled, wi.lane.parents[0..wi.lane.grown], wi.lane.own[0..wi.lane.grown], item.offered, kept);
        }
        if (!item.prompt) {
            if (item.offered > 0) {
                wi.lane.zero_streak = 0;
                wi.lane.plain_wait = 0;
            } else if (wi.lane.grown > 0) {
                wi.lane.zero_streak +|= 1;
                wi.lane.plain_wait = if (wi.lane.zero_streak <= 1)
                    0
                else
                    @min(@as(u32, 64), @as(u32, 1) << @intCast(@min(wi.lane.zero_streak - 1, 6)));
            } else if (!item.propose and wi.lane.plain_wait > 0) {
                wi.lane.plain_wait -= 1;
            }
            if (!item.propose and wi.lane.plain_wait == 0) {
                item.stream.draft_chances.clearRetainingCapacity();
                item.stream.draft_chances_cut = null;
            }
        }
        wi.lane.from = effectiveFrom(wi);
        wi.lane.end = wi.end;
        wi.lane.grown = 0;
        wi.lane.held_len = 0;
    }
}

pub fn encodeBlocks(d: anytype, e: mtl.ComputeEncoder, blocks: []const st.Block) !void {
    std.debug.assert(blocks.len > 0 and blocks.len <= d.max_batch);
    const c = &d.loaded.config;
    const s = &d.scratch;
    const n = w.block_size;
    const rows = blocks.len * n;
    const ids = s.ids.slice(u32, rows);
    const seq = s.seq.slice(u32, rows);
    for (blocks, 0..) |blk, bi| {
        const row0 = bi * n;
        ids[row0] = blk.pending;
        @memset(ids[row0 + 1 ..][0 .. n - 1], c.mask);
        seq[row0] = blk.pending;
    }
    k.embed(&d.pipelines, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.resid, 0), .{ .rows = @intCast(rows), .width = c.hidden, .use_norm = @intFromBool(d.use_embed_norm), .eps = c.eps });
    e.barrier();
    var block_qkv_args: k.BlockQkvArgs = .{ .rows = @intCast(rows), .block = @intCast(n), .eps = c.eps, .theta = c.theta, .pos0 = @splat(0) };
    var streams: [st.max_batch_streams]k.AttendStream = @splat(.{ .anchor = 0, .from = 0, .lo = 0, .retained = 0, .keys = 0, .chunks = 0 });
    var max_stream_chunks: u32 = 1;
    for (blocks, 0..) |blk, bi| {
        block_qkv_args.pos0[bi] = @intCast(blk.anchor);
        const lo = @max(blk.from, (blk.anchor + 1) -| c.window);
        const retained: u32 = @intCast(blk.anchor - lo);
        const keys: u32 = retained + @as(u32, n);
        const chunks: u32 = (keys + 127) / 128;
        max_stream_chunks = @max(max_stream_chunks, chunks);
        streams[bi] = .{ .anchor = @intCast(blk.anchor), .from = @intCast(blk.from), .lo = @intCast(lo), .retained = retained, .keys = keys, .chunks = chunks };
    }
    const attend_args: k.AttendArgs = .{ .rows = @intCast(rows), .block = @intCast(n), .window = c.window, .ring = @intCast(w.ring_size), .max_chunks = max_stream_chunks, .scale = 1.0 / @sqrt(128.0) };
    for (d.loaded.weights.layers, 0..) |layer, i| {
        k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.in_norm), at(s.x, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = @intFromBool(i > 0) });
        e.barrier();
        try project(d, e, layer.attn_conv.proj, at(s.x, 0), at(s.dyn, 0), rows);
        e.barrier();
        k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.xc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
        e.barrier();
        try project(d, e, layer.q_proj, at(s.xc, 0), at(s.q_raw, 0), rows);
        try project(d, e, layer.k_proj, at(s.xc, 0), at(s.k_raw, 0), rows);
        try project(d, e, layer.v_proj, at(s.xc, 0), at(s.v_raw, 0), rows);
        e.barrier();
        k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), block_qkv_args);
        e.barrier();
        var rings: [st.max_batch_streams]k.Ring = undefined;
        for (blocks, 0..) |blk, bi| rings[bi] = .{ .k = at(blk.lane.k[i], 0), .v = at(blk.lane.v[i], 0) };
        k.attendPart(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), rings[0..blocks.len], at(s.part, 0), attend_args, streams[0..blocks.len]);
        e.barrier();
        k.attendMerge(&d.pipelines, e, at(s.part, 0), at(s.attn, 0), attend_args, streams[0..blocks.len]);
        e.barrier();
        try project(d, e, layer.o_proj, at(s.attn, 0), at(s.y, 0), rows);
        e.barrier();
        k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.attn_conv.base), at(s.yc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
        e.barrier();
        k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(layer.post_norm), at(s.x, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 1 });
        e.barrier();
        try project(d, e, layer.mlp_conv.proj, at(s.x, 0), at(s.dyn, 0), rows);
        e.barrier();
        k.conv(&d.pipelines, e, at(s.x, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.xc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 0 });
        e.barrier();
        try project(d, e, layer.gate_proj, at(s.xc, 0), at(s.gate, 0), rows);
        try project(d, e, layer.up_proj, at(s.xc, 0), at(s.up, 0), rows);
        e.barrier();
        k.siluMul(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), @intCast(rows * w.intermediate));
        e.barrier();
        try project(d, e, layer.down_proj, at(s.act, 0), at(s.y, 0), rows);
        e.barrier();
        k.conv(&d.pipelines, e, at(s.y, 0), at(s.dyn, 0), r(layer.mlp_conv.base), at(s.yc, 0), .{ .rows = @intCast(rows), .width = c.hidden, .block = n, .groups = w.conv_groups, .side = 1 });
        e.barrier();
    }
    k.norm(&d.pipelines, e, at(s.resid, 0), at(s.yc, 0), r(d.loaded.weights.norm), at(s.x, 0), .{ .rows = @intCast(rows), .width = c.hidden, .eps = c.eps, .add = 1 });
    e.barrier();
    const logits_stride: u32 = if (d.loaded.draft_head_q4) |*head| blk: {
        std.debug.assert(head.n >= w.valid_vocab and head.n <= w.padded_vocab);
        q4.encodeBatchHead(d.fast.head_q4, e, head, at(s.x, 0), at(s.logits, 0), @intCast(rows), false);
        break :blk head.n;
    } else blk: {
        d.fast.dispatch(d.target, e, d.target.weights.unembed, at(s.x, 0), at(s.logits, 0), rows, true);
        break :blk @intCast(w.padded_vocab);
    };
    try project(d, e, d.loaded.weights.hidden_proj, at(s.x, 0), at(s.hid, 0), rows);
    e.barrier();
    k.topk16(&d.pipelines, e, at(s.logits, 0), at(s.tval, 0), at(s.tid, 0), .{ .rows = @intCast(rows), .vocab = w.valid_vocab, .k = c.top_k, .stride = logits_stride });
    if (d.tree_nodes == 0) {
        e.barrier();
        k.walk(&d.pipelines, e, at(s.tval, 0), at(s.tid, 0), at(s.hid, 0), r(d.loaded.weights.pred_cb), r(d.loaded.weights.succ_cb), at(s.seq, 0), .{ .slots = n - 1, .rank = c.rank, .top = c.top_k, .vocab = w.valid_vocab, .stride = n, .row0 = 1, .softcap = c.softcap }, @intCast(blocks.len));
    }
}

test "commitAbsorbs clears stale chances after one zero-offered round and backs off repeated trimmed trees" {
    const gpa = std.testing.allocator;
    const tree_mod = @import("dflash2_tree.zig");
    var d: struct { landing: tree_mod.Landing = .{} } = .{};
    var s = try Stream.init(gpa, .{ .id = "backoff", .prompt = &.{ 1, 2, 3 }, .max_new = 32 });
    defer s.deinit(gpa);
    var lane: st.Lane = .{ .k = undefined, .v = undefined, .grown = 4, .held_len = 4 };
    const feat: lanes.backend.Features = .{ .buffer = 0x1000, .rows = 1, .row_bytes = 64, .space = .device, .dtype = .bf16 };
    var work = [_]st.AbsorbWork{.{ .lane = &lane, .prev_end = 3, .start = 3, .keep_from = 3, .end = 4, .src = undefined, .offset = 0 }};

    // Single zero-offered round clears draft_chances immediately so the stream does not freeze at depth 0.
    try s.draft_chances.appendSlice(gpa, &.{ 0.2, 0.1 });
    commitAbsorbs(&d, &.{.{ .stream = &s, .features = feat, .start = 3, .follow = &.{4}, .prompt = false, .propose = false, .offered = 0 }}, &work);
    try std.testing.expectEqual(@as(usize, 0), s.draft_chances.items.len);
    try std.testing.expectEqual(@as(u32, 1), lane.zero_streak);
    try std.testing.expectEqual(@as(u32, 0), lane.plain_wait);

    // Second consecutive trimmed tree sets plain_wait=2 and retains low chances until backoff expires.
    lane.grown = 4;
    try s.draft_chances.appendSlice(gpa, &.{ 0.2, 0.1 });
    work[0].prev_end = 4;
    work[0].start = 4;
    work[0].keep_from = 4;
    work[0].end = 5;
    commitAbsorbs(&d, &.{.{ .stream = &s, .features = feat, .start = 4, .follow = &.{5}, .prompt = false, .propose = false, .offered = 0 }}, &work);
    try std.testing.expectEqual(@as(usize, 2), s.draft_chances.items.len);
    try std.testing.expectEqual(@as(u32, 2), lane.zero_streak);
    try std.testing.expectEqual(@as(u32, 2), lane.plain_wait);

    // Two plain rounds count down plain_wait to 0 and clear draft_chances for a fresh probe.
    for (0..2) |step_i| {
        work[0].prev_end = 5 + step_i;
        work[0].start = 5 + step_i;
        work[0].keep_from = 5 + step_i;
        work[0].end = 6 + step_i;
        commitAbsorbs(&d, &.{.{ .stream = &s, .features = feat, .start = 5 + step_i, .follow = &.{6}, .prompt = false, .propose = false, .offered = 0 }}, &work);
    }
    try std.testing.expectEqual(@as(usize, 0), s.draft_chances.items.len);
    try std.testing.expectEqual(@as(u32, 0), lane.plain_wait);

    // Any offered tree resets zero_streak back to 0.
    lane.grown = 1;
    lane.parents[0] = -1;
    lane.own[0] = -0.1;
    work[0].prev_end = 7;
    work[0].start = 7;
    work[0].keep_from = 7;
    work[0].end = 8;
    commitAbsorbs(&d, &.{.{ .stream = &s, .features = feat, .start = 7, .follow = &.{7}, .prompt = false, .propose = true, .offered = 1 }}, &work);
    try std.testing.expectEqual(@as(u32, 0), lane.zero_streak);
    try std.testing.expectEqual(@as(u32, 0), lane.plain_wait);
}
