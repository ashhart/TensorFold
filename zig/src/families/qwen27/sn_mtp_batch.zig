//! Multi-stream batched absorb, head pick, and chain steps for SenseNova MTP.
const std = @import("std");
const mtl = @import("metal");
const attn = @import("attention.zig");
const wts = @import("gpu_weights.zig");
const projection = @import("projection.zig");
const scratch_mod = @import("sn_mtp_scratch.zig");

const AbsorbItem = scratch_mod.AbsorbItem;
const HoldItem = scratch_mod.HoldItem;
const StreamCtx = scratch_mod.StreamCtx;
const max_streams: usize = scratch_mod.batch_streams;
const chain_cap: usize = scratch_mod.max_depth - 1;

pub fn canBatch(m: anytype, absorbs: []const AbsorbItem, holds: []const HoldItem) bool {
    if (absorbs.len == 0 and holds.len == 0) return false;
    if (absorbs.len > max_streams or holds.len > max_streams) return false;
    const row_cap = @min(m.abs_rows, attn.planning.max_rows);
    var total_rows: usize = 0;
    for (absorbs, 0..) |ab, i| {
        if (ab.prompt_only or ab.follow.len == 0) return false;
        total_rows += ab.follow.len;
        for (absorbs[0..i]) |prev| if (prev.stream == ab.stream) return false;
    }
    if (total_rows > row_cap) return false;
    for (holds, 0..) |hit, i| {
        for (holds[0..i]) |prev| if (prev.stream == hit.stream) return false;
    }
    return true;
}

pub fn run(m: anytype, absorbs: []const AbsorbItem, holds: []const HoldItem) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var abs_ctxs: [max_streams]*StreamCtx = undefined;
    var abs_begin: [max_streams]u64 = undefined;
    var abs_new_len: [max_streams]u64 = undefined;
    var abs_end_row: [max_streams]u32 = undefined;
    var abs_caches: [max_streams]attn.Cache = undefined;
    var abs_windows: [max_streams]attn.planning.Window = undefined;
    var abs_pick0: [max_streams]bool = @splat(false);

    var hold_ctxs: [max_streams]*StreamCtx = undefined;
    var hold_abs_idx: [max_streams]?usize = @splat(null);
    var hold_has_out: [max_streams]bool = @splat(false);
    var hold_begin: [max_streams]u64 = undefined;
    var hold_len: [max_streams]u64 = undefined;
    var hold_depth: [max_streams]u32 = undefined;

    for (absorbs, 0..) |ab, a| {
        const ctx = try m.ctxOf(ab.stream);
        const take = ab.follow.len;
        if (ab.rows) |sel| if (sel.len != take) return error.BadMtpAbsorb;
        if (ab.features.row_bytes != m.hidden * 2) return error.BadMtpFeatures;
        const start_begin = if (ctx.len == 0 and ab.start > 0) ab.start else ctx.begin;
        const start_len = if (ctx.len == 0 and ab.start > 0) ab.start else ctx.len;
        if (ab.start != start_len) return error.BadMtpPosition;
        var extra_chain: usize = 0;
        for (holds) |hit| if (hit.stream == ab.stream) {
            const d = @min(hit.depth, scratch_mod.max_depth);
            if (d > 1) extra_chain = d - 1;
            break;
        };
        if ((ab.start - start_begin) + take + extra_chain > m.capacity) return error.BadMtpPosition;
        abs_ctxs[a] = ctx;
        abs_begin[a] = start_begin;
        abs_new_len[a] = ab.start + take;
    }

    var pick_ctxs: [max_streams]*StreamCtx = undefined;
    var pick_count: usize = 0;
    var max_chain: usize = 0;

    for (holds, 0..) |hit, j| {
        const ctx = try m.ctxOf(hit.stream);
        hold_ctxs[j] = ctx;
        var ai: ?usize = null;
        for (absorbs, 0..) |ab, a| if (ab.stream == hit.stream) {
            ai = a;
            break;
        };
        hold_abs_idx[j] = ai;
        const has_out = (ai != null) or ctx.has_out;
        const base_begin = if (ai) |a| abs_begin[a] else ctx.begin;
        const base_len = if (ai) |a| abs_new_len[a] else ctx.len;
        const d = @min(hit.depth, scratch_mod.max_depth);
        if (has_out and d > 1) {
            if (base_len < base_begin or (base_len - base_begin) + (d - 1) > m.capacity) return error.BadMtpPosition;
            max_chain = @max(max_chain, d - 1);
        }
        hold_has_out[j] = has_out;
        hold_begin[j] = base_begin;
        hold_len[j] = base_len;
        hold_depth[j] = d;
    }

    for (absorbs, 0..) |ab, a| {
        var held_pos = false;
        for (holds, 0..) |hit, j| if (hit.stream == ab.stream) {
            held_pos = hold_depth[j] > 0;
            break;
        };
        abs_pick0[a] = ab.propose or held_pos;
        if (abs_pick0[a]) {
            pick_ctxs[pick_count] = abs_ctxs[a];
            pick_count += 1;
        }
    }
    for (holds, 0..) |_, j| {
        if (hold_abs_idx[j] == null and hold_depth[j] > 0 and hold_has_out[j] and !hold_ctxs[j].has_pick0) {
            pick_ctxs[pick_count] = hold_ctxs[j];
            pick_count += 1;
        }
    }

    if (absorbs.len == 0 and pick_count == 0 and max_chain == 0) {
        for (holds, 0..) |hit, j| {
            const ctx = hold_ctxs[j];
            if (hold_depth[j] == 0) {
                ctx.held_count = 0;
                continue;
            }
            for (ctx.held_tokens[0..hold_depth[j]]) |*t| t.* = hit.pending;
            ctx.held_count = hold_depth[j];
        }
        return;
    }

    var abs_plan: ?attn.Plan = null;
    defer if (abs_plan) |*p| p.deinit();
    var total_rows: u32 = 0;
    if (absorbs.len > 0) {
        const ids_slice = m.buf(.ids).slice(u32, m.abs_rows);
        const pos_slice = m.buf(.pos).slice(u32, m.abs_rows);
        var row_off: u32 = 0;
        for (absorbs, 0..) |ab, a| {
            const take: u32 = @intCast(ab.follow.len);
            try scratch_mod.stageFeaturesAt(m, ab.features, ab.rows, 0, take, row_off);
            @memcpy(ids_slice[row_off..][0..take], ab.follow);
            for (pos_slice[row_off..][0..take], 0..) |*p, r| p.* = @intCast(ab.start + r);
            abs_caches[a] = m.cacheViewAt(abs_ctxs[a], abs_begin[a]);
            abs_windows[a] = .{ .parents = m.parents_storage[0..take], .start = @intCast(ab.start - abs_begin[a]), .capacity = abs_caches[a].capacity };
            row_off += take;
            abs_end_row[a] = row_off;
        }
        total_rows = row_off;
        abs_plan = try attn.Plan.init(m.gpa, abs_windows[0..absorbs.len], m.geometry);
        try m.abs_scratch.upload(abs_plan.?);
        try m.abs_binding.upload(abs_plan.?, abs_caches[0..absorbs.len]);
    }

    var plans: [chain_cap]attn.Plan = undefined;
    var built: usize = 0;
    defer for (plans[0..built]) |*p| p.deinit();
    var ch_ctxs: [chain_cap][max_streams]*StreamCtx = undefined;
    var ch_caches: [chain_cap][max_streams]attn.Cache = undefined;
    var ch_counts: [chain_cap]usize = @splat(0);
    const cpos_ptr: [*]u32 = @ptrCast(@alignCast(m.buf(.chain_pos).contents()));
    for (0..max_chain) |s| {
        var ch_windows: [max_streams]attn.planning.Window = undefined;
        var k: usize = 0;
        for (0..holds.len) |j| {
            if (!hold_has_out[j] or hold_depth[j] <= s + 1) continue;
            const pos_k = hold_len[j] + s;
            cpos_ptr[s * max_streams + k] = @intCast(pos_k);
            ch_ctxs[s][k] = hold_ctxs[j];
            ch_caches[s][k] = m.cacheViewAt(hold_ctxs[j], hold_begin[j]);
            ch_windows[k] = .{ .parents = m.parents_storage[0..1], .start = @intCast(pos_k - hold_begin[j]), .capacity = ch_caches[s][k].capacity };
            k += 1;
        }
        ch_counts[s] = k;
        plans[s] = try attn.Plan.init(m.gpa, ch_windows[0..k], m.geometry);
        built += 1;
        try m.chain_scratch[s].upload(plans[s]);
        try m.chain_binding[s].upload(plans[s], ch_caches[s][0..k]);
    }

    const cb = m.model.queue.commandBuffer();
    const e = cb.compute(.serial);
    var ended = false;
    errdefer if (!ended) e.end();
    const hidden_u32: u32 = @intCast(m.hidden);
    const glue = m.model.glue;

    if (absorbs.len > 0) {
        try m.encodeTransformer(e, &m.abs_scratch, &m.abs_binding, abs_plan.?, abs_caches[0..absorbs.len], total_rows, m.ref(.gather), m.ref(.ids), m.ref(.pos), true, false, 0);
        for (0..absorbs.len) |a| {
            const src = projection.Ref{ .buffer = m.buf(.x), .offset = @as(usize, abs_end_row[a] - 1) * m.hidden * 2 };
            try glue.unstack(e, src, abs_ctxs[a].chainOutRef(0, m.hidden), .{ .rows = 1, .width = hidden_u32, .stride = hidden_u32, .offset = 0 });
        }
        e.barrier();
    }

    if (pick_count > 0) {
        for (pick_ctxs[0..pick_count], 0..) |ctx, i| {
            const dst = projection.Ref{ .buffer = m.buf(.cat), .offset = i * m.hidden * 2 };
            try glue.unstack(e, ctx.chainOutRef(0, m.hidden), dst, .{ .rows = 1, .width = hidden_u32, .stride = hidden_u32, .offset = 0 });
        }
        e.barrier();
        try m.encodeHeadBatch(e, m.ref(.cat), pick_ctxs[0..pick_count], 0);
    }

    for (0..max_chain) |s| {
        const k = ch_counts[s];
        const k_u32: u32 = @intCast(k);
        for (ch_ctxs[s][0..k], 0..) |ctx, i| {
            const hin_dst = projection.Ref{ .buffer = m.buf(.gather), .offset = i * m.hidden * 2 };
            try glue.unstack(e, ctx.chainOutRef(s, m.hidden), hin_dst, .{ .rows = 1, .width = hidden_u32, .stride = hidden_u32, .offset = 0 });
            const ids_ref = projection.Ref{ .buffer = ctx.picks_buf, .offset = s * 16 };
            const h_dst = projection.Ref{ .buffer = m.buf(.h), .offset = i * m.hidden * 2 };
            try glue.embeddingBf16(e, m.hidden, m.vocab, wts.ref(m.model.weights.embed_bf16.?), ids_ref, h_dst, 1);
        }
        const pos_ref = projection.Ref{ .buffer = m.buf(.chain_pos), .offset = s * max_streams * 4 };
        try m.encodeTransformer(e, &m.chain_scratch[s], &m.chain_binding[s], plans[s], ch_caches[s][0..k], k_u32, m.ref(.gather), null, pos_ref, false, false, s + 1);
        for (ch_ctxs[s][0..k], 0..) |ctx, i| {
            const src = projection.Ref{ .buffer = m.buf(.x), .offset = i * m.hidden * 2 };
            try glue.unstack(e, src, ctx.chainOutRef(s + 1, m.hidden), .{ .rows = 1, .width = hidden_u32, .stride = hidden_u32, .offset = 0 });
        }
        e.barrier();
        try m.encodeHeadBatch(e, m.ref(.x), ch_ctxs[s][0..k], s + 1);
    }

    e.end();
    ended = true;
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("sensenova mtp batch draftRound GPU failure: {s}", .{msg});
        return error.GpuFailed;
    }

    for (0..absorbs.len) |a| {
        abs_ctxs[a].begin = abs_begin[a];
        abs_ctxs[a].len = abs_new_len[a];
        abs_ctxs[a].has_out = true;
        abs_ctxs[a].has_pick0 = abs_pick0[a];
    }
    for (holds, 0..) |hit, j| {
        const ctx = hold_ctxs[j];
        if (hold_depth[j] == 0) {
            ctx.held_count = 0;
            continue;
        }
        if (!hold_has_out[j]) {
            for (ctx.held_tokens[0..hold_depth[j]]) |*t| t.* = hit.pending;
            ctx.held_count = hold_depth[j];
            continue;
        }
        ctx.has_pick0 = true;
        scratch_mod.populateHeld(ctx, hit.pending, hold_depth[j], m.vocab);
    }
}
