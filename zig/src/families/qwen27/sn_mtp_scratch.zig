//! Scratch buffer layout, stream KV context, and attention builder for SenseNova MTP.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const attn = @import("attention.zig");
const projection = @import("projection.zig");
const simd_attention = @import("simd_attention.zig");
const Config = @import("config.zig").Config;
const linear_gpu = core.sensenova_loop.linear_gpu;
const round_gpu = core.tree_round_gpu;

pub const max_depth: u32 = 4;
pub const batch_cap: u32 = 2048;
pub const batch_streams: u32 = 16;
pub const buf_opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Input hidden states slice passed to `Mtp.absorb` (layer `layers - 1` pre-final-norm output).
pub const Features = struct {
    buffer: usize,
    offset: usize,
    rows: u32,
    row_bytes: u32,
    host: bool = false,
};

pub const AbsorbItem = struct {
    stream: usize,
    features: Features,
    start: u64,
    follow: []const u32,
    rows: ?[]const u32 = null,
    propose: bool = true,
    prompt_only: bool = false,
};

pub const HoldItem = struct {
    stream: usize,
    pending: u32,
    position: u64,
    depth: u32,
};

/// Per-stream MTP KV cache and speculative state.
pub const StreamCtx = struct {
    keys: mtl.Buffer,
    values: mtl.Buffer,
    picks_buf: mtl.Buffer,
    chain_out_buf: mtl.Buffer,
    begin: u64 = 0,
    len: u64 = 0,
    has_out: bool = false,
    has_pick0: bool = false,
    in_use: bool = false,
    owner: usize = 0,
    held_tokens: [max_depth]u32 = @splat(0),
    held_count: u32 = 0,

    pub fn init(device: mtl.Device, capacity: usize, kv_dim: usize, hidden: usize) !StreamCtx {
        const kv_bytes = kv_dim * capacity * 2;
        const keys = try device.buffer(kv_bytes, buf_opts);
        errdefer keys.deinit();
        const values = try device.buffer(kv_bytes, buf_opts);
        errdefer values.deinit();
        const picks_buf = try device.buffer(max_depth * 16, buf_opts);
        errdefer picks_buf.deinit();
        const chain_out_buf = try device.buffer((max_depth + 1) * hidden * 2, buf_opts);
        return .{ .keys = keys, .values = values, .picks_buf = picks_buf, .chain_out_buf = chain_out_buf };
    }

    pub fn deinit(c: *StreamCtx) void {
        c.keys.deinit();
        c.values.deinit();
        c.picks_buf.deinit();
        c.chain_out_buf.deinit();
    }

    pub fn reset(c: *StreamCtx, owner: usize) void {
        c.begin = 0;
        c.len = 0;
        c.has_out = false;
        c.has_pick0 = false;
        c.in_use = owner != 0;
        c.owner = owner;
        c.held_count = 0;
    }

    pub fn chainOutRef(c: *const StreamCtx, slot_idx: usize, hidden: usize) projection.Ref {
        return .{ .buffer = c.chain_out_buf, .offset = slot_idx * hidden * 2 };
    }

    pub fn pickRef(c: *const StreamCtx, idx: usize) round_gpu.Ref {
        return .{ .buf = c.picks_buf, .off = idx * 16 };
    }
};

pub const ScratchField = enum(usize) {
    ids,
    pos,
    kept,
    hin,
    planes,
    cat,
    h,
    x,
    qg,
    kv_raw,
    k,
    v,
    norm_q,
    query,
    norm_k,
    rot_k,
    attn_out,
    r,
    gu,
    act,
    dummy_dims,
    dummy_sums,
    logits,
    chain_pos,
    gather,
};

pub const scratch_count: usize = 25;

pub fn lref(r: projection.Ref) linear_gpu.Ref {
    return .{ .buf = r.buffer, .off = r.offset };
}

pub fn scratchSizes(c: Config, rows: u32) [scratch_count]usize {
    const r: usize = rows;
    const h = c.hidden;
    const kv = c.kvDim();
    const qd = c.heads * c.head_dim;
    const inter = c.intermediate;
    return .{
        r * 4,
        r * 4,
        r * 4,
        r * h * 2,
        2 * r * h * 2,
        r * 2 * h * 2,
        r * h * 2,
        r * h * 2,
        r * 2 * qd * 2,
        16,
        r * kv * 2,
        r * kv * 2,
        r * qd * 2,
        r * qd * 2,
        r * kv * 2,
        r * kv * 2,
        r * qd * 2,
        r * h * 2,
        r * 2 * inter * 2,
        r * inter * 2,
        (max_depth + 1) * 16,
        @max(r, 16) * ((inter + 63) / 64) * 4,
        @as(usize, batch_streams) * c.vocab * 2,
        @as(usize, max_depth) * batch_streams * 4,
        r * h * 2,
    };
}

pub fn buildAttention(a: std.mem.Allocator, dev: mtl.Device, g: attn.planning.Geometry, units: bool) !attn.Attention {
    const text = try simd_attention.grouped(a, units, true, g.group(), g.kv_heads, false);
    defer a.free(text);
    const lib = mtl.Library.fromSource(dev, text, mtl.CompileOptions.mlx()) catch |err| blk: {
        if (units) return err;
        const base = try simd_attention.grouped(a, units, false, g.group(), g.kv_heads, false);
        defer a.free(base);
        break :blk try mtl.Library.fromSource(dev, base, mtl.CompileOptions.mlx());
    };
    defer lib.deinit();
    return attn.Attention.init(dev, lib);
}

pub fn encodeMat(
    sn: *linear_gpu.LinearGpu,
    e: mtl.ComputeEncoder,
    mat: core.sensenova_loop.mtp_weights.MtpMat,
    x: projection.Ref,
    y: projection.Ref,
    rows: u32,
    ldy: u32,
    prompt: bool,
    xs_opt: ?linear_gpu.Sums,
) !void {
    return encodeMatSplit(sn, e, mat, x, y, rows, ldy, prompt, xs_opt, .{});
}

pub fn encodeMatSplit(
    sn: *linear_gpu.LinearGpu,
    e: mtl.ComputeEncoder,
    mat: core.sensenova_loop.mtp_weights.MtpMat,
    x: projection.Ref,
    y: projection.Ref,
    rows: u32,
    ldy: u32,
    prompt: bool,
    xs_opt: ?linear_gpu.Sums,
    splits: linear_gpu.OutSplits,
) !void {
    switch (mat) {
        .bf16 => |m| try sn.encodeBf16Split(e, m, lref(x), lref(y), rows, ldy, prompt, null, splits),
        .awq => |m| try sn.encodeAwqSplit(e, m, lref(x), xs_opt, lref(y), rows, ldy, prompt, null, splits),
    }
}

pub fn populateHeld(ctx: *StreamCtx, pending: u32, depth: u32, vocab: usize) void {
    var prev = pending;
    for (0..depth) |d| {
        const p: *const round_gpu.contract.Pick = @ptrCast(@alignCast(ctx.picks_buf.contents() + d * 16));
        const tok = if (p.nonfinite == 0 and p.token < vocab) p.token else prev;
        ctx.held_tokens[d] = tok;
        prev = tok;
    }
    ctx.held_count = depth;
}

/// Copy `count` feature rows (selected or contiguous) into `gather` at `dst_row`.
pub fn stageFeaturesAt(m: anytype, feat: Features, rows: ?[]const u32, start_idx: usize, count: u32, dst_row: usize) !void {
    const rb = m.hidden * 2;
    const dst = m.buf(.gather).contents()[dst_row * rb ..][0 .. @as(usize, count) * rb];
    const src_ptr: [*]const u8 = if (!feat.host) (mtl.Buffer{ .id = @ptrFromInt(feat.buffer) }).contents() else @ptrFromInt(feat.buffer);
    if (rows) |sel| {
        for (0..count) |i| {
            const r = sel[start_idx + i];
            if (r >= feat.rows) return error.BadMtpFeatures;
            @memcpy(dst[i * rb ..][0..rb], src_ptr[feat.offset + @as(usize, r) * rb ..][0..rb]);
        }
    } else {
        @memcpy(dst, src_ptr[feat.offset + start_idx * rb ..][0 .. @as(usize, count) * rb]);
    }
}

pub fn stageRawFeatures(m: anytype, feat: Features, rows: ?[]const u32, start_idx: usize, count: u32) !projection.Ref {
    const rb = m.hidden * 2;
    if (rows == null and !feat.host and feat.offset % 16 == 0 and feat.buffer != 0) {
        return .{ .buffer = .{ .id = @ptrFromInt(feat.buffer) }, .offset = feat.offset + start_idx * rb };
    }
    try stageFeaturesAt(m, feat, rows, start_idx, count, 0);
    return m.ref(.gather);
}

pub fn appendMtpBuffers(m: anytype, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
    try m.weights.appendBuffers(a, list);
    if (m.draft_head) |dh| {
        try list.appendSlice(a, &.{ dh.w, dh.sb });
        if (dh.input_scale) |ib| try list.append(a, ib);
    }
    try list.appendSlice(a, &m.abs_scratch.buffers);
    try list.appendSlice(a, &m.abs_binding.buffers);
    for (m.chain_scratch, m.chain_binding) |s, b| {
        try list.appendSlice(a, &s.buffers);
        try list.appendSlice(a, &b.buffers);
    }
    try list.appendSlice(a, &m.bufs);
    for (m.ctxs) |slot_opt| if (slot_opt) |sc| {
        try list.appendSlice(a, &.{ sc.keys, sc.values, sc.picks_buf, sc.chain_out_buf });
    };
}

pub fn applyMtpResident(m: anytype, f: mtl.ResidencySet.Follow, joining: bool) void {
    m.weights.applyResident(f, joining);
    if (m.draft_head) |dh| {
        mtl.resident.fields.one(dh.w, f, joining);
        mtl.resident.fields.one(dh.sb, f, joining);
        if (dh.input_scale) |ib| mtl.resident.fields.one(ib, f, joining);
    }
    for (m.abs_scratch.buffers) |b| mtl.resident.fields.one(b, f, joining);
    for (m.abs_binding.buffers) |b| mtl.resident.fields.one(b, f, joining);
    for (m.chain_scratch, m.chain_binding) |s, b| {
        for (s.buffers) |sb| mtl.resident.fields.one(sb, f, joining);
        for (b.buffers) |bb| mtl.resident.fields.one(bb, f, joining);
    }
    for (m.bufs) |b| mtl.resident.fields.one(b, f, joining);
    for (m.ctxs) |slot_opt| if (slot_opt) |sc| {
        for ([_]mtl.Buffer{ sc.keys, sc.values, sc.picks_buf, sc.chain_out_buf }) |b| mtl.resident.fields.one(b, f, joining);
    };
}

/// Warm GEMM/splitK/attention pipelines before residency lock and time single-step `hold`.
pub fn warmup(m: anytype, io: std.Io) !void {
    const ctx = &(m.ctxs[0] orelse return error.MissingMtpSlot);
    ctx.reset(0);
    ctx.in_use = true;
    defer ctx.reset(0);
    const prompt_rows: u32 = @min(m.abs_rows, @min(512, @as(u32, @intCast(m.capacity - 8))));
    @memset(m.buf(.gather).slice(u16, prompt_rows * m.hidden), 0x3c00);
    const cache_view = m.cacheView(ctx);
    for (0..3) |pass| {
        ctx.begin = 0;
        ctx.len = 0;
        const rows: u32 = switch (pass) {
            0 => prompt_rows,
            1 => 2,
            else => 1,
        };
        const pos_dst = m.buf(.pos).slice(u32, rows);
        const id_dst = m.buf(.ids).slice(u32, rows);
        for (pos_dst, id_dst, 0..) |*p, *id, r| {
            p.* = @intCast(r);
            id.* = @intCast((r * 17 + 5) % m.vocab);
        }
        var plan = try attn.Plan.init(m.gpa, &.{.{ .parents = m.parents_storage[0..rows], .start = 0, .capacity = cache_view.capacity }}, m.geometry);
        defer plan.deinit();
        try m.abs_scratch.upload(plan);
        try m.abs_binding.upload(plan, &.{cache_view});
        const cb = m.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        var ended = false;
        errdefer if (!ended) e.end();
        try m.encodeStep(e, &m.abs_scratch, &m.abs_binding, plan, cache_view, rows, m.ref(.gather), m.ref(.ids), m.ref(.pos), true, ctx, true, 0);
        e.end();
        ended = true;
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("sensenova mtp warmup failure: {s}", .{msg});
            return error.GpuFailed;
        }
        ctx.len = rows;
    }
    var best: f64 = std.math.floatMax(f64);
    const cpos_ptr: [*]u32 = @ptrCast(@alignCast(m.buf(.chain_pos).contents()));
    for (0..8) |iter| {
        cpos_ptr[0] = @intCast(ctx.len);
        var plan = try attn.Plan.init(m.gpa, &.{.{ .parents = m.parents_storage[0..1], .start = @intCast(ctx.len), .capacity = cache_view.capacity }}, m.geometry);
        defer plan.deinit();
        try m.chain_scratch[0].upload(plan);
        try m.chain_binding[0].upload(plan, &.{cache_view});
        const t0 = std.Io.Clock.awake.now(io);
        const cb = m.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        var ended = false;
        errdefer if (!ended) e.end();
        try m.encodeStep(e, &m.chain_scratch[0], &m.chain_binding[0], plan, cache_view, 1, ctx.chainOutRef(0, m.hidden), .{ .buffer = ctx.picks_buf, .offset = 0 }, .{ .buffer = m.buf(.chain_pos), .offset = 0 }, false, ctx, true, 1);
        e.end();
        ended = true;
        cb.commit();
        cb.wait();
        const ns = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        if (iter >= 3) best = @min(best, @as(f64, @floatFromInt(ns)) / 1e6);
    }
    m.step_ms = best;
    std.log.info("sensenova mtp: bits={d} depth={d} abs_rows={d} step_ms={d:.2} plain_guard={} slot_bytes={d}", .{ m.bits, m.depth, m.abs_rows, m.step_ms, m.plain_guard, m.slotBytes() });
}

test "mtp scratch batch cap and row sizing" {
    try std.testing.expectEqual(@as(u32, 2048), batch_cap);
    var cfg = std.mem.zeroes(Config);
    cfg.hidden = 4096;
    cfg.heads = 16;
    cfg.kv_heads = 4;
    cfg.head_dim = 256;
    cfg.intermediate = 12288;
    cfg.vocab = 248320;
    const s128 = scratchSizes(cfg, 128);
    const s2048 = scratchSizes(cfg, 2048);
    try std.testing.expectEqual(s128[@backingInt(ScratchField.ids)] * 16, s2048[@backingInt(ScratchField.ids)]);
    try std.testing.expectEqual(s128[@backingInt(ScratchField.planes)] * 16, s2048[@backingInt(ScratchField.planes)]);
}
