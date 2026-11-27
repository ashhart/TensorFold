//! Named CPU-affine4 serial single-row reference (`cpu_affine4_serial_reference`) and recurrent-state byte comparators.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const ap = @import("attention_plan.zig");
const layer_plan = @import("../plan.zig");
const model_file = @import("model.zig");
const fwd = @import("forward.zig");
const conv_tables = @import("forward_conv.zig");
const st = @import("state.zig");
const ref = @import("affine4_ref.zig");

pub const reference_name = "cpu_affine4_serial_reference";
const worker_threads: usize = 8;

fn rawBytes(t: ck.Tensor) []const u8 {
    return (t.buffer.contents() + t.offset)[0..t.bytes];
}

fn denseRange(
    x_bf16: []const u16,
    w_bytes: []const u8,
    s_bytes: []const u8,
    b_bytes: []const u8,
    out_bf16: []u16,
    k: usize,
    c0: usize,
    c1: usize,
) void {
    const groups = k / ref.group_size;
    const words_per_row = k / 8;
    for (c0..c1) |col| {
        const w_base = col * words_per_row;
        const g_base = col * groups;
        var lane_acc: [32]f32 = @splat(0.0);
        for (0..groups) |g| {
            const sc = ref.fromBf16(std.mem.readInt(u16, s_bytes[(g_base + g) * 2 ..][0..2], .little));
            const bi = ref.fromBf16(std.mem.readInt(u16, b_bytes[(g_base + g) * 2 ..][0..2], .little));
            for (0..32) |lane| {
                const off = (w_base + g * 8 + (lane >> 2)) * 4;
                const word = std.mem.readInt(u32, w_bytes[off..][0..4], .little);
                const sh: u5 = @intCast((lane & 3) * 8);
                const dw0: f32 = @as(f32, @floatFromInt((word >> sh) & 0xf)) * sc + bi;
                const dw1: f32 = @as(f32, @floatFromInt((word >> (sh + 4)) & 0xf)) * sc + bi;
                const k0 = g * 64 + lane * 2;
                const p0: f32 = ref.fromBf16(x_bf16[k0]) * dw0;
                const p1: f32 = ref.fromBf16(x_bf16[k0 + 1]) * dw1;
                lane_acc[lane] = (lane_acc[lane] + p0) + p1;
            }
        }
        out_bf16[col] = ref.toBf16(ref.warpSum(lane_acc));
    }
}

/// Exact single-row CPU affine4 dense projection (column-disjoint threads, bit-identical to `affine4_ref.denseBytes`).
fn cpuDense1Row(x_buf: mtl.Buffer, weight: model_file.DenseWeight, out_buf: mtl.Buffer) !void {
    const q = switch (weight) {
        .affine4 => |t| t,
        .bf16 => return error.ExpectedAffine4Weight,
    };
    const n: usize = q.rows;
    const k: usize = q.cols;
    const x_bf = x_buf.slice(u16, k);
    const y_bf = out_buf.slice(u16, n);
    const wb = rawBytes(q.weight);
    const sb = rawBytes(q.scales);
    const bb = rawBytes(q.biases);
    if (n < 256) {
        denseRange(x_bf, wb, sb, bb, y_bf, k, 0, n);
        return;
    }
    const n_threads = @min(worker_threads, n);
    var threads: [worker_threads]std.Thread = undefined;
    var spawned: usize = 0;
    const chunk = (n + n_threads - 1) / n_threads;
    for (0..n_threads) |ti| {
        const c0 = ti * chunk;
        if (c0 >= n) break;
        const c1 = @min(n, c0 + chunk);
        if (ti + 1 == n_threads or c1 == n) {
            denseRange(x_bf, wb, sb, bb, y_bf, k, c0, c1);
            break;
        }
        threads[spawned] = try std.Thread.spawn(.{}, denseRange, .{ x_bf, wb, sb, bb, y_bf, k, c0, c1 });
        spawned += 1;
    }
    for (threads[0..spawned]) |th| th.join();
}

/// Exact single-row CPU affine4 logits (column-disjoint threads, bit-identical to `affine4_ref.logitsBytes`).
fn cpuLogits1Row(h_buf: mtl.Buffer, weight: model_file.DenseWeight, out_buf: mtl.Buffer, vocab: usize) !void {
    const q = switch (weight) {
        .affine4 => |t| t,
        .bf16 => return error.ExpectedAffine4Weight,
    };
    const h_bf = h_buf.slice(u16, st.hidden);
    const out_f32 = out_buf.slice(f32, vocab);
    const wb = rawBytes(q.weight);
    const sb = rawBytes(q.scales);
    const bb = rawBytes(q.biases);
    const n_threads = @min(worker_threads, vocab);
    var threads: [worker_threads]std.Thread = undefined;
    var spawned: usize = 0;
    const chunk = (vocab + n_threads - 1) / n_threads;
    for (0..n_threads) |ti| {
        const v0 = ti * chunk;
        if (v0 >= vocab) break;
        const v1 = @min(vocab, v0 + chunk);
        if (ti + 1 == n_threads or v1 == vocab) {
            ref.logitsRangeBytes(h_bf, wb, sb, bb, out_f32, 1, vocab, v0, v1);
            break;
        }
        threads[spawned] = try std.Thread.spawn(.{}, ref.logitsRangeBytes, .{ h_bf, wb, sb, bb, out_f32, 1, vocab, v0, v1 });
        spawned += 1;
    }
    for (threads[0..spawned]) |th| th.join();
}

fn commitWait(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("cpu_affine4_serial_reference GPU step failed: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Advance one token serially (`rows = 1`) using CPU `affine4_ref` for every 4-bit kernel (zero `affine4.metal`).
pub fn stepToken(m: *const model_file.Model, s: *st.Scratch, cache: *st.Cache, token: u32, compute_logits: bool, taps: ?st.Ref) !void {
    if (token >= m.config.text.vocab) return error.Inkling2Token;
    if (cache.len + 1 > cache.capacity) return error.Inkling2ContextFull;
    s.pos.slice(i32, 1)[0] = @intCast(cache.len);
    s.seg_of.slice(i32, 1)[0] = 0;
    s.segs.slice(ap.Seg, st.max_segments)[0] = .{ .row0 = 0, .n = 1, .p0 = @intCast(cache.len), .off = 0 };
    const segments = [_]conv_tables.Segment{.{ .cache = cache, .rows = 1 }};
    conv_tables.prepareConvTables(m, s, &segments);

    // Step 1: Pure CPU affine4 embed_norm (never invokes tf_ink2_embed_norm_q4).
    const embed_q = switch (m.weights.embed) {
        .affine4 => |t| t,
        .bf16 => return error.ExpectedAffine4Weight,
    };
    const ids = [_]u32{token};
    ref.embedNormBytes(&ids, rawBytes(embed_q.weight), rawBytes(embed_q.scales), rawBytes(embed_q.biases), rawBytes(m.weights.embed_norm), s.p.slice(u16, st.hidden));

    const visits = m.weight_plan.visits[0..m.weight_plan.n_visits];
    for (visits, 0..) |*v, vi| {
        const prev_fused = vi > 0 and fwd.canPreFuse(v);
        const next_v: ?*const layer_plan.Visit = if (vi + 1 < visits.len) &visits[vi + 1] else null;
        const lp = m.weight_plan.at(v.layer);
        const gated = v.hasMessages();
        const w = &m.weights.layers[v.layer];

        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.serial);
            try fwd.serialVisitPreAttn(m, e, s, v, prev_fused, taps);
            e.end();
            try commitWait(cb);
        }

        // Step 2: Pure CPU affine4 attention input projections (never invokes tf_ink2_dense_q4).
        const knorm = if (gated or lp.kind == .global) s.k_norm else s.q_norm;
        const vnorm = if (gated or lp.kind == .global) s.v_norm else s.q_norm;
        try cpuDense1Row(s.q_norm, w.attn.wq, s.q_raw);
        try cpuDense1Row(knorm, w.attn.wk, s.k_raw);
        try cpuDense1Row(vnorm, w.attn.wv, s.v_raw);
        try cpuDense1Row(s.q_norm, w.attn.wr, s.rel_raw);
        try cpuDense1Row(s.q_norm, w.attn.sink, s.sink);
        try cpuDense1Row(s.q_norm, w.attn.gate, s.gate);

        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.serial);
            try fwd.serialVisitAttend(m, e, s, v, cache);
            e.end();
            try commitWait(cb);
        }

        // Step 3: Pure CPU affine4 o_proj (never invokes tf_ink2_dense_q4).
        try cpuDense1Row(s.att, w.attn.wo, s.attn_dense);

        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.serial);
            try fwd.serialVisitPostAttn(m, e, s, v, next_v, cache);
            e.end();
            try commitWait(cb);
        }
    }

    {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        fwd.serialFinalNormAndCommit(m, e, s, cache, taps);
        e.end();
        try commitWait(cb);
    }

    // Step 4: Pure CPU affine4 lm_head logits (never invokes tf_ink2_logits_q4).
    if (compute_logits) try cpuLogits1Row(s.mlp_normed, m.weights.unembed, cache.logits, m.config.text.unpadded_vocab);
    cache.len += 1;
}

pub fn argmax(logits: []const f32) u32 {
    var best: usize = 0;
    var best_v: f32 = logits[0];
    for (logits[1..], 1..) |v, i| if (v > best_v) {
        best_v = v;
        best = i;
    };
    return @intCast(best);
}

pub const StateBytes = struct {
    len: usize,
    logits: []u8,
    conv: []u8,
    kv: []u8,
    taps: [st.feature_row_bytes]u8,

    pub fn deinit(self: StateBytes, gpa: std.mem.Allocator) void {
        gpa.free(self.logits);
        gpa.free(self.conv);
        gpa.free(self.kv);
    }
};

fn blitRead(queue: mtl.Queue, src: mtl.Buffer, dst: mtl.Buffer, bytes: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.copy(src, 0, dst, 0, bytes);
    e.end();
    try commitWait(cb);
}

/// Capture exact logit bytes, all 48-slot conv history bytes, all active KV bytes, and feature/attn_res tap bytes.
pub fn captureState(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, cache: *const st.Cache, vocab: usize, tap_bytes: []const u8) !StateBytes {
    std.debug.assert(tap_bytes.len == st.feature_row_bytes);
    const logits = try gpa.dupe(u8, cache.logits.slice(u8, vocab * 4));
    errdefer gpa.free(logits);

    const conv_stage = try device.buffer(st.conv_bytes, mtl.ResourceOptions.shared);
    defer conv_stage.deinit();
    try blitRead(queue, cache.conv, conv_stage, st.conv_bytes);
    const conv = try gpa.dupe(u8, conv_stage.slice(u8, st.conv_bytes));
    errdefer gpa.free(conv);

    const kv_total = st.kvBytes(cache.capacity);
    const kv_stage = try device.buffer(kv_total, mtl.ResourceOptions.shared);
    defer kv_stage.deinit();
    try blitRead(queue, cache.kv, kv_stage, kv_total);
    const kv_raw = kv_stage.slice(u8, kv_total);

    var active_bytes: usize = 0;
    for (0..st.visit_slots) |slot| {
        const count = if (!cache.globals[slot] and cache.len > 1024) 1024 else cache.len;
        active_bytes += 2 * 8 * count * 128 * 2;
    }
    const kv = try gpa.alloc(u8, active_bytes);
    errdefer gpa.free(kv);
    var out_off: usize = 0;
    for (0..st.visit_slots) |slot| {
        const ring = cache.ring(slot);
        const lo = if (!cache.globals[slot] and cache.len > 1024) cache.len - 1024 else 0;
        for ([_]usize{ cache.key_at[slot], cache.value_at[slot] }) |base| {
            for (0..8) |h| {
                for (lo..cache.len) |pos| {
                    const src_off = base + (h * ring + (pos % ring)) * 128 * 2;
                    @memcpy(kv[out_off..][0..256], kv_raw[src_off..][0..256]);
                    out_off += 256;
                }
            }
        }
    }
    std.debug.assert(out_off == active_bytes);
    var taps: [st.feature_row_bytes]u8 = undefined;
    @memcpy(&taps, tap_bytes);
    return .{ .len = cache.len, .logits = logits, .conv = conv, .kv = kv, .taps = taps };
}

pub fn expectStateBitEqual(want: StateBytes, got: StateBytes) !void {
    try std.testing.expectEqual(want.len, got.len);
    try std.testing.expectEqualSlices(u8, want.logits, got.logits);
    try std.testing.expectEqualSlices(u8, want.conv, got.conv);
    try std.testing.expectEqualSlices(u8, want.kv, got.kv);
    try std.testing.expectEqualSlices(u8, &want.taps, &got.taps);
}
