//! Folded Memory Weights v6 Metal adapter for Pure-Attention Llama models (`num_rec == 0`).
//! Stores ONLY compressed Pre-RoPE Q4 KV caches + dual-layer whitened gate prototypes — zero recurrent
//! state deltas, zero SVD overhead, instant RoPE-shifted composition, and bit-exact atomic unlearning!
const std = @import("std");
const mtl = @import("metal");
const living = @import("living");
const mem = living.memory;
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const Metal = @import("backend.zig").Metal;
const Allocator = std.mem.Allocator;

pub const Adapter = struct {
    metal: *Metal,

    pub fn init(gpa: Allocator, metal: *Metal) !*Adapter {
        const a = try gpa.create(Adapter);
        a.* = .{ .metal = metal };
        return a;
    }

    pub fn deinit(self: *Adapter, gpa: Allocator) void {
        gpa.destroy(self);
    }

    pub fn deinitOpaque(ptr: *anyopaque, gpa: Allocator) void {
        const self: *Adapter = @ptrCast(@alignCast(ptr));
        self.deinit(gpa);
    }

    pub fn archSpec(metal: *const Metal) mem.ArchSpec {
        const c = &metal.model.config;
        return .{
            .family = .llama,
            .hidden = c.hidden,
            .num_layers = c.layers,
            .layer_lex = 1,
            .layer_sem = (2 * c.layers) / 3,
            .num_attn = c.layers,
            .kv_heads = c.kv_heads,
            .head_dim = cfg.head_dim,
            .rotary_dim = cfg.head_dim, // full-head RoPE
            .rope_theta = c.theta,
            .num_rec = 0, // Pure-attention transformer -> ZERO recurrent overhead!
            .rec_heads = 0,
            .rec_rows = 0,
            .rec_cols = 0,
            .conv_elems = 0,
            .svd_rank = 0,
            .svd_rank_short = 0,
            .kv_bits = 4, // group-32 symmetric Q4 KV (0.5625 B/elem)
            .keep_conv = false,
        };
    }

    pub fn backend(self: *Adapter) mem.Backend {
        return .{
            .ctx = self,
            .arch = archSpec(self.metal),
            .extractHidden = extractHiddenFn,
            .compileState = compileStateFn,
            .buildSnap = buildSnapFn,
            .dropSnap = dropSnapFn,
        };
    }

    fn of(ctx: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(ctx));
    }

    fn extractHiddenFn(
        ctx: *anyopaque,
        ids: []const u32,
        layer_lex: usize,
        layer_sem: usize,
        out_lex: []f32,
        out_sem: []f32,
    ) anyerror!void {
        const self = of(ctx);
        const metal = self.metal;
        const c = &metal.model.config;
        const n = ids.len;
        const h = c.hidden;

        try metal.drain();
        const cap = std.mem.alignForward(usize, n + 16, 64);
        var cache = try st.Cache.init(metal.model.device, c, cap);
        defer cache.deinit();

        const lex_bf16 = try metal.gpa.alloc(u16, n * h);
        defer metal.gpa.free(lex_bf16);
        const sem_bf16 = try metal.gpa.alloc(u16, n * h);
        defer metal.gpa.free(sem_bf16);

        var pos: usize = 0;
        while (pos < n) {
            const step = @min(st.prompt_rows, n - pos);
            try fwd.runPartial(
                metal.model,
                &metal.prompt,
                &cache,
                ids[pos .. pos + step],
                layer_lex,
                layer_sem,
                lex_bf16[pos * h ..][0 .. step * h],
                sem_bf16[pos * h ..][0 .. step * h],
            );
            cache.len += step;
            pos += step;
        }
        mem.quant.bf16SliceToF32(lex_bf16, out_lex[0 .. n * h]);
        mem.quant.bf16SliceToF32(sem_bf16, out_sem[0 .. n * h]);
    }

    fn compileStateFn(
        ctx: *anyopaque,
        base: ?*const mem.RawState,
        ids: []const u32,
        out: *mem.RawState,
    ) anyerror!void {
        const self = of(ctx);
        const metal = self.metal;
        const c = &metal.model.config;
        const base_len: usize = if (base) |b| b.len else 0;
        const total = base_len + ids.len;
        const cap = std.mem.alignForward(usize, total + 16, 64);

        try metal.drain();
        var cache = try st.Cache.init(metal.model.device, c, cap);
        defer cache.deinit();

        if (base) |b| {
            try uploadRawToCache(metal, &cache, b);
            cache.len = b.len;
        }
        var pos: usize = 0;
        while (pos < ids.len) {
            const step = @min(st.prompt_rows, ids.len - pos);
            try fwd.run(metal.model, &metal.prompt, &cache, ids[pos .. pos + step], .{});
            cache.len += step;
            pos += step;
        }
        try downloadCacheSliceToRaw(metal, &cache, base_len, ids.len, out);
    }

    fn buildSnapFn(
        ctx: *anyopaque,
        composed: *const mem.RawState,
        suffix_ids: []const u32,
    ) anyerror!mem.SnapHandle {
        const self = of(ctx);
        const metal = self.metal;
        const c = &metal.model.config;
        const total = composed.len + suffix_ids.len;
        const cap = std.mem.alignForward(usize, total + 16, 64);

        try metal.drain();
        var cache = try st.Cache.init(metal.model.device, c, cap);
        defer cache.deinit();

        try uploadRawToCache(metal, &cache, composed);
        cache.len = composed.len;

        if (suffix_ids.len > 0) {
            try fwd.run(metal.model, &metal.prompt, &cache, suffix_ids, .{});
            cache.len += suffix_ids.len;
        }

        const snap = try metal.gpa.create(snapshot.Snap);
        errdefer metal.gpa.destroy(snap);
        snap.* = try snapshot.save(metal.model.device, metal.model.queue, c, &cache);
        return .{ .saved = snap, .at = @intCast(cache.len) };
    }

    fn dropSnapFn(ctx: *anyopaque, ptr: *anyopaque) void {
        const self = of(ctx);
        const snap: *snapshot.Snap = @ptrCast(@alignCast(ptr));
        snap.free();
        self.metal.gpa.destroy(snap);
    }
};

/// Uploads `RawState` (`[kv_heads, len, head_dim]`) into Llama's private KV buffer (`[len, kv_heads, head_dim]`).
fn uploadRawToCache(metal: *Metal, cache: *st.Cache, src: *const mem.RawState) !void {
    const c = &metal.model.config;
    const len: usize = src.len;
    if (len == 0) return;
    const kv_w = c.kvWidth();
    const layer_elems = len * kv_w;
    const stage_bytes = c.layers * 2 * layer_elems * 2;
    const stage = try metal.model.device.buffer(@max(stage_bytes, 16), mtl.ResourceOptions.shared);
    defer stage.deinit();

    const dst_all = stage.slice(u16, c.layers * 2 * layer_elems);
    for (0..c.layers) |l| {
        const dst_k = dst_all[(2 * l) * layer_elems ..][0..layer_elems];
        const dst_v = dst_all[(2 * l + 1) * layer_elems ..][0..layer_elems];
        for (0..len) |t| {
            for (0..c.kv_heads) |h| {
                const s_off = (h * len + t) * cfg.head_dim;
                const d_off = t * kv_w + h * cfg.head_dim;
                @memcpy(dst_k[d_off..][0..cfg.head_dim], src.kv_k[l][s_off..][0..cfg.head_dim]);
                @memcpy(dst_v[d_off..][0..cfg.head_dim], src.kv_v[l][s_off..][0..cfg.head_dim]);
            }
        }
    }

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = metal.model.queue.commandBuffer();
    const e = cb.blit();
    const bytes_each = layer_elems * 2;
    for (0..c.layers) |l| {
        e.copy(stage, (2 * l) * bytes_each, cache.kv, cache.keys(l), bytes_each);
        e.copy(stage, (2 * l + 1) * bytes_each, cache.kv, cache.values(l), bytes_each);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("llama memory upload: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Downloads positions `start .. start + count` from Llama's private KV buffer into `RawState` (`[kv_heads, count, head_dim]`).
fn downloadCacheSliceToRaw(metal: *Metal, cache: *const st.Cache, start: usize, count: usize, out: *mem.RawState) !void {
    const c = &metal.model.config;
    if (count == 0) return;
    const kv_w = c.kvWidth();
    const layer_elems = count * kv_w;
    const stage_bytes = c.layers * 2 * layer_elems * 2;
    const stage = try metal.model.device.buffer(@max(stage_bytes, 16), mtl.ResourceOptions.shared);
    defer stage.deinit();

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = metal.model.queue.commandBuffer();
    const e = cb.blit();
    const bytes_each = layer_elems * 2;
    const start_bytes = start * kv_w * 2;
    for (0..c.layers) |l| {
        e.copy(cache.kv, cache.keys(l) + start_bytes, stage, (2 * l) * bytes_each, bytes_each);
        e.copy(cache.kv, cache.values(l) + start_bytes, stage, (2 * l + 1) * bytes_each, bytes_each);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("llama memory download: {s}", .{msg});
        return error.GpuFailed;
    }

    const src_all = stage.slice(u16, c.layers * 2 * layer_elems);
    for (0..c.layers) |l| {
        const src_k = src_all[(2 * l) * layer_elems ..][0..layer_elems];
        const src_v = src_all[(2 * l + 1) * layer_elems ..][0..layer_elems];
        for (0..count) |t| {
            for (0..c.kv_heads) |h| {
                const s_off = t * kv_w + h * cfg.head_dim;
                const d_off = (h * count + t) * cfg.head_dim;
                @memcpy(out.kv_k[l][d_off..][0..cfg.head_dim], src_k[s_off..][0..cfg.head_dim]);
                @memcpy(out.kv_v[l][d_off..][0..cfg.head_dim], src_v[s_off..][0..cfg.head_dim]);
            }
        }
    }
}
