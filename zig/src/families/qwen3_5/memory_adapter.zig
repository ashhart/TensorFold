//! Folded Memory Weights v3 Metal adapter for Qwen3.5-2B (`zig/src/families/qwen3_5/`).
const std = @import("std");
const mtl = @import("metal");
const living = @import("living");
const mem = living.memory;
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const Metal = @import("backend.zig").Metal;
const Allocator = std.mem.Allocator;

pub const Adapter = struct {
    metal: *Metal,
    tap_lex: mtl.Buffer,

    pub fn init(gpa: Allocator, metal: *Metal) !*Adapter {
        const a = try gpa.create(Adapter);
        errdefer gpa.destroy(a);
        const bytes = metal.scratch.rows * c.hidden * 2;
        const tap = try metal.model.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        a.* = .{ .metal = metal, .tap_lex = tap };
        return a;
    }

    pub fn deinit(self: *Adapter, gpa: Allocator) void {
        self.tap_lex.deinit();
        gpa.destroy(self);
    }

    pub fn deinitOpaque(ptr: *anyopaque, gpa: Allocator) void {
        const self: *Adapter = @ptrCast(@alignCast(ptr));
        self.deinit(gpa);
    }

    pub fn archSpec() mem.ArchSpec {
        return .{
            .family = .qwen3_5,
            .hidden = c.hidden,
            .num_layers = c.layers,
            .layer_lex = 1,
            .layer_sem = (2 * c.layers) / 3, // layer 16
            .num_attn = st.attention_layers, // 6
            .kv_heads = c.kv_heads, // 2
            .head_dim = c.head_dim, // 256
            .rotary_dim = c.rotary_dim, // 64
            .rope_theta = @floatCast(c.theta), // 10_000_000.0
            .num_rec = c.layers - st.attention_layers, // 18
            .rec_heads = c.linear_heads, // 16
            .rec_rows = c.linear_dim, // 128
            .rec_cols = c.linear_dim, // 128
            .conv_elems = (c.conv_taps - 1) * c.conv_dim, // 3 * 6144
        };
    }

    pub fn backend(self: *Adapter) mem.Backend {
        return .{
            .ctx = self,
            .arch = archSpec(),
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
        const m = self.metal;
        if (ids.len == 0) return;
        std.debug.assert(out_lex.len == ids.len * c.hidden and out_sem.len == ids.len * c.hidden);

        var cache = try st.Cache.init(m.gpa, m.model.device, ids.len);
        defer cache.deinit();

        const step = m.scratch.rows;
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + step);
            const n = end - at;
            try fwd.runPartial(m.model, &m.scratch, &cache, ids[at..end], layer_lex, layer_sem, self.tap_lex);
            const lex_bf16 = self.tap_lex.slice(u16, n * c.hidden);
            const sem_bf16 = m.scratch.h.slice(u16, n * c.hidden);
            mem.quant.bf16SliceToF32(lex_bf16, out_lex[at * c.hidden ..][0 .. n * c.hidden]);
            mem.quant.bf16SliceToF32(sem_bf16, out_sem[at * c.hidden ..][0 .. n * c.hidden]);
            at = end;
        }
    }

    fn loadRawIntoCache(cache: *st.Cache, src: *const mem.RawState) void {
        const len: usize = src.len;
        std.debug.assert(len <= cache.capacity and cache.paged == null);
        var r_idx: usize = 0;
        var a_idx: usize = 0;
        for (cache.blocks) |block| switch (block) {
            .delta => |d| {
                @memcpy(d.recurrence.contents()[0..st.delta_bytes], std.mem.sliceAsBytes(src.rec_state[r_idx]));
                @memcpy(d.conv.contents()[0..st.conv_bytes], std.mem.sliceAsBytes(src.rec_conv[r_idx]));
                r_idx += 1;
            },
            .attention => |kv| {
                const k_dst = kv.keys.slice(u16, c.kv_heads * cache.capacity * c.head_dim);
                const v_dst = kv.values.slice(u16, c.kv_heads * cache.capacity * c.head_dim);
                for (0..c.kv_heads) |h| {
                    const elems = len * c.head_dim;
                    @memcpy(k_dst[h * cache.capacity * c.head_dim ..][0..elems], src.kv_k[a_idx][h * elems ..][0..elems]);
                    @memcpy(v_dst[h * cache.capacity * c.head_dim ..][0..elems], src.kv_v[a_idx][h * elems ..][0..elems]);
                }
                a_idx += 1;
            },
        };
        cache.len = len;
        cache.last = null;
    }

    fn copyCacheSliceToRaw(cache: *const st.Cache, start: usize, len: usize, dst: *mem.RawState) void {
        std.debug.assert(start + len <= cache.len and dst.len == len and cache.paged == null);
        var r_idx: usize = 0;
        var a_idx: usize = 0;
        for (cache.blocks) |block| switch (block) {
            .delta => |d| {
                @memcpy(std.mem.sliceAsBytes(dst.rec_state[r_idx]), d.recurrence.contents()[0..st.delta_bytes]);
                @memcpy(std.mem.sliceAsBytes(dst.rec_conv[r_idx]), d.conv.contents()[0..st.conv_bytes]);
                r_idx += 1;
            },
            .attention => |kv| {
                const k_src = kv.keys.slice(u16, c.kv_heads * cache.capacity * c.head_dim);
                const v_src = kv.values.slice(u16, c.kv_heads * cache.capacity * c.head_dim);
                for (0..c.kv_heads) |h| {
                    const elems = len * c.head_dim;
                    const src_off = (h * cache.capacity + start) * c.head_dim;
                    @memcpy(dst.kv_k[a_idx][h * elems ..][0..elems], k_src[src_off..][0..elems]);
                    @memcpy(dst.kv_v[a_idx][h * elems ..][0..elems], v_src[src_off..][0..elems]);
                }
                a_idx += 1;
            },
        };
    }

    fn compileStateFn(
        ctx: *anyopaque,
        base: ?*const mem.RawState,
        ids: []const u32,
        out: *mem.RawState,
    ) anyerror!void {
        const self = of(ctx);
        const m = self.metal;
        const base_len: usize = if (base) |b| b.len else 0;
        const cap = base_len + ids.len;
        if (cap == 0 or ids.len == 0) return error.EmptyCompile;

        var cache = try st.Cache.init(m.gpa, m.model.device, cap);
        defer cache.deinit();
        if (base) |b| loadRawIntoCache(&cache, b);

        const step = @min(m.options.chunk, m.scratch.rows);
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + step);
            try fwd.run(m.model, &m.scratch, &.{.{ .cache = &cache, .rows = end - at }}, ids[at..end], false, .last);
            at = end;
        }
        copyCacheSliceToRaw(&cache, base_len, ids.len, out);
    }

    fn buildSnapFn(
        ctx: *anyopaque,
        composed: *const mem.RawState,
        suffix_ids: []const u32,
    ) anyerror!mem.SnapHandle {
        const self = of(ctx);
        const m = self.metal;
        const total = @as(usize, composed.len) + suffix_ids.len;
        var cache = try st.Cache.init(m.gpa, m.model.device, total);
        defer cache.deinit();
        loadRawIntoCache(&cache, composed);

        if (suffix_ids.len > 0) {
            const step = @min(m.options.chunk, m.scratch.rows);
            var at: usize = 0;
            while (at < suffix_ids.len) {
                const end = @min(suffix_ids.len, at + step);
                try fwd.run(m.model, &m.scratch, &.{.{ .cache = &cache, .rows = end - at }}, suffix_ids[at..end], false, .last);
                at = end;
            }
        }
        const snap = try snapshot.save(m.gpa, &cache, @intCast(cache.len));
        return .{ .saved = @ptrCast(snap), .at = snap.at };
    }

    fn dropSnapFn(ctx: *anyopaque, ptr: *anyopaque) void {
        const self = of(ctx);
        snapshot.drop(self.metal.gpa, @ptrCast(@alignCast(ptr)));
    }
};
