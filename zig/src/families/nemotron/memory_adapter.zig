//! Folded Memory Weights v3 Metal adapter for Nemotron-H (Mamba-2 + MoE + NoPE Attention).
const std = @import("std");
const mtl = @import("metal");
const living = @import("living");
const mem = living.memory;
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const backend_mod = @import("backend.zig");
const Metal = backend_mod.Metal;
const Allocator = std.mem.Allocator;

pub const Adapter = struct {
    metal: *Metal,
    tap_lex: mtl.Buffer,
    ids_buf: mtl.Buffer,
    ids_cap: usize,

    pub fn init(gpa: Allocator, metal: *Metal) !*Adapter {
        const a = try gpa.create(Adapter);
        errdefer gpa.destroy(a);
        const c = metal.m.config;
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const tap_bytes = backend_mod.fused_rows * c.hidden * 2;
        const tap = try metal.m.device.buffer(tap_bytes, opts);
        errdefer tap.deinit();
        const cap: usize = 4096;
        const ids_buf = try metal.m.device.buffer(cap * 4, opts);
        a.* = .{ .metal = metal, .tap_lex = tap, .ids_buf = ids_buf, .ids_cap = cap };
        return a;
    }

    pub fn deinit(self: *Adapter, gpa: Allocator) void {
        self.tap_lex.deinit();
        self.ids_buf.deinit();
        gpa.destroy(self);
    }

    pub fn deinitOpaque(ptr: *anyopaque, gpa: Allocator) void {
        const self: *Adapter = @ptrCast(@alignCast(ptr));
        self.deinit(gpa);
    }

    pub fn archSpec(metal: *const Metal) mem.ArchSpec {
        const c = metal.m.config;
        const has_head = metal.head != null;
        return .{
            .family = .nemotron,
            .hidden = c.hidden,
            .num_layers = c.layers,
            .layer_lex = 1,
            .layer_sem = (2 * c.layers) / 3, // layer 34 of 52
            .num_attn = c.count(.attention) + @intFromBool(has_head), // 6 (+1 if MTP head)
            .kv_heads = c.kv_heads, // 2
            .head_dim = c.head_dim, // 128
            .rotary_dim = 0, // Nemotron-H is pure NoPE!
            .rope_theta = 10_000.0,
            .num_rec = c.count(.mamba), // 23
            .rec_heads = c.mamba_heads, // 64
            .rec_rows = c.mamba_head_dim, // 64
            .rec_cols = c.state, // 128
            .conv_elems = (c.conv_kernel - 1) * c.convDim(), // 3 * 6144
            .svd_rank = 0, // KV-only on doc chunks (zero Mamba SVD!)
            .svd_rank_short = 4, // rank-4 Mamba SVD on tiny facts & standing policies so 15-token rules steer all 23 Mamba layers!
            .kv_bits = 8, // Q8 KV across only 6+1 attention layers (+3.7 MiB vs Q4) -> zero decimal quantization drift!
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

    fn ensureIdsBuf(self: *Adapter, need: usize) !void {
        if (need <= self.ids_cap) return;
        const next_cap = std.math.ceilPowerOfTwo(usize, need) catch need;
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const nb = try self.metal.m.device.buffer(next_cap * 4, opts);
        self.ids_buf.deinit();
        self.ids_buf = nb;
        self.ids_cap = next_cap;
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
        const b = self.metal;
        const c = b.m.config;
        if (ids.len == 0) return;
        std.debug.assert(out_lex.len == ids.len * c.hidden and out_sem.len == ids.len * c.hidden);

        try b.drain();
        try self.ensureIdsBuf(ids.len);
        @memcpy(self.ids_buf.slice(u32, ids.len), ids);

        var cache = try st.Cache.init(b.m.device, c, ids.len + 1, false);
        defer cache.deinit(&b.pool);
        cache.untaught = true;

        const PartialJob = struct {
            cache: *st.Cache,
            ids_buf: mtl.Buffer,
            at: usize,
            rows: usize,
            layer_lex: usize,
            layer_sem: usize,
            tap_lex: mtl.Buffer,
            pub fn encode(j: @This(), m: *Metal, e: *fwd.Enc) !void {
                const fresh = try m.pool.take();
                const segs = [_]fwd.Seg{.{ .rows = j.rows, .cache = j.cache, .store = .full, .slot = @intCast(fresh) }};
                m.forward().bodyPartial(e, &segs, j.ids_buf, j.at * 4, j.layer_lex, j.layer_sem, j.tap_lex);
                j.cache.advance(&m.pool, j.rows, fresh);
            }
        };

        var at: usize = 0;
        while (at < ids.len) {
            const rows = @min(backend_mod.fused_rows, ids.len - at);
            try b.submit(.prefill, b.next, PartialJob{
                .cache = &cache,
                .ids_buf = self.ids_buf,
                .at = at,
                .rows = rows,
                .layer_lex = layer_lex,
                .layer_sem = layer_sem,
                .tap_lex = self.tap_lex,
            });
            try b.drain();
            const lex_bf16 = self.tap_lex.slice(u16, rows * c.hidden);
            const sem_bf16 = b.scratch.h.slice(u16, rows * c.hidden);
            mem.quant.bf16SliceToF32(lex_bf16, out_lex[at * c.hidden ..][0 .. rows * c.hidden]);
            mem.quant.bf16SliceToF32(sem_bf16, out_sem[at * c.hidden ..][0 .. rows * c.hidden]);
            at += rows;
        }
    }

    fn loadRawIntoCache(b: *Metal, cache: *st.Cache, src: *const mem.RawState) !void {
        const c = b.m.config;
        const len: usize = src.len;
        const slot = try b.pool.take();
        errdefer b.pool.give(slot);

        const conv_bytes = (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm_bytes = c.mamba_heads * c.mamba_head_dim * c.state * 4;
        for (0..b.pool.layers) |m| {
            @memcpy((b.pool.conv[m].contents() + slot * conv_bytes)[0..conv_bytes], std.mem.sliceAsBytes(src.rec_conv[m]));
            @memcpy((b.pool.ssm[m].contents() + slot * ssm_bytes)[0..ssm_bytes], std.mem.sliceAsBytes(src.rec_state[m]));
        }

        const n_kvs = cache.attentions + @intFromBool(cache.mtp != null);
        for (0..n_kvs) |i| {
            const kv = if (i < cache.attentions) cache.kv[i] else cache.mtp.?;
            const k_dst = kv.k.slice(u16, c.kv_heads * kv.capacity * c.head_dim);
            const v_dst = kv.v.slice(u16, c.kv_heads * kv.capacity * c.head_dim);
            const elems = len * c.head_dim;
            for (0..c.kv_heads) |h| {
                @memcpy(k_dst[h * kv.capacity * c.head_dim ..][0..elems], src.kv_k[i][h * elems ..][0..elems]);
                @memcpy(v_dst[h * kv.capacity * c.head_dim ..][0..elems], src.kv_v[i][h * elems ..][0..elems]);
            }
        }
        cache.advance(&b.pool, len, slot);
        if (cache.mtp != null) cache.mtp_len = len;
    }

    fn copyCacheSliceToRaw(b: *const Metal, cache: *const st.Cache, start: usize, len: usize, dst: *mem.RawState) void {
        const c = b.m.config;
        const slot: usize = cache.slot;
        const conv_bytes = (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm_bytes = c.mamba_heads * c.mamba_head_dim * c.state * 4;
        for (0..b.pool.layers) |m| {
            @memcpy(std.mem.sliceAsBytes(dst.rec_conv[m]), (b.pool.conv[m].contents() + slot * conv_bytes)[0..conv_bytes]);
            @memcpy(std.mem.sliceAsBytes(dst.rec_state[m]), (b.pool.ssm[m].contents() + slot * ssm_bytes)[0..ssm_bytes]);
        }

        const n_kvs = cache.attentions + @intFromBool(cache.mtp != null);
        for (0..n_kvs) |i| {
            const kv = if (i < cache.attentions) cache.kv[i] else cache.mtp.?;
            const k_src = kv.k.slice(u16, c.kv_heads * kv.capacity * c.head_dim);
            const v_src = kv.v.slice(u16, c.kv_heads * kv.capacity * c.head_dim);
            const elems = len * c.head_dim;
            for (0..c.kv_heads) |h| {
                const src_off = (h * kv.capacity + start) * c.head_dim;
                @memcpy(dst.kv_k[i][h * elems ..][0..elems], k_src[src_off..][0..elems]);
                @memcpy(dst.kv_v[i][h * elems ..][0..elems], v_src[src_off..][0..elems]);
            }
        }
    }

    /// Runs `ids` through `cache` (using full prefill chunks for speed), absorbing every row into
    /// the optional MTP head using `follow_token` as the next token after the final row.
    fn runForwardIntoCache(self: *Adapter, cache: *st.Cache, ids: []const u32, follow_token: u32) !void {
        const b = self.metal;
        try self.ensureIdsBuf(ids.len + 1);
        const id_slice = self.ids_buf.slice(u32, ids.len + 1);
        @memcpy(id_slice[0..ids.len], ids);
        id_slice[ids.len] = follow_token;

        const ChunkJob = struct {
            c: *st.Cache,
            ids_buf: mtl.Buffer,
            at: usize,
            rows: usize,
            pub fn encode(j: @This(), m: *Metal, e: *fwd.Enc) !void {
                const d = m.m.config.hidden * 2;
                const fresh = try m.pool.take();
                var out = m.scratch.x;
                if (j.rows <= backend_mod.fused_rows or m.wide == null) {
                    const segs = [_]fwd.Seg{.{ .rows = j.rows, .cache = j.c, .store = .full, .slot = @intCast(fresh) }};
                    m.forward().body(e, &segs, j.ids_buf, j.at * 4);
                } else {
                    const w = &m.wide.?;
                    w.chunk(e, m.forward(), j.c, j.ids_buf, j.at * 4, j.rows, fresh);
                    out = w.s.x;
                }
                j.c.advance(&m.pool, j.rows, fresh);
                if (m.head) |*h| {
                    var r: usize = 0;
                    while (r < j.rows) : (r += backend_mod.max_window) {
                        h.absorb(e, j.c, @min(backend_mod.max_window, j.rows - r), out, r * d, j.ids_buf, (j.at + 1 + r) * 4);
                    }
                }
            }
        };

        const step = if (b.wide != null) b.o.chunk else backend_mod.fused_rows;
        var at: usize = 0;
        while (at < ids.len) {
            const rows = @min(step, ids.len - at);
            try b.submit(.prefill, b.next, ChunkJob{
                .c = cache,
                .ids_buf = self.ids_buf,
                .at = at,
                .rows = rows,
            });
            at += rows;
        }
        try b.drain();
    }

    fn compileStateFn(
        ctx: *anyopaque,
        base: ?*const mem.RawState,
        ids: []const u32,
        out: *mem.RawState,
    ) anyerror!void {
        const self = of(ctx);
        const b = self.metal;
        const base_len: usize = if (base) |x| x.len else 0;
        const cap = base_len + ids.len + 2;
        if (ids.len == 0) return error.EmptyCompile;

        try b.drain();
        var cache = try st.Cache.init(b.m.device, b.m.config, cap, b.head != null);
        defer cache.deinit(&b.pool);
        cache.untaught = true;
        if (base) |x| try loadRawIntoCache(b, &cache, x);

        try self.runForwardIntoCache(&cache, ids, ids[0]);
        copyCacheSliceToRaw(b, &cache, base_len, ids.len, out);
    }

    fn buildSnapFn(
        ctx: *anyopaque,
        composed: *const mem.RawState,
        suffix_ids: []const u32,
    ) anyerror!mem.SnapHandle {
        const self = of(ctx);
        const b = self.metal;
        const total = @as(usize, composed.len) + suffix_ids.len;
        try b.drain();
        var cache = try st.Cache.init(b.m.device, b.m.config, total + 2, b.head != null);
        defer cache.deinit(&b.pool);
        cache.untaught = true;
        try loadRawIntoCache(b, &cache, composed);

        if (suffix_ids.len > 0) {
            const follow: u32 = if (suffix_ids.len > 0) suffix_ids[0] else 0;
            try self.runForwardIntoCache(&cache, suffix_ids, follow);
        }
        const snap = try snapshot.save(b, &cache, @intCast(cache.len));
        try b.drain();
        return .{ .saved = @ptrCast(snap), .at = snap.at };
    }

    fn dropSnapFn(ctx: *anyopaque, ptr: *anyopaque) void {
        const self = of(ctx);
        snapshot.drop(self.metal, @ptrCast(@alignCast(ptr)));
    }
};
