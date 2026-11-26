//! The Qwen MTP layer as the drafter on CUDA: chains or trees from a kept row's final normed state and next token.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const weights = @import("cuda_mtp_weights.zig");
const timing = @import("cuda_draft.zig").timing;
const precision = @import("cuda_draft_precision.zig");
const batched = @import("cuda_mtp_batch.zig");
pub const mtp_draw = @import("cuda_mtp_draw.zig");
const types = @import("cuda_mtp_types.zig");
const memory = @import("cuda_mtp_memory.zig");

fn arrayBytes(a: usize, b: usize, c: usize) !usize {
    return weights.checkedMul(try weights.checkedMul(a, b), c);
}

const E = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;
const dr = lanes.drafter;
const Stream = lanes.Stream;
const head_dim = types.head_dim;
const max_rows = types.max_rows;
const most_window = types.most_window;
pub const max_drafts = types.max_drafts;
pub const max_lanes = types.max_lanes;
const ranks = types.ranks;
pub const lp_words = types.lp_words;
pub const TapRows = types.TapRows;
pub const Room = memory.Room;
pub const Ctx = types.Ctx;
const Pass = types.Pass;
pub const RingOf = types.RingOf;

pub const Mtp = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    e: *E,
    projection_role: precision.Role = .proposal,
    w: weights.Weights,
    mod: cuda.Module,
    dmod: cuda.Module,
    fns: types.Fns,
    ring_fns: types.Ring,
    scratch: DeviceBuffer,
    capacity: usize, // ring rows: every position of the stream's window
    ring_bytes: usize = 0,
    step_offset: usize = 0,
    area_bytes: usize = 0,
    stream_bytes: usize = 0,
    scratch_bytes: usize = 0,
    depth: u32,
    /// Initial prompt rows retained for direct-normalized MTP; other paths keep the full tap window.
    initial_prompt_rows: u32 = types.default_initial_prompt_rows,
    trees: bool = false,
    tap: [1]u32,
    // Row scratch supports the KV-only prompt tile; decode rows remain at the lane verify width.
    taps_in: u64,
    hin: u64,
    hn: u64,
    emb: u64,
    en: u64,
    h: u64,
    r: u64,
    x: u64,
    qraw: u64,
    q: u64,
    kraw: u64,
    kb: u64,
    vb: u64,
    att: u64,
    xs: u64, // f32 group sums for the projections
    logits: u64, // max_lanes rows
    words: u64, // u32 logprob words a lane row
    picks: u64, // u32 zeros: the logprob rows' pick (unused)
    lane_state: u64, // each tree lane's normed output
    lane_k: u64, // each tree lane's key and value rows
    lane_v: u64,
    anc: u64, // int32 ancestor table a row: a count, then lane indices
    ids: u64, // u32 the rows' tokens
    tok: u64, // u32 the drawn tokens
    rules: u64 = 0, // sample.cu's Rule a row for sampled draws
    meta: u64 = 0, // sample.cu's position - 1 a row for sampled draws
    chain_tok: u64 = 0, // u32 chain tokens kept on device until the hold finishes
    staged: cuda.HostBuffer, // pinned: ids, ancestors and async batch slots up; token results down
    tab: u64, // a batched pass's RingRow table (cuda_mtp_batch.zig)
    batch: bool = true,
    batch_passes: u64 = 0, // batched passes run, for the GPU check
    streams: std.AutoHashMapUnmanaged(*Stream, *Ctx) = .empty,
    /// Contexts between streams, allocated and run once (`reserve`), as the driver stalls a buffer's first kernel.
    pool: std.ArrayList(*Ctx) = .empty,
    /// TF_QWEN_KV_PAGES: a stream's ring and state are views of its target lane's run (null: the contexts own theirs).
    ring_of: ?RingOf = null,
    /// One draft level's cost (timed at load): a chained row, or a tree's rows at one depth.
    step_ms: f64 = 1.0,
    /// The target's window ms by rows (cuda_native.zig times them), for the round loop's prices.
    costs: [most_window]lanes.config.Cost = undefined,
    n_costs: usize = 0,

    const anc_stride = 1 + max_lanes;
    const ids_at = 0;
    const absorb_id_rows = @import("cuda_cache.zig").tap_window;
    pub const max_streams = types.max_segs;
    pub const max_chain_tokens = max_lanes;
    pub const tok_at = absorb_id_rows * 4;
    pub const words_at = tok_at + max_lanes * 4;
    const anc_at = words_at + max_lanes * lp_words * 4;
    pub const chain_tok_at = anc_at + max_lanes * anc_stride * 4;
    pub const batch_ids_bytes = batched.most * 4;
    pub const batch_tab_at = engine.aligned(batch_ids_bytes);
    pub const batch_stage_stride = engine.aligned(batch_tab_at + batched.most * @sizeOf(batched.RingRow));
    pub const batch_slots = (most_window - 1) * ((types.max_segs + batched.most - 1) / batched.most);
    pub const batch_stage_at = chain_tok_at + types.max_segs * max_lanes * 4;
    pub const rules_at = std.mem.alignForward(usize, batch_stage_at + batch_slots * batch_stage_stride, @alignOf(cuda.sample.Rule));
    pub const meta_at = rules_at + max_lanes * @sizeOf(cuda.sample.Rule);
    pub const staged_len = meta_at + max_lanes * 4;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8, capacity: usize) !*Mtp {
        return initConfigured(gpa, io, e, dir, capacity, null, 0, null);
    }

    pub fn initConfigured(gpa: std.mem.Allocator, io: std.Io, e: *E, dir: []const u8, capacity: usize, requested_depth: ?u32, calibration_cache_bytes: usize, room: ?Room) !*Mtp {
        try weights.validateGeometry(e.g);
        const d = try gpa.create(Mtp);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .e = e, .w = undefined, .mod = undefined, .dmod = undefined, .fns = undefined, .ring_fns = undefined, .scratch = undefined, .capacity = capacity, .depth = requested_depth orelse types.default_depth, .tap = .{@intCast(e.g.layers - 1)}, .taps_in = 0, .hin = 0, .hn = 0, .emb = 0, .en = 0, .h = 0, .r = 0, .x = 0, .qraw = 0, .q = 0, .kraw = 0, .kb = 0, .vb = 0, .att = 0, .xs = 0, .logits = 0, .words = 0, .picks = 0, .lane_state = 0, .lane_k = 0, .lane_v = 0, .anc = 0, .ids = 0, .tok = 0, .staged = undefined, .tab = 0 };
        if (requested_depth == null) {
            d.depth = try platform.environment.read(u32, "TF_MTP_DRAFTS", types.default_depth);
        }
        d.trees = try platform.environment.read(bool, "TF_MTP_TREES", false);
        if (d.depth == 0) return error.MtpDepthZero;
        d.depth = @min(d.depth, max_drafts);
        d.batch = try platform.environment.read(bool, "TF_MTP_BATCH", true);
        try d.initializeMemorySizes();
        try memory.check(gpa, io, e.g, dir, d.scratch_bytes, d.ring_bytes, calibration_cache_bytes, room);
        d.mod = try cuda.Module.load(e.d, cuda.kernels.qwen36_mtp);
        errdefer d.mod.unload();
        d.dmod = try cuda.Module.load(e.d, cuda.kernels.qwen35_draft);
        errdefer d.dmod.unload();
        inline for (@typeInfo(types.Fns).@"struct".field_names) |name| @field(d.fns, name) = try d.mod.function("tf_q36m_" ++ name);
        inline for (@typeInfo(types.Ring).@"struct".field_names) |name| @field(d.ring_fns, name) = try d.dmod.function("tf_q35d_" ++ name);
        d.w = try weights.load(gpa, io, e, dir);
        errdefer d.w.deinit();
        d.scratch = try DeviceBuffer.alloc(e.d, d.scratch_bytes);
        errdefer d.scratch.free();
        try d.scratch.fill8(0, null);
        _ = try d.layout(true);
        d.staged = try cuda.HostBuffer.alloc(e.d, staged_len);
        errdefer d.staged.free();
        // the first context, and a chain timed on it: the quickest of sixteen, as the driver's stalls last a dozen
        const ctx = try d.create();
        errdefer d.drop(ctx);
        var quickest: f64 = std.math.inf(f64);
        for (0..16) |_| {
            try d.absorb(ctx, d.taps_in, 0, 1, null, &.{0});
            const t0 = std.Io.Clock.awake.now(io);
            try d.hold(ctx, 0, 1, d.depth, null);
            const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
            quickest = @min(quickest, ms);
        }
        d.step_ms = quickest / @as(f64, @floatFromInt(d.depth));
        d.drop(ctx);
        return d;
    }

    fn treeMode(raw: ?[]const u8) !bool {
        return platform.environment.value(bool, "TF_MTP_TREES", raw, false);
    }

    test "MTP drafter selection: tree drafting is an explicit strict opt-in" {
        try std.testing.expect(!try treeMode(null));
        try std.testing.expect(!try treeMode("0"));
        try std.testing.expect(try treeMode("1"));
        try std.testing.expectError(error.InvalidEnvironment, treeMode("yes"));
    }

    /// The drafter's tap: the target's last layer, whose residual the final norm then reads.
    pub fn taps(d: *const Mtp) []const u32 {
        return &d.tap;
    }

    pub fn setCost(d: *Mtp, rows: u32, ms: f64) void {
        if (platform.environment.read(bool, "TF_TIMING", false) catch false) std.debug.print("[timing] window {d} rows {d:.2} ms (a draft level {d:.2})\n", .{ rows, ms, d.step_ms });
        if (d.n_costs < d.costs.len) {
            d.costs[d.n_costs] = .{ .width = rows, .ms = ms };
            d.n_costs += 1;
        }
    }

    pub fn deinit(d: *Mtp) void {
        var it = d.streams.valueIterator();
        while (it.next()) |ctx| d.drop(ctx.*);
        d.streams.deinit(d.gpa);
        for (d.pool.items) |ctx| d.drop(ctx);
        d.pool.deinit(d.gpa);
        d.staged.free();
        d.scratch.free();
        d.w.deinit();
        d.dmod.unload();
        d.mod.unload();
        d.gpa.destroy(d);
    }

    pub fn kvWidth(d: *const Mtp) usize {
        return d.e.g.kv_heads * head_dim;
    }

    pub fn ringBytes(d: *const Mtp) usize {
        return d.ring_bytes;
    }

    /// A pool lane's MTP area: ring, two aligned states, and its held draft tokens.
    pub fn areaBytes(d: *const Mtp) usize {
        return d.area_bytes;
    }

    fn stepOffset(d: *const Mtp) usize {
        return d.step_offset;
    }

    pub fn initializeMemorySizes(d: *Mtp) !void {
        try weights.validateGeometry(d.e.g);
        const max_c_int: usize = @intCast(std.math.maxInt(c_int));
        if (d.capacity == 0 or d.capacity > max_c_int) return error.UnsupportedMtpGeometry;
        const hidden_bytes = try weights.checkedMul(d.e.g.hidden, 2);
        const target_tap_bytes = try weights.checkedMul(try weights.checkedMul(d.taps().len, d.e.tapRows()), hidden_bytes);
        const kv_width = try weights.checkedMul(d.e.g.kv_heads, head_dim);
        d.ring_bytes = try arrayBytes(try weights.checkedMul(2, d.capacity), kv_width, 2);
        const aligned_ring = try weights.checkedAligned(d.ring_bytes);
        d.step_offset = try weights.checkedAligned(try weights.checkedAdd(aligned_ring, hidden_bytes));
        d.area_bytes = try weights.checkedAdd(d.step_offset, hidden_bytes);
        d.area_bytes = try weights.checkedAdd(d.area_bytes, try weights.checkedMul(max_lanes, 4));
        d.stream_bytes = try weights.checkedAdd(d.ring_bytes, try weights.checkedMul(2, hidden_bytes));
        d.stream_bytes = try weights.checkedAdd(d.stream_bytes, try weights.checkedMul(max_lanes, 4));
        d.stream_bytes = try weights.checkedAdd(d.stream_bytes, target_tap_bytes);
        d.scratch_bytes = try d.layout(false);
    }

    /// The scratch rows, measured (nothing assigned) or assigned from the scratch buffer.
    pub fn layout(d: *Mtp, assign: bool) !usize {
        const g = d.e.g;
        const R = @import("cuda_mtp_absorb.zig").prompt_kv_tile;
        const L = max_lanes;
        const aw = g.attnWidth();
        const kw = d.kvWidth();
        const sizes = .{
            .{ "taps_in", try arrayBytes(R, g.hidden, 2) },                   .{ "hin", try arrayBytes(R, g.hidden, 2) },   .{ "hn", try arrayBytes(R, g.hidden, 2) },                                .{ "emb", try arrayBytes(R, g.hidden, 2) },
            .{ "en", try arrayBytes(R, g.hidden, 2) },                        .{ "h", try arrayBytes(R, g.hidden, 2) },     .{ "r", try arrayBytes(R, g.hidden, 2) },                                 .{ "x", try arrayBytes(R, g.hidden, 2) },
            .{ "qraw", try arrayBytes(R, try weights.checkedMul(2, aw), 2) }, .{ "q", try arrayBytes(R, aw, 2) },           .{ "kraw", try arrayBytes(R, kw, 2) },                                    .{ "kb", try arrayBytes(R, kw, 2) },
            .{ "vb", try arrayBytes(R, kw, 2) },                              .{ "att", try arrayBytes(R, aw, 2) },         .{ "xs", try arrayBytes(R, @max(g.hidden, aw, g.intermediate) / 64, 4) }, .{ "logits", try arrayBytes(L, g.vocab, 2) },
            .{ "words", try arrayBytes(L, lp_words, 4) },                     .{ "picks", try arrayBytes(L, 1, 4) },        .{ "lane_state", try arrayBytes(L, g.hidden, 2) },                        .{ "lane_k", try arrayBytes(L, kw, 2) },
            .{ "lane_v", try arrayBytes(L, kw, 2) },                          .{ "anc", try arrayBytes(L, anc_stride, 4) }, .{ "ids", try arrayBytes(absorb_id_rows, 1, 4) },                         .{ "tok", try arrayBytes(L, 1, 4) },
            .{ "rules", try arrayBytes(L, @sizeOf(cuda.sample.Rule), 1) },    .{ "meta", try arrayBytes(L, 1, 4) },         .{ "chain_tok", try arrayBytes(L, 1, 4) },                                .{ "tab", try arrayBytes(batched.most, @sizeOf(batched.RingRow), 1) },
        };
        var total: usize = 0;
        inline for (sizes) |s| {
            const at = try weights.checkedAligned(total);
            total = try weights.checkedAdd(at, s[1]);
            if (assign) @field(d, s[0]) = d.scratch.ptr + at;
        }
        return total;
    }

    pub fn drop(d: *Mtp, ctx: *Ctx) void {
        ctx.held_dev.free();
        ctx.step.free();
        ctx.last.free();
        ctx.ring.free();
        d.gpa.destroy(ctx);
    }

    fn resetCtx(d: *Mtp, ctx: *Ctx) void {
        _ = d;
        ctx.* = .{ .ring = ctx.ring, .last = ctx.last, .step = ctx.step, .held_dev = ctx.held_dev };
    }

    /// A new context, touched once (an absorb, then a chain), so no request pays the driver's stall.
    pub fn create(d: *Mtp) !*Ctx {
        const ctx = try d.gpa.create(Ctx);
        errdefer d.gpa.destroy(ctx);
        ctx.* = .{ .ring = try DeviceBuffer.alloc(d.e.d, d.ringBytes()), .last = undefined, .step = undefined, .held_dev = undefined };
        errdefer ctx.ring.free();
        ctx.last = try DeviceBuffer.alloc(d.e.d, d.e.g.hidden * 2);
        errdefer ctx.last.free();
        ctx.step = try DeviceBuffer.alloc(d.e.d, d.e.g.hidden * 2);
        errdefer ctx.step.free();
        ctx.held_dev = try DeviceBuffer.alloc(d.e.d, max_lanes * 4);
        errdefer ctx.held_dev.free();
        try ctx.step.fill8(0, null);
        try ctx.ring.fill8(0, null);
        try ctx.last.fill8(0, null);
        try d.absorb(ctx, d.taps_in, 0, 1, null, &.{0}); // any tap rows do: the drafter's own zeroed staging
        try d.hold(ctx, 0, 1, d.depth, null);
        d.resetCtx(ctx);
        return ctx;
    }

    /// `n` contexts ready in the pool, for the streams the server admits.
    pub fn reserve(d: *Mtp, n: usize) !void {
        // with a pool the other streams' buffers are lane runs (all: every stream's)
        const own = if (d.ring_of) |r| (if (r.all) 0 else @min(n, 1)) else n;
        try d.pool.ensureTotalCapacity(d.gpa, own);
        while (d.pool.items.len > own) d.drop(d.pool.pop().?);
        while (d.pool.items.len < own) d.pool.appendAssumeCapacity(try d.create());
    }

    pub fn ringK(d: *Mtp, ctx: *Ctx) u64 {
        _ = d;
        return ctx.ring.ptr;
    }

    pub fn ringV(d: *Mtp, ctx: *Ctx) u64 {
        return ctx.ring.ptr + d.capacity * d.kvWidth() * 2;
    }

    /// out (rows, n) = x (rows, k) through `l`, as the target's decode tiles or prompt GEMM, with the drafter's sums.
    pub fn proj(d: *Mtp, l: engine.Proj, x: u64, out: u64, rows: usize) !void {
        return precision.project(d.e, l, x, d.xs, out, rows, d.projection_role, precision.mtp_projection_prompt_cutoff);
    }

    pub fn routed(d: *Mtp, router: u64, experts: cuda.experts.Layer, x: u64, out: u64, rows: usize) !void {
        return precision.routed(d.e, router, experts, x, out, rows, d.projection_role);
    }

    /// Dense SwiGLU MTP and the existing routed MTP use the target's projection and activation arithmetic.
    pub fn mlp(d: *Mtp, x: u64, out: u64, rows: usize) !void {
        const e = d.e;
        switch (d.w.mlp) {
            .dense => |m| {
                try d.proj(m.gate, x, e.gate, rows);
                try d.proj(m.up, x, e.up, rows);
                const elements = rows * e.g.intermediate;
                try e.run(e.fns.mlp_act, .{ (elements + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, @as(c_longlong, @intCast(elements)), @as(u64, 0), @as(c_int, @intCast(e.g.intermediate)), @as(u64, 0) });
                try d.proj(m.down, e.act, out, rows);
            },
            .moe => |m| try d.routed(m.router, m.experts, x, out, rows),
        }
    }

    /// The layer's actual grouped Q/K/V dispatch, sharing only the input sums.
    pub fn projQkv(d: *Mtp, rows: usize) !void {
        try d.e.projManySums(&.{ d.w.q, d.w.k, d.w.v }, d.x, d.xs, &.{ d.qraw, d.kraw, d.vb }, rows, precision.prompt(d.projection_role, rows, precision.mtp_projection_prompt_cutoff), 0);
    }

    /// out = norm(h + r) with w; h takes the sum (r 0: no residual).
    pub fn norm(d: *Mtp, h: u64, r: u64, w: u64, out: u64, rows: usize) !void {
        const e = d.e;
        try e.run(e.fns.norm, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ h, r, w, out, e.g.eps, @as(c_int, @intCast(e.g.hidden)), @as(u64, 0), @as(u64, 0) });
    }

    /// q or k heads normed and roped from `src` into `out`, with the target's rope (its YaRN table when it has one).
    pub fn normRope(d: *Mtp, src: u64, w: u64, out: u64, heads: usize, stride: usize, in_head: usize, p0: c_longlong, step: c_int, rows: usize) !void {
        const e = d.e;
        try e.run(d.fns.norm_rope, .{ rows * heads, 1, 1 }, .{ 32, 1, 1 }, .{ src, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(stride)), @as(c_int, @intCast(in_head)), p0, step, e.g.eps, types.theta, e.rope_table });
    }

    /// The layer over `rows` rows at slots pos0..: d.ids their tokens, d.hin their states; x gets their normed outputs.
    pub fn layer(d: *Mtp, ctx: *Ctx, rows: usize, pos0: u64, pass: Pass) !void {
        return d.layerMode(ctx, rows, pos0, pass, true);
    }

    pub fn layerMode(d: *Mtp, ctx: *Ctx, rows: usize, pos0: u64, pass: Pass, propose: bool) !void {
        return d.layerModeWithIds(ctx, d.ids, rows, pos0, pass, propose);
    }

    /// Run a layer using token IDs already resident on the device.
    pub fn layerModeWithIds(d: *Mtp, ctx: *Ctx, ids: u64, rows: usize, pos0: u64, pass: Pass, propose: bool) !void {
        return d.layerModeWithInput(ctx, ids, rows, pos0, pass, propose, d.hin);
    }

    /// Run a layer over a caller-owned hidden row view; all reads are enqueued on the target stream.
    pub fn layerModeWithInput(d: *Mtp, ctx: *Ctx, ids: u64, rows: usize, pos0: u64, pass: Pass, propose: bool, hin: u64) !void {
        const e = d.e;
        const g = e.g;
        const w = &d.w;
        const aw = g.attnWidth();
        const kw = d.kvWidth();
        const p0: c_longlong = @intCast(pos0);
        // a tree pass's rows are sibling lanes at one depth, so they share pos0; chained or absorbed rows step by one
        const step: c_int = @intFromBool(pass == .ring);
        try e.embed(ids, d.emb, rows, 0);
        try d.norm(d.emb, 0, w.norm_e, d.en, rows);
        try d.norm(hin, 0, w.norm_h, d.hn, rows);
        try d.proj(w.fc_e, d.en, d.h, rows);
        try d.proj(w.fc_h, d.hn, d.r, rows);
        try d.norm(d.h, d.r, w.input_norm, d.x, rows);
        if (!propose) {
            // Prompt cuts that cannot feed a proposal need only build the ring's keys and values.
            try @import("cuda_mtp_absorb.zig").projKv(d, rows);
            try d.normRope(d.kraw, w.k_norm, d.kb, g.kv_heads, kw, head_dim, p0, step, rows);
            try d.putRing(ctx, rows, p0);
            return;
        }
        try d.projQkv(rows);
        try d.normRope(d.qraw, w.q_norm, d.q, g.q_heads, 2 * aw, 2 * head_dim, p0, step, rows);
        try d.normRope(d.kraw, w.k_norm, d.kb, g.kv_heads, kw, head_dim, p0, step, rows);
        const heads: c_int = @intCast(g.q_heads);
        const kvh: c_int = @intCast(g.kv_heads);
        const cap: c_int = @intCast(d.capacity);
        const begin: c_longlong = @intCast(ctx.begin);
        switch (pass) {
            .ring => try e.run(d.fns.attn, .{ rows, g.q_heads, 1 }, .{ 32, 1, 1 }, .{ d.q, d.qraw, @as(c_int, @intCast(2 * aw)), d.kb, d.vb, d.ringK(ctx), d.ringV(ctx), heads, kvh, cap, begin, p0, d.att }),
            .tree => |anc| try e.run(d.fns.attn_tree, .{ rows, g.q_heads, 1 }, .{ 32, 1, 1 }, .{ d.q, d.qraw, @as(c_int, @intCast(2 * aw)), d.kb, d.vb, d.lane_k, d.lane_v, anc, @as(c_int, anc_stride), d.ringK(ctx), d.ringV(ctx), heads, kvh, cap, begin, @as(c_longlong, @intCast(ctx.end)), d.att }),
        }
        try d.proj(w.o, d.att, d.r, rows);
        try d.norm(d.h, d.r, w.post_norm, d.x, rows);
        try d.mlp(d.x, d.r, rows);
        try d.norm(d.h, d.r, w.norm, d.x, rows);
        if (pass == .ring) try d.putRing(ctx, rows, p0);
    }

    fn putRing(d: *Mtp, ctx: *Ctx, rows: usize, p0: c_longlong) !void {
        try d.e.run(d.ring_fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kb, d.vb, d.ringK(ctx), d.ringV(ctx), @as(c_int, @intCast(d.kvWidth())), p0, @as(c_int, @intCast(d.capacity)) });
    }

    /// Rows [first, first + specs.len) of x through the target's head, drawn under each row's rule and position.
    pub fn drawSpecs(d: *Mtp, first: usize, specs: []const mtp_draw.Spec, cands: bool) !void {
        return mtp_draw.launch(d, d.x + first * d.e.g.hidden * 2, specs, cands);
    }

    /// Reads back the drawn tokens and optional tree candidates for `specs`.
    pub fn fetchSpecs(d: *Mtp, specs: []const mtp_draw.Spec, out_toks: []u32, cands: [][ranks]u32) !void {
        return mtp_draw.fetch(d, specs, out_toks, cands);
    }

    pub fn putIds(d: *Mtp, ids: []const u32) !void {
        if (ids.len > absorb_id_rows) return error.BadDraftRows;
        return d.putIdsReady(ids);
    }

    /// copy_source owns the host bytes before its async transfer; the shared stream orders device ID use.
    fn putIdsReady(d: *Mtp, ids: []const u32) !void {
        const e = d.e;
        @memcpy(std.mem.bytesAsSlice(u32, d.staged.bytes[ids_at..][0 .. ids.len * 4]), ids);
        try cuda.copy_source.enqueue(e.d, d.ids, d.staged.bytes[ids_at..][0 .. ids.len * 4], e.stream);
    }

    /// Read a completed chain in one host transfer after all of its rows have run.
    fn fetchChain(d: *Mtp, out: []u32) !void {
        if (out.len == 0) return;
        const e = d.e;
        const st = d.staged.bytes;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + chain_tok_at, d.chain_tok, out.len * 4, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        @memcpy(out, std.mem.bytesAsSlice(u32, st[chain_tok_at..][0 .. out.len * 4]));
    }

    /// Read a device-resident candidate and its draft suffix back together after the chain is complete.
    fn fetchHeld(d: *Mtp, ctx: *Ctx) !void {
        if (ctx.held_n == 0) return;
        if (ctx.held_n > max_chain_tokens) return error.TooManyDrafts;
        const e = d.e;
        const st = d.staged.bytes;
        const bytes = ctx.held_n * 4;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(st.ptr + chain_tok_at, ctx.held_dev.ptr, bytes, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        @memcpy(ctx.held[0..ctx.held_n], std.mem.bytesAsSlice(u32, st[chain_tok_at..][0..bytes]));
    }

    /// Target rows [start, start + n) from the tap ring with the token after each; `rows`, when given, picks them.
    pub fn absorb(d: *Mtp, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize, rows_: ?[]const u32, follow: []const u32) !void {
        return d.absorbRole(ctx, tap_ring, start_, n_, rows_, follow, true, true, null);
    }

    pub const absorbRole = @import("cuda_mtp_absorb.zig").absorbRole;
    pub const absorbNormalizedRole = @import("cuda_mtp_absorb.zig").absorbNormalizedRole;

    /// `depth` drafts after `pending` at pos0 (the pending token's slot): a chain, or the planner's tree of lanes.
    pub fn hold(d: *Mtp, ctx: *Ctx, pending: u32, pos0: u64, depth: usize, lanes_: ?*const lanes.shape.Shape) !void {
        return d.holdSampled(ctx, pending, pos0, depth, lanes_, null);
    }

    /// `hold` under `sampling` (target-coupled draws at pos0 + k + 1 when sampled, greedy argmax otherwise).
    pub fn holdSampled(d: *Mtp, ctx: *Ctx, pending: u32, pos0: u64, depth: usize, lanes_: ?*const lanes.shape.Shape, sampling: ?lanes.Sampling) !void {
        _ = pending; // the newest absorbed row's follow token, whose candidates ctx.cands are
        if (ctx.end != pos0) return error.DraftContextBehind;
        if (lanes_) |l| return d.holdTreeSampled(ctx, pos0, l, sampling);
        const e = d.e;
        const g = e.g;
        const device_candidate = !d.trees;
        if (!device_candidate) ctx.held[0] = ctx.cands[0];
        ctx.held_n = 1;
        if (depth > 1 and !device_candidate) try d.putIdsReady(ctx.held[0..1]);
        // Feed the device candidate buffer directly to embedding, without a host round-trip or ID upload.
        var ids = if (device_candidate) ctx.held_dev.ptr else d.ids;
        var k: usize = 1;
        while (k < depth and k < ctx.held.len) : (k += 1) {
            // row k: (the previous row's normed output, the draft before) at slot pos0 + k - 1
            const src = if (k == 1) ctx.last.ptr else d.x;
            try d.scratch.copyFrom(d.hin - d.scratch.ptr, src, g.hidden * 2, e.stream.handle);
            try d.layerModeWithIds(ctx, ids, 1, pos0 + k - 1, .ring, true);
            const spec = [_]mtp_draw.Spec{.{ .sampling = sampling, .position = pos0 + k + 1 }};
            try d.drawSpecs(0, &spec, false);
            if (device_candidate) {
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(ctx.held_dev.ptr + k * 4, d.tok, 4, e.stream.handle), "cuMemcpyDtoDAsync");
            } else {
                try d.scratch.copyFrom(@intCast(d.chain_tok - d.scratch.ptr + (k - 1) * 4), d.tok, 4, e.stream.handle);
            }
            ctx.held_n = k + 1;
            ids = d.tok;
        }
        if (device_candidate) try d.fetchHeld(ctx) else try d.fetchChain(ctx.held[1..ctx.held_n]);
    }

    /// Run each tree depth as one pass; lane i takes rank r from its parent's candidates.
    pub fn holdTree(d: *Mtp, ctx: *Ctx, pos0: u64, l: *const lanes.shape.Shape) !void {
        return @import("cuda_mtp_tree.zig").holdTreeSampled(d, ctx, pos0, l, null, anc_at);
    }

    /// `holdTree` with target-coupled sampling at each internal depth.
    pub fn holdTreeSampled(d: *Mtp, ctx: *Ctx, pos0: u64, l: *const lanes.shape.Shape, sampling: ?lanes.Sampling) !void {
        return @import("cuda_mtp_tree.zig").holdTreeSampled(d, ctx, pos0, l, sampling, anc_at);
    }

    // the Drafter contract

    pub const drafter_vtable: dr.Drafter.VTable = blk: {
        const adapter = @import("cuda_mtp_contract.zig").Adapter(Mtp);
        break :blk .{ .taps = adapter.taps, .facts = adapter.facts, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn };
    };

    pub fn drafter(d: *Mtp) dr.Drafter {
        return @import("cuda_prefill_trace.zig").wrapDrafter(d, .{ .kind = "MTP", .ptr = d, .vtable = &drafter_vtable });
    }

    fn openFn(p: *anyopaque, s: *Stream) anyerror!void {
        const d: *Mtp = @ptrCast(@alignCast(p));
        if (d.streams.contains(s)) return error.StreamOpen;
        if (d.ring_of) |r| if (r.area(r.ctx, s)) |base| { // the lane's run holds them, zeroed: a context borrowing it
            const ctx = try d.gpa.create(Ctx);
            errdefer d.gpa.destroy(ctx);
            const last = base + std.mem.alignForward(usize, d.ringBytes(), 256);
            const step = base + d.stepOffset();
            ctx.* = .{ .ring = .{ .d = d.e.d, .ptr = base, .len = d.ringBytes(), .borrowed = true }, .last = .{ .d = d.e.d, .ptr = last, .len = d.e.g.hidden * 2, .borrowed = true }, .step = .{ .d = d.e.d, .ptr = step, .len = d.e.g.hidden * 2, .borrowed = true }, .held_dev = .{ .d = d.e.d, .ptr = step + d.e.g.hidden * 2, .len = max_lanes * 4, .borrowed = true } };
            return d.streams.put(d.gpa, s, ctx);
        };
        const ctx = d.pool.pop() orelse try d.create();
        errdefer d.pool.append(d.gpa, ctx) catch d.drop(ctx);
        d.resetCtx(ctx);
        try d.streams.put(d.gpa, s, ctx);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d: *Mtp = @ptrCast(@alignCast(p));
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("mtp absorb", d.io, t0);
        // Batch shared rounds need proposals; route single-stream and KV-only prompt absorbs through serial.
        if (!d.trees and d.batch and items.len > 1 and
            @import("cuda_mtp_absorb.zig").batchProposalAllowed(items)) return batched.absorb(d, items);
        for (items) |it| try @import("cuda_mtp_absorb.zig").absorbOne(d, it);
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d: *Mtp = @ptrCast(@alignCast(p));
        const t0 = std.Io.Clock.awake.now(d.io);
        defer timing("mtp hold", d.io, t0);
        if (!d.trees and d.batch and items.len > 1) return batched.hold(d, items);
        for (items) |it| {
            const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
            try d.holdSampled(ctx, it.pending, it.position - 1, it.depth, it.lanes, it.sampling orelse it.stream.sampling);
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d: *Mtp = @ptrCast(@alignCast(p));
        for (streams, out) |s, o| {
            const ctx = d.streams.get(s) orelse return error.UnknownStream;
            if (o.len > ctx.held_n) return error.TooFewDrafts;
            @memcpy(o, ctx.held[0..o.len]);
        }
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const d: *Mtp = @ptrCast(@alignCast(p));
        if (d.streams.fetchRemove(s)) |kv| {
            if (kv.value.ring.borrowed) return d.drop(kv.value); // its lane's run goes back with the lane
            d.pool.append(d.gpa, kv.value) catch d.drop(kv.value);
        }
    }
};

test "TF_MTP_TREES: the platform parses unset, valid and malformed text" {
    try std.testing.expectEqual(@as(?u64, null), try platform.environment.parse("TF_MTP_TREES", null));
    try std.testing.expectEqual(@as(?u64, 1), try platform.environment.parse("TF_MTP_TREES", "1"));
    try std.testing.expectError(error.InvalidEnvironment, platform.environment.parse("TF_MTP_TREES", "on"));
}
