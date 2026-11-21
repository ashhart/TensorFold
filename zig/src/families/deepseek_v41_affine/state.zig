//! A stream's caches (by position or group: dropping draft rows moves `len`) and a forward's scratch.
const std = @import("std");
const confidence = @import("confidence.zig");
const mtl = @import("metal");
const c = @import("config.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Window ring rows a layer: the 128-key window behind the longest call's rows.
pub const ring = 16384;
pub const original_prefill_rows = 4096 - c.window;
/// Rows a forward takes at most (a prompt chunk).
pub const max_rows = ring - c.window;
/// Compressed picks a row: the indexer's top-k.
pub const picks = c.index_topk;
/// Rows that take the split attention (decode, windows, drafts), and its 16-key chunks a row.
pub const split_rows = 16;
pub const split_chunks = (c.window + picks + 15) / 16;
/// Routed slots a threadgroup of the prompt chunk's expert sort takes (v41_moe.metal's SORT_BLOCK).
pub const sort_block = 256;
/// Scores a prompt call's indexer holds at once (bf16), in bytes: rows are taken in slices of this many keys' worth.
pub const score_budget = 768 << 20;

pub const Storage = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Storage, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), opts);
        errdefer b.deinit();
        try self.buffers.append(self.gpa, b);
        return b;
    }

    pub fn deinit(self: *Storage) void {
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.gpa);
    }
};

/// A KV source's compressed latents (bf16 [groups, 512], RoPE'd and fp4-rounded) and index keys (bf16 [groups, 128]).
pub const Source = struct { latents: mtl.Buffer, keys: mtl.Buffer, groups: usize };

pub const Cache = struct {
    gpa: std.mem.Allocator,
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    /// Each layer's window keys (bf16 [ring, 512] by position).
    windows: [c.layers]mtl.Buffer,
    sources: [c.kv_sources.len]Source,
    /// The ratio-2 sources' projections (f32 [ring, 512] by position), pooled once a pair completes.
    proj_kv: [c.kv_sources.len]?mtl.Buffer,
    proj_score: [c.kv_sources.len]?mtl.Buffer,
    /// Compressed token ids by position, for the engram hashes.
    engram_ids: std.ArrayList(i32) = .empty,
    logits: mtl.Buffer,
    /// With the DSpark head: each stage's window keys of the target's positions (bf16 [ring, 512] by position).
    stage_rings: ?[c.stages]mtl.Buffer = null,
    /// Where the last call started, its rows and its first scratch row (a window's keep takes back to a prefix).
    last_start: usize = 0,
    last_rows: usize = 0,
    last_r0: usize = 0,
    /// The drafts held for the next window (the head's greedy chain).
    held: [c.draft_block]u32 = @splat(0),
    held_n: usize = 0,
    conf_hist: [c.draft_block + 1]u64 = @splat(0),
    attention_values_checked: usize = 0,
    attention_eight_checked: usize = 0,
    attention_stream_checked: usize = 0,
    hc_overlap_checked: usize = 0,
    router_variant_checked: usize = 0,
    hc_wide_rounds: [1 + c.draft_block]u64 = @splat(0),
    condition_audited: u64 = 0,
    short_index_layers: u64 = 0,
    wo_audited: u64 = 0,
    short_index_checked: u64 = 0,
    short_index_replays: usize = 0,
    projection_audited: u64 = 0,
    conf_offered: u64 = 0,
    conf_scored: u64 = 0,
    conf_examined: u64 = 0,
    carry_eligible: u64 = 0,
    carry_used: u64 = 0,
    carry_verified: u64 = 0,
    carry_proposed: u64 = 0,
    carry_accepted: u64 = 0,
    carry_first_hits: u64 = 0,
    carry_pending: bool = false,
    carry_ages: [c.draft_block]u64 = @splat(0),
    conf_digest: std.crypto.hash.sha2.Sha256 = std.crypto.hash.sha2.Sha256.init(.{}),

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize) !Cache {
        return initWith(gpa, device, capacity, false);
    }

    /// `drafts`: the DSpark head's rings.
    pub fn initWith(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize, drafts: bool) !Cache {
        var out: Cache = .{ .gpa = gpa, .memory = .{ .gpa = gpa, .device = device }, .capacity = capacity, .windows = undefined, .sources = undefined, .proj_kv = @splat(null), .proj_score = @splat(null), .logits = undefined };
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        for (&out.windows) |*w| w.* = try out.memory.alloc(ring * c.head_dim * 2);
        for (&out.sources, c.kv_sources, 0..) |*s, layer, i| {
            const groups = capacity / c.ratios[layer] + 1;
            s.* = .{ .latents = try out.memory.alloc(groups * c.head_dim * 2), .keys = try out.memory.alloc(groups * c.index_dim * 2), .groups = groups };
            if (c.ratios[layer] > 1) {
                out.proj_kv[i] = try out.memory.alloc(ring * c.head_dim * 4);
                out.proj_score[i] = try out.memory.alloc(ring * c.head_dim * 4);
            }
        }
        try out.engram_ids.ensureTotalCapacity(gpa, capacity);
        if (drafts) {
            var rings: [c.stages]mtl.Buffer = undefined;
            for (&rings) |*r| r.* = try out.memory.alloc(ring * c.head_dim * 2);
            out.stage_rings = rings;
        }
        return out;
    }

    pub fn deinit(self: *Cache) void {
        self.engram_ids.deinit(self.gpa);
        self.memory.deinit();
    }
};

/// A prompt chunk whose engram rows were gathered ahead: its cache and rows, its tokens' hash, the set holding them.
pub const Prepared = struct { cache: *Cache, base: usize, n: usize, hash: u64, set: u8 };

pub const Scratch = struct {
    memory: Storage,
    rows: usize,
    /// The bytes of `scores`: score_budget, or 0 in the draft head's scratch (it runs no indexer).
    score_bytes: usize,
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    x: mtl.Buffer, // the streams, bf16 [rows, 4, 5120]
    x2: mtl.Buffer, // the other streams buffer: a fused expand writes it, then the two swap
    normed: mtl.Buffer,
    branch: mtl.Buffer,
    pre: [2]mtl.Buffer, // f32 [rows, 4]: the carried collapse, and this boundary's
    post: mtl.Buffer,
    comb: mtl.Buffer,
    hc_part: mtl.Buffer, // f32 [rows, 20 slices, 25]: a boundary's partial mix sums
    /// float2 [2 variants (plain, compressed), trig_rows, 32]: the call's RoPE (cos, sin) at its rows' positions
    trig: mtl.Buffer,
    /// The rows the table holds this call (forward.runSegs); 0: none, the kernels compute their own.
    trig_rows: usize = 0,
    qr: mtl.Buffer,
    q: mtl.Buffer,
    kv: mtl.Buffer,
    o: mtl.Buffer,
    oa: mtl.Buffer,
    cproj: mtl.Buffer, // f32 [rows, 512]
    cscore: mtl.Buffer,
    latent: mtl.Buffer, // bf16 [rows, 512]
    iq: mtl.Buffer,
    iw: mtl.Buffer,
    ik: mtl.Buffer,
    scores: mtl.Buffer, // bf16, a slice of rows by the keys they score
    /// Split attention's chunk partials for up to `split_rows` rows: outputs f32 [rows, 64, chunks, 512], (max, sum).
    attn_po: mtl.Buffer,
    attn_ml: mtl.Buffer,
    attention_ref: mtl.Buffer,
    attention_differs: mtl.Buffer,
    blocks: mtl.Buffer, // f32: the slice's block maxima, a row's 8-key blocks together
    keep: mtl.Buffer, // i32 [rows, 2048]: the candidate blocks, in order
    picks: mtl.Buffer, // i32 [rows, 512]
    router: mtl.Buffer,
    route_pick: mtl.Buffer,
    route_mates: mtl.Buffer,
    route_trace: mtl.Buffer,
    /// The shared expert's own gate, up and activation rows (bf16 [rows, 2304]) when it runs beside the routed ones.
    sh_gate: mtl.Buffer,
    sh_up: mtl.Buffer,
    sh_act: mtl.Buffer,
    /// u32 [rows * 8]: slot i's wo_a group (i % 8), the decode path's expert ids for the stacked wo_a.
    wo_ids: mtl.Buffer,
    wo_reference: mtl.Buffer,
    wo_mismatch: mtl.Buffer,
    route_weight: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    experts: mtl.Buffer,
    shared: mtl.Buffer,
    /// A chunk's slots sorted by expert: offsets, order, weights, each block's counts, the tensor units' group sums.
    offs: mtl.Buffer,
    order: mtl.Buffer,
    wsort: mtl.Buffer,
    counts: mtl.Buffer,
    sums: mtl.Buffer,
    /// bf16 [rows, 24 x 256] a layer, filled on the host: two sets, so a chunk's gather overlaps the chunk before.
    engram_rows: [2][c.engram_layers.len]mtl.Buffer,
    /// The set this call's rows read.
    engram_set: u8,
    /// A chunk gathered ahead into the other set (forward.runChunk), or null.
    prepared: ?Prepared,
    /// Groups this call completes, a buffer a source (host-written as it encodes): positions and cache rows.
    gpos: [c.kv_sources.len]mtl.Buffer,
    gidx: [c.kv_sources.len]mtl.Buffer,
    engram_kv: mtl.Buffer,
    logits: mtl.Buffer,
    greedy_parts: mtl.Buffer,
    router_ref_logits: mtl.Buffer,
    router_ref_ids: mtl.Buffer,
    router_ref_weights: mtl.Buffer,
    router_variant_error: mtl.Buffer,
    hc_overlap_ref_norm: mtl.Buffer,
    hc_overlap_ref_pre: mtl.Buffer,
    hc_overlap_ref_post: mtl.Buffer,
    hc_overlap_ref_comb: mtl.Buffer,
    hc_overlap_error: mtl.Buffer,
    greedy_tokens: mtl.Buffer,
    /// DSpark: target layers' stream means and conditioning rows; a draft call's ids, own-key picks, argmax parts.
    mh: mtl.Buffer,
    /// Each row's image flag (i32: an image span's row) and code (forward.Vis).
    img: mtl.Buffer,
    img_code: mtl.Buffer,
    mainx: mtl.Buffer,
    condition_kv: [c.stages]mtl.Buffer,
    condition_ref: mtl.Buffer,
    condition_error: mtl.Buffer,
    short_index_status: mtl.Buffer,
    query_ref: mtl.Buffer,
    index_query_ref: mtl.Buffer,
    index_weight_ref: mtl.Buffer,
    projection_error: mtl.Buffer,
    draft_ids: mtl.Buffer,
    branch_ids: mtl.Buffer,
    draft_picks: mtl.Buffer,
    markov_part: mtl.Buffer,
    markov_runners: mtl.Buffer,
    markov_ranks: mtl.Buffer,
    transition: ?TransitionCache,
    markov_reference: mtl.Buffer,
    markov_mismatch: mtl.Buffer,
    conf_hidden: mtl.Buffer,
    conf_state: mtl.Buffer,
    conf_dispatch: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, rows: usize) !Scratch {
        return initScored(gpa, device, rows, score_budget);
    }

    /// The draft head's scratch: its stage attention reads fixed picks, so no index scores.
    const TransitionCache = struct {
        tag: mtl.Buffer,
        bias: mtl.Buffer,
        counts: mtl.Buffer,
        log_slots: u32,
        audit: bool,
    };

    pub fn initDraft(gpa: std.mem.Allocator, device: mtl.Device, rows: usize) !Scratch {
        var scratch = try initScored(gpa, device, rows, 0);
        errdefer scratch.deinit();
        {
            const log_slots: u5 = 8;
            if (log_slots < 4 or log_slots > 10) return error.InvalidDeepseekTransitionCacheSize;
            const slots = @as(usize, 1) << log_slots;
            comptime std.debug.assert(c.vocab % 256 == 0);
            const tags = slots * (c.vocab / 256);
            const tag = try scratch.memory.alloc(tags * @sizeOf(u32));
            @memset(tag.slice(u32, tags), std.math.maxInt(u32));
            scratch.transition = .{
                .tag = tag,
                .bias = try scratch.memory.alloc(slots * c.vocab * @sizeOf(f32)),
                .counts = try scratch.memory.alloc(3 * @sizeOf(u32)),
                .log_slots = log_slots,
                .audit = false,
            };
            std.log.info("deepseek_v41 bench: exact Markov transition cache slots={d} audit={}", .{ slots, scratch.transition.?.audit });
        }
        return scratch;
    }

    fn initScored(gpa: std.mem.Allocator, device: mtl.Device, rows: usize, score_bytes: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        out.score_bytes = score_bytes;
        const m = &out.memory;
        const slots = rows * c.topk;
        out.ids = try m.alloc(rows * 4);
        out.pos = try m.alloc(rows * 4);
        out.x = try m.alloc(rows * c.hc * c.hidden * 2);
        out.x2 = try m.alloc(rows * c.hc * c.hidden * 2);
        out.normed = try m.alloc(rows * c.hidden * 2);
        out.branch = try m.alloc(rows * c.hidden * 2);
        for (&out.pre) |*p| p.* = try m.alloc(rows * c.hc * 4);
        out.post = try m.alloc(rows * c.hc * 4);
        out.comb = try m.alloc(rows * c.hc * c.hc * 4);
        out.hc_part = try m.alloc(rows * (c.hc * c.hidden / 1024) * (c.hc_mix + 1) * 4);
        out.trig = try m.alloc(2 * rows * 32 * 8);
        out.trig_rows = 0;
        out.qr = try m.alloc(rows * c.q_rank * 2);
        out.q = try m.alloc(rows * c.heads * c.head_dim * 2);
        out.kv = try m.alloc(rows * c.head_dim * 2);
        out.o = try m.alloc(rows * c.heads * c.head_dim * 2);
        out.oa = try m.alloc(rows * c.o_groups * c.o_rank * 2);
        out.cproj = try m.alloc(rows * c.head_dim * 4);
        out.cscore = try m.alloc(rows * c.head_dim * 4);
        out.latent = try m.alloc(rows * c.head_dim * 2);
        out.iq = try m.alloc(rows * c.index_heads * c.index_dim * 2);
        out.iw = try m.alloc(rows * c.index_heads * 2);
        out.ik = try m.alloc(rows * c.index_dim * 2);
        out.scores = try m.alloc(score_bytes);
        out.attn_po = try m.alloc(@min(rows, split_rows) * @as(usize, c.heads) * split_chunks * c.head_dim * 4);
        out.attn_ml = try m.alloc(@min(rows, split_rows) * @as(usize, c.heads) * split_chunks * 8);
        out.attention_ref = try m.alloc(@as(usize, @min(rows, 1 + c.draft_block)) * c.heads * c.head_dim * 2);
        out.attention_differs = try m.alloc(4);
        out.blocks = try m.alloc(if (score_bytes == 0) 0 else score_bytes / 4 + rows * 4); // rows by keys / 8 + 1
        out.keep = try m.alloc(rows * c.candidate_blocks * 4);
        out.picks = try m.alloc(rows * picks * 4);
        out.router = try m.alloc(rows * c.experts * 4);
        out.route_pick = try m.alloc(slots * 4);
        out.route_mates = try m.alloc((1 + c.draft_block) * c.topk * 4);
        out.route_trace = try m.alloc(c.layers * (1 + c.draft_block) * c.topk * 4);
        out.sh_gate = try m.alloc(rows * c.moe_inter * 2);
        out.sh_up = try m.alloc(rows * c.moe_inter * 2);
        out.sh_act = try m.alloc(rows * c.moe_inter * 2);
        out.wo_ids = try m.alloc(rows * c.o_groups * 4);
        const wo_audit_rows: usize = if (false) 1 + c.draft_block else 0;
        out.wo_reference = try m.alloc(@max(4, wo_audit_rows * c.o_groups * c.o_rank * 2));
        out.wo_mismatch = try m.alloc(4);
        for (out.wo_ids.slice(u32, rows * c.o_groups), 0..) |*v, i| v.* = @intCast(i % c.o_groups);
        out.route_weight = try m.alloc(slots * 4);
        out.gate = try m.alloc(slots * c.moe_inter * 2);
        out.up = try m.alloc(slots * c.moe_inter * 2);
        out.act = try m.alloc(slots * c.moe_inter * 2);
        out.experts = try m.alloc(slots * c.hidden * 2);
        out.shared = try m.alloc(rows * c.hidden * 2);
        out.offs = try m.alloc((c.experts + 1) * 4);
        out.order = try m.alloc(slots * 4);
        out.wsort = try m.alloc(slots * 4);
        out.counts = try m.alloc((slots + sort_block - 1) / sort_block * c.experts * 4);
        out.sums = try m.alloc(@max(rows * (c.heads * c.head_dim) / 64, slots * c.hidden / 64) * 4);
        for (&out.engram_rows) |*set| for (set) |*b| {
            b.* = try m.alloc(rows * c.engram_cols * c.engram_dim * 2);
        };
        out.engram_set = 0;
        out.prepared = null;
        for (&out.gpos, &out.gidx) |*p, *g| {
            p.* = try m.alloc(rows * 4);
            g.* = try m.alloc(rows * 4);
        }
        out.engram_kv = try m.alloc(rows * (c.hc + 1) * c.hidden * 2);
        out.logits = try m.alloc(rows * c.vocab * 2);
        out.mh = try m.alloc(rows * c.draft_targets.len * c.hidden * 2);
        out.img = try m.alloc(rows * 4);
        out.img_code = try m.alloc(rows * 4);
        out.mainx = try m.alloc(rows * c.hidden * 2);
        for (&out.condition_kv) |*b| b.* = try m.alloc(@as(usize, @min(rows, 1 + c.draft_block)) * c.head_dim * 2);
        out.condition_ref = try m.alloc(c.stages * (1 + c.draft_block) * c.head_dim * 2);
        out.condition_error = try m.alloc(4);
        out.short_index_status = try m.alloc(4);
        const audit_rows: usize = if (false) 1 + c.draft_block else 0;
        out.query_ref = try m.alloc(@max(4, audit_rows * c.heads * c.head_dim * 2));
        out.index_query_ref = try m.alloc(@max(4, audit_rows * c.index_heads * c.index_dim * 2));
        out.index_weight_ref = try m.alloc(@max(4, audit_rows * c.index_heads * 2));
        out.projection_error = try m.alloc(4);
        out.greedy_parts = try m.alloc(rows * @import("gpu_greedy.zig").bytes_per_row);
        out.greedy_tokens = try m.alloc(rows * 4);
        const route_audit = false;
        out.router_ref_logits = try m.alloc(if (route_audit) (1 + c.draft_block) * c.experts * 4 else 4);
        out.router_ref_ids = try m.alloc(if (route_audit) (1 + c.draft_block) * c.topk * 4 else 4);
        out.router_ref_weights = try m.alloc(if (route_audit) (1 + c.draft_block) * c.topk * 4 else 4);
        out.router_variant_error = try m.alloc(4);
        const hc_audit = false;
        out.hc_overlap_ref_norm = try m.alloc(if (hc_audit) (1 + c.draft_block) * c.hidden * 2 else 4);
        out.hc_overlap_ref_pre = try m.alloc(if (hc_audit) (1 + c.draft_block) * c.hc * 4 else 4);
        out.hc_overlap_ref_post = try m.alloc(if (hc_audit) (1 + c.draft_block) * c.hc * 4 else 4);
        out.hc_overlap_ref_comb = try m.alloc(if (hc_audit) (1 + c.draft_block) * c.hc * c.hc * 4 else 4);
        out.hc_overlap_error = try m.alloc(4);
        out.draft_ids = try m.alloc((c.draft_block + 1) * 4);
        out.branch_ids = try m.alloc((c.draft_block + 1) * 4);
        out.draft_picks = try m.alloc(c.draft_block * c.draft_block * 4);
        for (out.draft_picks.slice(i32, c.draft_block * c.draft_block), 0..) |*v, i| v.* = @intCast(i % c.draft_block);
        out.markov_part = try m.alloc((c.vocab + 255) / 256 * 8);
        out.markov_runners = try m.alloc((c.vocab + 255) / 256 * 8);
        out.markov_ranks = try m.alloc(c.draft_block * 16);
        out.transition = null;
        out.markov_reference = try m.alloc((c.vocab + 255) / 256 * 8);
        out.markov_mismatch = try m.alloc(4);
        out.conf_hidden = try m.alloc(c.draft_block * 4);
        out.conf_state = try m.alloc(@sizeOf(confidence.State));
        out.conf_dispatch = try m.alloc(16);
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.memory.deinit();
    }
};
