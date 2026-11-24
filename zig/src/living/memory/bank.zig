//! Folded Memory Weights v6 Bank ("The Living Brain"): family-neutral weight compilation for both
//! hybrid (Mamba-2/GDN + Attention) and pure-attention (`num_rec == 0`) models, Q4/Q8 KV, spectral
//! low-rank SVD, temporal belief revision with undoable supersession history, 2-level ClusterIndex,
//! crash-safe CRC32 append-only delta journal (`living-weights.delta`), sleep-time spectral skill
//! consolidation, and zero-plaintext persistence (`living-weights.safetensors`).
const std = @import("std");
const quant = @import("quant.zig");
const whitener = @import("whitener.zig");
const gate = @import("gate.zig");
const store = @import("store.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const shared_sys_prefix =
    "<|im_start|>system\n" ++
    "Facts and reference documents previously shared by the user. " ++
    "Answer accurately from these entries when relevant (keeping separate entries distinct). " ++
    "If a detail is not mentioned in these entries, say you do not know.\n";

pub const shared_sys_suffix = "<|im_end|>\n";

pub const Family = enum {
    qwen3_5,
    qwen27,
    nemotron,
    llama,
    cuda_hybrid,
    cuda_attn,

    pub fn name(self: Family) []const u8 {
        return @tagName(self);
    }
};

/// Geometry & optimal compression settings of any hybrid / attention model for Folded Memory Weights.
pub const ArchSpec = struct {
    family: Family,
    hidden: usize,
    num_layers: usize,
    layer_lex: usize, // typically 1
    layer_sem: usize, // typically (2 * num_layers) / 3
    num_attn: usize, // attention KV layers (+1 if Nemotron MTP head KV included)
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize, // 64 for Qwen3.5/3.8, 0 for Nemotron NoPE
    rope_theta: f64, // 10_000_000.0 for Qwen, 10_000.0 for Nemotron
    num_rec: usize, // GDN or Mamba-2 recurrent layers
    rec_heads: usize, // 16 (2B), 48 (27B), 64 (Nemotron)
    rec_rows: usize, // 128 (Qwen dv), 64 (Nemotron head_dim)
    rec_cols: usize, // 128 (Qwen dk / Nemotron state)
    conv_elems: usize, // elements of bf16 per recurrent layer conv buffer
    /// Empirical optimum for document chunks: 4 for 2B, 0 for 27B/30B (KV-only!).
    svd_rank: usize = 4,
    /// Empirical optimum for short facts & standing policies (steers Mamba-2/GDN layers on 15-token rules!).
    svd_rank_short: usize = 4,
    /// KV quantization bit-width: 4 (symmetric group-32 Q4, 0.5625 B/elem) or 8 (per-row Q8).
    kv_bits: u8 = 4,
    /// Keep per-unit conv buffers (false omits conv on multi-chunk docs; short/standing units may keep if true).
    keep_conv: bool = false,

    pub fn recStateElems(self: ArchSpec) usize {
        return self.rec_heads * self.rec_rows * self.rec_cols;
    }

    pub fn kvLayerElems(self: ArchSpec, seq_len: usize) usize {
        return self.kv_heads * seq_len * self.head_dim;
    }
};

/// Backend-owned opaque snapshot handle returned by `Backend.buildSnap`.
pub const SnapHandle = struct {
    saved: *anyopaque,
    at: u32,
};

/// Uncompressed CPU-side hybrid state exchanged between `Bank` and a family's Metal adapter.
/// Layout conventions (exact match to TensorFold's snapshot layouts):
/// - `kv_k[a]`, `kv_v[a]`: `[kv_heads, len, head_dim]` BF16 (`u16`), post-RoPE at positions `0..len`
/// - `rec_state[r]`: `[rec_heads, rec_rows, rec_cols]` F32
/// - `rec_conv[r]`: `[conv_elems]` BF16 (`u16`)
pub const RawState = struct {
    len: u32,
    kv_k: [][]u16,
    kv_v: [][]u16,
    rec_state: [][]f32,
    rec_conv: [][]u16,

    pub fn init(gpa: Allocator, arch: ArchSpec, len: u32) !RawState {
        const kv_k = try gpa.alloc([]u16, arch.num_attn);
        errdefer gpa.free(kv_k);
        const kv_v = try gpa.alloc([]u16, arch.num_attn);
        errdefer gpa.free(kv_v);
        var ak: usize = 0;
        errdefer {
            for (0..ak) |i| {
                gpa.free(kv_k[i]);
                gpa.free(kv_v[i]);
            }
        }
        const kv_elems = arch.kvLayerElems(len);
        for (0..arch.num_attn) |a| {
            kv_k[a] = try gpa.alloc(u16, kv_elems);
            errdefer gpa.free(kv_k[a]);
            kv_v[a] = try gpa.alloc(u16, kv_elems);
            @memset(kv_k[a], 0);
            @memset(kv_v[a], 0);
            ak += 1;
        }

        const rec_state = try gpa.alloc([]f32, arch.num_rec);
        errdefer gpa.free(rec_state);
        const rec_conv = try gpa.alloc([]u16, arch.num_rec);
        errdefer gpa.free(rec_conv);
        var ar: usize = 0;
        errdefer {
            for (0..ar) |i| {
                gpa.free(rec_state[i]);
                gpa.free(rec_conv[i]);
            }
        }
        const st_elems = arch.recStateElems();
        for (0..arch.num_rec) |r| {
            rec_state[r] = try gpa.alloc(f32, st_elems);
            errdefer gpa.free(rec_state[r]);
            rec_conv[r] = try gpa.alloc(u16, arch.conv_elems);
            @memset(rec_state[r], 0);
            @memset(rec_conv[r], 0);
            ar += 1;
        }
        return .{ .len = len, .kv_k = kv_k, .kv_v = kv_v, .rec_state = rec_state, .rec_conv = rec_conv };
    }

    pub fn deinit(self: *RawState, gpa: Allocator) void {
        for (self.kv_k) |s| gpa.free(s);
        for (self.kv_v) |s| gpa.free(s);
        gpa.free(self.kv_k);
        gpa.free(self.kv_v);
        for (self.rec_state) |s| gpa.free(s);
        for (self.rec_conv) |s| gpa.free(s);
        gpa.free(self.rec_state);
        gpa.free(self.rec_conv);
        self.* = undefined;
    }
};

/// The 4-function family adapter vtable. Each Metal backend supplies these using its existing
/// forward pass and snapshot save/restore!
pub const Backend = struct {
    ctx: *anyopaque,
    arch: ArchSpec,
    /// Partial forward on `ids` through layer `layer_sem`, writing `[ids.len * hidden]` f32 into `out_lex` & `out_sem`.
    extractHidden: *const fn (ctx: *anyopaque, ids: []const u32, layer_lex: usize, layer_sem: usize, out_lex: []f32, out_sem: []f32) anyerror!void,
    /// Full forward on `ids` starting from `base` (null = empty zero state), writing ONLY the new `ids.len` KV rows
    /// (at post-RoPE positions `base.len .. base.len + ids.len`) and the final recurrent states into `out` (`out.len == ids.len`).
    compileState: *const fn (ctx: *anyopaque, base: ?*const RawState, ids: []const u32, out: *RawState) anyerror!void,
    /// Upload composed `RawState` (`prefix + active units`) into a scratch cache, run `suffix_ids`, and return a native Snap.
    buildSnap: *const fn (ctx: *anyopaque, composed: *const RawState, suffix_ids: []const u32) anyerror!SnapHandle,
    /// Destroy a Snap previously created by `buildSnap`.
    dropSnap: *const fn (ctx: *anyopaque, snap: *anyopaque) void,
};

pub const KvLayerData = union(enum) {
    q8: struct {
        kq: []i8, // [kv_heads * n_tokens * head_dim] (Pre-RoPE K_raw)
        ks: []f32, // [kv_heads * n_tokens]
        vq: []i8, // [kv_heads * n_tokens * head_dim]
        vs: []f32, // [kv_heads * n_tokens]
    },
    q4: struct {
        kq: []u8, // [kv_heads * n_tokens * (head_dim / 2)] packed nibbles
        ks: []u16, // [kv_heads * n_tokens * (head_dim / 32)] BF16 group-32 scales
        vq: []u8,
        vs: []u16,
    },

    pub fn bits(self: *const KvLayerData) u8 {
        return switch (self.*) {
            .q8 => 8,
            .q4 => 4,
        };
    }

    pub fn deinit(self: *KvLayerData, gpa: Allocator) void {
        switch (self.*) {
            .q8 => |d| {
                gpa.free(d.kq);
                gpa.free(d.ks);
                gpa.free(d.vq);
                gpa.free(d.vs);
            },
            .q4 => |d| {
                gpa.free(d.kq);
                gpa.free(d.ks);
                gpa.free(d.vq);
                gpa.free(d.vs);
            },
        }
    }
};

pub const RecLayerData = struct {
    rank: u16,
    uq: []i8, // [rec_heads * rec_rows * rank]
    us: []f32, // [rec_heads * rec_rows]
    vq: []i8, // [rec_heads * rank * rec_cols]
    vs: []f32, // [rec_heads * rank]
    conv: []u16, // [conv_elems] bf16 OR empty when `keep_conv == false`

    pub fn deinit(self: *RecLayerData, gpa: Allocator) void {
        gpa.free(self.uq);
        gpa.free(self.us);
        gpa.free(self.vq);
        gpa.free(self.vs);
        gpa.free(self.conv);
    }
};

pub const Unit = struct {
    unit_id: u32,
    parent_id: u32,
    n_tokens: u32,
    gate: gate.UnitGate,
    kv: []KvLayerData, // [num_attn]
    rec: []RecLayerData, // [num_rec] OR empty when `svd_rank == 0`!

    pub fn deinit(self: *Unit, gpa: Allocator) void {
        self.gate.deinit(gpa);
        for (self.kv) |*k| k.deinit(gpa);
        gpa.free(self.kv);
        for (self.rec) |*r| r.deinit(gpa);
        gpa.free(self.rec);
        self.* = undefined;
    }
};

pub const CalPrompt = struct {
    lex: whitener.Transformed,
    sem: whitener.Transformed,

    pub fn deinit(self: *CalPrompt, gpa: Allocator) void {
        self.lex.deinit(gpa);
        self.sem.deinit(gpa);
    }
};

pub const TeachChunkInput = struct {
    parent_id: u32,
    body_ids: []const u32, // framed entry text tokens
    gate_ids: []const u32, // unique section body tokens
    title_ids: []const u32 = &.{}, // optional repeated doc title tokens (0.72x weight)
    cue_ids: []const u32 = &.{}, // optional small-fact concept cue tokens
    is_small_fact: bool = false,
    is_standing: bool = false, // always-on standing policy/behavioral rule (bypasses episodic gate)
    is_digest: bool = false, // consolidated cross-section entity/spec digest for a multi-chunk parent
};

pub const ConsolidateReport = struct {
    units: u32 = 0,
    pruned_superseded: u32 = 0,
    consolidated_policies: u32 = 0,
    clusters: u32 = 0,
    compacted_deltas: u32 = 0,
    bytes: u64 = 0,
};

pub const Bank = struct {
    gpa: Allocator,
    io: Io,
    dir: []u8,
    backend: Backend,
    calibrated: bool = false,
    w_lex: ?whitener.LayerWhitener = null,
    w_sem: ?whitener.LayerWhitener = null,
    cal_prompts: std.ArrayList(CalPrompt) = .empty,
    prefix_ids: []u32 = &.{},
    suffix_ids: []u32 = &.{},
    prefix_state: ?RawState = null,
    units: std.ArrayList(Unit) = .empty,
    bridges: []gate.BridgeEdge = &.{},
    cluster_index: gate.ClusterIndex = .{},
    /// Sleep-time spectral-consolidated recurrent steering delta across active standing policies.
    skill_rec: []RecLayerData = &.{},
    skill_policy_count: u32 = 0,
    next_unit_id: u32 = 1,
    next_epoch: u64 = 1,
    base_saved: bool = false,
    pending_deltas: u32 = 0,
    unflushed_from: usize = 0,
    last_bytes: u64 = 0,
    route_cfg: gate.RouteConfig = .{},

    pub fn init(gpa: Allocator, io: Io, dir: []const u8, backend: Backend) !*Bank {
        const b = try gpa.create(Bank);
        errdefer gpa.destroy(b);
        b.* = .{
            .gpa = gpa,
            .io = io,
            .dir = try gpa.dupe(u8, dir),
            .backend = backend,
        };
        return b;
    }

    pub fn deinit(self: *Bank) void {
        self.clearMemory();
        self.gpa.free(self.dir);
        self.gpa.destroy(self);
    }

    fn freeSkillRec(self: *Bank) void {
        for (self.skill_rec) |*r| r.deinit(self.gpa);
        if (self.skill_rec.len > 0) self.gpa.free(self.skill_rec);
        self.skill_rec = &.{};
        self.skill_policy_count = 0;
    }

    fn clearMemory(self: *Bank) void {
        self.freeSkillRec();
        self.cluster_index.deinit(self.gpa);
        self.gpa.free(self.bridges);
        self.bridges = &.{};
        for (self.units.items) |*u| u.deinit(self.gpa);
        self.units.deinit(self.gpa);
        self.units = .empty;
        for (self.cal_prompts.items) |*c| c.deinit(self.gpa);
        self.cal_prompts.deinit(self.gpa);
        self.cal_prompts = .empty;
        if (self.w_lex) |*w| w.deinit(self.gpa);
        self.w_lex = null;
        if (self.w_sem) |*w| w.deinit(self.gpa);
        self.w_sem = null;
        if (self.prefix_state) |*p| p.deinit(self.gpa);
        self.prefix_state = null;
        self.gpa.free(self.prefix_ids);
        self.prefix_ids = &.{};
        self.gpa.free(self.suffix_ids);
        self.suffix_ids = &.{};
        self.calibrated = false;
        self.base_saved = false;
        self.pending_deltas = 0;
        self.unflushed_from = 0;
    }

    pub fn hasUnits(self: *const Bank) bool {
        if (!self.calibrated or self.prefix_state == null) return false;
        for (self.units.items) |u| if (u.gate.isActive()) return true;
        return false;
    }

    pub fn activeUnitCount(self: *const Bank) u32 {
        var c: u32 = 0;
        for (self.units.items) |u| if (u.gate.isActive()) {
            c += 1;
        };
        return c;
    }

    pub fn maxParentId(self: *const Bank) u32 {
        var m: u32 = 0;
        for (self.units.items) |u| {
            if (u.parent_id > m) m = u.parent_id;
        }
        return m;
    }

    /// Recomputes both the pure-vector associative bridge graph and the 2-level ClusterIndex (< 2 ms).
    pub fn rebuildBridges(self: *Bank) !void {
        self.gpa.free(self.bridges);
        self.bridges = &.{};
        self.cluster_index.deinit(self.gpa);
        if (self.units.items.len == 0) return;

        const views = try self.gpa.alloc(gate.UnitView, self.units.items.len);
        defer self.gpa.free(views);
        for (self.units.items, views, 0..) |*u, *v, idx| {
            v.* = .{ .index = idx, .parent_id = u.parent_id, .n_tokens = u.n_tokens, .gate = &u.gate };
        }
        if (self.units.items.len >= 2) {
            self.bridges = try gate.computeBridgeEdges(self.gpa, views);
        }
        try self.cluster_index.rebuild(self.gpa, views, self.backend.arch.hidden);
        if (std.c.getenv("TF_SLIDE_TRACE") != null) {
            std.debug.print("lw6 index: {d} bridges, {d} clusters across {d} units", .{ self.bridges.len, self.cluster_index.n_clusters, self.units.items.len });
            for (self.bridges[0..@min(self.bridges.len, 12)]) |e| {
                std.debug.print(" [u{d}->u{d}:{d:.2}]", .{ self.units.items[e.from].unit_id, self.units.items[e.to].unit_id, e.weight });
            }
            std.debug.print("\n", .{});
        }
    }

    /// Runs the 45-text calibration bank + shared prefix compilation once on a fresh folder.
    /// If already loaded from disk (calibrated == true), preserves existing calibration & units,
    /// rebuilding only `cal_prompts` if needed for teaching new units!
    pub fn ensureCalibrated(
        self: *Bank,
        cal_ids: []const []const u32,
        prefix_ids: []const u32,
        suffix_ids: []const u32,
    ) !void {
        const arch = self.backend.arch;
        if (self.calibrated and self.w_lex != null and self.w_sem != null and self.prefix_state != null) {
            if (self.cal_prompts.items.len == 0 and cal_ids.len > 0) {
                try self.populateCalPrompts(cal_ids);
            }
            return;
        }
        self.clearMemory();

        // Extract dual-layer hidden states on all calibration texts
        var lex_all: std.ArrayList(f32) = .empty;
        defer lex_all.deinit(self.gpa);
        var sem_all: std.ArrayList(f32) = .empty;
        defer sem_all.deinit(self.gpa);
        var lengths: std.ArrayList(usize) = .empty;
        defer lengths.deinit(self.gpa);

        for (cal_ids) |ids| {
            if (ids.len == 0) continue;
            const n = ids.len;
            const hl = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hl);
            const hs = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hs);
            try self.backend.extractHidden(self.backend.ctx, ids, arch.layer_lex, arch.layer_sem, hl, hs);
            try lex_all.appendSlice(self.gpa, hl);
            try sem_all.appendSlice(self.gpa, hs);
            try lengths.append(self.gpa, n);
        }
        const total_toks = lex_all.items.len / arch.hidden;
        if (total_toks == 0) return error.EmptyCalibration;

        self.w_lex = try whitener.LayerWhitener.fit(self.gpa, lex_all.items, total_toks, arch.hidden, 16, 560);
        self.w_sem = try whitener.LayerWhitener.fit(self.gpa, sem_all.items, total_toks, arch.hidden, 32, 480);

        var off: usize = 0;
        for (lengths.items) |n| {
            const hl = lex_all.items[off * arch.hidden ..][0 .. n * arch.hidden];
            const hs = sem_all.items[off * arch.hidden ..][0 .. n * arch.hidden];
            const tl = try self.w_lex.?.transform(self.gpa, hl, n);
            const ts = try self.w_sem.?.transform(self.gpa, hs, n);
            try self.cal_prompts.append(self.gpa, .{ .lex = tl, .sem = ts });
            off += n;
        }

        // Compile shared prefix state (KV + recurrent state)
        self.prefix_ids = try self.gpa.dupe(u32, prefix_ids);
        self.suffix_ids = try self.gpa.dupe(u32, suffix_ids);
        var pref = try RawState.init(self.gpa, arch, @intCast(prefix_ids.len));
        errdefer pref.deinit(self.gpa);
        try self.backend.compileState(self.backend.ctx, null, prefix_ids, &pref);
        self.prefix_state = pref;
        self.calibrated = true;
    }

    fn populateCalPrompts(self: *Bank, cal_ids: []const []const u32) !void {
        const arch = self.backend.arch;
        for (cal_ids) |ids| {
            if (ids.len == 0) continue;
            const n = ids.len;
            const hl = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hl);
            const hs = try self.gpa.alloc(f32, n * arch.hidden);
            defer self.gpa.free(hs);
            try self.backend.extractHidden(self.backend.ctx, ids, arch.layer_lex, arch.layer_sem, hl, hs);
            const tl = try self.w_lex.?.transform(self.gpa, hl, n);
            const ts = try self.w_sem.?.transform(self.gpa, hs, n);
            try self.cal_prompts.append(self.gpa, .{ .lex = tl, .sem = ts });
        }
    }

    /// Compiles one chunk (fact, document section, digest, or standing policy) into pure compressed weights
    /// + dual-layer gate prototypes, updating the pure-vector associative bridge graph.
    pub fn teachChunk(self: *Bank, inp: TeachChunkInput) !u32 {
        if (!self.calibrated or self.prefix_state == null or self.w_lex == null or self.w_sem == null) {
            return error.BankNotCalibrated;
        }
        if (inp.body_ids.len == 0) return error.EmptyChunk;
        if (!inp.is_standing and !inp.is_digest and inp.gate_ids.len == 0) return error.EmptyChunk;
        const arch = self.backend.arch;
        const pref = &self.prefix_state.?;
        const b_len: usize = inp.body_ids.len;

        // 1. Compile KV + recurrent state by running body_ids on top of shared prefix
        var body_raw = try RawState.init(self.gpa, arch, @intCast(b_len));
        defer body_raw.deinit(self.gpa);
        try self.backend.compileState(self.backend.ctx, pref, inp.body_ids, &body_raw);

        // 2. Inverse-RoPE K back to position-independent Pre-RoPE K_raw, then Q4 (group-32) or Q8 quantize
        const kv = try self.gpa.alloc(KvLayerData, arch.num_attn);
        var kv_made: usize = 0;
        errdefer {
            for (kv[0..kv_made]) |*k| k.deinit(self.gpa);
            self.gpa.free(kv);
        }
        const rows_kv = arch.kv_heads * b_len;
        const use_q4 = (arch.kv_bits == 4) and (arch.head_dim % quant.q4_group == 0);
        for (0..arch.num_attn) |a| {
            quant.ropeRotateKvBf16(
                body_raw.kv_k[a],
                arch.kv_heads,
                b_len,
                arch.head_dim,
                pref.len,
                arch.rotary_dim,
                arch.rope_theta,
                true,
            );
            if (use_q4) {
                const packed_elems = rows_kv * (arch.head_dim / 2);
                const group_scales = rows_kv * (arch.head_dim / quant.q4_group);
                const kq = try self.gpa.alloc(u8, packed_elems);
                errdefer self.gpa.free(kq);
                const ks = try self.gpa.alloc(u16, group_scales);
                errdefer self.gpa.free(ks);
                const vq = try self.gpa.alloc(u8, packed_elems);
                errdefer self.gpa.free(vq);
                const vs = try self.gpa.alloc(u16, group_scales);
                errdefer self.gpa.free(vs);
                quant.quantizeRowsBf16Q4(body_raw.kv_k[a], rows_kv, arch.head_dim, kq, ks);
                quant.quantizeRowsBf16Q4(body_raw.kv_v[a], rows_kv, arch.head_dim, vq, vs);
                kv[a] = .{ .q4 = .{ .kq = kq, .ks = ks, .vq = vq, .vs = vs } };
            } else {
                const elems_kv = rows_kv * arch.head_dim;
                const kq = try self.gpa.alloc(i8, elems_kv);
                errdefer self.gpa.free(kq);
                const ks = try self.gpa.alloc(f32, rows_kv);
                errdefer self.gpa.free(ks);
                const vq = try self.gpa.alloc(i8, elems_kv);
                errdefer self.gpa.free(vq);
                const vs = try self.gpa.alloc(f32, rows_kv);
                errdefer self.gpa.free(vs);
                quant.quantizeRowsBf16(body_raw.kv_k[a], rows_kv, arch.head_dim, kq, ks);
                quant.quantizeRowsBf16(body_raw.kv_v[a], rows_kv, arch.head_dim, vq, vs);
                kv[a] = .{ .q8 = .{ .kq = kq, .ks = ks, .vq = vq, .vs = vs } };
            }
            kv_made += 1;
        }

        // 3. Compress recurrent state deltas (S_after - S_prefix):
        // Pure-attention transformers (`num_rec == 0`) have zero recurrent cost!
        // Hybrid models use `arch.svd_rank_short` for short facts / standing policies and `arch.svd_rank` for doc chunks.
        const target_rank: usize = if (arch.num_rec == 0) 0 else if (inp.is_small_fact or inp.is_standing) arch.svd_rank_short else arch.svd_rank;
        const rank: usize = if (target_rank == 0) 0 else @min(b_len + 2, @min(target_rank, @min(arch.rec_rows, arch.rec_cols)));
        const rec: []RecLayerData = if (rank == 0)
            try self.gpa.alloc(RecLayerData, 0)
        else
            try compressRecurrentLayers(self.gpa, arch, pref, &body_raw, rank);
        errdefer {
            for (rec) |*r| r.deinit(self.gpa);
            self.gpa.free(rec);
        }

        // 4. Build dual-layer gate prototypes (standing & digest units skip Hop-1 gating -> 0 prototype bytes!)
        var ug: gate.UnitGate = undefined;
        if (inp.is_standing or inp.is_digest) {
            ug = .{
                .dim = arch.hidden,
                .lex_protos = try self.gpa.alloc(f32, 0),
                .lex_weights = try self.gpa.alloc(f32, 0),
                .n_lex = 0,
                .n_body_lex = 0,
                .sem_protos = try self.gpa.alloc(f32, 0),
                .sem_weights = try self.gpa.alloc(f32, 0),
                .n_sem = 0,
                .neg_lex_max = 0.0,
                .neg_sem_max = 0.0,
                .is_small_fact = false,
                .is_standing = inp.is_standing,
                .is_digest = inp.is_digest,
            };
        } else {
            var lex_vecs: std.ArrayList(f32) = .empty;
            errdefer lex_vecs.deinit(self.gpa);
            var lex_wts: std.ArrayList(f32) = .empty;
            errdefer lex_wts.deinit(self.gpa);
            var sem_vecs: std.ArrayList(f32) = .empty;
            errdefer sem_vecs.deinit(self.gpa);
            var sem_wts: std.ArrayList(f32) = .empty;
            errdefer sem_wts.deinit(self.gpa);

            var n_body_lex: usize = 0;
            {
                const n = inp.gate_ids.len;
                const hl = try self.gpa.alloc(f32, n * arch.hidden);
                defer self.gpa.free(hl);
                const hs = try self.gpa.alloc(f32, n * arch.hidden);
                defer self.gpa.free(hs);
                try self.backend.extractHidden(self.backend.ctx, inp.gate_ids, arch.layer_lex, arch.layer_sem, hl, hs);
                var tl = try self.w_lex.?.transform(self.gpa, hl, n);
                defer tl.deinit(self.gpa);
                var ts = try self.w_sem.?.transform(self.gpa, hs, n);
                defer ts.deinit(self.gpa);

                const max_lex = @min(@as(usize, 180), @max(@as(usize, 24), n / 2));
                const max_sem = @min(@as(usize, 96), @max(@as(usize, 16), n / 3));
                var lp = try whitener.selectPrototypes(self.gpa, &tl, arch.hidden, self.w_lex.?.norm_floor, max_lex, 0.35, 0.93);
                defer lp.deinit(self.gpa);
                var sp = try whitener.selectPrototypes(self.gpa, &ts, arch.hidden, self.w_sem.?.norm_floor, max_sem, 0.30, 0.87);
                defer sp.deinit(self.gpa);

                try lex_vecs.appendSlice(self.gpa, lp.vecs);
                try lex_wts.appendSlice(self.gpa, lp.weights);
                try sem_vecs.appendSlice(self.gpa, sp.vecs);
                try sem_wts.appendSlice(self.gpa, sp.weights);
                n_body_lex = lp.count;

                // Small-fact pooled semantic centroid (v6): strengthens deep semantic paraphrase matching
                // (e.g. "colour" <-> "shade/hue", "dog" <-> "pet") even without LLM cue words!
                if (inp.is_small_fact and sp.count >= 2) {
                    const mean_s = try self.gpa.alloc(f32, arch.hidden);
                    defer self.gpa.free(mean_s);
                    @memset(mean_s, 0.0);
                    var wsum: f32 = 0.0;
                    for (0..sp.count) |pi| {
                        const w = @max(sp.weights[pi], 0.15);
                        const v = sp.vecs[pi * arch.hidden ..][0..arch.hidden];
                        for (mean_s, v) |*dst, x| dst.* += w * x;
                        wsum += w;
                    }
                    var sq: f32 = 0.0;
                    for (mean_s) |x| sq += x * x;
                    if (sq > 1e-12) {
                        const inv = 1.0 / @sqrt(sq);
                        for (mean_s) |*x| x.* *= inv;
                        try sem_vecs.appendSlice(self.gpa, mean_s);
                        try sem_wts.append(self.gpa, @min(0.95, wsum / @as(f32, @floatFromInt(sp.count)) + 0.08));
                    }
                }
            }

            if (inp.title_ids.len > 0) {
                const n = inp.title_ids.len;
                const hl = try self.gpa.alloc(f32, n * arch.hidden);
                defer self.gpa.free(hl);
                const hs = try self.gpa.alloc(f32, n * arch.hidden);
                defer self.gpa.free(hs);
                try self.backend.extractHidden(self.backend.ctx, inp.title_ids, arch.layer_lex, arch.layer_sem, hl, hs);
                var tl = try self.w_lex.?.transform(self.gpa, hl, n);
                defer tl.deinit(self.gpa);
                var lp = try whitener.selectPrototypes(self.gpa, &tl, arch.hidden, self.w_lex.?.norm_floor, 12, 0.40, 0.92);
                defer lp.deinit(self.gpa);
                try lex_vecs.appendSlice(self.gpa, lp.vecs);
                for (lp.weights) |w| try lex_wts.append(self.gpa, w * 0.72);
            }

            if (inp.cue_ids.len > 0) {
                const n = inp.cue_ids.len;
                const hl = try self.gpa.alloc(f32, n * arch.hidden);
                defer self.gpa.free(hl);
                const hs = try self.gpa.alloc(f32, n * arch.hidden);
                defer self.gpa.free(hs);
                try self.backend.extractHidden(self.backend.ctx, inp.cue_ids, arch.layer_lex, arch.layer_sem, hl, hs);
                var tl = try self.w_lex.?.transform(self.gpa, hl, n);
                defer tl.deinit(self.gpa);
                var ts = try self.w_sem.?.transform(self.gpa, hs, n);
                defer ts.deinit(self.gpa);
                var lp = try whitener.selectPrototypes(self.gpa, &tl, arch.hidden, self.w_lex.?.norm_floor, 18, 0.35, 0.92);
                defer lp.deinit(self.gpa);
                var sp = try whitener.selectPrototypes(self.gpa, &ts, arch.hidden, self.w_sem.?.norm_floor, 20, 0.25, 0.87);
                defer sp.deinit(self.gpa);
                try lex_vecs.appendSlice(self.gpa, lp.vecs);
                for (lp.weights) |w| try lex_wts.append(self.gpa, w * 0.90);
                try sem_vecs.appendSlice(self.gpa, sp.vecs);
                for (sp.weights) |w| try sem_wts.append(self.gpa, w * 0.94);
            }

            const n_lex = lex_wts.items.len;
            const n_sem = sem_wts.items.len;
            // each slice freed by this block until the gate holds all four, then by the gate's cleanup alone
            ug = gate: {
                const lex_p = try lex_vecs.toOwnedSlice(self.gpa);
                errdefer self.gpa.free(lex_p);
                const lex_w = try lex_wts.toOwnedSlice(self.gpa);
                errdefer self.gpa.free(lex_w);
                const sem_p = try sem_vecs.toOwnedSlice(self.gpa);
                errdefer self.gpa.free(sem_p);
                break :gate .{
                    .dim = arch.hidden,
                    .lex_protos = lex_p,
                    .lex_weights = lex_w,
                    .n_lex = n_lex,
                    .n_body_lex = n_body_lex,
                    .sem_protos = sem_p,
                    .sem_weights = try sem_wts.toOwnedSlice(self.gpa),
                    .n_sem = n_sem,
                    .neg_lex_max = 0.0,
                    .neg_sem_max = 0.0,
                    .is_small_fact = inp.is_small_fact,
                    .is_standing = false,
                    .is_digest = inp.is_digest,
                };
            };
            errdefer ug.deinit(self.gpa);

            // Calibrate negative semantic ceiling against the background calibration prompts
            var neg_sem_max: f32 = 0.0;
            for (self.cal_prompts.items) |*cp| {
                const sc = try gate.scoreUnitDual(
                    self.gpa,
                    &ug,
                    &cp.lex,
                    &cp.sem,
                    self.w_lex.?.norm_floor,
                    self.w_sem.?.norm_floor,
                );
                if (sc.sem_score > neg_sem_max) neg_sem_max = sc.sem_score;
            }
            ug.neg_sem_max = neg_sem_max;
        }
        errdefer ug.deinit(self.gpa);

        const uid = self.next_unit_id;
        self.next_unit_id += 1;
        ug.epoch = self.next_epoch;
        self.next_epoch += 1;
        ug.superseded_by = 0;
        ug.reinforcements = 1;
        try ug.computeSummaries(self.gpa);
        if (ug.is_standing) self.freeSkillRec();
        if (std.c.getenv("TF_SLIDE_TRACE") != null) {
            std.debug.print(
                "lw6 teach: uid={d} pid={d} ep={d} tok={d} kv=q{d} rk={d} n_lex={d}(body={d}) n_sem={d} small={} standing={} digest={} neg_sem={d:.3}\n",
                .{ uid, inp.parent_id, ug.epoch, b_len, if (use_q4) @as(u8, 4) else @as(u8, 8), rank, ug.n_lex, ug.n_body_lex, ug.n_sem, ug.is_small_fact, ug.is_standing, ug.is_digest, ug.neg_sem_max },
            );
        }
        try self.units.append(self.gpa, .{
            .unit_id = uid,
            .parent_id = inp.parent_id,
            .n_tokens = @intCast(b_len),
            .gate = ug,
            .kv = kv,
            .rec = rec,
        });
        // the unit's parts stay this function's until the index takes it: a failed rebuild gives it back, freed once
        self.rebuildBridges() catch |e| {
            _ = self.units.pop();
            self.rebuildBridges() catch {}; // the index of the units before this one, as it was
            return e;
        };
        return uid;
    }

    /// Pure-vector (< 1 ms) entity conflict search for Temporal Belief Revision (v6).
    /// Returns up to `max_out` currently active small-fact parent IDs overlapping `gate_ids`.
    pub fn findConflictParents(self: *Bank, gate_ids: []const u32, max_out: usize) ![]gate.ConflictMatch {
        if (!self.hasUnits() or gate_ids.len == 0) return try self.gpa.alloc(gate.ConflictMatch, 0);
        const arch = self.backend.arch;
        const q_slice = if (gate_ids.len > 128) gate_ids[0..128] else gate_ids;
        const n = q_slice.len;
        const hl = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hl);
        const hs = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hs);
        try self.backend.extractHidden(self.backend.ctx, q_slice, arch.layer_lex, arch.layer_sem, hl, hs);

        var tl = try self.w_lex.?.transform(self.gpa, hl, n);
        defer tl.deinit(self.gpa);
        var ts = try self.w_sem.?.transform(self.gpa, hs, n);
        defer ts.deinit(self.gpa);

        const views = try self.gpa.alloc(gate.UnitView, self.units.items.len);
        defer self.gpa.free(views);
        for (self.units.items, views, 0..) |*u, *v, idx| {
            v.* = .{ .index = idx, .parent_id = u.parent_id, .n_tokens = u.n_tokens, .gate = &u.gate };
        }
        return gate.findConflictParents(self.gpa, views, &tl, &ts, self.w_lex.?.norm_floor, self.w_sem.?.norm_floor, max_out);
    }

    /// Marks all active units belonging to `old_parent_id` as superseded by `new_parent_id`.
    /// Superseded units stop routing immediately yet stay restorable if `new_parent_id` is later deleted!
    pub fn supersedeParent(self: *Bank, old_parent_id: u32, new_parent_id: u32) u32 {
        if (old_parent_id == 0 or new_parent_id == 0 or old_parent_id == new_parent_id) return 0;
        var changed: u32 = 0;
        var touched_standing = false;
        for (self.units.items) |*u| {
            if (u.parent_id == old_parent_id and u.gate.superseded_by == 0) {
                u.gate.superseded_by = new_parent_id;
                if (u.gate.is_standing) touched_standing = true;
                changed += 1;
            }
        }
        if (touched_standing) self.freeSkillRec();
        if (changed > 0) self.rebuildBridges() catch {};
        return changed;
    }

    /// Increments the reinforcement counter for all active units of `parent_id`.
    pub fn reinforceParent(self: *Bank, parent_id: u32) u32 {
        var changed: u32 = 0;
        for (self.units.items) |*u| {
            if (u.parent_id == parent_id and u.gate.superseded_by == 0) {
                u.gate.reinforcements +|= 1;
                changed += 1;
            }
        }
        return changed;
    }

    /// Removes all units belonging to `parent_id` AND automatically restores any prior belief that
    /// was superseded by `parent_id` (undoable temporal belief history!).
    pub fn forgetParent(self: *Bank, parent_id: u32) u32 {
        var removed: u32 = 0;
        var restored: u32 = 0;
        var touched_standing = false;
        var i: usize = 0;
        while (i < self.units.items.len) {
            if (self.units.items[i].parent_id == parent_id) {
                if (self.units.items[i].gate.is_standing) touched_standing = true;
                self.units.items[i].deinit(self.gpa);
                _ = self.units.orderedRemove(i);
                if (i < self.unflushed_from) self.unflushed_from -= 1;
                removed += 1;
            } else {
                if (self.units.items[i].gate.superseded_by == parent_id) {
                    self.units.items[i].gate.superseded_by = 0;
                    if (self.units.items[i].gate.is_standing) touched_standing = true;
                    restored += 1;
                }
                i += 1;
            }
        }
        if (touched_standing) self.freeSkillRec();
        if (removed > 0 or restored > 0) self.rebuildBridges() catch {};
        return removed;
    }

    /// Runs the dual-layer neural gate (2-level ClusterIndex + single-token semantic guard +
    /// submodular coverage + 2-hop associative bridges + standing policies) on `query_ids`.
    /// Returns `null` on miss!
    pub fn routeAndCompose(self: *Bank, query_ids: []const u32) !?SnapHandle {
        if (!self.hasUnits() or query_ids.len == 0) return null;
        const arch = self.backend.arch;

        // Clamp very long queries to the last 256 tokens for fast O(1) gating
        const q_slice = if (query_ids.len > 256) query_ids[query_ids.len - 256 ..] else query_ids;
        const n = q_slice.len;
        const hl = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hl);
        const hs = try self.gpa.alloc(f32, n * arch.hidden);
        defer self.gpa.free(hs);
        try self.backend.extractHidden(self.backend.ctx, q_slice, arch.layer_lex, arch.layer_sem, hl, hs);

        var tl = try self.w_lex.?.transform(self.gpa, hl, n);
        defer tl.deinit(self.gpa);
        var ts = try self.w_sem.?.transform(self.gpa, hs, n);
        defer ts.deinit(self.gpa);

        const views = try self.gpa.alloc(gate.UnitView, self.units.items.len);
        defer self.gpa.free(views);
        for (self.units.items, views, 0..) |*u, *v, idx| {
            v.* = .{ .index = idx, .parent_id = u.parent_id, .n_tokens = u.n_tokens, .gate = &u.gate };
        }

        const cidx: ?*const gate.ClusterIndex = if (self.cluster_index.isReady()) &self.cluster_index else null;
        const active = try gate.routeUnitsIndexed(
            self.gpa,
            views,
            self.bridges,
            cidx,
            &tl,
            &ts,
            self.w_lex.?.norm_floor,
            self.w_sem.?.norm_floor,
            self.route_cfg,
        );
        defer self.gpa.free(active);
        if (std.c.getenv("TF_SLIDE_TRACE") != null) {
            std.debug.print("lw6 route: q_len={d} active=[", .{n});
            for (active, 0..) |idx, k| {
                if (k > 0) std.debug.print(",", .{});
                const u = &self.units.items[idx];
                const tag: []const u8 = if (u.gate.is_standing) ":S" else if (u.gate.is_digest) ":D" else "";
                std.debug.print("u{d}(p{d}{s})", .{ u.unit_id, u.parent_id, tag });
            }
            std.debug.print("]", .{});
            for (views) |v| {
                if (!v.gate.isActive() or v.gate.is_standing or v.gate.is_digest) continue;
                const sc = gate.scoreUnitDual(self.gpa, v.gate, &tl, &ts, self.w_lex.?.norm_floor, self.w_sem.?.norm_floor) catch continue;
                const t_sem = @max(self.route_cfg.sem_floor, v.gate.neg_sem_max + self.route_cfg.sem_margin);
                if (sc.lex_gate >= 0.30 or sc.sem_score >= t_sem - 0.05) {
                    std.debug.print(" | u{d}(p{d}):lg={d:.2},ns={d},ss={d:.2}/{d:.2}", .{ self.units.items[v.index].unit_id, v.parent_id, sc.lex_gate, sc.n_strong_lex, sc.sem_score, t_sem });
                }
            }
            std.debug.print("\n", .{});
        }
        if (active.len == 0) return null;

        var composed = try self.composeRawState(active);
        defer composed.deinit(self.gpa);
        return try self.backend.buildSnap(self.backend.ctx, &composed, self.suffix_ids);
    }

    /// Assembles `prefix + active units` KV (dequantized Q4/Q8 & re-RoPE'd at contiguous offsets) and
    /// parent-balanced spectral-denoised SVD recurrent states (or clean prefix recurrent state if `rank == 0`).
    pub fn composeRawState(self: *const Bank, active_indices: []const usize) !RawState {
        const arch = self.backend.arch;
        const pref = &self.prefix_state.?;

        var total_len: usize = pref.len;
        for (active_indices) |idx| total_len += self.units.items[idx].n_tokens;

        var out = try RawState.init(self.gpa, arch, @intCast(total_len));
        errdefer out.deinit(self.gpa);

        // 1. Assemble Attention KV caches: prefix at [0..p_len) + each active unit dequantized & RoPE'd at `cur`
        var tmp_k: std.ArrayList(u16) = .empty;
        defer tmp_k.deinit(self.gpa);
        var tmp_v: std.ArrayList(u16) = .empty;
        defer tmp_v.deinit(self.gpa);

        for (0..arch.num_attn) |a| {
            for (0..arch.kv_heads) |h| {
                const src_k = pref.kv_k[a][h * pref.len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
                const src_v = pref.kv_v[a][h * pref.len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
                const dst_k = out.kv_k[a][h * total_len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
                const dst_v = out.kv_v[a][h * total_len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
                @memcpy(dst_k, src_k);
                @memcpy(dst_v, src_v);
            }

            var cur: usize = pref.len;
            for (active_indices) |idx| {
                const u = &self.units.items[idx];
                const u_len: usize = u.n_tokens;
                const rows = arch.kv_heads * u_len;
                const elems = rows * arch.head_dim;
                try tmp_k.resize(self.gpa, elems);
                try tmp_v.resize(self.gpa, elems);

                switch (u.kv[a]) {
                    .q8 => |d| {
                        quant.dequantizeRowsBf16(d.kq, d.ks, rows, arch.head_dim, tmp_k.items);
                        quant.dequantizeRowsBf16(d.vq, d.vs, rows, arch.head_dim, tmp_v.items);
                    },
                    .q4 => |d| {
                        quant.dequantizeRowsBf16Q4(d.kq, d.ks, rows, arch.head_dim, tmp_k.items);
                        quant.dequantizeRowsBf16Q4(d.vq, d.vs, rows, arch.head_dim, tmp_v.items);
                    },
                }
                quant.ropeRotateKvBf16(
                    tmp_k.items,
                    arch.kv_heads,
                    u_len,
                    arch.head_dim,
                    cur,
                    arch.rotary_dim,
                    arch.rope_theta,
                    false,
                );

                for (0..arch.kv_heads) |h| {
                    const sk = tmp_k.items[h * u_len * arch.head_dim ..][0 .. u_len * arch.head_dim];
                    const sv = tmp_v.items[h * u_len * arch.head_dim ..][0 .. u_len * arch.head_dim];
                    const dk = out.kv_k[a][(h * total_len + cur) * arch.head_dim ..][0 .. u_len * arch.head_dim];
                    const dv = out.kv_v[a][(h * total_len + cur) * arch.head_dim ..][0 .. u_len * arch.head_dim];
                    @memcpy(dk, sk);
                    @memcpy(dv, sv);
                }
                cur += u_len;
            }
        }

        // 2. Pure-attention transformers (`num_rec == 0`) finish immediately after KV assembly!
        if (arch.num_rec == 0) return out;

        // 3. Parent-balanced SVD recurrent state blending (plus sleep-time consolidated `skill_rec` steering!)
        var has_standing_active = false;
        var max_active_rank: usize = 0;
        for (active_indices) |idx| {
            const u = &self.units.items[idx];
            if (u.gate.is_standing) has_standing_active = true;
            if (u.rec.len > 0 and u.rec[0].rank > max_active_rank) max_active_rank = u.rec[0].rank;
        }
        const use_skill_rec = has_standing_active and self.skill_rec.len == arch.num_rec and self.skill_rec[0].rank > 0;
        if (use_skill_rec and self.skill_rec[0].rank > max_active_rank) {
            max_active_rank = self.skill_rec[0].rank;
        }

        if (max_active_rank == 0) {
            for (0..arch.num_rec) |r| {
                @memcpy(out.rec_state[r], pref.rec_state[r]);
                @memcpy(out.rec_conv[r], pref.rec_conv[r]);
            }
            return out;
        }

        var unique_parents: [16]u32 = undefined;
        var parent_counts: [16]usize = @splat(0);
        var n_parents: usize = 0;
        for (active_indices) |idx| {
            const u = &self.units.items[idx];
            if (use_skill_rec and u.gate.is_standing) continue; // represented jointly by `skill_rec`
            const pid = u.parent_id;
            var found_p: ?usize = null;
            for (0..n_parents) |pi| if (unique_parents[pi] == pid) {
                found_p = pi;
                break;
            };
            if (found_p) |pi| {
                parent_counts[pi] += 1;
            } else if (n_parents < unique_parents.len) {
                unique_parents[n_parents] = pid;
                parent_counts[n_parents] = 1;
                n_parents += 1;
            }
        }
        const total_groups: usize = @max(1, n_parents + @intFromBool(use_skill_rec));

        const weights = try self.gpa.alloc(f32, active_indices.len);
        defer self.gpa.free(weights);
        for (active_indices, weights) |idx, *w| {
            const u = &self.units.items[idx];
            if (use_skill_rec and u.gate.is_standing) {
                w.* = 0.0;
                continue;
            }
            const pid = u.parent_id;
            var pc: usize = 1;
            for (0..n_parents) |pi| if (unique_parents[pi] == pid) {
                pc = parent_counts[pi];
                break;
            };
            w.* = 1.0 / @as(f32, @floatFromInt(@max(1, total_groups * pc)));
        }
        const skill_w: f32 = if (use_skill_rec) 1.0 / @as(f32, @floatFromInt(total_groups)) else 0.0;

        const v_scratch = try self.gpa.alloc(f32, max_active_rank * arch.rec_cols);
        defer self.gpa.free(v_scratch);
        const head_mat = arch.rec_rows * arch.rec_cols;

        for (0..arch.num_rec) |r| {
            @memcpy(out.rec_state[r], pref.rec_state[r]);
            if (use_skill_rec) {
                const rl = &self.skill_rec[r];
                const rk: usize = rl.rank;
                for (0..arch.rec_heads) |h| {
                    const uq = rl.uq[h * arch.rec_rows * rk ..][0 .. arch.rec_rows * rk];
                    const us = rl.us[h * arch.rec_rows ..][0..arch.rec_rows];
                    const vq = rl.vq[h * rk * arch.rec_cols ..][0 .. rk * arch.rec_cols];
                    const vs = rl.vs[h * rk ..][0..rk];
                    const dst = out.rec_state[r][h * head_mat ..][0..head_mat];
                    quant.reconstructSvdAdd(uq, us, vq, vs, arch.rec_rows, rk, arch.rec_cols, skill_w, dst, v_scratch);
                }
            }
            for (active_indices, weights) |idx, w| {
                if (w == 0.0) continue;
                const u = &self.units.items[idx];
                if (u.rec.len <= r) continue;
                const rl = &u.rec[r];
                const rk: usize = rl.rank;
                if (rk == 0) continue;
                for (0..arch.rec_heads) |h| {
                    const uq = rl.uq[h * arch.rec_rows * rk ..][0 .. arch.rec_rows * rk];
                    const us = rl.us[h * arch.rec_rows ..][0..arch.rec_rows];
                    const vq = rl.vq[h * rk * arch.rec_cols ..][0 .. rk * arch.rec_cols];
                    const vs = rl.vs[h * rk ..][0..rk];
                    const dst = out.rec_state[r][h * head_mat ..][0..head_mat];
                    quant.reconstructSvdAdd(uq, us, vq, vs, arch.rec_rows, rk, arch.rec_cols, w, dst, v_scratch);
                }
            }
            const use_unit_conv = active_indices.len == 1 and
                self.units.items[active_indices[0]].rec.len > r and
                self.units.items[active_indices[0]].rec[r].conv.len == arch.conv_elems;
            const conv_src = if (use_unit_conv)
                self.units.items[active_indices[0]].rec[r].conv
            else
                pref.rec_conv[r];
            @memcpy(out.rec_conv[r], conv_src);
        }
        return out;
    }

    pub fn filePath(self: *const Bank, gpa: Allocator) ![]u8 {
        return std.fs.path.join(gpa, &.{ self.dir, store.file_name });
    }

    pub fn deltaPath(self: *const Bank, gpa: Allocator) ![]u8 {
        return std.fs.path.join(gpa, &.{ self.dir, store.delta_file_name });
    }

    /// Sleep-time consolidation (v6):
    /// 1. Optionally prunes permanently superseded tombstones (`prune_superseded`), reclaiming RAM & disk.
    /// 2. Compiles a unified spectral rank-8 recurrent steering delta (`skill_rec`) across all active
    ///    standing policies (`compact_policies`, hybrid models with `num_rec > 0`), verified for norm stability.
    /// 3. Rebuilds pure-vector bridges + 2-level ClusterIndex and compacts `living-weights.delta` into
    ///    a clean atomic `living-weights.safetensors`.
    pub fn consolidate(self: *Bank, prune_superseded: bool, compact_policies: bool) !ConsolidateReport {
        var rep = ConsolidateReport{ .compacted_deltas = self.pending_deltas };

        if (prune_superseded) {
            var i: usize = 0;
            while (i < self.units.items.len) {
                if (self.units.items[i].gate.superseded_by != 0) {
                    self.units.items[i].deinit(self.gpa);
                    _ = self.units.orderedRemove(i);
                    rep.pruned_superseded += 1;
                } else {
                    i += 1;
                }
            }
        }

        // Sleep-time spectral recurrent steering consolidation across active standing policies
        self.freeSkillRec();
        const arch = self.backend.arch;
        if (compact_policies and arch.num_rec > 0) {
            var pol_indices: std.ArrayList(usize) = .empty;
            defer pol_indices.deinit(self.gpa);
            for (self.units.items, 0..) |*u, idx| {
                if (u.gate.isActive() and u.gate.is_standing and u.rec.len == arch.num_rec and u.rec[0].rank > 0) {
                    try pol_indices.append(self.gpa, idx);
                }
            }
            if (pol_indices.items.len >= 1) {
                const target_rk = @min(@as(usize, 8), @min(arch.rec_rows, arch.rec_cols));
                try self.buildConsolidatedSkillRec(pol_indices.items, target_rk);
                rep.consolidated_policies = @intCast(pol_indices.items.len);
            }
        }

        try self.rebuildBridges();
        rep.bytes = try self.saveFull();
        rep.units = self.activeUnitCount();
        rep.clusters = @intCast(self.cluster_index.n_clusters);
        return rep;
    }

    fn buildConsolidatedSkillRec(self: *Bank, pol_indices: []const usize, rank: usize) !void {
        const arch = self.backend.arch;
        const head_mat = arch.rec_rows * arch.rec_cols;
        const n_pol_f: f32 = @floatFromInt(pol_indices.len);
        // Energy-preserving 1/sqrt(N) scaling preserves individual policy steering strength in Mamba-2/GDN
        const w_each: f32 = 1.0 / @sqrt(n_pol_f);

        var max_src_rk: usize = 0;
        for (pol_indices) |idx| {
            const rk = self.units.items[idx].rec[0].rank;
            if (rk > max_src_rk) max_src_rk = rk;
        }
        const v_scratch = try self.gpa.alloc(f32, @max(max_src_rk, rank) * arch.rec_cols);
        defer self.gpa.free(v_scratch);
        const acc = try self.gpa.alloc(f32, head_mat);
        defer self.gpa.free(acc);
        const u_tmp = try self.gpa.alloc(f32, arch.rec_rows * rank);
        defer self.gpa.free(u_tmp);
        const v_tmp = try self.gpa.alloc(f32, rank * arch.rec_cols);
        defer self.gpa.free(v_tmp);

        const rec = try self.gpa.alloc(RecLayerData, arch.num_rec);
        var made: usize = 0;
        errdefer {
            for (rec[0..made]) |*r| r.deinit(self.gpa);
            self.gpa.free(rec);
        }
        var scratch_arena = std.heap.ArenaAllocator.init(self.gpa);
        defer scratch_arena.deinit();

        for (0..arch.num_rec) |r| {
            const uq = try self.gpa.alloc(i8, arch.rec_heads * arch.rec_rows * rank);
            errdefer self.gpa.free(uq);
            const us = try self.gpa.alloc(f32, arch.rec_heads * arch.rec_rows);
            errdefer self.gpa.free(us);
            const vq = try self.gpa.alloc(i8, arch.rec_heads * rank * arch.rec_cols);
            errdefer self.gpa.free(vq);
            const vs = try self.gpa.alloc(f32, arch.rec_heads * rank);
            errdefer self.gpa.free(vs);
            const cv = try self.gpa.alloc(u16, 0);
            errdefer self.gpa.free(cv);

            for (0..arch.rec_heads) |h| {
                _ = scratch_arena.reset(.retain_capacity);
                @memset(acc, 0.0);
                for (pol_indices) |idx| {
                    const rl = &self.units.items[idx].rec[r];
                    const rk: usize = rl.rank;
                    if (rk == 0) continue;
                    quant.reconstructSvdAdd(
                        rl.uq[h * arch.rec_rows * rk ..][0 .. arch.rec_rows * rk],
                        rl.us[h * arch.rec_rows ..][0..arch.rec_rows],
                        rl.vq[h * rk * arch.rec_cols ..][0 .. rk * arch.rec_cols],
                        rl.vs[h * rk ..][0..rk],
                        arch.rec_rows,
                        rk,
                        arch.rec_cols,
                        w_each,
                        acc,
                        v_scratch,
                    );
                }
                _ = try quant.randomizedSvdBalanced(scratch_arena.allocator(), acc, arch.rec_rows, arch.rec_cols, rank, u_tmp, v_tmp);
                quant.quantizeRowsF32(u_tmp, arch.rec_rows, rank, uq[h * arch.rec_rows * rank ..][0 .. arch.rec_rows * rank], us[h * arch.rec_rows ..][0..arch.rec_rows]);
                quant.quantizeRowsF32(v_tmp, rank, arch.rec_cols, vq[h * rank * arch.rec_cols ..][0 .. rank * arch.rec_cols], vs[h * rank ..][0..rank]);
            }
            rec[r] = .{ .rank = @intCast(rank), .uq = uq, .us = us, .vq = vq, .vs = vs, .conv = cv };
            made += 1;
        }
        self.skill_rec = rec;
        self.skill_policy_count = @intCast(pol_indices.len);
    }

    /// Serializes a single unit's metadata + tensors into a self-contained safetensors byte slice
    /// for O(1) `< 1 ms` append-only delta journaling (`living-weights.delta`, ZERO plaintext!).
    fn serializeSingleUnit(self: *const Bank, gpa: Allocator, u: *const Unit) ![]u8 {
        const arch = self.backend.arch;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const rank: u16 = if (u.rec.len > 0) u.rec[0].rank else 0;
        const has_conv: bool = u.rec.len > 0 and u.rec[0].conv.len == arch.conv_elems;
        const kv_b: u8 = if (u.kv.len > 0) u.kv[0].bits() else arch.kv_bits;
        const ujson = try std.fmt.allocPrint(
            a,
            "{{\"uid\":{d},\"pid\":{d},\"tok\":{d},\"nl\":{d},\"nbl\":{d},\"ns\":{d},\"rk\":{d},\"kb\":{d},\"cv\":{s},\"sm\":{s},\"st\":{s},\"dg\":{s},\"nsem\":{d:.6},\"ep\":{d},\"sup\":{d},\"rf\":{d}}}",
            .{
                u.unit_id,
                u.parent_id,
                u.n_tokens,
                u.gate.n_lex,
                u.gate.bodyLexCount(),
                u.gate.n_sem,
                rank,
                kv_b,
                if (has_conv) "true" else "false",
                if (u.gate.is_small_fact) "true" else "false",
                if (u.gate.is_standing) "true" else "false",
                if (u.gate.is_digest) "true" else "false",
                u.gate.neg_sem_max,
                u.gate.epoch,
                u.gate.superseded_by,
                u.gate.reinforcements,
            },
        );
        const meta = [_]store.MetaKV{
            .{ .key = "format", .value = store.format_tag },
            .{ .key = "unit", .value = ujson },
        };
        var tensors: std.ArrayList(store.TensorSpec) = .empty;
        try appendUnitTensors(a, &tensors, arch, u, 0);
        return store.serializeToBytes(gpa, &meta, tensors.items);
    }

    /// Fast O(1) incremental persistence (v6):
    /// - When a base `living-weights.safetensors` already exists and `< 16` delta frames are pending,
    ///   appends ONLY newly added units to `living-weights.delta` in `< 1-2 ms`!
    /// - Otherwise performs a full atomic compaction (`saveFull`) and clears `living-weights.delta`.
    pub fn save(self: *Bank) !u64 {
        if (!self.calibrated or self.units.items.len == 0) {
            return self.saveFull();
        }
        if (self.base_saved and self.pending_deltas < 16 and self.unflushed_from < self.units.items.len and self.skill_rec.len == 0) {
            const dpath = try self.deltaPath(self.gpa);
            defer self.gpa.free(dpath);
            var delta_total: u64 = 0;
            for (self.units.items[self.unflushed_from..]) |*u| {
                const blob = try self.serializeSingleUnit(self.gpa, u);
                defer self.gpa.free(blob);
                delta_total = try store.appendDeltaFrame(self.io, dpath, .add_unit, blob);
                self.pending_deltas += 1;
            }
            self.unflushed_from = self.units.items.len;
            const path = try self.filePath(self.gpa);
            defer self.gpa.free(path);
            const base_len: u64 = blk: {
                const f = Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk self.last_bytes;
                defer f.close(self.io);
                break :blk f.length(self.io) catch self.last_bytes;
            };
            self.last_bytes = base_len + delta_total;
            return self.last_bytes;
        }
        return self.saveFull();
    }

    /// Appends a tiny (< 0.1 ms) delta tombstone frame (`forget_parent` or `supersede_parent`),
    /// or compacts cleanly if base isn't saved / delta threshold reached / bank became empty.
    pub fn persistTombstone(self: *Bank, op: store.DeltaOp, pid_a: u32, pid_b: u32) !u64 {
        if (!self.calibrated or self.units.items.len == 0) return self.saveFull();
        if (self.base_saved and self.pending_deltas < 16 and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) {
            const dpath = try self.deltaPath(self.gpa);
            defer self.gpa.free(dpath);
            var buf: [8]u8 = undefined;
            std.mem.writeInt(u32, buf[0..4], pid_a, .little);
            std.mem.writeInt(u32, buf[4..8], pid_b, .little);
            const payload = if (op == .supersede_parent) buf[0..8] else buf[0..4];
            const delta_total = try store.appendDeltaFrame(self.io, dpath, op, payload);
            self.pending_deltas += 1;
            const path = try self.filePath(self.gpa);
            defer self.gpa.free(path);
            const base_len: u64 = blk: {
                const f = Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk self.last_bytes;
                defer f.close(self.io);
                break :blk f.length(self.io) catch self.last_bytes;
            };
            self.last_bytes = base_len + delta_total;
            return self.last_bytes;
        }
        return self.saveFull();
    }

    /// Atomically compacts the entire bank into `living-weights.safetensors` (`tf-living-memory-v6`)
    /// and removes any pending `living-weights.delta` journal.
    pub fn saveFull(self: *Bank) !u64 {
        const path = try self.filePath(self.gpa);
        defer self.gpa.free(path);
        const dpath = try self.deltaPath(self.gpa);
        defer self.gpa.free(dpath);
        if (!self.calibrated or self.units.items.len == 0) {
            Io.Dir.cwd().deleteFile(self.io, path) catch {};
            Io.Dir.cwd().deleteFile(self.io, dpath) catch {};
            self.last_bytes = 0;
            self.base_saved = false;
            self.pending_deltas = 0;
            self.unflushed_from = 0;
            return 0;
        }
        const arch = self.backend.arch;
        const wl = &self.w_lex.?;
        const ws = &self.w_sem.?;
        const pref = &self.prefix_state.?;

        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // Build structural unit metadata JSON (NO raw text anywhere!)
        var unit_meta: std.ArrayList(u8) = .empty;
        try unit_meta.append(a, '[');
        for (self.units.items, 0..) |*u, i| {
            if (i > 0) try unit_meta.append(a, ',');
            const rank: u16 = if (u.rec.len > 0) u.rec[0].rank else 0;
            const has_conv: bool = u.rec.len > 0 and u.rec[0].conv.len == arch.conv_elems;
            const kv_b: u8 = if (u.kv.len > 0) u.kv[0].bits() else arch.kv_bits;
            const entry = try std.fmt.allocPrint(
                a,
                "{{\"uid\":{d},\"pid\":{d},\"tok\":{d},\"nl\":{d},\"nbl\":{d},\"ns\":{d},\"rk\":{d},\"kb\":{d},\"cv\":{s},\"sm\":{s},\"st\":{s},\"dg\":{s},\"nsem\":{d:.6},\"ep\":{d},\"sup\":{d},\"rf\":{d}}}",
                .{
                    u.unit_id,
                    u.parent_id,
                    u.n_tokens,
                    u.gate.n_lex,
                    u.gate.bodyLexCount(),
                    u.gate.n_sem,
                    rank,
                    kv_b,
                    if (has_conv) "true" else "false",
                    if (u.gate.is_small_fact) "true" else "false",
                    if (u.gate.is_standing) "true" else "false",
                    if (u.gate.is_digest) "true" else "false",
                    u.gate.neg_sem_max,
                    u.gate.epoch,
                    u.gate.superseded_by,
                    u.gate.reinforcements,
                },
            );
            try unit_meta.appendSlice(a, entry);
        }
        try unit_meta.append(a, ']');

        const skill_rk: u16 = if (self.skill_rec.len == arch.num_rec and arch.num_rec > 0) self.skill_rec[0].rank else 0;
        const meta = [_]store.MetaKV{
            .{ .key = "format", .value = store.format_tag },
            .{ .key = "family", .value = arch.family.name() },
            .{ .key = "hidden", .value = try std.fmt.allocPrint(a, "{d}", .{arch.hidden}) },
            .{ .key = "num_attn", .value = try std.fmt.allocPrint(a, "{d}", .{arch.num_attn}) },
            .{ .key = "num_rec", .value = try std.fmt.allocPrint(a, "{d}", .{arch.num_rec}) },
            .{ .key = "layer_lex", .value = try std.fmt.allocPrint(a, "{d}", .{arch.layer_lex}) },
            .{ .key = "layer_sem", .value = try std.fmt.allocPrint(a, "{d}", .{arch.layer_sem}) },
            .{ .key = "prefix_len", .value = try std.fmt.allocPrint(a, "{d}", .{pref.len}) },
            .{ .key = "lex_k", .value = try std.fmt.allocPrint(a, "{d}", .{wl.k_pcs}) },
            .{ .key = "lex_cal", .value = try std.fmt.allocPrint(a, "{d}", .{wl.n_cal}) },
            .{ .key = "lex_floor", .value = try std.fmt.allocPrint(a, "{d:.8}", .{wl.norm_floor}) },
            .{ .key = "sem_k", .value = try std.fmt.allocPrint(a, "{d}", .{ws.k_pcs}) },
            .{ .key = "sem_cal", .value = try std.fmt.allocPrint(a, "{d}", .{ws.n_cal}) },
            .{ .key = "sem_floor", .value = try std.fmt.allocPrint(a, "{d:.8}", .{ws.norm_floor}) },
            .{ .key = "next_uid", .value = try std.fmt.allocPrint(a, "{d}", .{self.next_unit_id}) },
            .{ .key = "next_epoch", .value = try std.fmt.allocPrint(a, "{d}", .{self.next_epoch}) },
            .{ .key = "skill_rk", .value = try std.fmt.allocPrint(a, "{d}", .{skill_rk}) },
            .{ .key = "skill_pols", .value = try std.fmt.allocPrint(a, "{d}", .{self.skill_policy_count}) },
            .{ .key = "units", .value = unit_meta.items },
        };

        var tensors: std.ArrayList(store.TensorSpec) = .empty;
        try tensors.append(a, .{ .name = "frame.prefix_ids", .dtype = .u32, .shape = try a.dupe(usize, &.{self.prefix_ids.len}), .bytes = std.mem.sliceAsBytes(self.prefix_ids) });
        try tensors.append(a, .{ .name = "frame.suffix_ids", .dtype = .u32, .shape = try a.dupe(usize, &.{self.suffix_ids.len}), .bytes = std.mem.sliceAsBytes(self.suffix_ids) });

        // Whiteners: mean & basis in F32 (tiny), cal_tokens quantized to Q8 (4x smaller, <0.0004 cosine error!)
        try tensors.append(a, .{ .name = "w.lex.mean", .dtype = .f32, .shape = try a.dupe(usize, &.{arch.hidden}), .bytes = std.mem.sliceAsBytes(wl.mean) });
        try tensors.append(a, .{ .name = "w.lex.basis", .dtype = .f32, .shape = try a.dupe(usize, &.{ wl.k_pcs, arch.hidden }), .bytes = std.mem.sliceAsBytes(wl.basis) });
        {
            const cq = try a.alloc(i8, wl.n_cal * arch.hidden);
            const cs = try a.alloc(f32, wl.n_cal);
            quant.quantizeRowsF32(wl.cal_tokens, wl.n_cal, arch.hidden, cq, cs);
            try tensors.append(a, .{ .name = "w.lex.cal.q", .dtype = .i8, .shape = try a.dupe(usize, &.{ wl.n_cal, arch.hidden }), .bytes = std.mem.sliceAsBytes(cq) });
            try tensors.append(a, .{ .name = "w.lex.cal.s", .dtype = .f32, .shape = try a.dupe(usize, &.{wl.n_cal}), .bytes = std.mem.sliceAsBytes(cs) });
        }
        try tensors.append(a, .{ .name = "w.sem.mean", .dtype = .f32, .shape = try a.dupe(usize, &.{arch.hidden}), .bytes = std.mem.sliceAsBytes(ws.mean) });
        try tensors.append(a, .{ .name = "w.sem.basis", .dtype = .f32, .shape = try a.dupe(usize, &.{ ws.k_pcs, arch.hidden }), .bytes = std.mem.sliceAsBytes(ws.basis) });
        {
            const cq = try a.alloc(i8, ws.n_cal * arch.hidden);
            const cs = try a.alloc(f32, ws.n_cal);
            quant.quantizeRowsF32(ws.cal_tokens, ws.n_cal, arch.hidden, cq, cs);
            try tensors.append(a, .{ .name = "w.sem.cal.q", .dtype = .i8, .shape = try a.dupe(usize, &.{ ws.n_cal, arch.hidden }), .bytes = std.mem.sliceAsBytes(cq) });
            try tensors.append(a, .{ .name = "w.sem.cal.s", .dtype = .f32, .shape = try a.dupe(usize, &.{ws.n_cal}), .bytes = std.mem.sliceAsBytes(cs) });
        }

        // Sleep-time consolidated spectral skill steering tensors (optional, pure weights!)
        if (skill_rk > 0) {
            const rk_u: usize = skill_rk;
            for (0..arch.num_rec) |r| {
                const rl = &self.skill_rec[r];
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.uq", .{r}), .dtype = .i8, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows, rk_u }), .bytes = std.mem.sliceAsBytes(rl.uq) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.us", .{r}), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows }), .bytes = std.mem.sliceAsBytes(rl.us) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.vq", .{r}), .dtype = .i8, .shape = try a.dupe(usize, &.{ arch.rec_heads, rk_u, arch.rec_cols }), .bytes = std.mem.sliceAsBytes(rl.vq) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.vs", .{r}), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, rk_u }), .bytes = std.mem.sliceAsBytes(rl.vs) });
            }
        }

        // Memory units
        for (self.units.items, 0..) |*u, ui| {
            try appendUnitTensors(a, &tensors, arch, u, ui);
        }

        const nbytes = try store.writeFileAtomic(self.gpa, self.io, path, &meta, tensors.items);
        Io.Dir.cwd().deleteFile(self.io, dpath) catch {};
        self.base_saved = true;
        self.pending_deltas = 0;
        self.unflushed_from = self.units.items.len;
        self.last_bytes = nbytes;
        return nbytes;
    }

    /// The prefix state a load reads or recompiles; freed here on failure, the caller's once returned.
    fn loadPrefix(self: *Bank, lf: *const store.LoadedFile, prefix_len: usize) !RawState {
        const arch = self.backend.arch;
        var pref = try RawState.init(self.gpa, arch, @intCast(prefix_len));
        errdefer pref.deinit(self.gpa);
        var name_buf: [96]u8 = undefined;
        if (lf.getTensor("p.kv.0.k") != null) {
            const p_kv_elems = arch.kvLayerElems(prefix_len);
            for (0..arch.num_attn) |l| {
                const tk = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.kv.{d}.k", .{l})) orelse return error.MissingTensor;
                if (tk.bytes.len != p_kv_elems * 2) return error.BadTensorShape;
                @memcpy(std.mem.sliceAsBytes(pref.kv_k[l]), tk.bytes);
                const tv = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.kv.{d}.v", .{l})) orelse return error.MissingTensor;
                if (tv.bytes.len != p_kv_elems * 2) return error.BadTensorShape;
                @memcpy(std.mem.sliceAsBytes(pref.kv_v[l]), tv.bytes);
            }
            const st_elems = arch.recStateElems();
            for (0..arch.num_rec) |r| {
                const ts = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.rec.{d}.st", .{r})) orelse return error.MissingTensor;
                if (ts.bytes.len != st_elems * 4) return error.BadTensorShape;
                @memcpy(std.mem.sliceAsBytes(pref.rec_state[r]), ts.bytes);
                const tc = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.rec.{d}.cv", .{r})) orelse return error.MissingTensor;
                if (tc.bytes.len != arch.conv_elems * 2) return error.BadTensorShape;
                @memcpy(std.mem.sliceAsBytes(pref.rec_conv[r]), tc.bytes);
            }
        } else {
            if (self.prefix_ids.len != prefix_len) return error.BadLivingWeights;
            try self.backend.compileState(self.backend.ctx, null, self.prefix_ids, &pref);
        }
        return pref;
    }

    /// Loads `<dir>/living-weights.safetensors` (`v6`, `v4`, or legacy `v3`) AND seamlessly replays
    /// any uncompacted frames from `<dir>/living-weights.delta`, recompiling prefix state & rebuilding indexes.
    pub fn loadIfPresent(self: *Bank) !bool {
        const path = try self.filePath(self.gpa);
        defer self.gpa.free(path);
        if (!try store.probeFoldedMemoryFile(self.gpa, self.io, path)) return false; // unreadable errs, not empty

        var lf = try store.readFile(self.gpa, self.io, path);
        defer lf.deinit();

        const fmt = lf.getMeta("format") orelse return error.BadLivingWeights;
        const is_v6 = std.mem.eql(u8, fmt, store.format_tag);
        const is_v4 = std.mem.eql(u8, fmt, store.format_tag_v4);
        const is_v3 = std.mem.eql(u8, fmt, store.format_tag_v3);
        if (!is_v6 and !is_v4 and !is_v3) return error.BadLivingWeights;

        const arch = self.backend.arch;
        const hidden = try parseUsizeMeta(&lf, "hidden");
        const num_attn = try parseUsizeMeta(&lf, "num_attn");
        const num_rec = try parseUsizeMeta(&lf, "num_rec");
        if (hidden != arch.hidden or num_attn != arch.num_attn or num_rec != arch.num_rec) {
            return error.LivingWeightsArchMismatch;
        }

        self.clearMemory();
        errdefer self.clearMemory();

        const prefix_len = try parseUsizeMeta(&lf, "prefix_len");
        const lex_k = try parseUsizeMeta(&lf, "lex_k");
        const lex_cal = try parseUsizeMeta(&lf, "lex_cal");
        const lex_floor = try parseF32Meta(&lf, "lex_floor");
        const sem_k = try parseUsizeMeta(&lf, "sem_k");
        const sem_cal = try parseUsizeMeta(&lf, "sem_cal");
        const sem_floor = try parseF32Meta(&lf, "sem_floor");
        self.next_unit_id = @intCast(try parseUsizeMeta(&lf, "next_uid"));
        self.next_epoch = if (lf.getMeta("next_epoch")) |s| (std.fmt.parseInt(u64, s, 10) catch 1) else 1;

        self.prefix_ids = try lf.copyU32(self.gpa, "frame.prefix_ids");
        self.suffix_ids = try lf.copyU32(self.gpa, "frame.suffix_ids");

        // each read whole before the bank owns it: from then on only clearMemory frees it
        self.w_lex = try loadWhitener(&lf, self.gpa, "w.lex", lex_k, lex_cal, lex_floor, arch.hidden);
        self.w_sem = try loadWhitener(&lf, self.gpa, "w.sem", sem_k, sem_cal, sem_floor, arch.hidden);

        // Recompile static prefix_state on load for v4/v6 (0 bytes on disk!); read from disk for legacy v3
        self.prefix_state = try self.loadPrefix(&lf, prefix_len);
        var name_buf: [96]u8 = undefined;

        // Optional sleep-time consolidated spectral skill steering
        const skill_rk: u16 = if (lf.getMeta("skill_rk")) |s| (std.fmt.parseInt(u16, s, 10) catch 0) else 0;
        const skill_pols: u32 = if (lf.getMeta("skill_pols")) |s| (std.fmt.parseInt(u32, s, 10) catch 0) else 0;
        if (skill_rk > 0 and arch.num_rec > 0) {
            const arr = try self.gpa.alloc(RecLayerData, arch.num_rec);
            var rc_done: usize = 0;
            errdefer {
                for (arr[0..rc_done]) |*r| r.deinit(self.gpa);
                self.gpa.free(arr);
            }
            const rk_u: usize = skill_rk;
            for (0..arch.num_rec) |r| {
                const uq = try lf.copyI8(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.uq", .{r}), arch.rec_heads * arch.rec_rows * rk_u);
                errdefer self.gpa.free(uq);
                const us = try lf.copyF32(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.us", .{r}), arch.rec_heads * arch.rec_rows);
                errdefer self.gpa.free(us);
                const vq = try lf.copyI8(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.vq", .{r}), arch.rec_heads * rk_u * arch.rec_cols);
                errdefer self.gpa.free(vq);
                const vs = try lf.copyF32(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.vs", .{r}), arch.rec_heads * rk_u);
                errdefer self.gpa.free(vs);
                const cv = try self.gpa.alloc(u16, 0);
                arr[r] = .{ .rank = skill_rk, .uq = uq, .us = us, .vq = vq, .vs = vs, .conv = cv };
                rc_done += 1;
            }
            self.skill_rec = arr;
            self.skill_policy_count = skill_pols;
        }

        const units_json = lf.getMeta("units") orelse return error.BadLivingWeights;
        var parse_arena = std.heap.ArenaAllocator.init(self.gpa);
        defer parse_arena.deinit();
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, parse_arena.allocator(), units_json, .{});
        if (parsed != .array) return error.BadLivingWeights;

        for (parsed.array.items, 0..) |item, ui| {
            var u = try loadUnitFromObject(self.gpa, &lf, arch, item, ui);
            errdefer u.deinit(self.gpa);
            if (u.gate.epoch >= self.next_epoch) self.next_epoch = u.gate.epoch + 1;
            try self.units.append(self.gpa, u);
        }

        self.calibrated = true;
        self.base_saved = true;
        self.last_bytes = lf.raw.len;

        // Replay any uncompacted crash-safe delta journal frames (`living-weights.delta`)!
        const dpath = try self.deltaPath(self.gpa);
        defer self.gpa.free(dpath);
        const frames = try store.readDeltaFrames(self.gpa, self.io, dpath);
        defer {
            for (frames) |*fr| fr.deinit(self.gpa);
            self.gpa.free(frames);
        }
        for (frames) |fr| {
            switch (fr.op) {
                .add_unit => {
                    const owned = try self.gpa.dupe(u8, fr.payload);
                    var dlf = store.parseOwnedBytes(self.gpa, owned) catch {
                        self.gpa.free(owned);
                        continue;
                    };
                    defer dlf.deinit();
                    const uj = dlf.getMeta("unit") orelse continue;
                    var ua = std.heap.ArenaAllocator.init(self.gpa);
                    defer ua.deinit();
                    const up = std.json.parseFromSliceLeaky(std.json.Value, ua.allocator(), uj, .{}) catch continue;
                    var u = loadUnitFromObject(self.gpa, &dlf, arch, up, 0) catch continue;
                    if (u.unit_id >= self.next_unit_id) self.next_unit_id = u.unit_id + 1;
                    if (u.gate.epoch >= self.next_epoch) self.next_epoch = u.gate.epoch + 1;
                    self.units.append(self.gpa, u) catch |e| {
                        u.deinit(self.gpa); // not the bank's yet: freed here, once
                        return e;
                    };
                },
                .forget_parent => {
                    if (fr.payload.len >= 4) {
                        const pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                        _ = self.forgetParentNoRebuild(pid);
                    }
                },
                .supersede_parent => {
                    if (fr.payload.len >= 8) {
                        const old_pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                        const new_pid = std.mem.readInt(u32, fr.payload[4..8], .little);
                        for (self.units.items) |*u| {
                            if (u.parent_id == old_pid and u.gate.superseded_by == 0) {
                                u.gate.superseded_by = new_pid;
                            }
                        }
                    }
                },
                .reinforce_parent => {
                    if (fr.payload.len >= 4) {
                        const pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                        _ = self.reinforceParent(pid);
                    }
                },
            }
            self.pending_deltas += 1;
            self.last_bytes += fr.payload.len + 24;
        }
        self.unflushed_from = self.units.items.len;
        try self.rebuildBridges();
        return true;
    }

    fn forgetParentNoRebuild(self: *Bank, parent_id: u32) u32 {
        var removed: u32 = 0;
        var i: usize = 0;
        while (i < self.units.items.len) {
            if (self.units.items[i].parent_id == parent_id) {
                self.units.items[i].deinit(self.gpa);
                _ = self.units.orderedRemove(i);
                removed += 1;
            } else {
                if (self.units.items[i].gate.superseded_by == parent_id) {
                    self.units.items[i].gate.superseded_by = 0;
                }
                i += 1;
            }
        }
        return removed;
    }
};

fn appendUnitTensors(a: Allocator, tensors: *std.ArrayList(store.TensorSpec), arch: ArchSpec, u: *const Unit, ui: usize) !void {
    if (u.gate.n_lex > 0) {
        const lq = try a.alloc(i8, u.gate.n_lex * arch.hidden);
        const ls = try a.alloc(f32, u.gate.n_lex);
        quant.quantizeRowsF32(u.gate.lex_protos, u.gate.n_lex, arch.hidden, lq, ls);
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.gl.q", .{ui}), .dtype = .i8, .shape = try a.dupe(usize, &.{ u.gate.n_lex, arch.hidden }), .bytes = std.mem.sliceAsBytes(lq) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.gl.s", .{ui}), .dtype = .f32, .shape = try a.dupe(usize, &.{u.gate.n_lex}), .bytes = std.mem.sliceAsBytes(ls) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.gl.w", .{ui}), .dtype = .f32, .shape = try a.dupe(usize, &.{u.gate.n_lex}), .bytes = std.mem.sliceAsBytes(u.gate.lex_weights) });
    }
    if (u.gate.n_sem > 0) {
        const sq = try a.alloc(i8, u.gate.n_sem * arch.hidden);
        const ss = try a.alloc(f32, u.gate.n_sem);
        quant.quantizeRowsF32(u.gate.sem_protos, u.gate.n_sem, arch.hidden, sq, ss);
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.gs.q", .{ui}), .dtype = .i8, .shape = try a.dupe(usize, &.{ u.gate.n_sem, arch.hidden }), .bytes = std.mem.sliceAsBytes(sq) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.gs.s", .{ui}), .dtype = .f32, .shape = try a.dupe(usize, &.{u.gate.n_sem}), .bytes = std.mem.sliceAsBytes(ss) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.gs.w", .{ui}), .dtype = .f32, .shape = try a.dupe(usize, &.{u.gate.n_sem}), .bytes = std.mem.sliceAsBytes(u.gate.sem_weights) });
    }
    for (0..arch.num_attn) |l| {
        switch (u.kv[l]) {
            .q4 => |d| {
                const sh_q4 = try a.dupe(usize, &.{ arch.kv_heads, u.n_tokens, arch.head_dim / 2 });
                const sh_s4 = try a.dupe(usize, &.{ arch.kv_heads, u.n_tokens, arch.head_dim / quant.q4_group });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.kq4", .{ ui, l }), .dtype = .u8, .shape = sh_q4, .bytes = d.kq });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.ks4", .{ ui, l }), .dtype = .bf16, .shape = sh_s4, .bytes = std.mem.sliceAsBytes(d.ks) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.vq4", .{ ui, l }), .dtype = .u8, .shape = sh_q4, .bytes = d.vq });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.vs4", .{ ui, l }), .dtype = .bf16, .shape = sh_s4, .bytes = std.mem.sliceAsBytes(d.vs) });
            },
            .q8 => |d| {
                const sh_kv = try a.dupe(usize, &.{ arch.kv_heads, u.n_tokens, arch.head_dim });
                const sh_s = try a.dupe(usize, &.{ arch.kv_heads, u.n_tokens });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.kq", .{ ui, l }), .dtype = .i8, .shape = sh_kv, .bytes = std.mem.sliceAsBytes(d.kq) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.ks", .{ ui, l }), .dtype = .f32, .shape = sh_s, .bytes = std.mem.sliceAsBytes(d.ks) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.vq", .{ ui, l }), .dtype = .i8, .shape = sh_kv, .bytes = std.mem.sliceAsBytes(d.vq) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.kv.{d}.vs", .{ ui, l }), .dtype = .f32, .shape = sh_s, .bytes = std.mem.sliceAsBytes(d.vs) });
            },
        }
    }
    if (u.rec.len == arch.num_rec) {
        for (0..arch.num_rec) |r| {
            const rk: usize = u.rec[r].rank;
            if (rk == 0) continue;
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.uq", .{ ui, r }), .dtype = .i8, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows, rk }), .bytes = std.mem.sliceAsBytes(u.rec[r].uq) });
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.us", .{ ui, r }), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows }), .bytes = std.mem.sliceAsBytes(u.rec[r].us) });
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.vq", .{ ui, r }), .dtype = .i8, .shape = try a.dupe(usize, &.{ arch.rec_heads, rk, arch.rec_cols }), .bytes = std.mem.sliceAsBytes(u.rec[r].vq) });
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.vs", .{ ui, r }), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, rk }), .bytes = std.mem.sliceAsBytes(u.rec[r].vs) });
            if (u.rec[r].conv.len == arch.conv_elems) {
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "u.{d}.rc.{d}.cv", .{ ui, r }), .dtype = .bf16, .shape = try a.dupe(usize, &.{arch.conv_elems}), .bytes = std.mem.sliceAsBytes(u.rec[r].conv) });
            }
        }
    }
}

fn loadUnitFromObject(gpa: Allocator, lf: *const store.LoadedFile, arch: ArchSpec, item: std.json.Value, ui: usize) !Unit {
    if (item != .object) return error.BadLivingWeights;
    const o = item.object;
    const uid: u32 = @intCast((o.get("uid") orelse return error.BadLivingWeights).integer);
    const pid: u32 = @intCast((o.get("pid") orelse return error.BadLivingWeights).integer);
    const n_tok: u32 = @intCast((o.get("tok") orelse return error.BadLivingWeights).integer);
    const n_lex: usize = @intCast((o.get("nl") orelse return error.BadLivingWeights).integer);
    const n_body_lex: usize = if (o.get("nbl")) |v| @intCast(v.integer) else n_lex;
    const n_sem: usize = @intCast((o.get("ns") orelse return error.BadLivingWeights).integer);
    const rk: u16 = @intCast((o.get("rk") orelse return error.BadLivingWeights).integer);
    const kv_b: u8 = if (o.get("kb")) |v| @intCast(v.integer) else 8;
    const has_conv: bool = if (o.get("cv")) |v| v.bool else (rk > 0);
    const is_sm = (o.get("sm") orelse return error.BadLivingWeights).bool;
    const is_st = if (o.get("st")) |v| v.bool else false;
    const is_dg = if (o.get("dg")) |v| v.bool else false;
    const epoch: u64 = if (o.get("ep")) |v| @intCast(v.integer) else 0;
    const sup: u32 = if (o.get("sup")) |v| @intCast(v.integer) else 0;
    const rf: u16 = if (o.get("rf")) |v| @intCast(v.integer) else 1;
    const nsem_val = o.get("nsem") orelse return error.BadLivingWeights;
    const neg_sem: f32 = switch (nsem_val) {
        .float => |fv| @floatCast(fv),
        .integer => |iv| @floatFromInt(iv),
        else => return error.BadLivingWeights,
    };

    var name_buf: [96]u8 = undefined;
    const gl_v = if (n_lex > 0) try loadUnitProtoMaybeQ8(lf, gpa, ui, "gl", n_lex, arch.hidden) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gl_v);
    const gl_w = if (n_lex > 0) try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.gl.w", .{ui}), n_lex) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gl_w);
    const gs_v = if (n_sem > 0) try loadUnitProtoMaybeQ8(lf, gpa, ui, "gs", n_sem, arch.hidden) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gs_v);
    const gs_w = if (n_sem > 0) try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.gs.w", .{ui}), n_sem) else try gpa.alloc(f32, 0);
    errdefer gpa.free(gs_w);

    const kv = try gpa.alloc(KvLayerData, arch.num_attn);
    var kv_done: usize = 0;
    errdefer {
        for (kv[0..kv_done]) |*k| k.deinit(gpa);
        gpa.free(kv);
    }
    const kv_rows = arch.kv_heads * n_tok;
    for (0..arch.num_attn) |l| {
        if (kv_b == 4) {
            const packed_elems = kv_rows * (arch.head_dim / 2);
            const group_scales = kv_rows * (arch.head_dim / quant.q4_group);
            const kq = try lf.copyU8(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.kq4", .{ ui, l }), packed_elems);
            errdefer gpa.free(kq);
            const ks = try lf.copyBf16(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.ks4", .{ ui, l }), group_scales);
            errdefer gpa.free(ks);
            const vq = try lf.copyU8(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.vq4", .{ ui, l }), packed_elems);
            errdefer gpa.free(vq);
            const vs = try lf.copyBf16(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.vs4", .{ ui, l }), group_scales);
            errdefer gpa.free(vs);
            kv[l] = .{ .q4 = .{ .kq = kq, .ks = ks, .vq = vq, .vs = vs } };
        } else {
            const kv_elems = kv_rows * arch.head_dim;
            const kq = try lf.copyI8(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.kq", .{ ui, l }), kv_elems);
            errdefer gpa.free(kq);
            const ks = try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.ks", .{ ui, l }), kv_rows);
            errdefer gpa.free(ks);
            const vq = try lf.copyI8(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.vq", .{ ui, l }), kv_elems);
            errdefer gpa.free(vq);
            const vs = try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.kv.{d}.vs", .{ ui, l }), kv_rows);
            errdefer gpa.free(vs);
            kv[l] = .{ .q8 = .{ .kq = kq, .ks = ks, .vq = vq, .vs = vs } };
        }
        kv_done += 1;
    }

    const rec: []RecLayerData = if (rk == 0 or arch.num_rec == 0)
        try gpa.alloc(RecLayerData, 0)
    else blk: {
        const arr = try gpa.alloc(RecLayerData, arch.num_rec);
        var rc_done: usize = 0;
        errdefer {
            for (arr[0..rc_done]) |*r| r.deinit(gpa);
            gpa.free(arr);
        }
        const rk_u: usize = rk;
        for (0..arch.num_rec) |r| {
            const uq = try lf.copyI8(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.uq", .{ ui, r }), arch.rec_heads * arch.rec_rows * rk_u);
            errdefer gpa.free(uq);
            const us = try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.us", .{ ui, r }), arch.rec_heads * arch.rec_rows);
            errdefer gpa.free(us);
            const vq = try lf.copyI8(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.vq", .{ ui, r }), arch.rec_heads * rk_u * arch.rec_cols);
            errdefer gpa.free(vq);
            const vs = try lf.copyF32(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.vs", .{ ui, r }), arch.rec_heads * rk_u);
            errdefer gpa.free(vs);
            const cv = if (has_conv)
                try lf.copyBf16(gpa, try std.fmt.bufPrint(&name_buf, "u.{d}.rc.{d}.cv", .{ ui, r }), arch.conv_elems)
            else
                try gpa.alloc(u16, 0);
            errdefer gpa.free(cv);
            arr[r] = .{ .rank = rk, .uq = uq, .us = us, .vq = vq, .vs = vs, .conv = cv };
            rc_done += 1;
        }
        break :blk arr;
    };
    errdefer {
        for (rec) |*r| r.deinit(gpa);
        gpa.free(rec);
    }

    var ug = gate.UnitGate{
        .dim = arch.hidden,
        .lex_protos = gl_v,
        .lex_weights = gl_w,
        .n_lex = n_lex,
        .n_body_lex = n_body_lex,
        .sem_protos = gs_v,
        .sem_weights = gs_w,
        .n_sem = n_sem,
        .neg_sem_max = neg_sem,
        .is_small_fact = is_sm,
        .is_standing = is_st,
        .is_digest = is_dg,
        .epoch = epoch,
        .superseded_by = sup,
        .reinforcements = rf,
    };
    // its four arrays stay this function's cleanups' to free: a failure frees only the summaries the gate made
    ug.computeSummaries(gpa) catch |e| {
        if (ug.lex_summary.len > 0) gpa.free(ug.lex_summary);
        if (ug.sem_summary.len > 0) gpa.free(ug.sem_summary);
        return e;
    };

    return .{
        .unit_id = uid,
        .parent_id = pid,
        .n_tokens = n_tok,
        .gate = ug,
        .kv = kv,
        .rec = rec,
    };
}

/// A saved whitener read whole; on failure nothing it read is kept, on success the caller owns every array.
fn loadWhitener(lf: *const store.LoadedFile, gpa: Allocator, comptime name: []const u8, k: usize, n_cal: usize, floor: f32, hidden: usize) !whitener.LayerWhitener {
    const cal = try loadMatrixMaybeQ8(lf, gpa, name ++ ".cal", n_cal, hidden);
    errdefer gpa.free(cal);
    const mean = try lf.copyF32(gpa, name ++ ".mean", hidden);
    errdefer gpa.free(mean);
    const basis = try lf.copyF32(gpa, name ++ ".basis", k * hidden);
    return .{ .dim = hidden, .k_pcs = k, .n_cal = n_cal, .mean = mean, .basis = basis, .cal_tokens = cal, .norm_floor = floor };
}

fn loadMatrixMaybeQ8(lf: *const store.LoadedFile, gpa: Allocator, base: []const u8, rows: usize, cols: usize) ![]f32 {
    var nb: [96]u8 = undefined;
    const qname = try std.fmt.bufPrint(&nb, "{s}.q", .{base});
    if (lf.getTensor(qname)) |tq| {
        var sb: [96]u8 = undefined;
        const ts = lf.getTensor(try std.fmt.bufPrint(&sb, "{s}.s", .{base})) orelse return error.MissingTensor;
        if (tq.dtype != .i8 or tq.bytes.len != rows * cols or ts.dtype != .f32 or ts.bytes.len != rows * 4) return error.BadTensorShape;
        const out = try gpa.alloc(f32, rows * cols);
        errdefer gpa.free(out);
        const q_slice: []const i8 = @ptrCast(tq.bytes);
        const s_slice = try gpa.alloc(f32, rows);
        defer gpa.free(s_slice);
        @memcpy(std.mem.sliceAsBytes(s_slice), ts.bytes);
        quant.dequantizeRowsF32(q_slice, s_slice, rows, cols, out);
        return out;
    }
    return lf.copyF32(gpa, base, rows * cols);
}

fn loadUnitProtoMaybeQ8(lf: *const store.LoadedFile, gpa: Allocator, ui: usize, kind: []const u8, rows: usize, cols: usize) ![]f32 {
    var qb: [96]u8 = undefined;
    const qname = try std.fmt.bufPrint(&qb, "u.{d}.{s}.q", .{ ui, kind });
    if (lf.getTensor(qname)) |tq| {
        var sb: [96]u8 = undefined;
        const ts = lf.getTensor(try std.fmt.bufPrint(&sb, "u.{d}.{s}.s", .{ ui, kind })) orelse return error.MissingTensor;
        if (tq.dtype != .i8 or tq.bytes.len != rows * cols or ts.dtype != .f32 or ts.bytes.len != rows * 4) return error.BadTensorShape;
        const out = try gpa.alloc(f32, rows * cols);
        errdefer gpa.free(out);
        const q_slice: []const i8 = @ptrCast(tq.bytes);
        const s_slice = try gpa.alloc(f32, rows);
        defer gpa.free(s_slice);
        @memcpy(std.mem.sliceAsBytes(s_slice), ts.bytes);
        quant.dequantizeRowsF32(q_slice, s_slice, rows, cols, out);
        return out;
    }
    var vb: [96]u8 = undefined;
    return lf.copyF32(gpa, try std.fmt.bufPrint(&vb, "u.{d}.{s}.v", .{ ui, kind }), rows * cols);
}

/// Parallelized per-layer SVD + Q8 compression of recurrent state deltas (`S_after - S_prefix`).
fn compressRecurrentLayers(
    gpa: Allocator,
    arch: ArchSpec,
    pref: *const RawState,
    body: *const RawState,
    rank: usize,
) ![]RecLayerData {
    const rec = try gpa.alloc(RecLayerData, arch.num_rec);
    var made: usize = 0;
    errdefer {
        for (rec[0..made]) |*r| r.deinit(gpa);
        gpa.free(rec);
    }
    const u_elems = arch.rec_heads * arch.rec_rows * rank;
    const u_scales = arch.rec_heads * arch.rec_rows;
    const v_elems = arch.rec_heads * rank * arch.rec_cols;
    const v_scales = arch.rec_heads * rank;

    for (0..arch.num_rec) |r| {
        const uq = try gpa.alloc(i8, u_elems);
        errdefer gpa.free(uq);
        const us = try gpa.alloc(f32, u_scales);
        errdefer gpa.free(us);
        const vq = try gpa.alloc(i8, v_elems);
        errdefer gpa.free(vq);
        const vs = try gpa.alloc(f32, v_scales);
        errdefer gpa.free(vs);
        const cv = if (arch.keep_conv) try gpa.dupe(u16, body.rec_conv[r]) else try gpa.alloc(u16, 0);
        errdefer gpa.free(cv);
        rec[r] = .{ .rank = @intCast(rank), .uq = uq, .us = us, .vq = vq, .vs = vs, .conv = cv };
        made += 1;
    }
    if (arch.num_rec == 0 or rank == 0) return rec;

    const WorkerCtx = struct {
        arch: ArchSpec,
        pref: *const RawState,
        body: *const RawState,
        rec: []RecLayerData,
        rank: usize,
        next_layer: std.atomic.Value(usize),
        failed: std.atomic.Value(bool),

        fn run(ctx: *@This()) void {
            const alloc = std.heap.page_allocator;
            const head_mat = ctx.arch.rec_rows * ctx.arch.rec_cols;
            const delta_buf = alloc.alloc(f32, head_mat) catch {
                ctx.failed.store(true, .release);
                return;
            };
            defer alloc.free(delta_buf);
            const u_tmp = alloc.alloc(f32, ctx.arch.rec_rows * ctx.rank) catch {
                ctx.failed.store(true, .release);
                return;
            };
            defer alloc.free(u_tmp);
            const v_tmp = alloc.alloc(f32, ctx.rank * ctx.arch.rec_cols) catch {
                ctx.failed.store(true, .release);
                return;
            };
            defer alloc.free(v_tmp);

            var scratch_arena = std.heap.ArenaAllocator.init(alloc);
            defer scratch_arena.deinit();

            while (true) {
                const r = ctx.next_layer.fetchAdd(1, .acq_rel);
                if (r >= ctx.arch.num_rec) break;
                const rl = &ctx.rec[r];
                for (0..ctx.arch.rec_heads) |h| {
                    _ = scratch_arena.reset(.retain_capacity);
                    const p_st = ctx.pref.rec_state[r][h * head_mat ..][0..head_mat];
                    const b_st = ctx.body.rec_state[r][h * head_mat ..][0..head_mat];
                    for (p_st, b_st, delta_buf) |pv, bv, *dv| dv.* = bv - pv;
                    _ = quant.randomizedSvdBalanced(
                        scratch_arena.allocator(),
                        delta_buf,
                        ctx.arch.rec_rows,
                        ctx.arch.rec_cols,
                        ctx.rank,
                        u_tmp,
                        v_tmp,
                    ) catch {
                        ctx.failed.store(true, .release);
                        return;
                    };
                    quant.quantizeRowsF32(
                        u_tmp,
                        ctx.arch.rec_rows,
                        ctx.rank,
                        rl.uq[h * ctx.arch.rec_rows * ctx.rank ..][0 .. ctx.arch.rec_rows * ctx.rank],
                        rl.us[h * ctx.arch.rec_rows ..][0..ctx.arch.rec_rows],
                    );
                    quant.quantizeRowsF32(
                        v_tmp,
                        ctx.rank,
                        ctx.arch.rec_cols,
                        rl.vq[h * ctx.rank * ctx.arch.rec_cols ..][0 .. ctx.rank * ctx.arch.rec_cols],
                        rl.vs[h * ctx.rank ..][0..ctx.rank],
                    );
                }
            }
        }
    };

    var ctx = WorkerCtx{
        .arch = arch,
        .pref = pref,
        .body = body,
        .rec = rec,
        .rank = rank,
        .next_layer = .init(0),
        .failed = .init(false),
    };
    const n_threads = @min(arch.num_rec, @max(@as(usize, 1), std.Thread.getCpuCount() catch 4));
    if (n_threads <= 1) {
        WorkerCtx.run(&ctx);
    } else {
        const threads = try gpa.alloc(std.Thread, n_threads - 1);
        defer gpa.free(threads);
        var spawned: usize = 0;
        for (0..n_threads - 1) |t| {
            threads[t] = std.Thread.spawn(.{}, WorkerCtx.run, .{&ctx}) catch break;
            spawned += 1;
        }
        WorkerCtx.run(&ctx);
        for (threads[0..spawned]) |th| th.join();
    }
    if (ctx.failed.load(.acquire)) return error.OutOfMemory;
    return rec;
}

fn parseUsizeMeta(lf: *const store.LoadedFile, key: []const u8) !usize {
    const s = lf.getMeta(key) orelse return error.MissingMetadata;
    return std.fmt.parseInt(usize, s, 10) catch return error.BadLivingWeights;
}

fn parseF32Meta(lf: *const store.LoadedFile, key: []const u8) !f32 {
    const s = lf.getMeta(key) orelse return error.MissingMetadata;
    return std.fmt.parseFloat(f32, s) catch return error.BadLivingWeights;
}
