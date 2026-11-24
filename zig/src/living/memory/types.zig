//! Living-memory bank's plain types: arch spec, raw state, backend vtable, units, teach input and reports.
const std = @import("std");
const whitener = @import("whitener.zig");
const gate = @import("gate.zig");
const Allocator = std.mem.Allocator;

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
    /// Keep per-unit conv buffers (false omits conv on multi-chunk docs; short/standing units may keep if true).
    keep_conv: bool = false,

    /// Every geometry field a saved unit depends on, as text: a bank loads only into the architecture that wrote it.
    pub fn signature(arch: ArchSpec, a: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(a, "{s};h{d};l{d};x{d};s{d};a{d};kv{d};hd{d};r{d};t{x};n{d};rh{d};rr{d};rc{d};c{d}", .{
            arch.family.name(), arch.hidden,                         arch.num_layers, arch.layer_lex, arch.layer_sem, arch.num_attn, arch.kv_heads,   arch.head_dim,
            arch.rotary_dim,    @as(u64, @bitCast(arch.rope_theta)), arch.num_rec,    arch.rec_heads, arch.rec_rows,  arch.rec_cols, arch.conv_elems,
        });
    }

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

/// CPU-side state for a family's Metal adapter: KV `[kv_heads, len, head_dim]` BF16 post-RoPE, rec F32, conv BF16.
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

/// Family adapter vtable of four functions, built on each Metal backend's forward pass and snapshot save/restore.
pub const Backend = struct {
    ctx: *anyopaque,
    arch: ArchSpec,
    /// Partial forward on `ids` through layer `layer_sem`, writing `[ids.len * hidden]` f32 into `out_lex` & `out_sem`.
    extractHidden: *const fn (ctx: *anyopaque, ids: []const u32, layer_lex: usize, layer_sem: usize, out_lex: []f32, out_sem: []f32) anyerror!void,
    /// Full forward on `ids` from `base` (null = empty); writes only new KV rows and final recurrent states to `out`.
    compileState: *const fn (ctx: *anyopaque, base: ?*const RawState, ids: []const u32, out: *RawState) anyerror!void,
    /// Uploads composed `RawState` (prefix + active units) to a scratch cache, runs `suffix_ids`, returns a Snap.
    buildSnap: *const fn (ctx: *anyopaque, composed: *const RawState, suffix_ids: []const u32) anyerror!SnapHandle,
    /// Destroy a Snap previously created by `buildSnap`.
    dropSnap: *const fn (ctx: *anyopaque, snap: *anyopaque) void,
};

/// One attention layer's K and V as the model computed them: BF16, K rotated at rows `prefix_len + t`.
pub const KvLayerData = struct {
    k: []u16, // [kv_heads * n_tokens * head_dim]
    v: []u16,

    pub fn deinit(self: *KvLayerData, gpa: Allocator) void {
        gpa.free(self.k);
        gpa.free(self.v);
    }
};

pub const RecLayerData = struct {
    rank: u16,
    u: []f32, // [rec_heads * rec_rows * rank], F32 as the SVD computed it
    v: []f32, // [rec_heads * rank * rec_cols]
    conv: []u16, // [conv_elems] bf16 OR empty when `keep_conv == false`

    pub fn deinit(self: *RecLayerData, gpa: Allocator) void {
        gpa.free(self.u);
        gpa.free(self.v);
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
    staged: bool = false, // held out of routing until its lesson is published (Bank.publishParent)
};

pub const ConsolidateReport = struct {
    units: u32 = 0,
    pruned_superseded: u32 = 0,
    consolidated_policies: u32 = 0,
    clusters: u32 = 0,
    compacted_deltas: u32 = 0,
    bytes: u64 = 0,
};
