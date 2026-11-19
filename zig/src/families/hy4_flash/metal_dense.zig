//! The native Metal attention and dense feed-forward layer passes for Hy4-flash.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("row_projection");
const dense = @import("draft_ops");
const core = @import("core");
const cfg = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_gpu = @import("ihc_metal.zig");
const mla_gpu = @import("mla_metal.zig");
const mha = @import("mla_attention.zig");
const shape_ = @import("mla_weights.zig");
const weights = @import("model_weights.zig");
const rope = @import("rope.zig");
const gpu_qkv = @import("metal_tensor_qkv.zig");
const lane = @import("lane_projection");
const index = @import("metal_index.zig");

pub const Ref = mla_gpu.Ref;

pub const Buffers = struct {
    a: std.mem.Allocator,
    d: mtl.Device,
    list: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Buffers, bytes: usize) !mtl.Buffer {
        const b = try self.d.buffer(bytes, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        try self.list.append(self.a, b);
        return b;
    }

    pub fn upload(self: *Buffers, bytes: []const u8) !mtl.Buffer {
        const b = try self.alloc(bytes.len);
        @memcpy(b.slice(u8, bytes.len), bytes);
        return b;
    }

    pub fn clear(self: *Buffers) void {
        for (self.list.items) |b| b.deinit();
        self.list.deinit(self.a);
    }

    pub fn matrix(self: *Buffers, w: core.quant.mlx.Host) !affine.Weights {
        const view: affine.Weights = .{
            .w = try self.upload(w.words.bytes),
            .scales = try self.upload(w.scales.bytes),
            .biases = try self.upload(w.biases.bytes),
            .n = w.words.shape[0],
            .k = w.scales.shape[1] * w.group,
            .bits = w.bits,
            .group = w.group,
        };
        try view.validate();
        return view;
    }

    /// A dense projection's weights: packed for the tensor units' lane projection when `tensors`, else the row kernels'.
    pub fn linear(self: *Buffers, w: core.quant.mlx.Host, tensors: bool) !Linear {
        const n: u32 = @intCast(w.words.shape[0]);
        const k: u32 = @intCast(w.scales.shape[1] * w.group);
        if (tensors and n % 32 == 0 and w.group == 64 and (w.bits == 4 or w.bits == 8) and std.c.getenv("TF_HY4_DENSE_ROWS") == null) {
            const packed_ = try gpu_qkv.Packed.upload(self, &.{w}, gpu_qkv.denseLayout(w.bits, n, k));
            // decode-sized passes read the checkpoint layout through the decode kernels
            const plain = if (decodeRows() > 0) try self.matrix(w) else null;
            return .{ .tensor = .{ .w = packed_.w, .sb = packed_.sb, .n = n, .k = k, .plain = plain } };
        }
        return .{ .portable = try self.matrix(w) };
    }

    pub fn rows(self: *Buffers, count: u32, columns: u32) !Ref {
        return .{ .buffer = try self.alloc(@as(usize, count) * columns * @sizeOf(u16)) };
    }
};

/// A dense affine projection [n, k]: the core row kernels, or the tensor units' lane projection over packed weights.
/// Passes of at most this many rows (one-row decode, the drafter's columns) take the decode GEMV: fp32 sums, each
/// weight read once, in a different order from the tensor units' (TF_HY4_DECODE_ROWS, default 2, where the GEMV is
/// the faster; 0 keeps the tensor units everywhere). Quality is the same either way (fp32 sums, NLL-checked).
/// A probe's override of decodeRows (kernel checks).
pub var decode_rows_override: ?u32 = null;

fn envRows(name: [*:0]const u8, default: u32) u32 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

/// Routed-expert passes of at most this many rows (6: the lane verify widths) run every (row, pick) pair as GEMVs
/// (TF_HY4_EXPERT_PAIRS); against the tensor gathers 4 rows 24.6 -> 23.1 ms, 5 rows 26.6 -> 24.4, 6 rows 27.8 -> 26.4.
pub fn expertPairRows() u32 {
    return 0; // row-invariant bits: every pass takes the tensor gathers the prompt chunks take
}

/// Attention passes of at most this many rows (4 at most) read kv_b's affine codes (TF_HY4_MLA_Q; 0: the BF16 copies).
/// One row by default: 15.85 -> 15.29 ms a token; from two rows the fp32 dequant is compute-bound, slower than BF16
/// on tensor units (4 rows: 23.0 -> 24.2 ms).
pub fn mlaRows() u32 {
    return 0; // row-invariant bits: every pass reads the BF16 copies on the tensor units, as prompt chunks do
}

/// Larger passes up to this many rows (at most one router window, 16) run the grouped GEMVs (TF_HY4_EXPERT_GROUPS).
pub fn expertGroupRows() u32 {
    return @min(envRows("TF_HY4_EXPERT_GROUPS", 0), 16);
}

pub fn decodeRows() u32 {
    return decode_rows_override orelse 0; // row-invariant bits: the lane projection at every row count
}

pub const Linear = union(enum) {
    portable: affine.Weights,
    tensor: struct { w: lane.Ref, sb: lane.Ref, n: u32, k: u32, plain: ?affine.Weights = null },

    /// The checkpoint-layout weights the decode kernels read, when this projection has them.
    fn plainWeights(self: Linear) ?affine.Weights {
        return switch (self) {
            .portable => |w| w,
            .tensor => |t| t.plain,
        };
    }

    pub fn outputs(self: Linear) u32 {
        return switch (self) {
            .portable => |w| @intCast(w.n),
            .tensor => |t| t.n,
        };
    }
};

/// The lane projections for every dense shape a model has, made once at load, and one group-sums buffer that the
/// projections reading the same input share (sums once, then every main kernel without a barrier between them).
pub const Linears = struct {
    shapes: [16][2]u32 = undefined,
    projections: [16]lane.Projection = undefined,
    /// The same projections adding their own group sums (4-bit reg layouts), for small passes.
    inlines: [16]?lane.Projection = @splat(null),
    count: usize = 0,
    sums: ?mtl.Buffer = null,
    const sum_rows = gpu_qkv.prompt_rows;

    pub fn load(a: std.mem.Allocator, device: mtl.Device, t: cfg.Text, bits: u8) !Linears {
        var out: Linears = .{};
        errdefer out.deinit();
        const v = shape_.value_width;
        const wanted = [_][2]u32{
            .{ t.hidden_size, t.swa_num_attention_heads * v },
            .{ t.hidden_size, t.num_attention_heads * v },
            .{ t.routed_expert_hidden_size, t.hidden_size },
            .{ t.moe_intermediate_size * t.n_shared_experts, t.hidden_size },
            .{ t.hidden_size, t.routed_expert_hidden_size },
            .{ t.hidden_size, t.moe_intermediate_size * t.n_shared_experts },
            .{ t.intermediate_size, t.hidden_size },
            .{ t.hidden_size, t.intermediate_size },
            .{ t.vocab_size, t.hidden_size },
            // Engram's value and stream keys over a lookup row: every column's slice of the n-gram embedding
            .{ t.hidden_size, @import("engram.zig").columns * (t.engram_n_embed_per_ngram / @import("engram.zig").heads) },
        };
        var groups: usize = 0;
        for (wanted) |nk| {
            if (out.find(nk[0], nk[1]) != null or nk[0] % 32 != 0) continue;
            out.projections[out.count] = try lane.Projection.init(a, device, gpu_qkv.denseLayout(bits, nk[0], nk[1]));
            out.inlines[out.count] = if (gpu_qkv.withInlineSums(gpu_qkv.denseLayout(bits, nk[0], nk[1]))) |l| try lane.Projection.init(a, device, l) else null;
            out.shapes[out.count] = nk;
            out.count += 1;
            groups = @max(groups, nk[1] / 64);
        }
        out.sums = try device.buffer(sum_rows * groups * 4, mtl.ResourceOptions.shared);
        return out;
    }

    pub fn deinit(self: *Linears) void {
        for (self.projections[0..self.count]) |p| p.deinit();
        for (self.inlines[0..self.count]) |p| if (p) |x| x.deinit();
        if (self.sums) |b| b.deinit();
        self.count = 0;
    }

    fn find(self: *const Linears, n: u32, k: u32) ?*const lane.Projection {
        for (self.shapes[0..self.count], self.projections[0..self.count]) |nk, *p| if (nk[0] == n and nk[1] == k) return p;
        return null;
    }

    fn findInline(self: *const Linears, n: u32, k: u32) ?*const lane.Projection {
        for (self.shapes[0..self.count], self.inlines[0..self.count]) |nk, *p| if (nk[0] == n and nk[1] == k) if (p.*) |*x| return x;
        return null;
    }
};

pub const Merge = struct {
    project: mtl.Buffer,
    bias: mtl.Buffer,
    scale: mtl.Buffer,
    shape: ihc.Shape,

    pub fn upload(b: *Buffers, w: ihc.Weights) !Merge {
        return .{
            .project = try b.upload(w.project.bytes),
            .bias = try b.upload(w.bias.bytes),
            .scale = try b.upload(w.scale.bytes),
            .shape = w.shape,
        };
    }
};

const InputWeights = union(enum) {
    portable: struct { q_a: affine.Weights, q_b: affine.Weights, kv_a: affine.Weights, gates: affine.Weights },
    tensor: struct { qkv: gpu_qkv.Packed, q_b: gpu_qkv.Packed, flavor: usize },
};

pub const Attention = struct {
    own: Buffers,
    hidden: u32,
    query_rank: u32,
    heads: u32,
    epsilon: f32,
    rope: rope.Parameters,
    merge: Merge,
    norm: mtl.Buffer,
    inputs: InputWeights,
    q_norm: mtl.Buffer,
    kv_norm: mtl.Buffer,
    o: Linear,
    o_norm: mtl.Buffer,
    wk: mtl.Buffer,
    wv: mtl.Buffer,
    /// kv_b's affine codes, scales and biases for the decode GEMVs (null under TF_HY4_MLA_BF16).
    kv_q: ?[3]mtl.Buffer = null,
    bits: u8 = 4,
    sinks: mtl.Buffer,
    /// A sparse layer's compressed indexer (null on sliding layers).
    index: ?index.Weights = null,

    pub fn upload(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, t: cfg.Text) !Attention {
        return uploadForModel(a, device, w, t, false);
    }

    pub fn uploadForModel(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, t: cfg.Text, use_tensors: bool) !Attention {
        if (w.attention.owns_output) return error.Hy4MetalAttentionNeedsFullHeadRange;
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const heads = w.attention.heads.count;
        const bits = w.project_query.bits;
        const inputs: InputWeights = if (use_tensors)
            .{ .tensor = .{
                .qkv = try gpu_qkv.Packed.upload(&b, &.{ w.project_query, w.project_kv, w.attention.gate }, gpu_qkv.inputLayout(t, bits, heads)),
                .q_b = try gpu_qkv.Packed.upload(&b, &.{w.attention.query}, gpu_qkv.queryLayout(t, bits, heads)),
                .flavor = gpu_qkv.flavor(t, heads),
            } }
        else
            .{ .portable = .{
                .q_a = try b.matrix(w.project_query),
                .q_b = try b.matrix(w.attention.query),
                .kv_a = try b.matrix(w.project_kv),
                .gates = try b.matrix(w.attention.gate),
            } };
        const out: Attention = .{
            .own = undefined,
            .hidden = t.hidden_size,
            .query_rank = t.q_lora_rank,
            .heads = heads,
            .epsilon = t.rms_norm_eps,
            .rope = w.position,
            .merge = try Merge.upload(&b, w.attention_hc),
            .norm = try b.upload(w.attention_norm.bytes),
            .inputs = inputs,
            .q_norm = try b.upload(w.norm_query.bytes),
            .kv_norm = try b.upload(w.norm_kv.bytes),
            .o = try b.linear(w.attention.output, use_tensors),
            .o_norm = try b.upload(w.attention_output_norm.bytes),
            .wk = try b.upload(std.mem.sliceAsBytes(w.absorbed.key)),
            .wv = try b.upload(std.mem.sliceAsBytes(w.absorbed.value)),
            .bits = if (w.absorbed.packed_rows) |p| p.bits else 4,
            .kv_q = if (w.absorbed.packed_rows) |p| if (std.c.getenv("TF_HY4_MLA_BF16") == null) .{ try b.upload(p.words), try b.upload(p.scales), try b.upload(p.biases) } else null else null,
            .sinks = try b.upload(std.mem.sliceAsBytes(w.sinks[0..heads])),
            .index = if (w.index) |*ix| try index.Weights.upload(&b, ix, use_tensors) else null,
        };
        return withAllocation(out, b);
    }

    pub fn deinit(self: *Attention) void {
        self.own.clear();
        self.* = undefined;
    }
};

pub const Feed = struct {
    own: Buffers,
    merge: Merge,
    norm: mtl.Buffer,
    gate: Linear,
    up: Linear,
    down: Linear,
    output_norm: mtl.Buffer,

    pub fn upload(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer) !Feed {
        return uploadForModel(a, device, w, false);
    }

    pub fn uploadForModel(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, use_tensors: bool) !Feed {
        if (w.feed != .dense) return error.Hy4DenseFeedRequired;
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const feed = w.feed.dense;
        const out: Feed = .{
            .own = undefined,
            .merge = try Merge.upload(&b, w.mlp_hc),
            .norm = try b.upload(w.mlp_norm.bytes),
            .gate = try b.linear(feed.projections.gate, use_tensors),
            .up = try b.linear(feed.projections.up, use_tensors),
            .down = try b.linear(feed.projections.down, use_tensors),
            .output_norm = try b.upload(feed.norm.bytes),
        };
        return withAllocation(out, b);
    }

    pub fn deinit(self: *Feed) void {
        self.own.clear();
        self.* = undefined;
    }
};

fn withAllocation(out: anytype, b: Buffers) @TypeOf(out) {
    var owned = out;
    owned.own = b;
    return owned;
}

pub const Scratch = struct {
    own: Buffers,
    capacity: mha.Shape,
    shape: mha.Shape,
    inner_capacity: u32,
    merged: Ref,
    ihc_partials: Ref,
    /// A count and a generation a row for the fused seams' in-kernel barrier (zeroed; each launch leaves count 0).
    seam_barriers: Ref,
    qkv: ?Ref = null,
    post: Ref,
    normalized: Ref,
    query_a: Ref,
    query_norm: Ref,
    query: Ref,
    kv: Ref,
    value_logits: Ref,
    values: Ref,
    absorbed: Ref,
    rotary: Ref,
    latent: Ref,
    attention: Ref,
    key_scores: Ref,
    flash_parts: ?Ref = null,
    branch: Ref,
    gate: Ref,
    up: Ref,
    inner: Ref,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, layer: *const Attention, inner: u32, keys: u32, rows: u32) !Scratch {
        return initForModel(a, device, layer, inner, keys, rows, false);
    }

    pub fn initForModel(a: std.mem.Allocator, device: mtl.Device, layer: *const Attention, inner: u32, keys: u32, rows: u32, use_tensors: bool) !Scratch {
        var b: Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const r = rows;
        const h = layer.hidden;
        const heads = layer.heads;
        const shape = try mha.Shape.init(r, heads, keys);
        const out: Scratch = .{
            .own = undefined,
            .capacity = shape,
            .shape = shape,
            .inner_capacity = inner,
            .merged = try b.rows(r, h),
            .ihc_partials = .{ .buffer = try b.alloc(@as(usize, r) * ihc_gpu.partial_floats * @sizeOf(f32)) },
            .seam_barriers = blk: {
                const bars = try b.alloc(ihc_gpu.Kernels.seam_rows * 8);
                @memset(bars.slice(u8, ihc_gpu.Kernels.seam_rows * 8), 0);
                break :blk .{ .buffer = bars };
            },
            .qkv = if (use_tensors) try b.rows(r, layer.query_rank + shape_.absorbed_stride + heads * shape_.value_width) else null,
            .post = .{ .buffer = try b.alloc(@as(usize, r) * ihc.streams * @sizeOf(f32)) },
            .normalized = try b.rows(r, h),
            .query_a = try b.rows(r, layer.query_rank),
            .query_norm = try b.rows(r, layer.query_rank),
            .query = try b.rows(r, heads * shape_.query_stride),
            .kv = try b.rows(r, shape_.absorbed_stride),
            .value_logits = try b.rows(r, heads * shape_.value_width),
            .values = try b.rows(r, heads * shape_.value_width),
            .absorbed = try b.rows(r, heads * shape_.absorbed_stride),
            .rotary = try b.rows(r, shape_.rope),
            .latent = try b.rows(r, heads * shape_.latent),
            .attention = .{ .buffer = try b.alloc(shape.scratchBytes()) },
            .key_scores = .{ .buffer = try b.alloc(@as(usize, r) * heads * keys * @sizeOf(f32)) },
            .flash_parts = if (use_tensors) blk: {
                const parts = try b.alloc(mla_gpu.Nax.partBytes(heads));
                @memset(parts.slice(u8, mla_gpu.Nax.partBytes(heads)), 0);
                break :blk .{ .buffer = parts };
            } else null,
            .branch = try b.rows(r, h),
            .gate = try b.rows(r, inner),
            .up = try b.rows(r, inner),
            .inner = try b.rows(r, inner),
        };
        return withAllocation(out, b);
    }

    /// Keep one allocation at the widest layer and bind the current lane batch and head count before encoding.
    pub fn bind(self: *Scratch, rows: u32, heads: u32, keys: u32) !void {
        if (rows > self.capacity.rows or heads > self.capacity.heads or keys > self.capacity.key_stride)
            return error.ShortHy4Scratch;
        self.shape = try mha.Shape.init(rows, heads, keys);
    }

    pub fn deinit(self: *Scratch) void {
        self.own.clear();
        self.* = undefined;
    }
};

/// Lane windows of 2-8 rows through one weight read: tf_hy4_rows, the core row projection's per-row arithmetic
/// (bit-equal to its plain kernel at every row count) with RPS columns a simdgroup and 4 simdgroups a threadgroup.
pub const Rows = struct {
    q4: mtl.Pipeline,
    q8: mtl.Pipeline,
    rps: u32,
    /// Greedy tokens on the GPU (first index of the largest logit), so draft columns chain without the host.
    argmax: mtl.Pipeline,
    /// Decode-sized passes (1-8 rows) of 4- and 8-bit group-64 matrices, each weight read once.
    qmv4: mtl.Pipeline,
    qmv8: mtl.Pipeline,
    /// Output columns a decode GEMV threadgroup covers.
    qmv_cols: u32 = 8,
    /// 1- to 4-row decode GEMVs with four columns a simdgroup ([bits 4, 8][rows 1-4]; the same bits as qmv2,
    /// 14.83 -> 14.23 ms a token); TF_HY4_QMV=2 keeps qmv2, TF_HY4_QMV=8 takes eight columns a simdgroup.
    qmv3: ?[2][4]mtl.Pipeline = null,
    qmv3_cols: u32 = 16,
    /// Routed experts for 1-2 row passes: gate/up into the clipped SwiGLU, then down (q4, q8).
    expert_glu: [2]mtl.Pipeline,
    expert_down: [2]mtl.Pipeline,
    /// Routed experts for 3-16 row passes, grouped by the router window's jobs (each expert's codes unpacked once).
    expert_glu_groups: [2]mtl.Pipeline,
    expert_down_groups: [2]mtl.Pipeline,
    /// The expert down GEMVs with the routed weighted sum folded in (q4, q8; TF_HY4_NO_DOWN_SUM keeps two launches).
    expert_down_sum: [2]mtl.Pipeline,
    /// MLA absorb and expand-and-gate for 1-4 row passes over kv_b's affine codes (q4, q8).
    absorb_q: [2]mtl.Pipeline,
    expand_q: [2]mtl.Pipeline,
    /// The expert branches' add fused into the norm that reads them (bit-equal to add then norm).
    add_norm: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Rows {
        const text = try std.mem.concat(a, u8, &.{ affine.source(device.tensorUnits()), @import("kernel_sources").hy4_flash_rows });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        // TF_HY4_ROWS: the variant, r2 (default), r4, r2s, r1s or r1 (columns a simdgroup; s: shared unpack)
        const variant: []const u8 = if (std.c.getenv("TF_HY4_ROWS")) |v| std.mem.span(v) else "r2";
        const rps: u32 = if (variant[1] == '1') 1 else if (variant[1] == '4') 4 else 2;
        var name4: [32]u8 = undefined;
        var name8: [32]u8 = undefined;
        var q4 = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name4, "tf_hy4_rows_{s}_q4", .{variant}), false);
        errdefer q4.deinit();
        var q8 = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name8, "tf_hy4_rows_{s}_q8", .{variant}), false);
        errdefer q8.deinit();
        var argmax = try mtl.Pipeline.init(device, lib, "tf_hy4_argmax", false);
        errdefer argmax.deinit();
        // TF_HY4_QMV=1 keeps the first decode GEMV (four columns a simdgroup, codes unpacked a row at a time)
        const first = if (std.c.getenv("TF_HY4_QMV")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        var qmv4 = try mtl.Pipeline.init(device, lib, if (first) "tf_hy4_qmv_q4" else "tf_hy4_qmv2_q4", false);
        errdefer qmv4.deinit();
        var qmv8 = try mtl.Pipeline.init(device, lib, if (first) "tf_hy4_qmv_q8" else "tf_hy4_qmv2_q8", false);
        errdefer qmv8.deinit();
        var qmv3: ?[2][4]mtl.Pipeline = null;
        const qmv_mode: []const u8 = if (std.c.getenv("TF_HY4_QMV")) |v| std.mem.span(v) else "3";
        const wide8 = std.mem.eql(u8, qmv_mode, "8");
        if (std.mem.eql(u8, qmv_mode, "3") or wide8) {
            var p3: [2][4]mtl.Pipeline = undefined;
            const set4 = [_][4][:0]const u8{ .{ "tf_hy4_qmv3_r1_q4", "tf_hy4_qmv3_r2_q4", "tf_hy4_qmv3_r3_q4", "tf_hy4_qmv3_r4_q4" }, .{ "tf_hy4_qmv3_r1_q8", "tf_hy4_qmv3_r2_q8", "tf_hy4_qmv3_r3_q8", "tf_hy4_qmv3_r4_q8" } };
            const set8 = [_][4][:0]const u8{ .{ "tf_hy4_qmv3x8_r1_q4", "tf_hy4_qmv3x8_r2_q4", "tf_hy4_qmv3x8_r3_q4", "tf_hy4_qmv3x8_r4_q4" }, .{ "tf_hy4_qmv3x8_r1_q8", "tf_hy4_qmv3x8_r2_q8", "tf_hy4_qmv3x8_r3_q8", "tf_hy4_qmv3x8_r4_q8" } };
            for (if (wide8) set8 else set4, 0..) |names, bi| {
                for (names, 0..) |name, ri| p3[bi][ri] = try mtl.Pipeline.init(device, lib, name, false);
            }
            qmv3 = p3;
        }
        errdefer if (qmv3) |p3| for (p3) |row| for (row) |pipe| pipe.deinit();
        var experts: [14]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (experts[0..made]) |pipe| pipe.deinit();
        for ([_][:0]const u8{ "tf_hy4_expert_glu_q4", "tf_hy4_expert_glu_q8", "tf_hy4_expert_down_q4", "tf_hy4_expert_down_q8", "tf_hy4_expert_glu_groups_q4", "tf_hy4_expert_glu_groups_q8", "tf_hy4_expert_down_groups_q4", "tf_hy4_expert_down_groups_q8", "tf_hy4_absorb_qmv_q4", "tf_hy4_absorb_qmv_q8", "tf_hy4_expand_gated_qmv_q4", "tf_hy4_expand_gated_qmv_q8", "tf_hy4_expert_down_sum_q4", "tf_hy4_expert_down_sum_q8" }, &experts) |name, *dst| {
            dst.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return .{ .q4 = q4, .q8 = q8, .rps = rps, .argmax = argmax, .qmv4 = qmv4, .qmv8 = qmv8, .qmv_cols = if (first) 16 else 8, .qmv3 = qmv3, .qmv3_cols = if (wide8) 32 else 16, .expert_glu = .{ experts[0], experts[1] }, .expert_down = .{ experts[2], experts[3] }, .expert_glu_groups = .{ experts[4], experts[5] }, .expert_down_groups = .{ experts[6], experts[7] }, .absorb_q = .{ experts[8], experts[9] }, .expand_q = .{ experts[10], experts[11] }, .expert_down_sum = .{ experts[12], experts[13] }, .add_norm = try mtl.Pipeline.init(device, lib, "tf_hy4_add_rms", false) };
    }

    pub fn deinit(self: Rows) void {
        self.q4.deinit();
        self.q8.deinit();
        self.argmax.deinit();
        self.qmv4.deinit();
        self.qmv8.deinit();
        if (self.qmv3) |p3| for (p3) |row| for (row) |pipe| pipe.deinit();
        for (self.expert_glu) |pipe| pipe.deinit();
        for (self.expert_down) |pipe| pipe.deinit();
        for (self.expert_glu_groups) |pipe| pipe.deinit();
        for (self.expert_down_groups) |pipe| pipe.deinit();
        for (self.absorb_q) |pipe| pipe.deinit();
        for (self.expand_q) |pipe| pipe.deinit();
        for (self.expert_down_sum) |pipe| pipe.deinit();
        self.add_norm.deinit();
    }

    /// y = norm(a + b) * gain over `rows` rows of `width`, as dense.Ops.add then dense.Ops.norm.
    pub fn addNorm(self: Rows, e: mtl.ComputeEncoder, x: Ref, branch: Ref, gain: mtl.Buffer, y: Ref, rows: u32, width: u32, epsilon: f32) !void {
        const p: dense.Norm = .{ .rows = rows, .heads = 1, .dim = width, .x_stride = width, .y_stride = width, .eps = epsilon };
        const threads = try dense.normThreads(p);
        e.setPipeline(self.add_norm);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(branch.buffer, branch.offset, 1);
        e.setBuffer(gain, 0, 2);
        e.setValue(p, 3);
        e.setBuffer(y.buffer, y.offset, 4);
        e.dispatchGroups(mtl.Size.of(1, rows, 1), mtl.Size.of(threads, 1, 1));
    }

    /// out[at + r] = the greedy token of logits row r, for `rows` rows of `vocab`.
    pub fn pick(self: Rows, e: mtl.ComputeEncoder, logits: Ref, out: Ref, at: u32, rows: u32, vocab: u32) void {
        e.setPipeline(self.argmax);
        e.setBuffer(logits.buffer, logits.offset, 0);
        e.setBuffer(out.buffer, out.offset, 1);
        e.setValue([2]u32{ vocab, at }, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    fn sg(self: Rows) u32 {
        return if (self.rps == 1) 8 else 4;
    }

    fn choose(self: Rows, w: affine.Weights, rows: u32) ?mtl.Pipeline {
        if (rows < 2 or w.sum != .f32 or w.n % (self.rps * self.sg()) != 0) return null;
        return switch (w.bits) {
            4 => self.q4,
            8 => self.q8,
            else => null,
        };
    }
};

fn twoStep(keys: u32) bool {
    if (std.c.getenv("TF_HY4_ATTEND")) |v| return std.mem.eql(u8, std.mem.span(v), "two");
    return keys >= 1024;
}

pub const CacheRows = struct {
    positions: Ref,
    write_slots: Ref,
    retained: Ref,
    counts: Ref,
    cache: Ref,
    capacity: u32,
    /// When every row's keys are one contiguous run: their ranges and tiles, for the tensor units' flash kernel.
    ranges: ?mla_gpu.KeyRanges = null,
    /// The widest row's live key count this pass (0: unknown), which sizes the decode attention split.
    live_keys: u32 = 0,
    /// A sparse layer's indexer for this pass (null: it attends `retained`, the probe's contexts under 2,048).
    index: ?index.Pass = null,
};

pub const Kernels = struct {
    affine: affine.Pipelines,
    dense: dense.Ops,
    merge: ihc_gpu.Kernels,
    projections: mla_gpu.Projections,
    attention: mla_gpu.Attention,
    tensors: ?gpu_qkv.Kernels = null,
    rows: ?Rows = null,
    linears: ?Linears = null,

    pub fn load(device: mtl.Device) !Kernels {
        var row_project = try affine.Pipelines.load(device);
        errdefer row_project.deinit();
        const normalize = try dense.Ops.init(device);
        errdefer normalize.deinit();
        var merge = try ihc_gpu.Kernels.load(device);
        errdefer merge.deinit();
        var proj = try mla_gpu.Projections.load(device);
        errdefer proj.deinit();
        return .{
            .affine = row_project,
            .dense = normalize,
            .merge = merge,
            .projections = proj,
            .attention = try mla_gpu.Attention.load(device),
        };
    }

    pub fn deinit(self: *Kernels) void {
        if (self.linears) |*l| l.deinit();
        if (self.rows) |r| r.deinit();
        if (self.tensors) |t| t.deinit();
        self.attention.deinit();
        self.projections.deinit();
        self.merge.deinit();
        self.dense.deinit();
        self.affine.deinit();
    }

    pub fn loadForModel(a: std.mem.Allocator, device: mtl.Device, t: cfg.Text, bits: u8) !Kernels {
        var out = try load(device);
        errdefer out.deinit();
        if (device.tensorUnits()) out.tensors = try gpu_qkv.Kernels.load(a, device, t, bits);
        if (device.tensorUnits() and std.c.getenv("TF_HY4_DENSE_ROWS") == null) out.linears = try Linears.load(a, device, t, bits);
        if (std.c.getenv("TF_HY4_ROWS_OFF") == null) out.rows = try Rows.load(a, device);
        if (device.tensorUnits() and std.c.getenv("TF_HY4_MLA_FMA") == null) out.projections.nax = try mla_gpu.Nax.load(a, device);
        return out;
    }

    pub fn project(self: *const Kernels, e: mtl.ComputeEncoder, w: affine.Weights, x: Ref, y: Ref, rows: u32) !void {
        if (self.rows) |r| if (r.qmv3) |p3| if (rows <= decodeRows() and rows <= 4 and w.group == 64 and w.sum == .f32 and w.n % r.qmv3_cols == 0 and w.k % 32 == 0 and (w.bits == 4 or w.bits == 8)) {
            e.setPipeline(p3[@intFromBool(w.bits == 8)][rows - 1]);
            e.setBuffer(x.buffer, x.offset, 0);
            e.setBuffer(w.w, w.w_off, 1);
            e.setBuffer(w.scales, w.s_off, 2);
            e.setBuffer(w.biases, w.b_off, 3);
            e.setValue([6]i32{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), 1, 1 }, 4);
            e.setBuffer(y.buffer, y.offset, 5);
            e.dispatchGroups(mtl.Size.of(w.n / r.qmv3_cols, 1, 1), mtl.Size.of(128, 1, 1));
            return;
        };
        if (self.rows) |r| if (rows <= decodeRows() and rows <= 8 and w.group == 64 and w.sum == .f32 and w.n % r.qmv_cols == 0 and w.k % 32 == 0 and (w.bits == 4 or w.bits == 8)) {
            e.setPipeline(if (w.bits == 4) r.qmv4 else r.qmv8);
            e.setBuffer(x.buffer, x.offset, 0);
            e.setBuffer(w.w, w.w_off, 1);
            e.setBuffer(w.scales, w.s_off, 2);
            e.setBuffer(w.biases, w.b_off, 3);
            e.setValue([6]i32{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), 1, 1 }, 4);
            e.setBuffer(y.buffer, y.offset, 5);
            e.dispatchGroups(mtl.Size.of(w.n / r.qmv_cols, 1, 1), mtl.Size.of(128, 1, 1));
            return;
        };
        var call = try affine.call(&self.affine, w, rows, false);
        if (self.rows) |r| if (r.choose(w, rows)) |pipe| {
            call.pipeline = pipe;
            call.groups = w.n / (r.rps * r.sg());
            call.threads = 32 * r.sg();
        };
        e.setPipeline(call.pipeline);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(w.w, w.w_off, 1);
        e.setBuffer(w.scales, w.s_off, 2);
        e.setBuffer(w.biases, w.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(y.buffer, y.offset, 5);
        e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    /// y = x W^T for a dense projection, on whichever kernels its weights were packed for.
    pub fn linear(self: *const Kernels, e: mtl.ComputeEncoder, w: Linear, x: Ref, y: Ref, rows: u32) !void {
        return self.linearsFrom(e, x, &.{.{ .w = w, .y = y }}, rows);
    }

    pub const Out = struct { w: Linear, y: Ref };

    /// Every projection of the same input x: on tensor units one group-sums pass, a barrier, then each projection's
    /// main kernel with those sums and no barrier between them (the sums are what each would have made itself).
    pub fn linearsFrom(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, outs: []const Out, rows: u32) !void {
        if (outs.len == 0) return;
        if (rows <= decodeRows()) decode: {
            for (outs) |o| if (o.w.plainWeights() == null) break :decode;
            for (outs) |o| try self.project(e, o.w.plainWeights().?, x, o.y, rows);
            return;
        }
        if (outs[0].w == .portable) {
            for (outs) |o| try self.project(e, o.w.portable, x, o.y, rows);
            return;
        }
        const all = &(self.linears orelse return error.Hy4LinearsMissing);
        const k = outs[0].w.tensor.k;
        // small passes: the kernels that add their own group sums (the same words, no sums pass or barrier)
        if (rows <= gpu_qkv.inline_rows) inline_sums: {
            for (outs) |o| if (all.findInline(o.w.tensor.n, o.w.tensor.k) == null) break :inline_sums;
            for (outs) |o| try all.findInline(o.w.tensor.n, o.w.tensor.k).?.encode(e, tref(x), o.w.tensor.w, o.w.tensor.sb, tref(o.y), rows);
            return;
        }
        const first = all.find(outs[0].w.tensor.n, k) orelse return error.Hy4LinearShapeMissing;
        const shared = first.reg and first.sums_pipe != null and rows <= Linears.sum_rows;
        const stride: u32 = 16 * ((rows + 15) / 16);
        if (shared) {
            e.setPipeline(first.sums_pipe.?);
            e.setBuffer(x.buffer, x.offset, 0);
            e.setValue(@as(i32, @intCast(rows)), 1);
            e.setBuffer(all.sums.?, 0, 2);
            e.setValue(@as(i32, @intCast(stride)), 3);
            e.dispatchThreads(.{ .width = stride * (k / 64) }, .{ .width = 256 });
            e.barrier();
        }
        for (outs) |o| {
            const m = o.w.tensor;
            if (m.k != k) return error.Hy4LinearInputsDiffer;
            const p = all.find(m.n, m.k) orelse return error.Hy4LinearShapeMissing;
            if (shared)
                try p.encodeSums(e, tref(x), m.w, m.sb, .{ .ref = .{ .buf = all.sums.? }, .stride = stride }, tref(o.y), rows)
            else
                try p.encode(e, tref(x), m.w, m.sb, tref(o.y), rows);
        }
    }

    pub fn dref(x: Ref) dense.Ref {
        return .{ .buf = x.buffer, .off = x.offset };
    }

    fn tref(x: Ref) lane.Ref {
        return .{ .buf = x.buffer, .off = x.offset };
    }

    pub fn norm(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, gain: mtl.Buffer, y: Ref, rows: u32, width: u32, epsilon: f32) !void {
        try self.dense.norm(e, dref(x), .{ .buf = gain }, dref(y), .{
            .rows = rows,
            .heads = 1,
            .dim = width,
            .x_stride = width,
            .y_stride = width,
            .eps = epsilon,
        });
    }

    /// Where a sub-block sits between seams: `ready` when the previous tail already wrote this collapse's partials,
    /// `next` the collapse after this sub-block (its tail writes that one's partials).
    /// ready: the collapse's partials are written; headed: its gates, row and norm too (a fused seam before it).
    /// next_gain: the next sub-block's input norm, which lets the tail run that collapse's head in its launch.
    pub const Seam = struct { ready: bool = false, headed: bool = false, next: ?Merge = null, next_gain: ?mtl.Buffer = null };

    /// Decode passes fuse each tail with the next head (TF_HY4_SEAM_FUSE=0 keeps two launches).
    pub fn fusedSeams(rows: u32) bool {
        if (seamsOff() or rows > ihc_gpu.Kernels.seam_rows) return false;
        const v = std.c.getenv("TF_HY4_SEAM_FUSE") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    fn seamsOff() bool {
        return std.c.getenv("TF_HY4_NO_SEAMS") != null;
    }

    /// The collapse and input norm in two launches at most (partials unless `ready`, then gates, row and norm).
    pub fn collapseSeam(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Merge, gain: mtl.Buffer, s: *const Scratch, ready: bool) !ihc.Shape {
        return self.collapseSeamFrom(e, x, w, gain, s, .{ .ready = ready });
    }

    pub fn collapseSeamFrom(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Merge, gain: mtl.Buffer, s: *const Scratch, seam: Seam) !ihc.Shape {
        const ready = seam.ready;
        if (seamsOff()) return self.collapse(e, x, w, gain, s);
        var shape = w.shape;
        shape.rows = s.shape.rows;
        if (seam.headed) return shape;
        const part: ihc_gpu.Ref = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset };
        const hidden: ihc_gpu.Ref = .{ .buffer = x.buffer, .offset = x.offset };
        if (!ready) {
            try self.merge.seamPartials(e, hidden, .{ .buffer = w.project }, part, shape);
            e.barrier();
        }
        try self.merge.headPass(e, .{
            .hidden = hidden,
            .partials = part,
            .bias = .{ .buffer = w.bias },
            .scale = .{ .buffer = w.scale },
            .gain = .{ .buffer = gain },
            .collapsed = .{ .buffer = s.merged.buffer, .offset = s.merged.offset },
            .normalized = .{ .buffer = s.normalized.buffer, .offset = s.normalized.offset },
            .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
        }, shape, shape.eps);
        e.barrier();
        return shape;
    }

    /// A sub-block's end: its branch (two added, for the experts) through the output norm, mixed into `out`, and the
    /// next collapse's partials when there is one.
    pub fn tail(self: *const Kernels, e: mtl.ComputeEncoder, branch: Ref, branch2: ?Ref, gain: mtl.Buffer, eps: f32, x: Ref, out: Ref, s: *const Scratch, shape: ihc.Shape, next: ?Merge) !void {
        return self.tailTo(e, branch, branch2, gain, eps, x, out, s, shape, .{ .next = next });
    }

    pub fn tailTo(self: *const Kernels, e: mtl.ComputeEncoder, branch: Ref, branch2: ?Ref, gain: mtl.Buffer, eps: f32, x: Ref, out: Ref, s: *const Scratch, shape: ihc.Shape, seam: Seam) !void {
        const next = seam.next;
        if (next) |n| if (seam.next_gain) |ng| if (fusedSeams(shape.rows) and shape.width == 12 * 256) {
            const r = ihc_gpu.Ref;
            return self.merge.seamPass(e, .{
                .branch = r{ .buffer = branch.buffer, .offset = branch.offset },
                .branch2 = if (branch2) |b2| r{ .buffer = b2.buffer, .offset = b2.offset } else null,
                .gain = .{ .buffer = gain },
                .residual = .{ .buffer = x.buffer, .offset = x.offset },
                .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
                .output = .{ .buffer = out.buffer, .offset = out.offset },
                .next = r{ .buffer = n.project },
                .partials = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset },
            }, .{
                .bias = .{ .buffer = n.bias },
                .scale = .{ .buffer = n.scale },
                .gain = .{ .buffer = ng },
                .collapsed = .{ .buffer = s.merged.buffer, .offset = s.merged.offset },
                .normalized = .{ .buffer = s.normalized.buffer, .offset = s.normalized.offset },
                .shape = n.shape,
                .barriers = .{ .buffer = s.seam_barriers.buffer, .offset = s.seam_barriers.offset },
            }, shape.rows, shape.width, eps);
        };
        if (seamsOff()) {
            if (branch2) |b2| {
                if (self.rows) |r| try r.addNorm(e, branch, b2, gain, s.normalized, shape.rows, shape.width, eps) else return error.Hy4SeamsNeedRows;
            } else try self.norm(e, branch, gain, s.normalized, shape.rows, shape.width, eps);
            e.barrier();
            return self.residual(e, x, s.normalized, s.post, out, shape);
        }
        const r = ihc_gpu.Ref;
        self.merge.tailPass(e, .{
            .branch = r{ .buffer = branch.buffer, .offset = branch.offset },
            .branch2 = if (branch2) |b2| r{ .buffer = b2.buffer, .offset = b2.offset } else null,
            .gain = .{ .buffer = gain },
            .residual = .{ .buffer = x.buffer, .offset = x.offset },
            .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
            .output = .{ .buffer = out.buffer, .offset = out.offset },
            .next = if (next) |n| r{ .buffer = n.project } else null,
            .partials = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset },
        }, shape.rows, shape.width, eps, if (next) |n| n.shape.gates else 8);
    }

    pub fn collapse(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Merge, gain: mtl.Buffer, s: *const Scratch) !ihc.Shape {
        var shape = w.shape;
        shape.rows = s.shape.rows;
        try self.merge.collapse(e, .{
            .hidden = .{ .buffer = x.buffer, .offset = x.offset },
            .projection = .{ .buffer = w.project },
            .bias = .{ .buffer = w.bias },
            .scale = .{ .buffer = w.scale },
            .output = .{ .buffer = s.merged.buffer, .offset = s.merged.offset },
            .post = .{ .buffer = s.post.buffer, .offset = s.post.offset },
            .partials = .{ .buffer = s.ihc_partials.buffer, .offset = s.ihc_partials.offset },
        }, shape);
        e.barrier();
        try self.norm(e, s.merged, gain, s.normalized, shape.rows, shape.width, shape.eps);
        e.barrier();
        return shape;
    }

    pub fn residual(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, branch: Ref, post: Ref, out: Ref, shape: ihc.Shape) !void {
        try self.merge.add(e, .{
            .residual = .{ .buffer = x.buffer, .offset = x.offset },
            .branch = .{ .buffer = branch.buffer, .offset = branch.offset },
            .post = .{ .buffer = post.buffer, .offset = post.offset },
            .output = .{ .buffer = out.buffer, .offset = out.offset },
        }, shape);
    }

    /// The caller supplies the retained list for this lane from its sliding or sparse index state.
    pub fn attend(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Attention, s: *const Scratch, cache: CacheRows, x: Ref, out: Ref) !void {
        return self.attendSeam(e, w, s, cache, x, out, .{});
    }

    pub fn attendSeam(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Attention, s: *const Scratch, cache: CacheRows, x: Ref, out: Ref, seam: Seam) !void {
        if (s.shape.heads != w.heads) return error.Hy4WrongAttentionHeadCount;
        if (w.inputs == .tensor and (self.tensors == null or s.qkv == null)) return error.ShortHy4Scratch;
        const shape = try self.collapseSeamFrom(e, x, w.merge, w.norm, s, seam);
        const rows = shape.rows;
        var q_a = s.query_a;
        var kv = s.kv;
        var gates = s.value_logits;
        switch (w.inputs) {
            .portable => |m| {
                try self.project(e, m.q_a, s.normalized, q_a, rows);
                try self.project(e, m.kv_a, s.normalized, kv, rows);
                try self.project(e, m.gates, s.normalized, gates, rows);
                e.barrier();
                try self.norm(e, q_a, w.q_norm, s.query_norm, rows, w.query_rank, w.epsilon);
                e.barrier();
                try self.project(e, m.q_b, s.query_norm, s.query, rows);
            },
            .tensor => |m| {
                const kernels = &self.tensors.?;
                const qkv = s.qkv.?;
                try kernels.inputFor(m.flavor, rows).encode(e, tref(s.normalized), m.qkv.w, m.qkv.sb, tref(qkv), rows);
                e.barrier();
                if (rows == 1) {
                    q_a = qkv;
                    kv = .{ .buffer = qkv.buffer, .offset = @as(usize, w.query_rank) * @sizeOf(u16) };
                    gates = .{ .buffer = qkv.buffer, .offset = (@as(usize, w.query_rank) + shape_.absorbed_stride) * @sizeOf(u16) };
                } else {
                    kernels.split(e, tref(qkv), tref(q_a), tref(kv), tref(gates), rows, w.query_rank, w.heads * shape_.value_width);
                    e.barrier();
                }
                try self.norm(e, q_a, w.q_norm, s.query_norm, rows, w.query_rank, w.epsilon);
                e.barrier();
                try kernels.queryFor(m.flavor, rows).encode(e, tref(s.query_norm), m.q_b.w, m.q_b.sb, tref(s.query), rows);
            },
        }
        e.barrier();
        const ix_pass: ?index.Pass = if (cache.index) |p| (if (w.index != null) p else null) else null;
        if (ix_pass) |p| {
            if (p.scratch.rows != rows) return error.Hy4IndexRowsOutOfStep;
            try index.run(p, self, self.projections, e, &w.index.?, s.normalized, s.query_norm, cache.write_slots, cache.positions, s.rotary, w.rope, w.epsilon);
            e.barrier();
        }
        // past 2,048 keys a sparse layer attends each row's recalled slots
        const lists = if (ix_pass) |p| p.scratch.selects() else false;
        // rows of a pass across 2,048 keys that sit before it attend as they would in any other pass
        const lists_from: ?u32 = if (lists and cache.ranges != null and ix_pass.?.scratch.first < index.keys_a_row) index.keys_a_row - ix_pass.?.scratch.first else null;
        const quantized: ?mla_gpu.Quantized = if (self.rows) |r| if (w.kv_q) |q| if (rows <= mlaRows() and (w.bits == 4 or w.bits == 8)) .{
            .absorb = r.absorb_q[@intFromBool(w.bits == 8)],
            .expand = r.expand_q[@intFromBool(w.bits == 8)],
            .words = .{ .buffer = q[0] },
            .scales = .{ .buffer = q[1] },
            .biases = .{ .buffer = q[2] },
        } else null else null else null;
        try mla_gpu.forward(self.projections, self.attention, e, .{
            .quantized = quantized,
            .query = s.query,
            .key_weights = .{ .buffer = w.wk },
            .value_weights = .{ .buffer = w.wv },
            .value_logits = gates,
            .positions = cache.positions,
            .rotary_table = s.rotary,
            .values = s.values,
            .kv = .{ .projection = kv, .gain = .{ .buffer = w.kv_norm }, .write_slots = cache.write_slots, .capacity = cache.capacity, .epsilon = w.epsilon },
            .attention = .{
                .query = s.absorbed,
                .cache = cache.cache,
                .slots = if (lists) ix_pass.?.cache.tokens else cache.retained,
                .counts = if (lists) ix_pass.?.cache.counts else cache.counts,
                .sinks = .{ .buffer = w.sinks },
                .scratch = s.attention,
                .output = s.latent,
                // two steps pay off from two chunks of keys; below that the one-step kernel is quicker
                .key_scores = if (twoStep(s.shape.key_stride)) s.key_scores else null,
                .ranges = if (lists and lists_from == null) null else cache.ranges,
                .lists_from = lists_from,
                .flash_parts = s.flash_parts,
                .live_keys = if (lists) index.keys_a_row else cache.live_keys,
            },
        }, s.shape, w.rope);
        e.barrier();
        try self.linear(e, w.o, s.values, s.branch, rows);
        e.barrier();
        try self.tailTo(e, s.branch, null, w.o_norm, w.epsilon, x, out, s, shape, seam);
    }

    pub fn feed(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Feed, s: *const Scratch, x: Ref, out: Ref) !void {
        return self.feedSeam(e, w, s, x, out, .{});
    }

    pub fn feedSeam(self: *const Kernels, e: mtl.ComputeEncoder, w: *const Feed, s: *const Scratch, x: Ref, out: Ref, seam: Seam) !void {
        const inner: u32 = w.gate.outputs();
        if (inner > s.inner_capacity) return error.ShortHy4Scratch;
        const shape = try self.collapseSeamFrom(e, x, w.merge, w.norm, s, seam);
        try self.linearsFrom(e, s.normalized, &.{ .{ .w = w.gate, .y = s.gate }, .{ .w = w.up, .y = s.up } }, shape.rows);
        e.barrier();
        try self.dense.swiglu(e, dref(s.gate), dref(s.up), dref(s.inner), .{
            .rows = shape.rows,
            .width = inner,
            .gate_stride = inner,
            .up_stride = inner,
            .y_stride = inner,
        });
        e.barrier();
        try self.linear(e, w.down, s.inner, s.branch, shape.rows);
        e.barrier();
        try self.tailTo(e, s.branch, null, w.output_norm, shape.eps, x, out, s, shape, seam);
    }
};
