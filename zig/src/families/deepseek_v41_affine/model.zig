//! A native DeepSeek-V4.1-Flash model: config, checkpoint views, engram tables, RoPE tables and pipelines.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const engram = @import("engram.zig");
const vision = @import("vision.zig");

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    engram: engram.Engram,
    /// The image tower and aligner, when the checkpoint has them.
    vision: ?vision.Weights,
    /// RoPE frequencies (f32 [32]): plain for window-only layers and drafts, YaRN for compressed-key layers.
    freq_plain: mtl.Buffer,
    markov_packed: ?mtl.Buffer = null,
    confidence_probe: ?weights.Tensor = null,
    markov_audit: bool = false,
    markov_top_two: bool = false,
    markov_top_two_audit: bool = false,
    markov_parallel: bool = false,
    markov_group_policy: bool = false,
    clip_attention: bool = false,
    attention_eight: bool = false,
    hc_overlap: ?@import("hc_overlap_live.zig").Kernels = null,
    hc_overlap_cap: usize = 2,
    router_variant: ?@import("router_live.zig").Kernels = null,
    attention_eight_audit: bool = false,
    attention_stream_merge: bool = false,
    attention_stream_audit: bool = false,
    woa_batch: ?@import("woa_batch.zig").Kernels = null,
    projection_overlap: bool = false,
    projection_audit: bool = false,
    condition_overlap: bool = false,
    short_index: bool = false,
    short_index_audit: bool = false,
    short_index_force: ?usize = null,
    condition_overlap_audit: bool = false,
    tight_attention_merge: bool = false,
    audit_attention_clip: bool = false,
    confidence: ?weights.Tensor = null,
    freq_compressed: mtl.Buffer,
    /// The checkpoint's buffers held resident on the queue (none before macOS 15).
    resident: ?mtl.ResidencySet = null,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, drafts: bool) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = config;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);
        // held resident as read: ~190 GiB of untouched buffers invite the compressor, then jetsam kills the load
        m.resident = m.device.residencySet(files.len) catch null;
        errdefer if (m.resident) |set| set.deinit();
        for (files) |path| {
            try m.checkpoint.addFile(m.device, path, "");
            if (m.resident) |set| {
                set.add(m.checkpoint.shards.items[m.checkpoint.shards.items.len - 1].buffer);
                set.commit();
                set.requestResidency();
            }
        }
        // the DSpark head when asked for and the checkpoint has it (mtp.*)
        m.weights = try weights.load(&m.checkpoint, &m.config, drafts and m.checkpoint.has("mtp.0.attn.wq_a.weight"));
        m.vision = try vision.load(&m.checkpoint, m.config.dense);
        const wq_b = m.weights.layers[0].attn.wq_b;
        m.kernels = try kernels.load(m.device, &m.config, .{ .bits = @intCast(wq_b.bits), .group = @intCast(wq_b.group) });
        errdefer m.kernels.deinit();
        const l0 = &m.weights.layers[0];
        try kernels.checkTwins(&m.kernels, m.device, &.{ l0.attn.wq_a, l0.attn.wq_b, l0.attn.wkv, l0.attn.wo_b, l0.moe.shared.w1, l0.moe.shared.w2, l0.moe.shared.w3, m.weights.head });
        m.engram = try engram.Engram.open(gpa, io, dir, &m.config);
        errdefer m.engram.deinit();
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        m.freq_plain = try m.device.buffer(32 * 4, opts);
        errdefer m.freq_plain.deinit();
        m.freq_compressed = try m.device.buffer(32 * 4, opts);
        frequencies(m.config.rope, false, m.freq_plain.slice(f32, 32));
        frequencies(m.config.rope, true, m.freq_compressed.slice(f32, 32));
        errdefer m.freq_compressed.deinit();
        if (m.kernels.route_pair) |p| {
            const pair = @import("route_pair.zig");
            var seen: [9]bool = @splat(false);
            var values: usize = 0;
            var shape_audit: [4]usize = @splat(0);
            for (&m.weights.layers) |*l| {
                const mo = &l.moe;
                const bits = mo.w1.one.bits;
                if ((bits != 2 and bits != 3) or seen[bits]) continue;
                seen[bits] = true;
                for ([_]usize{ 2, 3, 4, 6 }) |rows| {
                    if (p.inline_cap == 0 and rows == 3) continue;
                    for ([_]@TypeOf(mo.w1){ mo.w1, mo.w3, mo.w2 }) |wt| {
                        const checked = try pair.check(p, &m.kernels.rows, m.device, m.queue, wt, rows, cfg.topk);
                        values += checked;
                        if (p.fixed != null) if (pair.fixedIndex(wt)) |index| {
                            shape_audit[index] += checked;
                        };
                    }
                }
            }
            if (values == 0) return error.NoPairedExperts;
            if (p.fixed != null) {
                for (shape_audit) |count| if (count == 0) return error.MissingFixedShapeFixture;
                std.log.info("deepseek_v41 bench: fixed expert shape exact BF16 comparisons={any}", .{shape_audit});
            }
            if (p.inline_cap != 0) std.log.info("deepseek_v41 bench: inline route cap={d}", .{p.inline_cap});
            std.log.info("deepseek_v41 bench: expert row pairs on, {d} identical BF16 output values (q2={}, q3={})", .{ values, seen[2], seen[3] });
        }
        if (m.kernels.greedy) |p| try p.check(gpa, m.device, m.queue);
        if (m.weights.stages != null and m.kernels.confidence != null) {
            const projection = try m.checkpoint.get("mtp.2.confidence_head.proj.weight");
            if (projection.dtype != .bf16 or projection.rank != 2 or
                projection.shape[0] != 1 or projection.shape[1] != cfg.hidden + cfg.markov_rank)
                return error.InvalidDeepseekConfidenceHead;
            m.confidence = projection;
        }
        m.attention_eight = true;
        m.clip_attention = true;
        m.tight_attention_merge = true;
        m.attention_stream_merge = true;
        m.markov_top_two = true;
        m.markov_parallel = true;
        m.markov_group_policy = m.confidence != null;
        m.condition_overlap = true;
        m.projection_overlap = true;
        m.short_index = @import("short_index.zig").compatible(m);
        m.woa_batch = try @import("woa_batch.zig").Kernels.load(m.device, .{ 1, 2, 3, 2, 3, 2 }, false);
        errdefer if (m.woa_batch) |p| p.deinit();
        m.hc_overlap_cap = 1 + cfg.draft_block;
        m.hc_overlap = try @import("hc_overlap_live.zig").Kernels.load(m.device, false);
        errdefer if (m.hc_overlap) |p| p.deinit();
        m.router_variant = try @import("router_live.zig").Kernels.load(m.device, true, false);
        return m;
    }

    /// The frequency table a layer's queries and keys rotate by (layers 40.. are the draft stages, window-only).
    pub fn freq(m: *const Model, layer: usize) mtl.Buffer {
        return if (layer < cfg.layers and cfg.ratios[layer] != 0) m.freq_compressed else m.freq_plain;
    }

    pub fn deinit(m: *Model) void {
        if (m.router_variant) |p| p.deinit();
        if (m.hc_overlap) |p| p.deinit();
        if (m.woa_batch) |p| p.deinit();
        if (m.resident) |set| {
            set.deinit();
        }
        if (m.markov_packed) |packed_weights| packed_weights.deinit();
        m.freq_plain.deinit();
        m.freq_compressed.deinit();
        m.engram.deinit();
        m.kernels.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

pub fn frequencies(r: cfg.Rope, compressed: bool, out: []f32) void {
    const dim: f64 = cfg.rope_dim;
    const base: f64 = if (compressed) r.compress_theta else r.theta;
    for (out, 0..) |*f, i| {
        const e: f64 = @as(f64, @floatFromInt(2 * i)) / dim;
        f.* = @floatCast(1.0 / std.math.pow(f64, base, e));
    }
    if (!compressed) return;
    const orig: f64 = @floatFromInt(r.original);
    const corrected = struct {
        fn at(rotations: f64, d: f64, o: f64, b: f64) f64 {
            return d * @log(o / (rotations * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.at;
    const low = @max(@floor(corrected(r.beta_fast, dim, orig, base)), 0);
    const high = @min(@ceil(corrected(r.beta_slow, dim, orig, base)), dim - 1);
    for (out, 0..) |*f, i| {
        const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / @max(high - low, 1e-3), 0, 1);
        const smooth = 1 - ramp;
        const v: f64 = f.*;
        f.* = @floatCast(v / r.factor * (1 - smooth) + v * smooth);
    }
}
