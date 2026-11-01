//! A native text model keeps immutable checkpoint bytes and prepared affine layouts under one owner.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const core = @import("core");
const cfg = @import("config.zig");
const affine = @import("affine.zig");
const widths = @import("widths.zig");
const wts = @import("gpu_weights.zig");
const quant = @import("quant_gpu.zig");
const glue = @import("glue_gpu.zig");
const Frame = @import("gpu_frame.zig").Frame;
const ane_mlp = @import("ane_mlp.zig");

const ane_policy = @import("ane_policy.zig");

pub const Model = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    /// This rank's share of the family's weights; `local` shapes are what this graph builds, `global` the file's.
    shard: cfg.Shard,
    /// The local config, so downstream shapes are the rank's own.
    config: cfg.Config,
    checkpoint: core.checkpoint_metal.Checkpoint,
    weights: wts.Weights,
    kernels: quant.Kernels,
    glue: glue.Kernels,
    frame: Frame,
    /// The Neural Engine's share of prompt-chunk MLPs, when it loaded.
    ane: ?*ane_mlp.Share = null,
    /// A deferred share still needs the checkpoint mappings, until loadShare decides and closes them.
    share_pending: bool = false,
    /// widths.Widths.tag: 0 for a uniform 4-bit g64 checkpoint.
    widths_tag: u64 = 0,

    /// Immediate share loading for tools; `share_fits` null keeps their existing unrestricted behavior.
    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: u32, share_fits: ?*const fn (u64) bool) !*Model {
        return loadWith(a, io, dir, rows, share_fits, false);
    }

    /// Serving decides the share after the drafter, residency and slots are accounted for.
    pub fn loadDeferred(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: u32) !*Model {
        return loadWith(a, io, dir, rows, null, true);
    }

    /// One rank of a `ranks`-way tensor-parallel run: the same checkpoint, sliced in the engine.
    pub fn loadShard(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: u32, rank: u32, ranks: u32, mode: cfg.Mode, share_fits: ?*const fn (u64) bool) !*Model {
        return loadShared(a, io, dir, rows, rank, ranks, mode, share_fits, false);
    }

    /// One rank of a pair, its share decided after everything else is accounted for (the serving order).
    pub fn loadShardDeferred(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: u32, rank: u32, ranks: u32, mode: cfg.Mode) !*Model {
        return loadShared(a, io, dir, rows, rank, ranks, mode, null, true);
    }

    fn loadWith(
        a: std.mem.Allocator,
        io: std.Io,
        dir: []const u8,
        rows: u32,
        share_fits: ?*const fn (u64) bool,
        later: bool,
    ) !*Model {
        return loadShared(a, io, dir, rows, 0, 1, .full, share_fits, later);
    }

    fn loadShared(
        a: std.mem.Allocator,
        io: std.Io,
        dir: []const u8,
        rows: u32,
        rank: u32,
        ranks: u32,
        mode: cfg.Mode,
        share_fits: ?*const fn (u64) bool,
        later: bool,
    ) !*Model {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
        defer a.free(text);
        const c = try cfg.parse(a, text, null);
        var local = c;
        const shard = if (ranks == 1) cfg.Shard{ .rank = 0, .ranks = 1, .global = c, .local = local } else try cfg.split(&local, rank, ranks, mode);
        try supported(shard.global);
        var formats = try affine.Formats.init(a, text);
        defer formats.deinit();
        var ws = try widths.Widths.resolve(a, &formats, shard.local);
        defer ws.deinit();
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.allocator = a;
        m.shard = shard;
        m.config = shard.local;
        m.widths_tag = ws.tag();
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        if (!qualifiedGeneration(m.device.generation())) return error.QwenMetalQualificationRequired;
        var why: widths.Why = .{};
        widths.admit(&ws, shard.local, .{ .tensor_units = m.device.tensorUnits() }, &why) catch |err| {
            std.log.err("qwen27: {s} (layer {?d}) is {d}-bit group {d}; this chip's projection kernels do not run it ({s})", .{ why.module, why.layer, why.width.bits, why.width.group, @errorName(why.reason) });
            return err;
        };
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = core.checkpoint_metal.Checkpoint.init(a);
        errdefer m.checkpoint.deinit();
        const files = try core.checkpoint_host.shardFiles(a, io, dir);
        defer core.checkpoint_host.freeShardFiles(a, files);
        // projections stay in the files until load repacks them: the weights live once, in their lane layouts
        for (files) |file| try m.checkpoint.addFileSplit(m.device, file, "", "language_model.", wts.repacked);
        m.weights = try wts.load(a, m.device, &m.checkpoint, &m.shard, &ws);
        errdefer m.weights.deinit();
        m.kernels = .{ .allocator = a, .device = m.device, .raw_tiles = !m.device.tensorUnits(), .long_prompt = quant.longPrompt(std.mem.span(m.device.name()), m.device.tensorUnits()) };
        errdefer m.kernels.deinit();
        m.glue = try glue.Kernels.init(m.device);
        errdefer m.glue.deinit();
        // Serving defers automatic widening until its context and streams are known; explicit widths stay exact.
        m.frame = try Frame.init(m.device, shard.local, @max(rows, promptRows(if (later) 0 else m.device.generation())));
        errdefer m.frame.deinit();
        m.ane = null;
        m.share_pending = true;
        if (!later) try m.finishShare(null, share_fits);
        return m;
    }

    /// A deferred share takes at most `room` bytes; zero refuses it, and decided models stay as they are.
    pub fn loadShare(m: *Model, room: u64) !void {
        try m.finishShare(room, null);
    }

    fn finishShare(m: *Model, room: ?u64, share_fits: ?*const fn (u64) bool) !void {
        if (!m.share_pending) return;
        m.share_pending = false;
        defer m.checkpoint.closeFiles();
        const c = m.config;
        const columns = try ane_policy.columns(if (platform.c.getenv("TF_ANE_COLUMNS")) |v| std.mem.span(v) else null);
        // The share is a whole MLP's columns: a rank holds half of them, so tensor parallel keeps prompts on the GPU.
        const share = m.shard.ranks == 1 and ane_policy.requestedOnChip(columns, m.device.tensorUnits()) and mtl.ane.available() and m.weights.uniformMlp();
        if (m.shard.ranks != 1) std.log.info("qwen27: tensor parallel rank {d}/{d} runs prompts on the GPU alone", .{ m.shard.rank, m.shard.ranks });
        const bytes = ane_mlp.Share.bytes(c, columns);
        const slices = (m.frame.capacity + ane_mlp.width - 1) / ane_mlp.width;
        const required: ?u64 = if (share) ane_mlp.Share.totalBytes(c, columns, slices, ane_policy.build_runtime_overhead) catch |err| switch (err) {
            error.MissingAneOverhead => null,
            else => return err,
        } else 0;
        const fits = !share or if (required) |total| (if (room) |free| total <= free else if (share_fits) |f| f(total) else true) else false;
        if (required == null) {
            std.log.info("qwen27: the Neural Engine share has no qualified build/runtime overhead bound; prompts run on the GPU alone", .{});
        } else if (!fits) {
            if (room) |free| {
                std.log.info("qwen27: the Neural Engine's prompt share ({d} MiB of fp16 weights) would take this Mac " ++
                    "past its load limit ({d} MiB left after its drafter, streams and images); prompts run on the GPU alone", .{ bytes >> 20, free >> 20 });
            } else {
                std.log.info("qwen27: the Neural Engine's prompt share ({d} MiB of fp16 weights) would take this Mac past its load limit; prompts run on the GPU alone", .{bytes >> 20});
            }
        }
        if (share and fits) m.ane = ane_mlp.Share.init(m.allocator, m.device, m.queue, &m.checkpoint, c, columns, slices) catch |err| blk: {
            std.log.warn("qwen27: the Neural Engine's prompt share did not load ({s}); prompts run on the GPU alone", .{@errorName(err)});
            break :blk null;
        };
        const loaded_columns = if (m.ane) |loaded| loaded.columns else 0;
        std.log.info("qwen27: Neural Engine prompt share requested_columns={d} loaded_columns={d} program_weight_bytes={d} loaded={any} exact_output_qualified={any}", .{ columns, loaded_columns, ane_mlp.Share.bytes(c, loaded_columns), m.ane != null, ane_policy.exact_output_qualified });
    }

    /// The loaded share's price still absent from the process footprint; no loaded share reserves nothing.
    pub fn shareCacheBytes(m: *const Model) !u64 {
        return if (m.ane) |share| share.cacheBytes(m.config) else 0;
    }

    pub fn deinit(m: *Model) void {
        if (m.ane) |share| share.deinit();
        m.frame.deinit();
        m.glue.deinit();
        m.kernels.deinit();
        m.weights.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.allocator.destroy(m);
    }
};

/// A prompt pass's requested rows, before the runner's frame and attention limits.
pub fn promptRows(generation: u32) u32 {
    const value = if (std.c.getenv("TF_PROMPT_ROWS")) |v| std.mem.span(v) else null;
    return @import("prompt_rows.zig").requested(generation, value);
}

/// Kernel-compatible geometry from config; checkpoint qualification is a separate device gate.
pub const supported = @import("geometry.zig").supported;

fn qualifiedGeneration(generation: u32) bool {
    return generation >= 13 and generation <= 18;
}

test "unknown and older GPU generations refuse before any weight load" {
    try std.testing.expect(qualifiedGeneration(13));
    try std.testing.expect(qualifiedGeneration(15));
    try std.testing.expect(qualifiedGeneration(17));
    try std.testing.expect(!qualifiedGeneration(0));
    try std.testing.expect(!qualifiedGeneration(12));
    try std.testing.expect(qualifiedGeneration(18));
    try std.testing.expect(!qualifiedGeneration(19));
}

test "the 27B's Neural Engine share of 3,072 columns holds 5,760 MiB of fp16 weights" {
    var c: cfg.Config = undefined;
    c.hidden = 5120;
    c.layers = 64;
    try std.testing.expectEqual(@as(u64, 5760 << 20), ane_mlp.Share.bytes(c, 3072));
}
