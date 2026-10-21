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
const moe_gpu = @import("moe_gpu.zig");
const Frame = @import("gpu_frame.zig").Frame;
const ane_mlp = @import("ane_mlp.zig");

const ane_policy = @import("ane_policy.zig");

pub const Model = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: core.checkpoint_metal.Checkpoint,
    weights: wts.Weights,
    kernels: quant.Kernels,
    glue: glue.Kernels,
    /// The routed block's kernels, with experts.
    moe: ?moe_gpu.Kernels = null,
    frame: Frame,
    /// The Neural Engine's share of prompt-chunk MLPs, when it loaded.
    ane: ?*ane_mlp.Share = null,
    /// widths.Widths.tag: 0 for a uniform 4-bit g64 checkpoint.
    widths_tag: u64 = 0,

    /// `share_fits(bytes)`: whether the Neural Engine's share fits beside what is loaded (null: wherever it can load).
    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: u32, share_fits: ?*const fn (u64) bool) !*Model {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
        defer a.free(text);
        var c = try cfg.parse(a, text, null);
        try supported(c);
        // TF_LAYERS=N runs the first N layers only (a port's layer-by-layer check against a reference)
        if (std.c.getenv("TF_LAYERS")) |v| c.layers = @min(c.layers, std.fmt.parseInt(usize, std.mem.span(v), 10) catch c.layers);
        var formats = try affine.Formats.init(a, text);
        defer formats.deinit();
        var ws = try widths.Widths.resolve(a, &formats, c);
        defer ws.deinit();
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.allocator = a;
        m.config = c;
        m.widths_tag = ws.tag();
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        if (!qualifiedGeneration(m.device.generation())) return error.QwenMetalQualificationRequired;
        var why: widths.Why = .{};
        widths.admit(&ws, c, .{ .tensor_units = m.device.tensorUnits() }, &why) catch |err| {
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
        m.weights = try wts.load(a, m.device, &m.checkpoint, c, &ws);
        errdefer m.weights.deinit();
        m.kernels = .{ .allocator = a, .device = m.device, .raw_tiles = !m.device.tensorUnits() };
        errdefer m.kernels.deinit();
        m.glue = try glue.Kernels.init(m.device);
        errdefer m.glue.deinit();
        m.moe = if (c.moe()) try moe_gpu.Kernels.init(a, m.device) else null;
        errdefer if (m.moe) |*k| k.deinit();
        m.frame = try Frame.init(m.device, c, @max(rows, promptRows(c)));
        errdefer m.frame.deinit();
        m.ane = null;
        const columns = try ane_policy.columns(if (platform.c.getenv("TF_ANE_COLUMNS")) |v| std.mem.span(v) else null);
        const share = ane_policy.requestedOnChip(columns, m.device.tensorUnits()) and mtl.ane.available() and m.weights.uniformMlp();
        const fits = if (share_fits) |f| !share or f(ane_mlp.Share.bytes(c, columns)) else true;
        if (!fits) std.log.info("qwen27: the Neural Engine's prompt share ({d} MiB of fp16 weights) would take this Mac past its load limit; prompts run on the GPU alone", .{ane_mlp.Share.bytes(c, columns) >> 20});
        if (share and fits) m.ane = ane_mlp.Share.init(a, m.device, m.queue, &m.checkpoint, c, columns, (m.frame.capacity + ane_mlp.width - 1) / ane_mlp.width) catch |err| blk: {
            std.log.warn("qwen27: the Neural Engine's prompt share did not load ({s}); prompts run on the GPU alone", .{@errorName(err)});
            break :blk null;
        };
        const loaded_columns = if (m.ane) |loaded| loaded.columns else 0;
        std.log.info("qwen27: Neural Engine prompt share requested_columns={d} loaded_columns={d} program_weight_bytes={d} loaded={any} exact_output_qualified={any}", .{ columns, loaded_columns, ane_mlp.Share.bytes(c, loaded_columns), m.ane != null, ane_policy.exact_output_qualified });
        m.checkpoint.closeFiles();
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.ane) |share| share.deinit();
        m.frame.deinit();
        if (m.moe) |*k| k.deinit();
        m.glue.deinit();
        m.kernels.deinit();
        m.weights.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.allocator.destroy(m);
    }
};

/// A prompt pass's rows: 512, or 1,024 with experts (a chunk reads every expert once); TF_PROMPT_ROWS: 16 to 2048.
pub fn promptRows(c: cfg.Config) u32 {
    const default: u32 = if (c.moe()) 1024 else 512;
    const v = std.c.getenv("TF_PROMPT_ROWS") orelse return default;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return default;
    return if (n >= 16 and n <= @import("gpu_frame.zig").max_rows and n % 16 == 0) n else default;
}

/// The qualified shape and layer schedule; per-module widths are admitted by widths.admit once the chip is known.
pub fn supported(c: cfg.Config) !void {
    const dense = c.hidden == 5120 and c.intermediate == 17408 and c.layers == 64 and c.heads == 24 and c.kv_heads == 4 and c.k_heads == 16 and c.v_heads == 48 and !c.moe();
    // Ornith-1.5-122B: 256 experts of 1,024 and a shared expert of 1,024 over 3,072, 32 heads over 2 of 256
    const routed = c.hidden == 3072 and c.intermediate == 1024 and c.layers == 48 and c.heads == 32 and c.kv_heads == 2 and c.k_heads == 16 and c.v_heads == 64 and c.experts == 256 and c.top_k == 8 and c.moe_intermediate == 1024;
    if (!(dense or routed) or c.vocab != 248320 or c.head_dim != 256 or c.dk != 128 or c.dv != 128 or c.conv_kernel != 4 or c.rope_dims != 64) return error.UnqualifiedQwenShape;
    for (c.kinds[0..c.layers], 0..) |kind, i| if (kind != (if ((i + 1) % 4 == 0) cfg.Kind.attention else cfg.Kind.linear)) return error.UnqualifiedLayerSchedule;
}

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
