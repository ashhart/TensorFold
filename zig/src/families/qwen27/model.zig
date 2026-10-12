//! A native text model keeps immutable checkpoint bytes and prepared affine layouts under one owner.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const cfg = @import("config.zig");
const affine = @import("affine.zig");
const widths = @import("widths.zig");
const wts = @import("gpu_weights.zig");
const quant = @import("quant_gpu.zig");
const glue = @import("glue_gpu.zig");
const Frame = @import("gpu_frame.zig").Frame;

/// Gate/up columns a layer the Neural Engine computes for prompt chunks (TF_ANE_COLUMNS overrides); M5 only.
const ane_columns = 3072;

pub const Model = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: core.checkpoint_metal.Checkpoint,
    weights: wts.Weights,
    kernels: quant.Kernels,
    glue: glue.Kernels,
    frame: Frame,
    /// The Neural Engine's share of prompt-chunk MLPs, when it loaded.
    ane: ?*@import("ane_mlp.zig").Share = null,
    /// widths.Widths.tag: 0 for a uniform 4-bit g64 checkpoint.
    widths_tag: u64 = 0,

    pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, rows: u32) !*Model {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
        defer a.free(text);
        const c = try cfg.parse(a, text, null);
        try supported(c);
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
        m.kernels = .{ .allocator = a, .device = m.device };
        errdefer m.kernels.deinit();
        m.glue = try glue.Kernels.init(m.device);
        errdefer m.glue.deinit();
        m.frame = try Frame.init(m.device, c, rows);
        errdefer m.frame.deinit();
        m.ane = null;
        const columns = if (std.c.getenv("TF_ANE_COLUMNS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else ane_columns;
        if (columns > 0 and m.device.tensorUnits() and mtl.ane.available() and m.weights.uniformMlp()) m.ane = @import("ane_mlp.zig").Share.init(a, m.device, m.queue, &m.checkpoint, c, columns) catch |err| blk: {
            std.log.warn("qwen27: the Neural Engine's prompt share did not load ({s}); prompts run on the GPU alone", .{@errorName(err)});
            break :blk null;
        };
        m.checkpoint.closeFiles();
        return m;
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

/// The qualified shape and layer schedule; per-module widths are admitted by widths.admit once the chip is known.
pub fn supported(c: cfg.Config) !void {
    if (c.hidden != 5120 or c.intermediate != 17408 or c.layers != 64 or c.vocab != 248320 or c.heads != 24 or c.kv_heads != 4 or c.head_dim != 256 or c.k_heads != 16 or c.v_heads != 48 or c.dk != 128 or c.dv != 128 or c.conv_kernel != 4 or c.rope_dims != 64) return error.UnqualifiedQwenShape;
    for (c.kinds[0..c.layers], 0..) |kind, i| if (kind != (if ((i + 1) % 4 == 0) cfg.Kind.attention else cfg.Kind.linear)) return error.UnqualifiedLayerSchedule;
}

fn qualifiedGeneration(generation: u32) bool {
    return generation >= 13 and generation <= 17;
}

test "unknown and older GPU generations refuse before any weight load" {
    try std.testing.expect(qualifiedGeneration(13));
    try std.testing.expect(qualifiedGeneration(15));
    try std.testing.expect(qualifiedGeneration(17));
    try std.testing.expect(!qualifiedGeneration(0));
    try std.testing.expect(!qualifiedGeneration(12));
    try std.testing.expect(!qualifiedGeneration(18));
}
