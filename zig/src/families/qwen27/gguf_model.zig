//! GGUF model ownership joins the same runner and frame after exact host adaptation and source admission.
const std = @import("std");
const core = @import("core");
const mtl = @import("metal");
const model = @import("model.zig");
const adapter = @import("gguf.zig");
pub fn load(a: std.mem.Allocator, io: std.Io, path: []const u8, rows: u32, problem: *adapter.Problem) !*model.Model {
    var issue: core.gguf.FileProblem = .{};
    defer issue.deinit(a);
    var file = core.gguf.File.openDiagnosed(a, io, path, .{}, &issue) catch |err| {
        if (issue.name) |name| return problem.fail("{s}: unsupported GGML quantization type {d}", .{ name, issue.kind });
        return err;
    };
    defer file.close(io);
    try @import("gguf_projection.zig").admit(&file.document, problem);
    const c = try adapter.metadata.parse(&file.document, problem);
    try model.supported(c);
    if (c.hidden != 5120 or c.layers != 64) return problem.fail("qwen35 GGUF serving: only the dense 27B geometry is supported", .{});
    var source = try adapter.Adapter.open(a, file.document.bytes, problem);
    defer source.deinit();
    const m = try a.create(model.Model);
    errdefer a.destroy(m);
    m.allocator = a;
    m.config = c;
    m.shard = .{ .rank = 0, .ranks = 1, .global = c, .local = c };
    m.device = try mtl.Device.init();
    errdefer m.device.deinit();
    if (m.device.generation() < 13 or m.device.generation() > 18) return error.QwenMetalQualificationRequired;
    m.queue = try m.device.queue();
    errdefer m.queue.deinit();
    m.checkpoint = core.checkpoint_metal.Checkpoint.init(a);
    errdefer m.checkpoint.deinit();
    m.weights = try @import("gguf_weights.zig").load(a, m.device, &source, problem);
    errdefer m.weights.deinit();
    m.kernels = .{ .allocator = a, .device = m.device };
    errdefer m.kernels.deinit();
    m.kernels.gguf = try @import("gguf_ops.zig").Kernels.init(a, m.device, &m.weights.owned, c.hidden, c.intermediate);
    m.glue = try @import("glue_gpu.zig").Kernels.init(m.device);
    errdefer m.glue.deinit();
    m.glue.gain = .f32;
    m.frame = try @import("gpu_frame.zig").Frame.init(m.device, c, @max(rows, model.promptRows(c, 0)));
    m.ane = null;
    m.share_pending = false;
    m.widths_tag = std.hash.Wyhash.hash(0x47475546, file.document.bytes[0..file.document.data_offset]);
    return m;
}
