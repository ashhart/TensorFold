//! Generated draft tensors exercise the real callback backend without checkpoint files.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const df = q.dflash;
const Backend = df.runtime_backend.Backend;
const Model = df.runtime_model.Model;
pub const config = df.config.Config{ .hidden = 512, .intermediate = 512, .vocab = 96, .heads = 8, .kv_heads = 2, .head_dim = 64, .target_layers = 5, .mask = 95, .window = 64, .rank = 32, .topk = 4, .taps = .{ 0, 1, 2, 3, 4 } };
fn dimensions(name: []const u8) []const usize {
    if (std.mem.eql(u8, name, "fc.weight")) return &.{ 512, 2560 };
    if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return &.{ 32, 512 };
    if (std.mem.startsWith(u8, name, "candidate_selector.")) return &.{ 96, 32 };
    if (std.mem.endsWith(u8, name, "base_kernel")) return &.{ 2, 2, 512 };
    if (std.mem.endsWith(u8, name, "kernel_projection.weight")) return &.{ 128, 512 };
    if (std.mem.endsWith(u8, name, "k_proj.weight") or std.mem.endsWith(u8, name, "v_proj.weight")) return &.{ 128, 512 };
    if (std.mem.endsWith(u8, name, "q_norm.weight") or std.mem.endsWith(u8, name, "k_norm.weight")) return &.{64};
    if (std.mem.endsWith(u8, name, "proj.weight")) return &.{ 512, 512 };
    return &.{512};
}
pub fn create(a: std.mem.Allocator, target: *q.model.Model) !*Model {
    const m = try a.create(Model);
    errdefer a.destroy(m);
    m.allocator = a;
    m.tree_block = 16;
    m.committed_end = 0;
    m.checkpoint = tf.checkpoint_metal.Checkpoint.init(a);
    errdefer m.checkpoint.deinit();
    const names = try df.weights.names(a, config);
    defer df.weights.freeNames(a, names);
    var random: std.Random.DefaultPrng = .init(5051);
    const rng = random.random();
    for (names) |name| {
        const shape = dimensions(name);
        var count: usize = 1;
        for (shape) |dimension| count *= dimension;
        const buffer = try target.device.buffer(count * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        m.checkpoint.shards.append(a, .{ .buffer = buffer, .bytes = count * 2 }) catch |err| {
            buffer.deinit();
            return err;
        };
        const norm = std.mem.endsWith(u8, name, "norm.weight") or std.mem.endsWith(u8, name, "layernorm.weight");
        for (buffer.slice(u16, count), 0..) |*item, i| {
            const value: f32 = if (norm) 1 else if (std.mem.endsWith(u8, name, "base_kernel")) (if ((i / 512) % 2 == 0) @as(f32, 1) else 0) else @as(f32, @floatFromInt(rng.intRangeLessThan(i32, -8, 9))) / 1024;
            item.* = @truncate(@as(u32, @bitCast(value)) >> 16);
        }
        var tensor = tf.checkpoint_metal.Tensor{ .buffer = buffer, .offset = 0, .bytes = count * 2, .dtype = .bf16, .rank = shape.len };
        @memcpy(tensor.shape[0..shape.len], shape);
        try m.checkpoint.tensors.put(a, try a.dupe(u8, name), tensor);
    }
    m.backend = try Backend.init(a, target, &m.checkpoint, config);
    errdefer m.backend.deinit();
    m.graph = try df.runtime_model.loadGraph(a, &m.backend);
    m.pending = try target.device.buffer(@as(usize, config.window) * config.tapWidth() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer m.pending.deinit();
    m.session = try df.session.Session.init(config, 1, 0);
    m.preparation = .{ .mode = .bf16_reference };
    return m;
}
