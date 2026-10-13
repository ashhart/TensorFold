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
/// A tensor's generated bf16 values: norms 1, base kernels a fixed pattern, the rest small random multiples of 1/1024.
fn fill(name: []const u8, out: []u16, rng: std.Random) void {
    const norm = std.mem.endsWith(u8, name, "norm.weight") or std.mem.endsWith(u8, name, "layernorm.weight");
    for (out, 0..) |*item, i| {
        const value: f32 = if (norm) 1 else if (std.mem.endsWith(u8, name, "base_kernel")) (if ((i / 512) % 2 == 0) @as(f32, 1) else 0) else @as(f32, @floatFromInt(rng.intRangeLessThan(i32, -8, 9))) / 1024;
        item.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
}

/// The same generated tensors written as a safetensors file and opened as the server opens a q4 drafter.
pub fn createQ4(a: std.mem.Allocator, target: *q.model.Model) !*Model {
    const names = try df.weights.names(a, config);
    defer df.weights.freeNames(a, names);
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    var data: std.ArrayList(u16) = .empty;
    defer data.deinit(a);
    var random: std.Random.DefaultPrng = .init(5051);
    try header.writer.writeByte('{');
    for (names, 0..) |name, n| {
        const shape = dimensions(name);
        var count: usize = 1;
        for (shape) |dimension| count *= dimension;
        const first = data.items.len * 2;
        fill(name, try data.addManyAsSlice(a, count), random.random());
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"BF16\",\"shape\":{f},\"data_offsets\":[{d},{d}]}}", .{ if (n == 0) "" else ",", name, std.json.fmt(shape, .{}), first, data.items.len * 2 });
    }
    try header.writer.writeByte('}');
    var path_buf: [96]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-q27-q4-draft-{d}.safetensors", .{std.c.getpid()}, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.Create;
    defer _ = std.c.unlink(path);
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, header.written().len, .little);
    for ([_][]const u8{ &head, header.written(), std.mem.sliceAsBytes(data.items) }) |part| {
        if (std.c.write(fd, part.ptr, part.len) != @as(isize, @intCast(part.len))) {
            _ = std.c.close(fd);
            return error.ShortWrite;
        }
    }
    _ = std.c.close(fd);
    return Model.open(a, target, config, &.{path}, .prepared_q4_reference);
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
        fill(name, buffer.slice(u16, count), rng);
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
