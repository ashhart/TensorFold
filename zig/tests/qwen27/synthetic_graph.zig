//! Small hybrid graph: generated affine weights, three GDN layers, one gated attention layer, no files.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const core = @import("tensorfold");

const Builder = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    cp: *core.checkpoint_metal.Checkpoint,
    seed: u32 = 0x2729,
    fn tensor(b: *Builder, name: []const u8, shape: []const usize, dtype: core.checkpoint_metal.DType, fill: f32) !void {
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        const bytes = count * dtype.size();
        const buf = try b.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        b.cp.shards.append(b.a, .{ .buffer = buf, .bytes = bytes }) catch |err| {
            buf.deinit();
            return err;
        };
        if (dtype == .u32) {
            for (buf.slice(u32, count)) |*v| {
                b.seed ^= b.seed << 13;
                b.seed ^= b.seed >> 17;
                b.seed ^= b.seed << 5;
                v.* = b.seed;
            }
        } else @memset(buf.slice(u16, count), @as(u16, @truncate(@as(u32, @bitCast(fill)) >> 16)));
        var t = core.checkpoint_metal.Tensor{ .buffer = buf, .offset = 0, .bytes = bytes, .dtype = dtype, .rank = shape.len };
        @memcpy(t.shape[0..shape.len], shape);
        const owned = try std.mem.concat(b.a, u8, &.{ "language_model.", name });
        errdefer b.a.free(owned);
        try b.cp.tensors.put(b.a, owned, t);
    }
    fn matrix(b: *Builder, name: []const u8, n: usize, k: usize) !void {
        try b.tensor(try std.fmt.allocPrint(b.a, "{s}.weight", .{name}), &.{ n, k / 8 }, .u32, 0);
        try b.tensor(try std.fmt.allocPrint(b.a, "{s}.scales", .{name}), &.{ n, k / 64 }, .bf16, 0.0078125);
        try b.tensor(try std.fmt.allocPrint(b.a, "{s}.biases", .{name}), &.{ n, k / 64 }, .bf16, -0.05859375);
    }
    fn layerName(b: Builder, layer: usize, leaf: []const u8) ![]const u8 {
        return std.fmt.allocPrint(b.a, "model.layers.{d}.{s}", .{ layer, leaf });
    }
};
pub fn create(a: std.mem.Allocator) !*q.model.Model {
    return createLayers(a, 4);
}
pub fn createLayers(a: std.mem.Allocator, layers: usize) !*q.model.Model {
    if (layers == 0 or layers > 5) return error.BadSyntheticLayerCount;
    var c = q.config.Config{
        .hidden = 512,
        .intermediate = 512,
        .layers = layers,
        .vocab = 96,
        .heads = 24,
        .kv_heads = 4,
        .head_dim = 256,
        .k_heads = 16,
        .v_heads = 48,
        .dk = 128,
        .dv = 128,
        .conv_kernel = 4,
        .eps = 1e-6,
        .rope_dims = 64,
        .rope_theta = 10000000,
    };
    c.kinds[3] = .attention;
    c.eos[0] = 95;
    c.eos_count = 1;
    const device = try mtl.Device.init();
    errdefer device.deinit();
    const queue = try device.queue();
    errdefer queue.deinit();
    var cp = core.checkpoint_metal.Checkpoint.init(a);
    errdefer cp.deinit();
    var builder = Builder{ .a = a, .device = device, .cp = &cp };
    try builder.matrix("model.embed_tokens", c.vocab, c.hidden);
    try builder.matrix("lm_head", c.vocab, c.hidden);
    try builder.tensor("model.norm.weight", &.{c.hidden}, .bf16, 1);
    for (0..c.layers) |i| {
        try builder.tensor(try builder.layerName(i, "input_layernorm.weight"), &.{c.hidden}, .bf16, 1);
        try builder.tensor(try builder.layerName(i, "post_attention_layernorm.weight"), &.{c.hidden}, .bf16, 1);
        inline for (.{ "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |leaf| try builder.matrix(try builder.layerName(i, leaf), 512, 512);
        if (c.kind(i) == .linear) {
            inline for (.{ .{ "linear_attn.in_proj_qkv", 10240 }, .{ "linear_attn.in_proj_z", 6144 }, .{ "linear_attn.in_proj_b", 48 }, .{ "linear_attn.in_proj_a", 48 } }) |part| try builder.matrix(try builder.layerName(i, part[0]), part[1], 512);
            try builder.matrix(try builder.layerName(i, "linear_attn.out_proj"), 512, 6144);
            try builder.tensor(try builder.layerName(i, "linear_attn.conv1d.weight"), &.{ 10240, 4, 1 }, .bf16, 0.25);
            try builder.tensor(try builder.layerName(i, "linear_attn.A_log"), &.{48}, .bf16, -0.5);
            try builder.tensor(try builder.layerName(i, "linear_attn.dt_bias"), &.{48}, .bf16, 0);
            try builder.tensor(try builder.layerName(i, "linear_attn.norm.weight"), &.{128}, .bf16, 1);
        } else {
            inline for (.{ .{ "self_attn.q_proj", 12288 }, .{ "self_attn.k_proj", 1024 }, .{ "self_attn.v_proj", 1024 } }) |part| try builder.matrix(try builder.layerName(i, part[0]), part[1], 512);
            try builder.matrix(try builder.layerName(i, "self_attn.o_proj"), 512, 6144);
            inline for (.{ "self_attn.q_norm.weight", "self_attn.k_norm.weight" }) |leaf| try builder.tensor(try builder.layerName(i, leaf), &.{256}, .bf16, 1);
        }
    }
    var weights = try q.gpu_weights.load(a, device, &cp, c);
    errdefer weights.deinit();
    const glue = try q.glue_gpu.Kernels.init(device);
    errdefer glue.deinit();
    const frame = try q.gpu_frame.Frame.init(device, c, 128);
    errdefer {
        var f = frame;
        f.deinit();
    }
    const m = try a.create(q.model.Model);
    m.* = .{ .allocator = a, .device = device, .queue = queue, .config = c, .checkpoint = cp, .weights = weights, .kernels = .{ .allocator = a, .device = device }, .glue = glue, .frame = frame };
    return m;
}
