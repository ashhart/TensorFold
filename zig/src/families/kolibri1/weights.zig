//! Kolibri-1's weights on the GPU: each layer packed on the host, then uploaded; projections as qlinear fp8g weights.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const names = @import("names.zig");
const pack = @import("pack.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const fp8 = cuda.fp8;

pub const Layer = struct {
    input_norm: u64, // fp32 [D]
    qkv: cuda.qlinear.Weight, // fp8g
    q_norm: u64, // fp32 [head_dim]
    k_norm: u64,
    o: cuda.qlinear.Weight,
    post_attn_norm: u64,
    pre_moe_norm: u64,
    router: u64, // bf16 [E, D]
    bias: u64, // fp32 [E]
    experts: cuda.experts.Layer, // E routed, then the shared expert (format fp8g)
    post_moe_norm: u64,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    embed: u64 = 0, // bf16 [V, D]
    norm: u64 = 0, // fp32 [D]
    head: u64 = 0, // bf16 [V, D]
    layers: []Layer = &.{},
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        const b = try DeviceBuffer.fromHost(d, bytes);
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }

    fn widened(w: *Weights, d: *const cuda.Driver, t: core.checkpoint.Tensor) !u64 {
        const v = try pack.widen(w.gpa, t);
        defer w.gpa.free(v);
        return w.upload(d, std.mem.sliceAsBytes(v));
    }

    fn linear(w: *Weights, d: *const cuda.Driver, l: pack.Linear) !cuda.qlinear.Weight {
        return .{ .fp8g = fp8.weight(try w.upload(d, l.w8), try w.upload(d, std.mem.sliceAsBytes(l.bs)), l.n, l.k) };
    }
};

/// config.json, every tensor mapped and checked, then layer by layer; `layers` caps them (tests).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, layers: ?usize) !Weights {
    const c = try Config.read(gpa, io, dir);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var m = try names.map(gpa, &ck, c);
    defer m.deinit();
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    const count = @min(layers orelse c.layers, c.layers);
    w.layers = try gpa.alloc(Layer, count);
    w.embed = try w.upload(d, m.embed.bytes);
    for (w.layers, m.layers[0..count]) |*out, ml| {
        var h = try pack.layer(gpa, ml, c);
        defer h.deinit(gpa);
        out.* = .{
            .input_norm = try w.upload(d, std.mem.sliceAsBytes(h.input_norm)),
            .qkv = try w.linear(d, h.qkv),
            .q_norm = try w.upload(d, std.mem.sliceAsBytes(h.q_norm)),
            .k_norm = try w.upload(d, std.mem.sliceAsBytes(h.k_norm)),
            .o = try w.linear(d, h.o),
            .post_attn_norm = try w.upload(d, std.mem.sliceAsBytes(h.post_attn_norm)),
            .pre_moe_norm = try w.upload(d, std.mem.sliceAsBytes(h.pre_moe_norm)),
            .router = try w.upload(d, h.router),
            .bias = try w.upload(d, std.mem.sliceAsBytes(h.bias)),
            .experts = .{
                .format = .fp8g,
                .up = try w.upload(d, std.mem.sliceAsBytes(h.up)),
                .down = try w.upload(d, std.mem.sliceAsBytes(h.down)),
                .up_scale = try w.upload(d, std.mem.sliceAsBytes(h.up_scale)),
                .down_scale = try w.upload(d, std.mem.sliceAsBytes(h.down_scale)),
                .width = c.moe_width,
                .dims = c.hidden,
                .experts = c.experts + 1,
            },
            .post_moe_norm = try w.widened(d, ml.post_moe_norm),
        };
    }
    w.norm = try w.widened(d, m.norm);
    w.head = try w.upload(d, m.head.bytes);
    return w;
}
