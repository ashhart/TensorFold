//! The forward's view of an uploaded model: projections, activation dtype and normalize_qk's constants.

const std = @import("std");
const hip = @import("hip");
const view = @import("view.zig");
const weights = @import("weights.zig");

pub const Error = error{UnsupportedTables} || hip.Error || std.mem.Allocator.Error;

/// A projection (N, K) from its uploaded tensors, in whichever format they are.
pub fn projection(p: weights.Projection) Error!view.Projection {
    return hip.quant.view(p, false);
}

/// A projection split along K: its fp32 output is one rank's share of the sum.
fn share(p: weights.Projection, sliced: bool) Error!view.Projection {
    var a = try projection(p);
    a.partial = sliced;
    return a;
}

/// An uploaded MLP as the forward reads it: a dense one's projections, or the routed experts (rank share if `sliced`).
pub fn mlpView(m: weights.Mlp, sliced: bool) Error!view.Mlp {
    return switch (m) {
        .dense => |d| .{ .dense = .{ .gate = try projection(d.gate), .up = try projection(d.up), .down = try share(d.down, sliced) } },
        .routed => |r| .{ .moe = .{
            .rows32 = r.rows32.ptr,
            .top_k = r.top_k,
            .rows = r.router.dim(0),
            .remap = if (r.remap) |b| b.ptr else 0,
            .experts = .{
                .fused = try hip.quant.view(r.experts.fused, true),
                .down = try hip.quant.view(r.experts.down, true),
                .count = r.experts.count,
                .width = r.experts.width,
                .dims = r.experts.dims,
            },
        } },
    };
}

pub const Bridge = struct {
    gpa: std.mem.Allocator,
    layers: []view.Layer,
    q_weight: hip.DeviceBuffer,
    k_weight: hip.DeviceBuffer,
    model: view.Model,

    /// The view of `m` for activations of `act` (fp16 on RDNA2, bf16 on gfx11 / gfx12).
    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, m: *const weights.Model, act: view.Kind) !*Bridge {
        const s = m.spec;
        const b = try gpa.create(Bridge);
        errdefer gpa.destroy(b);
        b.gpa = gpa;
        b.layers = try gpa.alloc(view.Layer, m.layers.len);
        errdefer gpa.free(b.layers);
        for (m.layers, b.layers) |l, *out| out.* = switch (l) {
            .full => |f| .{ .full = .{ .input_norm = f.input_norm.ptr, .post_norm = f.post_norm.ptr, .q = try projection(f.q), .k = try projection(f.k), .v = try projection(f.v), .o = try share(f.o, m.sliced), .q_norm = f.q_norm.ptr, .k_norm = f.k_norm.ptr, .mlp = try mlpView(f.mlp, m.sliced) } },
            .linear => |x| .{ .linear = .{ .input_norm = x.input_norm.ptr, .post_norm = x.post_norm.ptr, .qkv = try projection(x.qkv), .z = try projection(x.z), .a = try projection(x.a), .b = try projection(x.b), .out = try share(x.out, m.sliced), .conv = x.conv.ptr, .a_log = x.a_log.ptr, .dt_bias = x.dt_bias.ptr, .gnorm = x.gnorm.ptr, .mlp = try mlpView(x.mlp, m.sliced) } },
        };
        const c = view.qkConstants(s.key_dim, s.eps);
        const qs = try gpa.alloc(f32, s.key_dim);
        defer gpa.free(qs);
        @memset(qs, c.q);
        b.q_weight = try hip.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(qs));
        errdefer b.q_weight.free();
        @memset(qs, c.k);
        b.k_weight = try hip.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(qs));
        b.model = .{
            .spec = s,
            .act = act,
            .embed = try projection(m.embed),
            .layers = b.layers,
            .final_norm = m.final_norm.ptr,
            .head = try projection(m.outputHead()),
            .qk = .{ .q_weight = b.q_weight.base(), .k_weight = b.k_weight.base(), .eps = c.eps },
        };
        return b;
    }

    pub fn deinit(b: *Bridge) void {
        b.q_weight.free();
        b.k_weight.free();
        b.gpa.free(b.layers);
        b.gpa.destroy(b);
    }
};
