//! The complete target graph borrows native checkpoint tensors; only its layer records are owned here.
const std = @import("std");
const Config = @import("config.zig").Config;
const Kind = @import("config.zig").Kind;
const ck = @import("checkpoint.zig");
const Formats = @import("affine.zig").Formats;
pub const Tensor = ck.Tensor;
pub const Linear = ck.Linear;
pub const Mlp = struct { gate: Linear, up: Linear, down: Linear };
pub const Gdn = struct { qkv: Linear, z: Linear, b: Linear, a: Linear, out: Linear, conv: Tensor, a_log: Tensor, dt_bias: Tensor, norm: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, o: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Mixer = union(Kind) { linear: Gdn, attention: Attention };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, mlp: Mlp, mixer: Mixer };
pub const Weights = struct {
    gpa: std.mem.Allocator,
    prefix: []const u8,
    embed: Linear,
    norm: Tensor,
    head: Linear,
    layers: []Layer,

    pub fn deinit(w: *Weights) void {
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    pub fn tensorBytes(w: Weights) !usize {
        var total = try std.math.add(usize, linearBytes(w.embed), linearBytes(w.head));
        total = try std.math.add(usize, total, w.norm.bytes.len);
        for (w.layers) |l| {
            total = try std.math.add(usize, total, l.input_norm.bytes.len + l.post_norm.bytes.len);
            for ([_]Linear{ l.mlp.gate, l.mlp.up, l.mlp.down }) |p| total = try std.math.add(usize, total, linearBytes(p));
            switch (l.mixer) {
                .linear => |g| {
                    for ([_]Linear{ g.qkv, g.z, g.b, g.a, g.out }) |p| total = try std.math.add(usize, total, linearBytes(p));
                    for ([_]Tensor{ g.conv, g.a_log, g.dt_bias, g.norm }) |t| total = try std.math.add(usize, total, t.bytes.len);
                },
                .attention => |a| {
                    for ([_]Linear{ a.q, a.k, a.v, a.o }) |p| total = try std.math.add(usize, total, linearBytes(p));
                    total = try std.math.add(usize, total, a.q_norm.bytes.len + a.k_norm.bytes.len);
                },
            }
        }
        return total;
    }
};

fn linearBytes(p: Linear) usize {
    return switch (p) {
        .dense => |t| t.bytes.len,
        .quantized => |q| q.weight.bytes.len + q.scales.bytes.len + q.biases.bytes.len,
    };
}

fn present(src: ck.Source, name: []const u8) !bool {
    _ = src.get(name) catch |err| switch (err) {
        error.MissingTensor => return false,
        else => return err,
    };
    return true;
}

fn namespace(src: ck.Source) ![]const u8 {
    var found: ?[]const u8 = null;
    for ([_][]const u8{ "", "language_model." }) |prefix| {
        var name: [96]u8 = undefined;
        if (try present(src, try std.fmt.bufPrint(&name, "{s}model.embed_tokens.weight", .{prefix}))) {
            if (found != null) return error.AmbiguousTextNamespace;
            found = prefix;
        }
    }
    return found orelse error.MissingTensor;
}

const Builder = struct {
    gpa: std.mem.Allocator,
    src: ck.Source,
    formats: *const Formats,
    prefix: []const u8,

    fn name(b: Builder, out: []u8, module: []const u8) ![]const u8 {
        return std.fmt.bufPrint(out, "{s}{s}", .{ b.prefix, module });
    }

    fn linear(b: Builder, module: []const u8, n: usize, k: usize) !Linear {
        var out: [160]u8 = undefined;
        return ck.linear(b.gpa, b.src, b.formats, try b.name(&out, module), n, k);
    }

    fn tensor(b: Builder, module: []const u8, dims: []const usize) !Tensor {
        var out: [160]u8 = undefined;
        const t = try b.src.get(try b.name(&out, module));
        if (!ck.floating(t.dtype)) return error.ProjectionDType;
        try ck.shape(t, dims);
        return t;
    }

    fn conv(b: Builder, module: []const u8, c: Config) !Tensor {
        var out: [160]u8 = undefined;
        const t = try b.src.get(try b.name(&out, module));
        if (!ck.floating(t.dtype)) return error.ProjectionDType;
        if (t.rank == 2) {
            try ck.shape(t, &.{ c.gdnQkvDim(), c.conv_kernel });
        } else if (t.rank == 3 and t.shape[1] == c.conv_kernel and t.shape[2] == 1) {
            try ck.shape(t, &.{ c.gdnQkvDim(), c.conv_kernel, 1 });
        } else if (t.rank == 3 and t.shape[1] == 1 and t.shape[2] == c.conv_kernel) {
            try ck.shape(t, &.{ c.gdnQkvDim(), 1, c.conv_kernel });
        } else return error.ProjectionShape;
        return t;
    }

    fn layer(b: Builder, c: Config, i: usize) !Layer {
        var name_buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.", .{i});
        var arena = std.heap.ArenaAllocator.init(b.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const N = struct {
            fn full(alloc: std.mem.Allocator, prefix: []const u8, suffix: []const u8) ![]const u8 {
                return std.mem.concat(alloc, u8, &.{ prefix, suffix });
            }
        };
        const mixer: Mixer = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.linear(try N.full(a, p, "linear_attn.in_proj_qkv"), c.gdnQkvDim(), c.hidden),
                .z = try b.linear(try N.full(a, p, "linear_attn.in_proj_z"), c.gdnValueDim(), c.hidden),
                .b = try b.linear(try N.full(a, p, "linear_attn.in_proj_b"), c.v_heads, c.hidden),
                .a = try b.linear(try N.full(a, p, "linear_attn.in_proj_a"), c.v_heads, c.hidden),
                .out = try b.linear(try N.full(a, p, "linear_attn.out_proj"), c.hidden, c.gdnValueDim()),
                .conv = try b.conv(try N.full(a, p, "linear_attn.conv1d.weight"), c),
                .a_log = try b.tensor(try N.full(a, p, "linear_attn.A_log"), &.{c.v_heads}),
                .dt_bias = try b.tensor(try N.full(a, p, "linear_attn.dt_bias"), &.{c.v_heads}),
                .norm = try b.tensor(try N.full(a, p, "linear_attn.norm.weight"), &.{c.dv}),
            } },
            .attention => .{ .attention = .{
                .q = try b.linear(try N.full(a, p, "self_attn.q_proj"), c.qDim(), c.hidden),
                .k = try b.linear(try N.full(a, p, "self_attn.k_proj"), c.kvDim(), c.hidden),
                .v = try b.linear(try N.full(a, p, "self_attn.v_proj"), c.kvDim(), c.hidden),
                .o = try b.linear(try N.full(a, p, "self_attn.o_proj"), c.hidden, c.heads * c.head_dim),
                .q_norm = try b.tensor(try N.full(a, p, "self_attn.q_norm.weight"), &.{c.head_dim}),
                .k_norm = try b.tensor(try N.full(a, p, "self_attn.k_norm.weight"), &.{c.head_dim}),
            } },
        };
        return .{
            .input_norm = try b.tensor(try N.full(a, p, "input_layernorm.weight"), &.{c.hidden}),
            .post_norm = try b.tensor(try N.full(a, p, "post_attention_layernorm.weight"), &.{c.hidden}),
            .mixer = mixer,
            .mlp = .{
                .gate = try b.linear(try N.full(a, p, "mlp.gate_proj"), c.intermediate, c.hidden),
                .up = try b.linear(try N.full(a, p, "mlp.up_proj"), c.intermediate, c.hidden),
                .down = try b.linear(try N.full(a, p, "mlp.down_proj"), c.hidden, c.intermediate),
            },
        };
    }
};

pub fn load(gpa: std.mem.Allocator, src: ck.Source, formats: *const Formats, c: Config) !Weights {
    const b = Builder{ .gpa = gpa, .src = src, .formats = formats, .prefix = try namespace(src) };
    const layers = try gpa.alloc(Layer, c.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| l.* = try b.layer(c, i);
    return .{ .gpa = gpa, .prefix = b.prefix, .layers = layers, .embed = try b.linear("model.embed_tokens", c.vocab, c.hidden), .norm = try b.tensor("model.norm.weight", &.{c.hidden}), .head = try b.linear("lm_head", c.vocab, c.hidden) };
}
