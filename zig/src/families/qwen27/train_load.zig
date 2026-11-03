//! Original affine4 words for the Metal input-gradient products; no serving pack or weight is changed.
const std = @import("std");
const core = @import("core");
const wts = @import("gpu_weights.zig");
const to = @import("train_ops.zig");
const keepLow = @import("train.zig").keepLow;
pub const Mlp = struct { gu: to.Q4 = undefined, down: to.Q4 = undefined };
pub const Gdn = struct { input: to.Q4, out: to.Q4, conv: to.Ref, a_log: to.Ref, dt: to.Ref, norm: to.Ref };
pub const Attn = struct { q: to.Q4, k: to.Q4, v: to.Q4, out: to.Q4, qs: to.Ref, ks: to.Ref, inv: to.Ref };
pub const Layer = struct { mlp: Mlp = .{}, gdn: ?Gdn = null, attn: ?Attn = null };
const Reader = struct {
    t: *@import("train.zig").Trainer,
    prefix: []const u8,
    ck: *core.Checkpoint,

    fn get(r: Reader, buf: *[3][]const u8, name: []const u8) ![3]core.safetensors.Tensor {
        var nb: [192]u8 = undefined;
        var out: [3]core.safetensors.Tensor = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |s, i| {
            out[i] = try r.ck.get(try std.fmt.bufPrint(&nb, "{s}{s}.{s}", .{ r.prefix, name, s }));
            buf[i] = out[i].bytes;
        }
        return out;
    }

    /// rows [0, n) and columns [0, k) of `name` (stored [n_all, k_all]) as one Q4 on the device.
    fn q4(r: Reader, name: []const u8, n: usize, k: usize) !to.Q4 {
        var raw: [3][]const u8 = undefined;
        const ts = try r.get(&raw, name);
        const k_all = ts[1].dim(1) * 64;
        if (ts[0].dim(0) < n or k_all < k or ts[0].dim(1) * 8 != k_all or
            !ts[0].is(.u32, &.{ ts[0].dim(0), k_all / 8 }) or !ts[1].is(.bf16, &.{ ts[0].dim(0), k_all / 64 }) or
            !ts[2].is(.bf16, &.{ ts[0].dim(0), k_all / 64 })) return error.UnexpectedTensor;
        const per = [3]usize{ 4 * k / 8, 2 * k / 64, 2 * k / 64 };
        const per_all = [3]usize{ 4 * k_all / 8, 2 * k_all / 64, 2 * k_all / 64 };
        var ptrs: [3]to.Ref = undefined;
        for (0..3) |j| {
            if (k == k_all) {
                ptrs[j] = try r.t.bytes(raw[j][0 .. n * per[j]]);
                continue;
            }
            const cut = try r.t.gpa.alloc(u8, n * per[j]);
            defer r.t.gpa.free(cut);
            for (0..n) |row| @memcpy(cut[row * per[j] ..][0..per[j]], raw[j][row * per_all[j] ..][0..per[j]]);
            ptrs[j] = try r.t.bytes(cut);
        }
        return .{ .words = ptrs[0], .scales = ptrs[1], .biases = ptrs[2], .n = n, .k = k };
    }

    /// Several projections over the same inputs stacked by rows: [n0 + n1 + ..., k].
    fn stack(r: Reader, names: []const []const u8, ns: []const usize, k: usize) !to.Q4 {
        var total: usize = 0;
        for (ns) |n| total += n;
        var parts: [3][]u8 = undefined;
        const per = [3]usize{ 4 * k / 8, 2 * k / 64, 2 * k / 64 };
        var made: usize = 0;
        defer for (parts[0..made]) |p| r.t.gpa.free(p);
        for (&parts, per) |*p, b| {
            p.* = try r.t.gpa.alloc(u8, total * b);
            made += 1;
        }
        var at: usize = 0;
        for (names, ns) |name, n| {
            var raw: [3][]const u8 = undefined;
            const ts = try r.get(&raw, name);
            if (ts[0].dim(0) < n or ts[0].dim(1) * 8 != k) return error.UnexpectedTensor;
            for (0..3) |j| @memcpy(parts[j][at * per[j] ..][0 .. n * per[j]], raw[j][0 .. n * per[j]]);
            at += n;
        }
        return .{ .words = try r.t.bytes(parts[0]), .scales = try r.t.bytes(parts[1]), .biases = try r.t.bytes(parts[2]), .n = total, .k = k };
    }
};

pub fn weights(t: anytype, io: std.Io) !void {
    const g = t.g;
    var ck = try core.Checkpoint.openModel(t.gpa, io, t.dir);
    defer ck.close();
    const prefix: []const u8 = if (ck.has("language_model.model.layers.0.mlp.gate_proj.weight")) "language_model." else "";
    const r: Reader = .{ .t = t, .ck = &ck, .prefix = prefix };
    var nb: [192]u8 = undefined;
    var names: [4][192]u8 = undefined;
    const inv_host = try t.gpa.alloc(f32, g.rope_dims / 2);
    defer t.gpa.free(inv_host);
    for (inv_host, 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, g.rope_theta, -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(g.rope_dims / 2))));
    const inv = try t.bytes(std.mem.sliceAsBytes(inv_host));
    t.w = try t.gpa.alloc(Layer, g.layers);
    @memset(t.w, .{});
    for (keepLow(g.layers)..g.layers) |l| {
        const lw = &t.w[l];
        const I = g.intermediate;
        lw.mlp = .{
            .gu = try r.stack(&.{
                try std.fmt.bufPrint(&names[0], "model.layers.{d}.mlp.gate_proj", .{l}),
                try std.fmt.bufPrint(&names[1], "model.layers.{d}.mlp.up_proj", .{l}),
            }, &.{ I, I }, g.hidden),
            .down = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.down_proj", .{l}), g.hidden, I),
        };
        switch (t.m.weights.layers[l].mixer) {
            .linear => |d| {
                const modules = [_][]const u8{
                    try std.fmt.bufPrint(&names[0], "model.layers.{d}.linear_attn.in_proj_qkv", .{l}),
                    try std.fmt.bufPrint(&names[1], "model.layers.{d}.linear_attn.in_proj_z", .{l}),
                    try std.fmt.bufPrint(&names[2], "model.layers.{d}.linear_attn.in_proj_b", .{l}),
                    try std.fmt.bufPrint(&names[3], "model.layers.{d}.linear_attn.in_proj_a", .{l}),
                };
                lw.gdn = .{
                    .input = try r.stack(&modules, &.{ g.gdnQkvDim(), g.gdnValueDim(), g.v_heads, g.v_heads }, g.hidden),
                    .out = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.linear_attn.out_proj", .{l}), g.hidden, g.gdnValueDim()),
                    .conv = wts.ref(d.conv),
                    .norm = wts.ref(d.norm),
                    .a_log = try widen(t, d.a_log),
                    .dt = try widen(t, d.dt),
                };
            },
            .attention => |a| {
                lw.attn = .{
                    .q = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.q_proj", .{l}), g.qDim(), g.hidden),
                    .k = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.k_proj", .{l}), g.kvDim(), g.hidden),
                    .v = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.v_proj", .{l}), g.kvDim(), g.hidden),
                    .out = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.o_proj", .{l}), g.hidden, g.heads * g.head_dim),
                    .qs = try widen(t, a.q_norm),
                    .ks = try widen(t, a.k_norm),
                    .inv = inv,
                };
            },
        }
    }
    t.head = try r.q4("lm_head", g.vocab, g.hidden);
}
fn widen(t: anytype, tensor: wts.Tensor) !to.Ref {
    const dtype: @import("train_scalars.zig").Kind = switch (tensor.dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        else => return error.UnsupportedLearningScalar,
    };
    const floats = try @import("train_scalars.zig").values(t.gpa, dtype, tensor.host(u8));
    defer t.gpa.free(floats);
    return t.bytes(std.mem.sliceAsBytes(floats));
}
