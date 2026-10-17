//! The 27B trainer's MLX copies of the projections its backward reads (layers keep_low up, and the head).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = @import("cuda_learn_shape.zig");
const to = @import("cuda_train_ops.zig");
const train = @import("cuda_train.zig");

const gt = cuda.gdn_train;
const Trainer = train.Trainer;
const prefix = "language_model.";
const keepLow = train.keepLow;

/// Projections as stored, read through O_DIRECT slots; each cut to its first `n` rows and `k` columns.
const Reader = struct {
    t: *Trainer,
    ck: *core.Checkpoint,

    fn get(r: Reader, buf: *[3][]const u8, name: []const u8) ![3]core.checkpoint.Tensor {
        var nb: [192]u8 = undefined;
        var out: [3]core.checkpoint.Tensor = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |s, i| {
            out[i] = try r.ck.get(try std.fmt.bufPrint(&nb, "{s}{s}.{s}", .{ prefix, name, s }));
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
        var ptrs: [3]u64 = undefined;
        for (0..3) |j| {
            if (k == k_all) {
                ptrs[j] = try r.t.devBytes(raw[j][0 .. n * per[j]]);
                continue;
            }
            const cut = try r.t.gpa.alloc(u8, n * per[j]);
            defer r.t.gpa.free(cut);
            for (0..n) |row| @memcpy(cut[row * per[j] ..][0..per[j]], raw[j][row * per_all[j] ..][0..per[j]]);
            ptrs[j] = try r.t.devBytes(cut);
        }
        return .{ .w = ptrs[0], .s = ptrs[1], .b = ptrs[2], .n = n, .k = k };
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
            if (ts[0].dim(0) != n or ts[0].dim(1) * 8 != k) return error.UnexpectedTensor;
            for (0..3) |j| @memcpy(parts[j][at * per[j] ..][0 .. n * per[j]], raw[j][0 .. n * per[j]]);
            at += n;
        }
        return .{ .w = try r.t.devBytes(parts[0]), .s = try r.t.devBytes(parts[1]), .b = try r.t.devBytes(parts[2]), .n = total, .k = k };
    }
};

/// The projections a backward reads, for layers keep_low and up, and the head.
pub fn weights(t: *Trainer, io: std.Io) !void {
    const e = t.e;
    const g = t.g;
    var ck = try core.Checkpoint.openModelPrefix(t.gpa, io, e.dir, prefix);
    defer ck.close();
    const r: Reader = .{ .t = t, .ck = &ck };
    var nb: [160]u8 = undefined;
    var nb2: [160]u8 = undefined;
    var nb3: [160]u8 = undefined;
    var nb4: [160]u8 = undefined;
    var inv_host: [c.rotary_dim / 2]f32 = undefined;
    for (&inv_host, 0..) |*v, i| v.* = std.math.pow(f32, c.theta, -@as(f32, @floatFromInt(i)) / (c.rotary_dim / 2));
    const inv = try t.devBytes(std.mem.sliceAsBytes(&inv_host));
    for (keepLow(g.layers)..g.layers) |l| {
        const lw = &t.w[l];
        const I = g.intermediate;
        lw.mlp = .{
            .gate = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.gate_proj", .{l}), I, g.hidden),
            .up = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.up_proj", .{l}), I, g.hidden),
            .down = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.down_proj", .{l}), g.hidden, I),
        };
        switch (e.layers[l].mixer) {
            .delta => |d| {
                const names = [_][]const u8{
                    try std.fmt.bufPrint(&nb, "model.layers.{d}.linear_attn.in_proj_qkv", .{l}),
                    try std.fmt.bufPrint(&nb2, "model.layers.{d}.linear_attn.in_proj_z", .{l}),
                    try std.fmt.bufPrint(&nb3, "model.layers.{d}.linear_attn.in_proj_b", .{l}),
                    try std.fmt.bufPrint(&nb4, "model.layers.{d}.linear_attn.in_proj_a", .{l}),
                };
                const in_proj = try r.stack(&names, &.{ g.convDim(), g.vInner(), g.linear_v_heads, g.linear_v_heads }, g.hidden);
                const out = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.linear_attn.out_proj", .{l}), g.hidden, g.vInner());
                lw.gdn = .{ .nk = g.linear_k_heads, .nv = g.linear_v_heads, .hidden = g.hidden, .eps = c.eps, .conv_w = d.conv, .a_log = d.a_log, .dt_bias = try widen(t, d.dt_bias, g.linear_v_heads), .norm_w = d.norm, .in_proj = gq(in_proj), .out_proj = gq(out) };
            },
            .attention => |a| {
                lw.attn = .{
                    .q = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.q_proj", .{l}), 2 * g.qInner(), g.hidden),
                    .k = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.k_proj", .{l}), g.kvInner(), g.hidden),
                    .v = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.v_proj", .{l}), g.kvInner(), g.hidden),
                    .o = try r.q4(try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.o_proj", .{l}), g.hidden, g.qInner()),
                    .qscale = try widen(t, a.q_norm, c.head_dim),
                    .kscale = try widen(t, a.k_norm, c.head_dim),
                    .inv = inv,
                    .heads = g.query_heads,
                    .kv_heads = g.kv_heads,
                    .dim = c.head_dim,
                    .half = c.rotary_dim / 2,
                    .eps = c.eps,
                };
            },
        }
    }
    t.head = try r.q4("lm_head", c.vocab, g.hidden);
}

fn gq(m: to.Q4) gt.Q4 {
    return .{ .w = m.w, .s = m.s, .b = m.b, .n = m.n, .k = m.k, .group = m.group };
}

/// An engine bf16 vector as fp32 on the device.
fn widen(t: *Trainer, ptr: u64, n: usize) !u64 {
    const raw = try t.gpa.alloc(u16, n);
    defer t.gpa.free(raw);
    try t.e.stream.synchronize();
    const d = t.e.d;
    try d.check(d.api.cuMemcpyDtoH_v2(raw.ptr, ptr, n * 2), "cuMemcpyDtoH");
    const out = try t.gpa.alloc(f32, n);
    defer t.gpa.free(out);
    for (out, raw) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    return t.devBytes(std.mem.sliceAsBytes(out));
}
