//! The production native attention chain is compared with step one's independent host backward and scalar chain rule.
const std = @import("std");
const core = @import("core");
const n = @import("learning_numeric.zig");
const back = @import("attention_backward.zig");
const fixture = @import("../qwen4/learning_fixture.zig");
const a = std.testing.allocator;
const Reader = core.Checkpoint;
fn decode(mem: std.mem.Allocator, reader: *Reader, stem: []const u8, rows: usize, cols: usize) ![]f64 {
    const w = (try reader.get(try std.fmt.allocPrint(mem, "{s}.weight", .{stem}))).bytes;
    const s = (try reader.get(try std.fmt.allocPrint(mem, "{s}.scales", .{stem}))).bytes;
    const b = (try reader.get(try std.fmt.allocPrint(mem, "{s}.biases", .{stem}))).bytes;
    const out = try mem.alloc(f64, rows * cols);
    for (out, 0..) |*v, i| {
        const code = (std.mem.readInt(u32, w[i / 8 * 4 ..][0..4], .little) >> @intCast(i % 8 * 4)) & 15;
        v.* = @as(f64, @floatFromInt(code)) * n.f(std.mem.readInt(u16, s[i / 64 * 2 ..][0..2], .little)) + n.f(std.mem.readInt(u16, b[i / 64 * 2 ..][0..2], .little));
    }
    return out;
}
fn transpose(mem: std.mem.Allocator, dy: []const f64, w: []const f64, rows: usize, outputs: usize, inputs: usize) ![]f64 {
    const out = try mem.alloc(f64, rows * inputs);
    for (out, 0..) |*v, i| {
        v.* = 0;
        for (0..outputs) |j| v.* += dy[(i / inputs) * outputs + j] * w[j * inputs + i % inputs];
    }
    return out;
}
fn tape(mem: std.mem.Allocator, t: anytype, l: usize, node: @import("learning_hook.zig").Node) ![]f64 {
    const v = try t.taped(l, node);
    const out = try mem.alloc(f64, v.len / 2);
    for (out, 0..) |*x, i| x.* = n.f(std.mem.readInt(u16, v.buffer[v.offset + 2 * i ..][0..2], .little));
    return out;
}
fn prepare(mem: std.mem.Allocator, e: *n.E, reader: *Reader, l: usize, raw: []const f64, grad: []const f64, dim: usize, norm: ?[]const u8, rotated: bool) ![]f64 {
    const out = try mem.alloc(f64, grad.len);
    const tmp = try mem.dupe(f64, grad);
    if (rotated) {
        var freq: [256]f32 = undefined;
        e.frequencies(l, freq[0 .. dim / 2]);
        for (0..grad.len / dim) |block| for (0..dim / 2) |i| {
            const row = block / (if (norm != null and std.mem.endsWith(u8, norm.?, "q_norm.weight")) e.r.shape.heads else e.r.shape.kv(l));
            const angle = @as(f64, @floatFromInt(row)) * freq[i];
            const left = grad[block * dim + i];
            const right = grad[block * dim + dim / 2 + i];
            tmp[block * dim + i] = left * @cos(angle) + right * @sin(angle);
            tmp[block * dim + dim / 2 + i] = right * @cos(angle) - left * @sin(angle);
        };
    }
    const weights = if (norm) |name| (try reader.get(name)).bytes else &.{};
    for (0..raw.len / dim) |block| {
        var square: f64 = 0;
        var dot: f64 = 0;
        for (0..dim) |i| {
            const x = raw[block * dim + i];
            const w: f64 = if (norm != null) n.f(std.mem.readInt(u16, weights[i * 2 ..][0..2], .little)) else 1;
            square += x * x;
            dot += x * w * tmp[block * dim + i];
        }
        const inv = 1 / @sqrt(square / @as(f64, @floatFromInt(dim)) + 1e-6);
        for (0..dim) |i| {
            const w: f64 = if (norm != null) n.f(std.mem.readInt(u16, weights[i * 2 ..][0..2], .little)) else 1;
            out[block * dim + i] = inv * w * tmp[block * dim + i] - inv * inv * inv * dot * raw[block * dim + i] / @as(f64, @floatFromInt(dim));
        }
    }
    return out;
}
test "Gemma4 learning: native projection attention RMS RoPE and tied-value chain equals step-one binary64 reference" {
    var f = try @import("learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const T = @import("learning_train.zig").Trainer(n.E);
    const t = try T.init(a, std.testing.io, f.e, &f.e.r.pockets, f.e);
    defer t.deinit();
    const ids = [_]u32{ 3, 5, 7, 2, 9 };
    t.nudge.layers[0] = 1;
    _ = try t.step(&ids, 3, .target, .{ .decide = 0, .whole = true });
    for ([_]usize{ 4, 5 }) |l| {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const mem = arena.allocator();
        const s = t.shape;
        const dim = s.dim(l);
        const kv = s.kv(l);
        const qw = s.heads * dim;
        const kw = kv * dim;
        const stem = try std.fmt.allocPrint(mem, "language_model.model.layers.{d}.", .{l});
        const dy = try mem.alloc(f32, t.rows * s.hidden);
        for (dy, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16;
        t.encoding = try f.b.begin();
        try t.write(t.vec(.branch, s.hidden), std.mem.sliceAsBytes(dy));
        try t.ops().zero(t.buf(.sparse));
        try @import("learning_back.zig").attention(t, l);
        try f.b.submit(t.encoding);
        try f.b.drain();
        const gd = try mem.alloc(f64, dy.len);
        for (gd, dy) |*v, x| v.* = x;
        const dy_attention = try transpose(mem, gd, try decode(mem, &t.reader, try std.mem.concat(mem, u8, &.{ stem, "self_attn.o_proj" }), s.hidden, qw), t.rows, s.hidden, qw);
        const q = try tape(mem, t, l, .query);
        const k = try tape(mem, t, l, .key);
        const v = try tape(mem, t, l, .value);
        const dq = try mem.alloc(f64, q.len);
        const dk = try mem.alloc(f64, k.len);
        const dv = try mem.alloc(f64, v.len);
        const masks = try mem.alloc(back.Row, t.rows);
        for (masks, 0..) |*m, i| m.* = .{ .end = i + 1 };
        try back.backward(f64, mem, .{ .rows = t.rows, .heads = s.heads, .kv = kv, .width = dim, .capacity = t.rows }, masks, q, k, v, dy_attention, dq, dk, dv);
        const gq = try prepare(mem, f.e, &t.reader, l, try tape(mem, t, l, .q_raw), dq, dim, try std.mem.concat(mem, u8, &.{ stem, "self_attn.q_norm.weight" }), true);
        const gk = try prepare(mem, f.e, &t.reader, l, try tape(mem, t, l, .k_raw), dk, dim, try std.mem.concat(mem, u8, &.{ stem, "self_attn.k_norm.weight" }), true);
        const gv = try prepare(mem, f.e, &t.reader, l, try tape(mem, t, l, .v_raw), dv, dim, null, false);
        if (l % 6 == 5) for (gk, gv) |*x, y| {
            x.* += y;
        };
        const want = try transpose(mem, gq, try decode(mem, &t.reader, try std.mem.concat(mem, u8, &.{ stem, "self_attn.q_proj" }), qw, s.hidden), t.rows, qw, s.hidden);
        const kg = try transpose(mem, gk, try decode(mem, &t.reader, try std.mem.concat(mem, u8, &.{ stem, "self_attn.k_proj" }), kw, s.hidden), t.rows, kw, s.hidden);
        for (want, kg) |*x, y| x.* += y;
        if (l % 6 != 5) {
            const vg = try transpose(mem, gv, try decode(mem, &t.reader, try std.mem.concat(mem, u8, &.{ stem, "self_attn.v_proj" }), kw, s.hidden), t.rows, kw, s.hidden);
            for (want, vg) |*x, y| x.* += y;
        }
        const result = t.vec(.mixed, s.hidden);
        const got = std.mem.bytesAsSlice(f32, result.buffer[result.offset..][0..result.len]);
        for (got, want) |actual, expected| try fixture.near(actual, expected);
    }
}
