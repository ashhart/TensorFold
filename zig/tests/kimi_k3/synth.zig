//! Synthetic K3 tensors at real shapes, made on the GPU from their names (tools/zig/k3_synth.py makes the same).
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const store = k3.store;
const kernels = k3.kernels;

pub fn fnv1a(name: []const u8) u32 {
    var h: u32 = 0x811C9DC5;
    for (name) |b| h = (h ^ b) *% 0x01000193;
    return h;
}

pub fn mix(seed: u32, i: u32) u32 {
    var x = i ^ (seed *% 0x9E3779B9);
    x ^= x >> 16;
    x *%= 0x7FEB352D;
    x ^= x >> 15;
    x *%= 0x846CA68B;
    return x ^ (x >> 16);
}

/// k3_synth.py's 'w' values: k 2^-(e0 + two hash bits), k a hash byte minus 128.
pub fn value(x: u32, e0: u32) f32 {
    const k: f32 = @floatFromInt(@as(i32, @intCast(x & 0xFF)) - 128);
    return std.math.ldexp(k, -@as(i32, @intCast(e0 + ((x >> 8) & 3))));
}

pub fn fanExp(fan_in: u32) u32 {
    return @intFromFloat(@round(@log2(74.0 * @sqrt(@as(f64, @floatFromInt(fan_in))))) - 1);
}

/// The fill rule for a tensor name (mirrors k3_synth.rule): kind and base exponent.
pub fn rule(name: []const u8, shape: []const u32) struct { kind: u32, e0: u32 } {
    const ends = struct {
        fn f(n: []const u8, s: []const u8) bool {
            return std.mem.endsWith(u8, n, s);
        }
    }.f;
    if (ends(name, "_packed")) return .{ .kind = kernels.fill_u8, .e0 = 0 };
    if (ends(name, "_scale")) return .{ .kind = kernels.fill_e8, .e0 = 7 };
    if (ends(name, "A_log")) return .{ .kind = kernels.fill_w, .e0 = 7 };
    if (ends(name, "dt_bias") or ends(name, "conv1d.weight") or ends(name, "e_score_correction_bias")) return .{ .kind = kernels.fill_w, .e0 = 8 };
    if (ends(name, "norm.weight")) return .{ .kind = kernels.fill_one, .e0 = 0 };
    if (ends(name, "res_proj.weight")) return .{ .kind = kernels.fill_w, .e0 = 6 };
    if (ends(name, "embed_tokens.weight")) return .{ .kind = kernels.fill_w, .e0 = 7 };
    return .{ .kind = kernels.fill_w, .e0 = if (shape.len == 2) fanExp(shape[1]) else 6 };
}

pub const Synth = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: *const kernels.Kernels,
    bufs: std.ArrayList(mtl.Buffer) = .empty,
    made: std.StringHashMapUnmanaged(store.Tensor) = .empty,
    bytes: usize = 0,

    pub fn source(s: *Synth) store.Source {
        return .{ .ptr = s, .getFn = getFn };
    }

    fn getFn(ptr: *anyopaque, name: []const u8, dtype: store.DType, shape_in: []const u32) anyerror!store.Tensor {
        const s: *Synth = @ptrCast(@alignCast(ptr));
        if (s.made.get(name)) |t| return t;
        const a_log = [_]u32{128};
        const shape = if (shape_in.len == 0) a_log[0..] else shape_in;
        var t = store.Tensor{ .ref = undefined, .dtype = dtype, .rank = @intCast(shape.len) };
        @memcpy(t.shape[0..shape.len], shape);
        const buf = try s.device.buffer(@max(t.bytes(), 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        try s.bufs.append(s.gpa, buf);
        t.ref = .{ .buf = buf };
        const r = rule(name, shape);
        const cb = s.queue.commandBuffer();
        const e = cb.compute(.serial);
        s.k.fill(e, t.ref, switch (dtype) {
            .bf16 => .bf16,
            .f32 => .f32,
            .u8 => .u8,
        }, .{ .seed = fnv1a(name), .count = @intCast(t.count()), .kind = r.kind, .e0 = r.e0 });
        e.end();
        cb.commit();
        s.bytes += t.bytes();
        try s.made.put(s.gpa, try s.gpa.dupe(u8, name), t);
        return t;
    }

    /// Wait for the fills, then free nothing: `deinit` releases every tensor at once.
    pub fn sync(s: *Synth) void {
        const cb = s.queue.commandBuffer();
        cb.commit();
        cb.wait();
    }

    pub fn deinit(s: *Synth) void {
        var it = s.made.keyIterator();
        while (it.next()) |key| s.gpa.free(key.*);
        s.made.deinit(s.gpa);
        for (s.bufs.items) |b| b.deinit();
        s.bufs.deinit(s.gpa);
        s.bytes = 0;
    }
};

/// One row's hash-made bf16 inputs (seeded by a name the checker rebuilds), written as bf16 words.
pub fn rowValues(out: []u16, name: []const u8) void {
    const seed = fnv1a(name);
    for (out, 0..) |*o, i| o.* = @intCast(@as(u32, @bitCast(value(mix(seed, @intCast(i)), 7))) >> 16);
}

test "hash, values and fan exponent agree with k3_synth.py" {
    try std.testing.expectEqual(@as(u32, 0x69f35565), fnv1a("layers.1.self_attn.q_proj.weight"));
    try std.testing.expectEqual(@as(u32, 2939736086), mix(12345, 3));
    try std.testing.expectEqual(@as(f32, -0.828125), value(mix(12345, 3), 7));
    try std.testing.expectEqual(@as(u32, 12), fanExp(7168));
    try std.testing.expectEqual(@as(u32, 9), fanExp(128));
}
