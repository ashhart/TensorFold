//! Optional tests against the original Mini snapshot (`TF_INKLING2_MINI`).
const std = @import("std");
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const op = @import("metal_loop.zig");
const Mapped = @import("metal_mapped.zig").Mapped;
const fixture = @import("metal_weights_fixture.zig");
const t = std.testing;


fn tensor(x: ck.Tensor) op.View { return .{ .buffer = x.buffer, .offset = x.offset }; }
fn view(b: mtl.Buffer, offset: usize) op.View { return .{ .buffer = b, .offset = offset }; }
fn val(b: u16) f64 { return @as(f32, @bitCast(@as(u32, b) << 16)); }
fn rounded(x: f64) f64 {
    const narrow: f32 = @floatCast(x);
    const bits: u32 = @bitCast(narrow);
    return val(@truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16));
}

const h = 2304;
const rank = 64;
const names = [_][]const u8{ "attn", "k", "v", "mlp" };
const samples = [_]f32{ 0.25, -0.5, 1.5, -2 };
const prefix = "model.llm.layers.6.controllers.";

// Check the offsets that the mapped GPU binding uses, rather than just the folded shape fixture.
test "original Mini header maps small FP32 offsets and weights beyond four GiB" {
    const path = try fixture.checkpoint();
    defer t.allocator.free(path);
    var mapped = try Mapped.open(path);
    defer mapped.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const header = try st.parseHeader(arena.allocator(), mapped.header(), mapped.dataLen());
    try t.expectEqual(@as(usize, 148416), mapped.data_start);
    try t.expectEqual(@as(usize, 1244), header.count());
    const bias = header.get("model.llm.layers.6.mlp.gate.bias").?;
    try t.expectEqual(@as(usize, 0), (mapped.data_start + bias.begin) % 4);
    const expert = header.get("model.llm.layers.23.mlp.experts.w13_weight").?;
    try t.expect(mapped.data_start + expert.begin > std.math.maxInt(u32));
    const q = header.get("model.llm.middle_cycle_cross_pass_kv_queries").?;
    try t.expectEqual(@as(usize, 24 * h * 4), q.end - q.begin);
}

// A constant hidden and constant messages make both input norms round to ±1 in BF16.
// The independent CPU calculation can then evaluate the trained controllers with exact signed sums before gate tanh.
test "original Mini's four trained core controllers agree with FP64 scalar math" {
    const path = try fixture.checkpoint();
    defer t.allocator.free(path);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var c = ck.Checkpoint.init(t.allocator);
    defer c.deinit();
    try c.addFileSelected(device, path, "", prefix);
    var ops = try op.Ops.load(device);
    defer ops.deinit();
    const shared = mtl.ResourceOptions.shared;
    const x = try device.buffer(h * 4, shared);
    defer x.deinit();
    const m = try device.buffer(4 * h * 4, shared);
    defer m.deinit();
    const w = try device.buffer(h * 2, shared);
    defer w.deinit();
    const d = try device.buffer(3 * rank * 2, shared);
    defer d.deinit();
    const y = try device.buffer(4 * h * 2, shared);
    defer y.deinit();
    @memset(x.slice(f32, h), 1);
    @memset(w.slice(u16, h), 0x3f80);
    for (samples, 0..) |s, i| @memset(m.slice(f32, 4 * h)[i * h ..][0..h], s);
    var wd: [4]ck.Tensor = undefined;
    var wf: [4]ck.Tensor = undefined;
    var wg: [4]ck.Tensor = undefined;
    inline for (names, 0..) |name, i| {
        wd[i] = try c.get(prefix ++ name ++ ".down_weight");
        wf[i] = try c.get(prefix ++ name ++ ".forget_weight");
        wg[i] = try c.get(prefix ++ name ++ ".gain_weight");
    }
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    inline for (.{ @as(usize, 0), 3 }) |first| {
        const n = if (first == 0) 3 else 1;
        const xviews: [3]op.View = @splat(view(x, 0));
        const norms: [3]op.View = @splat(view(w, 0));
        const slots: [3]u32 = if (first == 0) .{ 0, 1, 2 } else .{ 3, 3, 3 };
        const dw: [3]op.View = if (first == 0) .{ tensor(wd[0]), tensor(wd[1]), tensor(wd[2]) } else @splat(tensor(wd[3]));
        const fw: [3]op.View = if (first == 0) .{ tensor(wf[0]), tensor(wf[1]), tensor(wf[2]) } else @splat(tensor(wf[3]));
        const gw: [3]op.View = if (first == 0) .{ tensor(wg[0]), tensor(wg[1]), tensor(wg[2]) } else @splat(tensor(wg[3]));
        try ops.gateDown(e, .{ .h = xviews, .msg = view(m, 0), .msg_stride = h, .slots = slots, .gates = n, .down_w = dw, .down = view(d, 0), .rows = 1, .hidden = h, .rank = rank });
        try ops.gateMix(e, .{ .h = xviews, .msg = view(m, 0), .msg_stride = h, .slots = slots, .gates = n, .down = view(d, 0), .forget = fw, .gain = gw, .norms = norms, .out = view(y, first * h * 2), .out_stride = h, .rows = 1, .hidden = h, .rank = rank, .eps = 1e-6 });
    }
    e.end();
    cb.commit();
    cb.wait();
    try t.expect(cb.failure() == null);
    const outputs = y.slice(u16, 4 * h);
    for (samples, 0..) |s, j| {
        var down: [rank]f64 = undefined;
        const down_weights = wd[j].host(u16);
        const forget_weights = wf[j].host(u16);
        const gain_weights = wg[j].host(u16);
        const sign: f64 = if (s < 0) -1 else 1;
        for (&down, 0..) |*a, r| {
            var sum: f64 = 0;
            for (0..h) |i| sum += val(down_weights[r * 2 * h + i]) + sign * val(down_weights[r * 2 * h + h + i]);
            a.* = rounded(sum);
        }
        var sum: f64 = 0;
        var want: [h]f64 = undefined;
        for (&want, 0..) |*v, i| {
            var f: f64 = 0;
            var g: f64 = 0;
            for (down, 0..) |z, r| {
                f += val(forget_weights[i * rank + r]) * z;
                g += val(gain_weights[i * rank + r]) * z;
            }
            v.* = 1 - std.math.tanh(f) + @as(f64, s) * std.math.tanh(g);
            sum += v.* * v.*;
        }
        const rms = @sqrt(sum / h + 1e-6);
        for (want, outputs[j * h ..][0..h]) |v, out| try t.expectApproxEqAbs(rounded(v / rms), val(out), 0.008);
    }
}
