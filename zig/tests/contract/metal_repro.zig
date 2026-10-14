//! The smallest inputs behind each finding, run on this GPU with the float64 outputs beside the kernel's.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const mm = tf.affine_mm;
const row = tf.row_projection;

/// One row of 64 inputs, +Inf first and zeros after; 64 outputs; every code 1, every scale and bias +0.01 (bf16).
pub fn affineInf(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !void {
    const n = 64;
    const k = 64;
    var r = rig_.Rig{ .a = a, .device = device, .queue = queue };
    defer r.deinit();
    var x: [k]u16 = @splat(0);
    x[0] = c.toBf16(std.math.inf(f32));
    var words: [n * k / 8]u32 = @splat(0x11111111);
    const s: [n]u16 = @splat(c.toBf16(0.01));
    const xr = try r.input(std.mem.sliceAsBytes(&x));
    const wr = try r.input(std.mem.sliceAsBytes(&words));
    const sr = try r.input(std.mem.sliceAsBytes(&s));
    const br = try r.input(std.mem.sliceAsBytes(&s));
    const sums = try r.scratch(4);
    const y_mm = try r.output(2 * n);
    const y_row = try r.output(2 * n);
    const src = try mm.source(device, a, .{ .bits = 4, .group = 64 });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: mm.Pipes = undefined;
    for (mm.names, 0..) |name, i| pipes[i] = try mtl.Pipeline.init(device, lib, name, false);
    defer for (pipes) |p| p.deinit();
    var rp = try row.Pipelines.load(device);
    defer rp.deinit();
    const pass = r.begin();
    const q = .{ .w = .{ .buf = wr.buf, .off = wr.off }, .s = .{ .buf = sr.buf, .off = sr.off }, .b = .{ .buf = br.buf, .off = br.off }, .n = @as(u32, n), .k = @as(u32, k) };
    mm.rowSums(pass.e, pipes, 64, .{ .buf = xr.buf, .off = xr.off }, .{ .buf = sums.buf, .off = sums.off }, 1, k);
    mm.dense(pass.e, pipes, .{ .buf = xr.buf, .off = xr.off }, .{ .buf = sums.buf, .off = sums.off }, q, .{ .buf = y_mm.buf, .off = y_mm.off }, 1);
    const cl = try row.call(&rp, .{ .w = wr.buf, .w_off = wr.off, .scales = sr.buf, .s_off = sr.off, .biases = br.buf, .b_off = br.off, .n = n, .k = k }, 1, false);
    pass.e.setPipeline(cl.pipeline);
    for ([_]rig_.Ref{ xr, wr, sr, br }, 0..) |ref, i| pass.e.setBuffer(ref.buf, ref.off, i);
    pass.e.setValue(cl.dims, 4);
    pass.e.setBuffer(y_row.buf, y_row.off, 5);
    pass.e.dispatchGroups(mtl.Size.of(cl.groups, cl.slots, 1), mtl.Size.of(cl.threads, 1, 1));
    try r.finish(pass);
    const sv: f64 = c.fromBf16(s[0]);
    const inf = std.math.inf(f64);
    std.debug.print("REPRO affine_mm q4 g64, x = [+Inf, 0 x 63], codes 1, scale = bias = {d} (bf16 0x{x:0>4})\n", .{ sv, s[0] });
    std.debug.print("  float64, MLX's s*sum(q x) + b*sum(x): {d}; dequantized first, x (s q + b): {d}\n", .{ sv * 1 * inf + sv * inf, inf * (sv + sv) });
    std.debug.print("  float64, the kernel's s*sum((q + 128) x) + (b - 128 s)*sum(x): {d}\n", .{sv * 129 * inf + (sv - 128 * sv) * inf});
    for ([_]struct { []const u8, rig_.Ref }{ .{ "affine_mm (tf_affine_mm)", y_mm }, .{ "row projection (tf_row_projection_q4_f32)", y_row } }) |o| {
        const got = r.bytes(o[1], 2 * n);
        const h0 = std.mem.readInt(u16, got[0..2], .little);
        var same = true;
        for (0..n) |j| same = same and std.mem.readInt(u16, got[2 * j ..][0..2], .little) == h0;
        std.debug.print("  {s}: every output 0x{x:0>4} = {d} ({s})\n", .{ o[0], h0, c.fromBf16(h0), if (same) "all 64 equal" else "outputs differ" });
    }
}
