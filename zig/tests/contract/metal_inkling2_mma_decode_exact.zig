//! The production MMA decoder's operands, before any multiply can hide a wrong value through clipping or FTZ.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const host = @import("tensorfold").inkling2_metal_host.experts;
const plan = @import("inkling2_experts_plan");
const controls = @import("metal_inkling2_mma_controls.zig");
const rig_ = @import("rig.zig");
const label = "metal inkling2 mma MXFP8 decoder";

fn exact(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !bool {
    const production = try controls.apply(a, if (control) .mx_gu_scale else .none);
    defer a.free(production);
    const text = try std.mem.concat(a, u8, &.{ production, "\n", @embedFile("metal_inkling2_mma_decode_probe.metal") });
    defer a.free(text);
    const pipe = try rig_.pipeline(device, text, "tf_ink2_mma_decode_exact");
    defer pipe.deinit();
    if (pipe.maxThreads() < 1 or rig_.tgMemory(pipe) > 8192) return error.MmaDecoderLimits;
    var r: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer r.deinit();
    const out = try r.output(65536 * 2);
    var codes: [65536]u8 = undefined;
    var scales: [256 * 8]u8 = undefined;
    for (&codes, 0..) |*code, i| code.* = @truncate(i);
    for (&scales, 0..) |*scale, i| scale.* = @intCast(i / 8);
    const rw = try r.input(&codes);
    const rs = try r.input(&scales);
    const baseline = try r.scratch(65536 * 2);
    var scalar_pipes = try host.Pipelines.loadSource(device, src.experts);
    defer scalar_pipes.deinit();
    const scalar_pass = r.begin();
    try host.mxfp8Quantize(&scalar_pipes, scalar_pass.e, .{ .buf = rw.buf, .off = rw.off }, .{ .buf = rs.buf, .off = rs.off }, .{ .buf = baseline.buf, .off = baseline.off }, 256, 256);
    try r.finish(scalar_pass);
    const scalar = r.bytes(baseline, 65536 * 2);
    const first = try a.alloc(u8, 65536 * 2);
    defer a.free(first);
    var equal = true;
    for (0..5) |repeat| {
        const pass = r.begin();
        pass.e.setPipeline(pipe);
        pass.e.setBuffer(out.buf, out.off, 0);
        pass.e.dispatchGroups(mtl.Size.of(4096, 1, 1), mtl.Size.of(1, 1, 1));
        try r.finish(pass);
        const bytes = try r.outputBytes();
        if (repeat == 0) @memcpy(first, bytes) else if (!std.mem.eql(u8, first, bytes)) return error.MmaDecoderUnstable;
        if (!try r.intact()) return error.MmaDecoderGuard;
        if (!std.mem.eql(u8, &codes, r.bytes(rw, codes.len)) or !std.mem.eql(u8, &scales, r.bytes(rs, scales.len))) return error.MmaDecoderInput;
        if (!std.mem.eql(u8, scalar, bytes)) equal = false;
        for (0..256) |scale| for (0..256) |code| {
            const bits = std.mem.readInt(u16, bytes[2 * (256 * scale + code) ..][0..2], .little);
            const want: u16 = @truncate(@as(u32, @bitCast(plan.mxWeight(@intCast(code), @intCast(scale)))) >> 16);
            const both_nan = (bits & 0x7fff) > 0x7f80 and (want & 0x7fff) > 0x7f80;
            if (!both_nan and bits != want) equal = false;
        };
    }
    return equal;
}

pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8) !usize {
    if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) return 0;
    const good = try exact(a, device, queue, false);
    const control_red = !try exact(a, device, queue, true);
    std.debug.print("{s} {s}: all65536 operand bits, five repeats/guards and scale control red={}\n", .{ if (good and control_red) "PASS" else "FAIL", label, control_red });
    return @intFromBool(!good or !control_red);
}
