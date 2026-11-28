//! Exhaustive scalar GPU MXFP8 decoder across all 65,536 (e4m3, ue8m0) pairs, exact in BF16 words.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const plan = @import("tensorfold").inkling2_metal_host.experts_plan;
const label = "metal inkling2 experts MXFP8 decoder";
const from = "ink2_mx_bf(W[at], S[row * (a.k / 32u) + col / 32u])";
const to = "ink2_mx_bf(W[at], min(uint(S[row * (a.k / 32u) + col / 32u]) + 1u, 255u))";

fn hr(b: rig_.Ref) host.Ref {
    return .{ .buf = b.buf, .off = b.off };
}

fn exact(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !bool {
    const text = if (control) blk: {
        if (std.mem.count(u8, src.experts, from) != 1) return error.ControlEditCount;
        break :blk try std.mem.replaceOwned(u8, a, src.experts, from, to);
    } else try a.dupe(u8, src.experts);
    defer a.free(text);
    var pipes = try host.Pipelines.loadSource(device, text);
    defer pipes.deinit();
    var r: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer r.deinit();
    var codes: [65536]u8 = undefined;
    var scales: [256 * 8]u8 = undefined;
    for (0..256) |row| {
        for (0..256) |col| codes[row * 256 + col] = @intCast(col);
        for (0..8) |blk_i| scales[row * 8 + blk_i] = @intCast(row);
    }
    const rw = try r.input(&codes);
    const rs = try r.input(&scales);
    const ro = try r.output(65536 * 2);
    const first = try a.alloc(u8, 65536 * 2);
    defer a.free(first);
    var equal = true;
    for (0..5) |repeat| {
        const pass = r.begin();
        try host.mxfp8Quantize(&pipes, pass.e, hr(rw), hr(rs), hr(ro), 256, 256);
        try r.finish(pass);
        const bytes = try r.outputBytes();
        if (repeat == 0) @memcpy(first, bytes) else if (!std.mem.eql(u8, first, bytes)) return error.ScalarDecoderUnstable;
        if (!try r.intact()) return error.ScalarDecoderGuard;
        if (!std.mem.eql(u8, &codes, r.bytes(rw, codes.len)) or !std.mem.eql(u8, &scales, r.bytes(rs, scales.len))) return error.ScalarDecoderInput;
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
    std.debug.print("{s} {s}: all65536 BF16 words, five repeats/guards and scale control red={}\n", .{ if (good and control_red) "PASS" else "FAIL", label, control_red });
    return @intFromBool(!good or !control_red);
}
