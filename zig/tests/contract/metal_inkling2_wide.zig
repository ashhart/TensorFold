//! Integer grouping has no floating-point sum: compare all table bytes, live order, tails and the serial launch.
const std = @import("std");
const mtl = @import("metal");
const host = @import("tensorfold").inkling2_metal_host.experts;
const plan = @import("tensorfold").inkling2_metal_host.experts_plan;
const src = @import("inkling2_metal");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_wide_controls.zig");
const label = "metal inkling2 mma tf_ink2_group_wide";
const words = plan.expert_count * (1 + host.mma_cap);

fn checkCase(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, rows: u32, topk: u32, experts: u32, same: bool, control: bool) !bool {
    var pipes = try host.Pipelines.loadSource(device, src.experts);
    defer pipes.deinit();
    const text = if (control) try controls.apply(a) else try a.dupe(u8, src.experts_mma);
    defer a.free(text);
    try pipes.loadMma(device, text);
    pipes.groups.?.deinit();
    pipes.groups = null;
    pipes.groups = try device.buffer(words * 4 + rig_.guard, mtl.ResourceOptions.shared);
    const table = pipes.groups.?;
    var r = rig_.Rig{ .a = a, .device = device, .queue = queue };
    defer r.deinit();
    const ids = try a.alloc(i32, rows * topk);
    defer a.free(ids);
    for (ids, 0..) |*v, i| {
        v.* = if (same) 0 else if (i % 17 == 0) -1 else if (i % 19 == 0) @intCast(experts) else @intCast((i * 37 + i / 3) % experts);
    }
    const input = try r.input(std.mem.sliceAsBytes(ids));
    const want = try a.alloc(u32, words);
    defer a.free(want);
    @memset(want, 0xffff_ffff);
    plan.groupReference(ids, experts, host.mma_cap, want[0..experts], want[plan.expert_count..]);
    const scalar = try a.alloc(u8, words * 4);
    defer a.free(scalar);
    const call = host.Call{ .rows = rows, .topk = topk, .experts = experts, .k = 64, .n = 64, .format = .mxfp8 };
    for (0..7) |pass_i| {
        @memset(table.contents()[0 .. words * 4], 0xff);
        @memset(table.contents()[words * 4 ..][0..rig_.guard], 0xa5);
        const pass = r.begin();
        const ref = host.Ref{ .buf = input.buf, .off = input.off };
        if (pass_i == 0) try host.groupScalar(&pipes, pass.e, ref, call) else if (pass_i == 1) try host.group(&pipes, pass.e, ref, call) else try host.groupWide(&pipes, pass.e, ref, call);
        try r.finish(pass);
        const got = table.contents()[0 .. words * 4];
        for (table.contents()[words * 4 ..][0..rig_.guard]) |b| if (b != 0xa5) return false;
        if (!try r.intact() or !std.mem.eql(u8, std.mem.sliceAsBytes(ids), r.bytes(input, ids.len * 4))) return false;
        if (pass_i == 0) {
            if (!std.mem.eql(u8, got, std.mem.sliceAsBytes(want))) return false;
            @memcpy(scalar, got);
        } else if (control) {
            if (pass_i >= 2 and std.mem.eql(u8, got, scalar)) return false;
        } else if (!std.mem.eql(u8, got, scalar) or !std.mem.eql(u8, got, std.mem.sliceAsBytes(want))) return false;
    }
    return true;
}

pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8) !usize {
    if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) return 0;
    var failures: usize = 0;
    const cases = [_][3]u32{ .{ 1, 1, 1 }, .{ 31, 8, 127 }, .{ 32, 8, 128 }, .{ 33, 8, 128 }, .{ 128, 8, 128 }, .{ 255, 1, 128 }, .{ 256, 1, 128 }, .{ 256, 8, 128 } };
    for (cases) |case| for ([_]bool{ false, true }) |same| {
        if (!try checkCase(a, device, queue, case[0], case[1], case[2], same, false)) failures += 1;
    };
    if (!try checkCase(a, device, queue, 256, 8, 128, true, true)) failures += 1;
    std.debug.print("{s} {s}: serial and host tables, dispatch cutoff, five forced-wide repeats, invalid IDs, cap tails and ordinal control ({d} failures)\n", .{ if (failures == 0) "PASS" else "FAIL", label, failures });
    return failures;
}
