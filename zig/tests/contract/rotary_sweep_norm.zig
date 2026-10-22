//! Normalized vectors come from the real production kernels at zero phase, over every fixture phase.
const std = @import("std");
const mtl = @import("metal");
const f = @import("tensorfold").flashnext_vision_contract;
const accuracy = f.rotary_accuracy;
const prep = @import("metal_flashnext.zig");
const pool = @import("metal_flashnext_pool.zig");

pub fn which(role: accuracy.Role) f.Which {
    return switch (role) {
        .mrope => .mrope,
        .target => .target,
        .mtp => .mtp,
        .pooled => .pooled,
        .absolute => .absolute,
        .positioned => .positioned,
    };
}
fn capture(a: std.mem.Allocator, width: usize, weight: f32, input: []const u16, output: []const u8) !accuracy.Norm {
    _ = a;
    var n: accuracy.Norm = .{ .width = width, .weight = @bitCast(weight), .epsilon = @bitCast(@as(f32, 1e-5)) };
    @memcpy(n.input[0..width], input);
    for (n.output[0..width], 0..) |*h, i| h.* = std.mem.readInt(u16, output[i * 2 ..][0..2], .little);
    return n;
}
pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, role: accuracy.Role) ![]const accuracy.Norm {
    const w = which(role);
    var rows: std.ArrayList(accuracy.Norm) = .empty;
    errdefer rows.deinit(a);
    if (!f.isPool(w)) {
        var p = try prep.Probe.init(a, device, queue, w);
        defer p.deinit();
        for (0..17) |seed| {
            _ = try p.prepare(.{ .rows = 1, .n = f.output_width, .k = 256 }, .random, seed);
            const before = try a.dupe(u8, try p.rig.inputBytes());
            defer a.free(before);
            try p.run();
            if (!try p.rig.intact() or !std.mem.eql(u8, before, try p.rig.inputBytes())) return error.SweepChangedInputs;
            const output = try p.rig.outputBytes();
            for ([_]usize{ 0, 24 * 512, 24 * 512 + 4 * 256 }, [_]usize{ 0, 24 * 256, 26 * 256 }, [_]usize{ 256, 256, 128 }, [_]f32{ 1, 0.75, 1.25 }) |src, dst, width, weight| {
                try rows.append(a, try capture(a, width, weight, p.host[src..][0..width], output[dst * 2 ..][0 .. width * 2]));
            }
        }
    } else {
        var p = try pool.Probe.init(a, device, queue, w);
        defer p.deinit();
        p.calibration = true;
        for (0..17) |seed| {
            _ = try p.prepare(.{ .rows = 1, .n = 128, .k = 128 }, .random, seed);
            const before = try a.dupe(u8, try p.rig.inputBytes());
            defer a.free(before);
            try p.run();
            if (!try p.rig.intact() or !std.mem.eql(u8, before, try p.rig.inputBytes())) return error.SweepChangedInputs;
            var mean: [128]u16 = undefined;
            for (&mean, 0..) |*h, d| {
                var total: f32 = @as(f32, @bitCast(@as(u32, p.host[d]) << 16));
                for (1..4) |k| total += @as(f32, @bitCast(@as(u32, p.host[k * 128 + d]) << 16));
                const bits: u32 = @bitCast(total * 0.25);
                h.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
            }
            try rows.append(a, try capture(a, 128, 1.25, &mean, (try p.rig.outputBytes())[0..256]));
        }
    }
    return rows.toOwnedSlice(a);
}
