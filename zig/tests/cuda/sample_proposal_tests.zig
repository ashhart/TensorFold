//! Device-only parity for pitched proposal rows against the existing contiguous keyed draw.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
pub fn run(gpu: check.Gpu) !void {
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer mod.unload();
    const draws = try cuda.sample.Draws.resolveProposal(mod);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var cases: usize = 0;
    for ([_]usize{ 3, 257, 1025 }) |logical| {
        const stride = std.mem.alignForward(usize, logical, 128);
        const contiguous = try gpu.gpa.alloc(u16, 64 * logical);
        defer gpu.gpa.free(contiguous);
        const padded = try gpu.gpa.alloc(u16, 64 * stride);
        defer gpu.gpa.free(padded);
        @memset(padded, 0x7f7f);
        for (0..64) |r| for (0..logical) |col| {
            const bits: u16 = @intCast(0x3f00 + ((col * 7 + r) % 128));
            contiguous[r * logical + col] = bits;
            padded[r * stride + col] = bits;
        };
        const ids = try gpu.gpa.alloc(u32, logical);
        defer gpu.gpa.free(ids);
        const wide = try gpu.gpa.alloc(i64, logical);
        defer gpu.gpa.free(wide);
        for (ids, wide, 0..) |*id, *w, col| {
            id.* = @intCast(col * 4 + 1);
            w.* = id.*;
        }
        var tight = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(contiguous));
        defer tight.free();
        var pitch = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(padded));
        defer pitch.free();
        var mapped = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wide));
        defer mapped.free();
        var meta = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&[4]i32{ 127, 0, 0, 0 }));
        defer meta.free();
        var tokens = try cuda.DeviceBuffer.alloc(gpu.d, 128 * 4);
        defer tokens.free();
        var probabilities = try cuda.DeviceBuffer.alloc(gpu.d, 128 * 4);
        defer probabilities.free();
        for ([_]u32{ 0, 20, 1024 }) |k| for ([_]bool{ false, true }) |keyed| {
            const rule: cuda.sample.Rule = .{ .seed = 0x5eed, .inv_t = 1.0, .top_p = 0.9, .near = 20, .min_log = -3, .top_k = k };
            var rules = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&rule));
            defer rules.free();
            for ([_]usize{ 1, 17, 64 }) |rows| {
                const view = .{ .vocab = @as(u32, @intCast(if (keyed) logical * 4 else logical)), .logical = @as(u32, @intCast(logical)), .stride = @as(u32, @intCast(stride)), .ids = if (keyed) @as(?[]const u32, ids) else null };
                const map: ?u64 = if (keyed) mapped.ptr else null;
                try draws.draw(stream, tight.ptr, logical, rules.ptr, meta.ptr, 4, tokens.ptr, rows, map, probabilities.ptr);
                try draws.proposal(stream, pitch.ptr, view, map, rules.ptr, meta.ptr, 4, tokens.ptr + 64 * 4, rows, probabilities.ptr + 64 * 4);
                try stream.synchronize();
                var got: [128]u32 = undefined;
                var shares: [128]u32 = undefined;
                try tokens.download(0, std.mem.sliceAsBytes(&got));
                try probabilities.download(0, std.mem.sliceAsBytes(&shares));
                try check.expect(std.mem.eql(u32, got[0..rows], got[64..][0..rows]), "pitched proposal IDs differ", .{});
                try check.expect(std.mem.eql(u32, shares[0..rows], shares[64..][0..rows]), "pitched proposal probabilities differ", .{});
                cases += 1;
            }
        };
    }
    check.pass("pitched proposal: {d} keyed and full cases equal contiguous IDs and probability bits", .{cases});
}
