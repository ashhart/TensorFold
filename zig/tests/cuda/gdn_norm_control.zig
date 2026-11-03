//! Frozen old sigmoid launches must equal Flash Next and must disagree with the 27B's SiLU gradients.
const std = @import("std");
const cuda = @import("cuda");
const gt = cuda.gdn_train;
const checks = @import("check.zig");

pub fn check(gpu: checks.Gpu, stream: cuda.Stream, sc: gt.Scratch, shape: gt.Shape, p: u64, w: u64, eps: f32, dy: []const f32, dp: []const f32, silu: bool) !void {
    if (cuda.kernels.gdn_train.len == 0) return error.MissingGdnControl;
    var module = try cuda.Module.load(gpu.d, cuda.kernels.gdn_train);
    defer module.unload();
    const f = try module.function("tf_ctl_gdn_norm_back_sigmoid");
    var args: cuda.Args = .{};
    for ([_]u64{ sc.y, p, w, sc.dout, sc.dy, sc.dproj }) |v| args.add(v);
    args.add(shape);
    args.add(eps);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(shape.rows), .y = @intCast(shape.nv) }, .block = .{ .x = 128 } }, stream, &args);
    try stream.synchronize();
    const old = try gpu.gpa.alloc(f32, dy.len);
    defer gpu.gpa.free(old);
    const oldp = try gpu.gpa.alloc(f32, dp.len);
    defer gpu.gpa.free(oldp);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = sc.dy, .len = dy.len * 4 }).download(0, std.mem.sliceAsBytes(old));
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = sc.dproj, .len = dp.len * 4 }).download(0, std.mem.sliceAsBytes(oldp));
    if (!silu) {
        try checks.expect(std.mem.eql(u8, std.mem.sliceAsBytes(dy), std.mem.sliceAsBytes(old)), "Flash Next norm dy replacement changed bytes", .{});
        const width: usize = @intCast(shape.nv * 128);
        const stride: usize = @intCast(shape.proj);
        const start: usize = @intCast(shape.conv);
        for (0..@intCast(shape.rows)) |r| {
            try checks.expect(std.mem.eql(u8, std.mem.sliceAsBytes(dp[r * stride + start ..][0..width]), std.mem.sliceAsBytes(oldp[r * stride + start ..][0..width])), "Flash Next norm dz replacement changed bytes", .{});
        }
        checks.pass("gdn-back: frozen Flash Next norm dy and dz byte-equal", .{});
    } else {
        var error2: f64 = 0;
        var norm2: f64 = 0;
        for (dy, old) |a, b| {
            error2 += (@as(f64, a) - b) * (@as(f64, a) - b);
            norm2 += @as(f64, a) * a;
        }
        try checks.expect(norm2 > 0 and @sqrt(error2 / norm2) > 0.15, "old sigmoid gate control failed to go red on the 27B", .{});
        checks.pass("CONTROL RED gdn-back: old sigmoid gate on 27B SiLU input gradients", .{});
    }
}
