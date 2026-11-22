//! H3 video VAE CNN encoder: unit test of the causal conv3d.
const std = @import("std");
const enc = @import("vae_encoder");

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1 and std.mem.eql(u8, args[1], "conv")) {
        var wgt = [_]f32{1.0};
        var bias = [_]f32{0.0};
        var input = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const out = try enc.conv3d(gpa, &input, &wgt, &bias, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, false, 0);
        defer gpa.free(out);
        for (0..8) |i| std.debug.print("out[{d}]={d} ", .{ i, out[i] });
        std.debug.print("", .{});
        return;
    }
    if (args.len > 1 and std.mem.eql(u8, args[1], "conv3x3")) {
        const wgt = [_]f32{ 1, 1, 1 };
        const bias = [_]f32{0.0};
        const input = [_]f32{ 1, 1, 2, 2 };
        const out = try enc.conv3d(gpa, &input, &wgt, &bias, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, true, 0);
        defer gpa.free(out);
        std.debug.print("CONV-CAUSAL out[0]={d} out[1]={d} (expect 1, 2)", .{ out[0], out[1] });
        return;
    }
    std.debug.print("usage: tf-h3-vae-encode [conv|conv3x3]", .{});
}
