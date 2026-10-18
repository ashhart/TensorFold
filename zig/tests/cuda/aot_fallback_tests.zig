//! `tf-cuda-test aot-fallback <dir>`: a launch's 1 from the runtime variant gives the folded variant's bits (#1003).

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

pub fn run(gpu: Gpu, dir: []const u8) !void {
    var set = try cuda.aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, dir);
    defer set.deinit();
    try check.expect(set.variants.len == 2, "the fixture holds both specializations, got {d}", .{set.variants.len});

    const args = [_]cuda.aot.Arg{ cuda.aot.ptr("X", "*i32", 0), cuda.aot.ptr("OUT", "*i32", 0), cuda.aot.int("R", 1) };
    const consts = [_]cuda.aot.Const{cuda.aot.ci("BLOCK", 128)};

    // the full set: the constexpr-1 variant wins, in the exact pass
    const both = try set.find("tf_aot_fallback_k", &args, &consts);
    try check.expect(both.spec.params.len == 2, "the folded variant carries no runtime R", .{});

    // the same set without the folded variant: the runtime variant serves the launch by the fallback
    const only_runtime = cuda.aot.Set{ .parsed = set.parsed, .variants = blk: {
        for (set.variants, 0..) |*v, i| if (v.spec.params.len == 3) break :blk set.variants[i .. i + 1];
        return error.MissingRuntimeVariant;
    }, .gpa = set.gpa };
    const rt = try only_runtime.find("tf_aot_fallback_k", &args, &consts);
    try check.expect(rt.spec.params.len == 3, "the fallback serves the runtime-parameter variant", .{});

    // both on device, same inputs: byte-equal outputs
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, &xBytes);
    defer x.free();
    var out_a = try cuda.DeviceBuffer.alloc(gpu.d, xBytes.len);
    defer out_a.free();
    var out_b = try cuda.DeviceBuffer.alloc(gpu.d, xBytes.len);
    defer out_b.free();

    const full_args = [_]cuda.aot.Arg{ cuda.aot.ptr("X", "*i32", x.ptr), cuda.aot.ptr("OUT", "*i32", out_a.ptr), cuda.aot.int("R", 1) };
    try set.run(stream, "tf_aot_fallback_k", .{ 1, 1, 1 }, &full_args, &consts);
    const rt_args = [_]cuda.aot.Arg{ cuda.aot.ptr("X", "*i32", x.ptr), cuda.aot.ptr("OUT", "*i32", out_b.ptr), cuda.aot.int("R", 1) };
    try only_runtime.run(stream, "tf_aot_fallback_k", .{ 1, 1, 1 }, &rt_args, &consts);
    try stream.synchronize();

    const a = try gpu.gpa.alloc(u8, xBytes.len);
    defer gpu.gpa.free(a);
    try download(gpu.d, stream, a, out_a.ptr);
    const b = try gpu.gpa.alloc(u8, xBytes.len);
    defer gpu.gpa.free(b);
    try download(gpu.d, stream, b, out_b.ptr);
    try check.expect(std.mem.eql(u8, a, b), "the fallback's output is the folded variant's, byte for byte", .{});
    check.pass("aot-fallback: the runtime-parameter variant serves a launch of 1 with the folded variant's bits", .{});
}

const xBytes: [128 * 4]u8 = blk: {
    var b: [128 * 4]u8 = undefined;
    for (0..128) |i| std.mem.writeInt(i32, b[i * 4 ..][0..4], @intCast(@as(isize, @intCast(i * 7)) - 40), .little);
    break :blk b;
};

fn download(d: *const cuda.Driver, stream: cuda.Stream, dst: []u8, src: u64) !void {
    try d.check(d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, stream.handle), "cuMemcpyDtoHAsync");
    try stream.synchronize();
}
