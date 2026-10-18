//! #927's staged tower GEMV is a candidate; every probe compares it with the merged numerical baseline.
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("engine.zig").Engine;

pub fn allowed(sm: u32, blas: usize, n: usize, k: usize) bool {
    return sm == 121 and blas == 130101 and n > 0 and n % 8 == 0 and k > 0 and k % 256 == 0;
}

pub fn verify(e: *Engine) !void {
    const sm = try e.ctx.capability();
    var library = try cuda.cublaslt.Library.open();
    defer library.close();
    const version = library.api.cublasLtGetVersion();
    if (!allowed(sm, version, 2048, 4096)) return error.UnqualifiedSpeedDevice;
    var image = try cuda.Module.load(&e.driver, cuda.kernels.drama3_staged_gemv);
    defer image.unload();
    const f = try image.function("drama3_staged_gemv");
    const a = e.gpa;
    var x = try cuda.DeviceBuffer.alloc(&e.driver, 4096 * 2);
    defer x.free();
    var ref = try cuda.DeviceBuffer.alloc(&e.driver, 2048 * 2);
    defer ref.free();
    var got = try cuda.DeviceBuffer.alloc(&e.driver, 2048 * 2);
    defer got.free();
    const host = try a.alloc(u16, 4 * 2048);
    defer a.free(host);
    defer e.stream.synchronize() catch {};
    for ([_]u16{ 0x3f80, 0xbf80, 0x3b80, 0x0000 }) |bits| {
        @memset(host[0..4096], bits);
        try x.upload(0, std.mem.sliceAsBytes(host[0..4096]));
        try e.k.gemv(e.stream, e.w.tower[0].wo, x.ptr, 4096, 2048, ref.ptr, 0);
        var args: cuda.Args = .{};
        args.add(x.ptr);
        args.add(e.w.tower[0].wo);
        args.add(got.ptr);
        args.add(@as(c_int, 4096));
        try cuda.launch.launch(f, .{ .grid = .{ .x = 256 }, .block = .{ .x = 32 } }, e.stream, &args);
        try e.stream.synchronize();
        try ref.download(0, std.mem.sliceAsBytes(host[4096..6144]));
        try got.download(0, std.mem.sliceAsBytes(host[6144..]));
        if (!std.mem.eql(u16, host[4096..6144], host[6144..])) return error.SpeedCandidateDiffers;
    }
    std.debug.print("staged GEMV probe matched; promotion still requires full-frame and PCM replay\n", .{});
}

test "unqualified architectures, libraries and partial row tiles keep the baseline" {
    try std.testing.expect(allowed(121, 130101, 2048, 2048));
    try std.testing.expect(!allowed(89, 130101, 2048, 2048));
    try std.testing.expect(!allowed(121, 130100, 2048, 2048));
    try std.testing.expect(!allowed(121, 130101, 4097, 2048));
}
