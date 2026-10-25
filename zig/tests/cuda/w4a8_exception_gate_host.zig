//! Drive the real nonfinite gates and production consumer through an in-process memory/warp shim.
const std = @import("std");
const cuda = @import("cuda");
const cases = @import("contract_w4a8.zig");
const g = @import("core").contract.gates;
const t = std.testing;
const abi = cuda.abi;
extern fn tf_w4a8_quantize_host([*]const u16, [*]u8, [*]f32, [*]f32, c_int, c_int, c_int, *u32) c_int;
const Fake = struct {
    var buffers: std.ArrayList([]u64) = .empty;
    var consumers: usize = 0;
    var miss_flag = false;
    fn alloc(out: *u64, n: usize) callconv(.c) abi.Result {
        const data = t.allocator.alloc(u64, (n + 7) / 8) catch return 2;
        buffers.append(t.allocator, data) catch return 2;
        out.* = @intFromPtr(data.ptr);
        return 0;
    }
    fn free(ptr: u64) callconv(.c) abi.Result {
        for (buffers.items, 0..) |b, i| if (@intFromPtr(b.ptr) == ptr) {
            t.allocator.free(buffers.swapRemove(i));
            return 0;
        };
        return 1;
    }
    fn upload(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return 0;
    }
    fn asyncDownload(dst: ?*anyopaque, src: u64, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        return download(dst, src, n);
    }
    fn zero(dst: u64, v: u8, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        @memset(@as([*]u8, @ptrFromInt(dst))[0..n], v);
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn read(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: abi.Function, rows: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        if (@intFromPtr(f) == 1) {
            const flag: *u32 = @ptrFromInt(read(u64, args, 6));
            const invalid = tf_w4a8_quantize_host(@ptrFromInt(read(u64, args, 0)), @ptrFromInt(read(u64, args, 1)), @ptrFromInt(read(u64, args, 2)), @ptrFromInt(read(u64, args, 3)), @intCast(rows), read(c_int, args, 4), read(c_int, args, 5), flag);
            if (invalid != 0) return 1;
            if (miss_flag) flag.* = 0;
        } else {
            consumers += 1;
            const out: [*]u8 = @ptrFromInt(read(u64, args, 6));
            @memset(out[0 .. @as(usize, @intCast(read(c_int, args, 7))) * @as(usize, @intCast(read(c_int, args, 8))) * 2], 0);
        }
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = upload;
        d.api.cuMemcpyDtoH_v2 = download;
        d.api.cuMemcpyDtoHAsync_v2 = asyncDownload;
        d.api.cuMemsetD8Async = zero;
        d.api.cuStreamSynchronize = sync;
        d.api.cuLaunchKernel = launch;
        return d;
    }
};
fn kernels(d: *cuda.Driver) cuda.w4a8.Kernels {
    return .{ .quantize = .{ .d = d, .handle = @ptrFromInt(1) }, .prefill = .{ .d = d, .handle = @ptrFromInt(2) } };
}
test "w4a8 host: real gates pass the source-produced flag, and both missed-flag plants go red" {
    const quiet = g.quiet;
    g.quiet = true;
    defer g.quiet = quiet;
    defer Fake.buffers.deinit(t.allocator);
    var d = Fake.driver();
    for (std.enums.values(cases.Kind)) |kind| for ([_]cases.Fault{ .none, .missed_flag }) |fault| {
        var p: cases.Case = .{ .rig = .{ .gpu = .{ .gpa = t.allocator, .io = undefined, .d = &d, .ctx = undefined }, .s = .borrowed(&d, null) }, .kind = kind, .fault = fault, .kernels = kernels(&d), .arena = .init(t.allocator) };
        defer p.deinit();
        Fake.consumers = 0;
        try t.expectEqual(if (fault == .none) g.Verdict.pass else g.Verdict.fail, try g.nonfinite(t.allocator, p.kernel(), 7));
        try t.expectEqual(@as(usize, 0), Fake.consumers);
        if (fault == .missed_flag) try t.expectEqual(@import("core").contract.spec.Gate.nonfinite, p.kernel().control.?);
    };
}
test "w4a8 host: production consumer skips exceptional scratch, clears the flag next time, and exposes a lost flag" {
    var d = Fake.driver();
    const kk = kernels(&d);
    var x: [64]u16 = @splat(0x3f80);
    var ws: [9]u64 = @splat(0);
    var flag: u32 = 0;
    var verdict: u32 = 0;
    var out: u16 = 0xa5a5;
    const q: cuda.qlinear.Affine4 = .{ .w = 1, .s = 2, .b = 3, .n = 1, .k = 64, .npad = 64 };
    Fake.consumers = 0;
    x[63] = 0x7f80;
    try kk.prompt(.borrowed(&d, null), @intFromPtr(&x), q, @intFromPtr(&out), 1, @intFromPtr(&ws), @intFromPtr(&flag), &verdict);
    try t.expect(verdict != 0 and flag != 0 and out == 0xa5a5 and Fake.consumers == 0);
    x[63] = 0x3f80;
    try kk.prompt(.borrowed(&d, null), @intFromPtr(&x), q, @intFromPtr(&out), 1, @intFromPtr(&ws), @intFromPtr(&flag), &verdict);
    try t.expect(verdict == 0 and flag == 0 and out == 0 and Fake.consumers == 1);
    Fake.miss_flag = true;
    defer Fake.miss_flag = false;
    x[0] = 0x7fc0;
    try kk.prompt(.borrowed(&d, null), @intFromPtr(&x), q, @intFromPtr(&out), 1, @intFromPtr(&ws), @intFromPtr(&flag), &verdict);
    try t.expect(verdict == 0 and Fake.consumers == 2);
}
