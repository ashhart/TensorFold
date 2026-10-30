//! The actual CUDA producer opens under an owned driver and its uploaded frequency bytes are checked before use.
const std = @import("std");
const C = @import("cuda");
const family = @import("qwen4").family;
const Gpu = family.cuda_gpu.Gpu(C);
const t = std.testing;
const Trace = struct {
    var modules: usize = 0;
    var allocated: usize = 0;
    var uploads: usize = 0;
    var words: [64]f32 align(16) = undefined;
    fn load(out: *C.abi.Module, _: ?*const anyopaque) callconv(.c) C.abi.Result {
        modules += 1;
        out.* = @ptrFromInt(modules);
        return 0;
    }
    fn unload(_: C.abi.Module) callconv(.c) C.abi.Result {
        modules -= 1;
        return 0;
    }
    fn function(out: *C.abi.Function, _: C.abi.Module, _: [*:0]const u8) callconv(.c) C.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn attribute(out: *c_int, _: C.abi.FunctionAttribute, _: C.abi.Function) callconv(.c) C.abi.Result {
        out.* = 1024;
        return 0;
    }
    fn setAttribute(_: C.abi.Function, _: C.abi.FunctionAttribute, bytes: c_int) callconv(.c) C.abi.Result {
        std.debug.assert(bytes == 26624);
        return 0;
    }
    fn current(_: C.abi.Context) callconv(.c) C.abi.Result {
        return 0;
    }
    fn drain(_: C.abi.Stream) callconv(.c) C.abi.Result {
        return 0;
    }
    fn allocate(out: *C.abi.DevicePtr, bytes: usize) callconv(.c) C.abi.Result {
        std.debug.assert(allocated == 0 and bytes > 0 and bytes <= @sizeOf(@TypeOf(words)));
        allocated = bytes;
        out.* = @intFromPtr(&words);
        return 0;
    }
    fn free(pointer: C.abi.DevicePtr) callconv(.c) C.abi.Result {
        std.debug.assert(pointer == @intFromPtr(&words) and allocated > 0);
        allocated = 0;
        return 0;
    }
    fn copy(to: C.abi.DevicePtr, from: ?*const anyopaque, bytes: usize) callconv(.c) C.abi.Result {
        std.debug.assert(to == @intFromPtr(&words) and bytes == allocated);
        @memcpy(std.mem.sliceAsBytes(&words)[0..bytes], @as([*]const u8, @ptrCast(from.?))[0..bytes]);
        uploads += 1;
        return 0;
    }
    fn driver() C.Driver {
        var d: C.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuModuleLoadData = load;
        d.api.cuModuleUnload = unload;
        d.api.cuModuleGetFunction = function;
        d.api.cuFuncGetAttribute = attribute;
        d.api.cuFuncSetAttribute = setAttribute;
        d.api.cuCtxSetCurrent = current;
        d.api.cuStreamSynchronize = drain;
        d.api.cuMemAlloc_v2 = allocate;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        return d;
    }
};
test "Qwen4 CUDA open: real rotary allocation uploads frequency zero without the Metal header" {
    var problem: family.Problem = .{};
    var c = try family.config.parse(t.allocator, family.device_fixture.config, &problem);
    defer c.deinit();
    var d = Trace.driver();
    const context: C.Context = .{ .d = &d, .device = 0, .handle = @ptrFromInt(0x1234) };
    const before = C.copy_source.current();
    defer C.copy_source.setCurrent(before);
    var backend: Gpu.Storage = .{ .driver = &d, .stream = C.Stream.borrowed(&d, null), .context = &context };
    const image: [8]u8 align(8) = @splat(1);
    const images: Gpu.Images = .{ .projection = &image, .hc_scalar = &image, .hc = &image, .gdn = &image, .attention = &image, .batch = &image, .ple = &image };
    const expected = [_]u32{ 0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c3, 0x3d0186e2, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e37, 0x3ae91528, 0x3a83126f, 0x3a136a16, 0x39a5cb5f, 0x393a7753 };
    for ([_]usize{ 32, 64, 128, 0 }) |rotary| {
        c.rotary = rotary;
        const uploads = Trace.uploads;
        {
            var gpu = try Gpu.load(t.allocator, &backend, &c, &problem, images, .native);
            defer gpu.close() catch unreachable;
            try t.expectEqual(@max(@as(usize, 1), rotary / 2) * @sizeOf(f32), gpu.rope.?.len);
            try t.expectEqual(uploads + 1, Trace.uploads);
            try t.expectEqual(@as(u32, 0x3f800000), @as(u32, @bitCast(Trace.words[0])));
            if (rotary == 32) for (Trace.words[0..16], expected) |word, want|
                try t.expectEqual(want, @as(u32, @bitCast(word)));
            try t.expect(gpu.attention != null and gpu.lexical != null and gpu.gdn != null);
        }
        try t.expectEqual(@as(usize, 0), Trace.modules);
        try t.expectEqual(@as(usize, 0), Trace.allocated);
    }
    c.theta = std.math.nan(f64);
    const uploads = Trace.uploads;
    try t.expectError(error.Qwen4RotaryTable, Gpu.load(t.allocator, &backend, &c, &problem, images, .native));
    try t.expectEqual(uploads, Trace.uploads);
    try t.expectEqual(@as(usize, 0), Trace.modules);
    try t.expectEqual(@as(usize, 0), Trace.allocated);
}
