//! Geometry and real launch ABI are recorded for Q24/KV4; no attention arithmetic or device is executed here.
const std = @import("std");
const cuda = @import("cuda");
const family = @import("qwen4").family;
const qsa = family.cuda_attention;
const K = qsa.Kernels(cuda);
const T = std.testing;
const Trace = struct {
    var live: usize = 0;
    var calls: usize = 0;
    var fail_at: ?usize = null;
    var fail_drain = false;
    var seen: [7]bool = @splat(false);
    fn load(out: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        live += 1;
        out.* = @ptrFromInt(0x100 + live);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        for (K.names, 0..) |n, i| if (std.mem.eql(u8, n, std.mem.span(name))) {
            out.* = @ptrFromInt(i + 1);
            return 0;
        };
        return cuda.abi.error_not_found;
    }
    fn attribute(out: *c_int, _: cuda.abi.FunctionAttribute, f: cuda.abi.Function) callconv(.c) cuda.abi.Result {
        out.* = if (fail_at == @intFromPtr(f) - 1) 64 else 1024;
        return 0;
    }
    fn setAttribute(_: cuda.abi.Function, _: cuda.abi.FunctionAttribute, value: c_int) callconv(.c) cuda.abi.Result {
        std.debug.assert(value == 26624);
        return 0;
    }
    fn drain(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fail_drain) cuda.abi.error_not_ready else 0;
    }
    fn arg(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn integer(args: ?[*]?*anyopaque, i: usize, want: c_int) void {
        std.debug.assert(arg(c_int, args, i) == want);
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const i = @intFromPtr(f) - 1;
        seen[i] = true;
        calls += 1;
        std.debug.assert(by == 1 and bz == 1 and gx > 0 and gy > 0 and gz > 0);
        switch (i) {
            0 => {
                std.debug.assert(gx == 3 and gy == 33 and bx == 256);
                integer(args, 12, 14976);
                integer(args, 13, 24);
                integer(args, 14, 4);
                integer(args, 15, 4);
                integer(args, 16, 128);
                integer(args, 17, 32);
            },
            1 => {
                std.debug.assert(gx == 2 and bx == 128);
                integer(args, 6, 3);
                integer(args, 7, 128);
                integer(args, 8, 32);
                integer(args, 9, 4);
            },
            2 => {
                std.debug.assert(gx == 3 and gy == 16 and bx == 256);
                integer(args, 4, 1024);
                integer(args, 5, 4);
                integer(args, 6, 4);
                integer(args, 7, 512);
            },
            3 => {
                std.debug.assert(gx == 3 and bx == 1024);
                integer(args, 5, 1024);
                integer(args, 6, 2052);
                integer(args, 7, 512);
                integer(args, 8, 4);
            },
            4 => {
                std.debug.assert(gx == 3 and gy == 4 and gz == 5 and bx == 128 and shared == 26624);
                integer(args, 10, 24);
                integer(args, 11, 4);
                integer(args, 12, 6);
                integer(args, 14, 5);
                integer(args, 16, 2052);
                integer(args, 17, 1);
            },
            5 => {
                std.debug.assert(gx == 3 and gy == 24 and bx == 256);
                integer(args, 7, 24);
                integer(args, 8, 4);
                integer(args, 9, 6);
                integer(args, 11, 5);
                integer(args, 12, 1);
            },
            6 => {
                std.debug.assert(gx == 3 and gy == 24 and bx == 256);
                integer(args, 4, 14976);
                integer(args, 5, 24);
            },
            else => unreachable,
        }
        if (i != 4) std.debug.assert(shared == 0);
        return 0;
    }
};
fn driver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleLoadData = Trace.load;
    d.api.cuModuleUnload = Trace.unload;
    d.api.cuModuleGetFunction = Trace.function;
    d.api.cuFuncGetAttribute = Trace.attribute;
    d.api.cuFuncSetAttribute = Trace.setAttribute;
    d.api.cuStreamSynchronize = Trace.drain;
    d.api.cuLaunchKernel = Trace.launch;
    return d;
}
test "Qwen4 QSA launch: native Q24 KV4 geometry and seven real launch ABIs; all refusals precede writes" {
    var d = driver();
    const s = cuda.Stream.borrowed(&d, null);
    const image: [8]u8 align(8) = @splat(1);
    var k = try K.load(&d, &image, &image);
    const g: qsa.Geometry = .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 4, .top = 512 };
    const p: qsa.Call = .{ .rows = 3, .start = 2100, .capacity = 4096, .eps = 1e-6 };
    const z = try qsa.sizes(g, p);
    var buffers: [23]cuda.DeviceBuffer = undefined;
    var views: [23]K.View = undefined;
    for (&buffers, &views, z.bytes, 0..) |*buffer, *v, n, i| {
        buffer.* = .{ .d = &d, .ptr = (i + 1) * 0x1000000, .len = n };
        v.* = .{ .buffer = buffer, .len = n };
    }
    try k.encode(s, g, p, views);
    for (Trace.seen) |seen| try T.expect(seen);
    const count = Trace.calls;
    var invalid = views;
    invalid[21] = views[20];
    try T.expectError(error.Qwen4AttentionAlias, k.encode(s, g, p, invalid));
    invalid = views;
    invalid[17].offset = 4;
    invalid[17].len -= 4;
    try T.expectError(error.Qwen4AttentionBuffer, k.encode(s, g, p, invalid));
    var foreign = d;
    try T.expectError(error.Qwen4AttentionContext, k.encode(cuda.Stream.borrowed(&foreign, null), g, p, views));
    var wrong = p;
    wrong.start = 4095;
    try T.expectError(error.Qwen4AttentionShape, k.encode(s, g, wrong, views));
    try T.expectEqual(count, Trace.calls);
    Trace.fail_drain = true;
    try T.expectError(error.NotReady, k.close(s));
    try T.expectEqual(@as(usize, 2), Trace.live);
    Trace.fail_drain = false;
    try k.close(s);
    try T.expectEqual(@as(usize, 0), Trace.live);
    for (0..7) |i| {
        Trace.fail_at = i;
        try T.expectError(error.Qwen4AttentionThreads, K.load(&d, &image, &image));
        try T.expectEqual(@as(usize, 0), Trace.live);
    }
    Trace.fail_at = null;
}
