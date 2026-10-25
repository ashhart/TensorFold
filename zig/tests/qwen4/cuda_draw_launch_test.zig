//! The actual CUDA drawer emits its ABI into owned memory; neither a CUDA driver nor sampler arithmetic runs.
const std = @import("std");
const C = @import("cuda");
const F = @import("qwen4").family;
const Core = @import("core");
const L = Core.lanes;
const t = std.testing;
const a = t.allocator;
const Trace = struct {
    var modules: usize = 0;
    var launches: usize = 0;
    var context_calls: usize = 0;
    var sampled_threads: c_int = 1024;
    var positions: [2]i32 = undefined;
    var rule: C.sample.Rule = undefined;
    const names = [_][]const u8{ "tf_draw", "tf_draw_ids", "tf_logprob_rows", "tf_topk_slices", "tf_topk_merge", "tf_draw_rows", "tf_grammar_mask", "tf_argmax_rows_i32_kernel" };
    fn load(out: *C.abi.Module, _: ?*const anyopaque) callconv(.c) c_int {
        modules += 1;
        out.* = @ptrFromInt(modules);
        return 0;
    }
    fn unload(_: C.abi.Module) callconv(.c) c_int {
        modules -= 1;
        return 0;
    }
    fn function(out: *C.abi.Function, _: C.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        for (names, 0..) |n, i| if (std.mem.eql(u8, n, std.mem.span(name))) {
            out.* = @ptrFromInt(i + 1);
            return 0;
        };
        out.* = @ptrFromInt(9);
        return 0;
    }
    fn attribute(out: *c_int, _: C.abi.FunctionAttribute, f: C.abi.Function) callconv(.c) c_int {
        out.* = if (@intFromPtr(f) == 6) sampled_threads else 256;
        return 0;
    }
    fn get(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: C.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: C.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        std.debug.assert(gx == 2 and gy == 1 and gz == 1 and by == 1 and bz == 1 and shared == 0);
        launches += 1;
        const index = @intFromPtr(f) - 1;
        if (index == 6) {
            std.debug.assert(bx == 256 and get(u32, args, 1) == 33 and get(u32, args, 3) == 2);
            return 0;
        }
        var output: u64 = 0;
        if (index == 5) {
            std.debug.assert(bx == 1024 and get(u32, args, 1) == 33);
            rule = @as(*const C.sample.Rule, @ptrFromInt(get(u64, args, 2))).*;
            positions = @as(*const [2]i32, @ptrFromInt(get(u64, args, 3))).*;
            output = get(u64, args, 4);
        } else {
            std.debug.assert(index == 7 and bx == 256 and get(i64, args, 2) == 2 and get(i64, args, 3) == 33 and get(i64, args, 4) == 33 and get(i32, args, 5) == 0);
            output = get(u64, args, 1);
        }
        const ids: *[2]u32 = @ptrFromInt(output);
        ids.* = .{ 7, 8 };
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) c_int {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return 0;
    }
    fn current(_: C.abi.Context) callconv(.c) c_int {
        context_calls += 1;
        return 0;
    }
    fn synchronize(_: C.abi.Stream) callconv(.c) c_int {
        return 0;
    }
};
fn driver() C.Driver {
    var d: C.Driver = .{ .api = undefined, .lib = undefined };
    d.api.cuModuleLoadData = Trace.load;
    d.api.cuModuleUnload = Trace.unload;
    d.api.cuModuleGetFunction = Trace.function;
    d.api.cuFuncGetAttribute = Trace.attribute;
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuMemcpyDtoH_v2 = Trace.download;
    d.api.cuCtxSetCurrent = Trace.current;
    d.api.cuStreamSynchronize = Trace.synchronize;
    return d;
}
const B = struct {
    pub const Buffer = C.DeviceBuffer;
    driver: *const C.Driver,
    stream: C.Stream,
    held: usize = 0,
    fail_drain: bool = false,
    writes: usize = 0,
    pub fn enter(_: *@This()) !void {}
    pub fn allocate(b: *@This(), n: usize) !Buffer {
        const bytes = try a.alignedAlloc(u8, .@"16", n);
        b.held += n;
        return .{ .d = b.driver, .ptr = @intFromPtr(bytes.ptr), .len = n };
    }
    pub fn release(b: *@This(), x: Buffer) void {
        const bytes: []align(16) u8 = @as([*]align(16) u8, @ptrFromInt(x.ptr))[0..x.len];
        b.held -= x.len;
        a.free(bytes);
    }
    pub fn write(b: *@This(), x: Buffer, off: usize, bytes: []const u8) !void {
        @memcpy(@as([*]u8, @ptrFromInt(x.ptr + off))[0..bytes.len], bytes);
        b.writes += 1;
    }
    pub fn drain(b: *@This()) !void {
        if (b.fail_drain) return error.PlantedDrawDrain;
    }
};
test "Qwen4 drawer launch: CUDA batch rules, position-minus-one ABI, BF16 greedy and retained resources" {
    const D = F.cuda_draw.Drawer(C, L, B);
    var driver_ = driver();
    var b: B = .{ .driver = &driver_, .stream = C.Stream.borrowed(&driver_, null) };
    Trace.modules = 0;
    Trace.launches = 0;
    Trace.sampled_threads = 1024;
    const image: [8]u8 align(8) = @splat(1);
    var d = try D.load(&b, 33, 4, 2, &image, &image);
    const x = try b.allocate(16 + 2 * 33 * 2);
    defer b.release(x);
    const tensor: F.arena.Arena(B).Tensor = .{ .buffer = x, .offset = 16, .rows = 2, .width = 33, .kind = .bf16 };
    const rule: L.Sampling = .{ .seed = 0x0123456789abcdef, .temperature = 0.5, .top_k = 7, .top_p = 0.75, .min_p = 0.125 };
    try d.draw(tensor, 16, &.{ 7, 19 }, rule, &.{ 0xffffffff, 1, 0xffffffff, 1 });
    try t.expectEqualSlices(i32, &.{ 6, 18 }, &Trace.positions);
    try t.expectEqual(rule.seed, Trace.rule.seed);
    try t.expectEqual(@as(f32, 2), Trace.rule.inv_t);
    try t.expectEqual(@as(u32, 7), Trace.rule.top_k);
    try t.expectEqual(@as(f32, 0.75), Trace.rule.top_p);
    const launched = Trace.launches;
    const writes = b.writes;
    try t.expectError(error.Qwen4DrawPosition, d.draw(tensor, 0, &.{ 0, 2 }, rule, &.{}));
    try t.expectEqual(launched, Trace.launches);
    try t.expectEqual(writes, b.writes);
    try d.draw(tensor, 16, &.{ 7, 19 }, null, &.{});
    var ids: [2]u32 = undefined;
    try d.read(16, &ids);
    try t.expectEqualSlices(u32, &.{ 7, 8 }, &ids);
    const held = b.held;
    b.fail_drain = true;
    try t.expectError(error.PlantedDrawDrain, d.close());
    try t.expectEqual(held, b.held);
    try t.expectEqual(@as(usize, 2), Trace.modules);
    b.fail_drain = false;
    try d.close();
    try t.expectEqual(@as(usize, 0), Trace.modules);
    Trace.sampled_threads = 512;
    try t.expectError(error.Qwen4DrawThreads, D.load(&b, 33, 4, 2, &image, &image));
    try t.expectEqual(x.len, b.held);
    try t.expectEqual(@as(usize, 0), Trace.modules);
    Trace.sampled_threads = 1024;
}
test "Qwen4 CUDA storage: every worker entry binds its stable context, missing and foreign contexts refuse" {
    const Storage = F.cuda_storage.Backend(C);
    var d = driver();
    var other = driver();
    var context: C.Context = .{ .d = &d, .device = 0, .handle = @ptrFromInt(17) };
    var b: Storage = .{ .driver = &d, .stream = C.Stream.borrowed(&d, null) };
    Trace.context_calls = 0;
    try t.expectError(error.Qwen4ContextMissing, b.enter());
    try t.expectEqual(@as(usize, 0), Trace.context_calls);
    b.context = &context;
    b.stream = C.Stream.borrowed(&other, null);
    try t.expectError(error.Qwen4DeviceContext, b.enter());
    try t.expectEqual(@as(usize, 0), Trace.context_calls);
    b.stream = C.Stream.borrowed(&d, null);
    try b.enter();
    try b.drain();
    try t.expectEqual(@as(usize, 2), Trace.context_calls);
}
