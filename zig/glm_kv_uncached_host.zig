//! Actual wide host-KV gather dispatches must retire the old lists before rewriting their row slots.
const std = @import("std");
const cuda = @import("cuda");
const Gather = @import("glm").kv_host.Gather;
const abi = cuda.abi;
const t = std.testing;
const Params = ?[*]?*anyopaque;
const Recording = struct {
    var counts: [16]u32 = undefined;
    var cleared_before_copy = false;
    var cleared_before_cached = false;
    var calls: [16]usize = undefined;
    var used: usize = 0;
    const names = [_][]const u8{ "tf_glm_kvh_mark", "tf_glm_kvh_copy", "tf_glm_kvh_remap", "tf_glm_kvh_reset", "tf_glm_kvh_prev_mark", "tf_glm_kvh_copy_cached", "tf_glm_kvh_unmark", "tf_glm_kvh_keep" };
    fn function(out: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
        for (names, 1..) |s, i| if (std.mem.eql(u8, s, std.mem.span(name))) {
            out.* = @ptrFromInt(i);
            return 0;
        };
        return 500;
    }
    fn launch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: Params, _: Params) callconv(.c) abi.Result {
        const i = @intFromPtr(f);
        calls[used] = i;
        used += 1;
        if (i == 2) cleared_before_copy = std.mem.allEqual(u32, &counts, 0);
        if (i == 5) cleared_before_cached = std.mem.allEqual(u32, &counts, 0);
        return 0;
    }
    fn set(dst: u64, value: c_uint, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        const ptr: [*]u32 = @ptrFromInt(dst);
        @memset(ptr[0..n], value);
        return 0;
    }
};

test "GLM host KV uncached: a 32-row call invalidates every old cached layer before copying, then a cached call sees none" {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleGetFunction = Recording.function;
    d.api.cuLaunchKernel = Recording.launch;
    d.api.cuMemsetD32Async = Recording.set;
    var n: u32 = 0;
    var g: Gather = undefined;
    g.n = .{ .d = &d, .ptr = @intFromPtr(&n), .len = 4, .borrowed = true };
    g.prev_n = .{ .d = &d, .ptr = @intFromPtr(&Recording.counts), .len = @sizeOf(@TypeOf(Recording.counts)), .borrowed = true };
    g.slot_of = .{ .d = &d, .ptr = 1, .len = 4096, .borrowed = true };
    g.list = g.slot_of;
    g.slots = g.slot_of;
    g.prev_of = g.slot_of;
    g.prev_rows = g.slot_of;
    g.cache = g.slot_of;
    g.over_dev = 2;
    g.cap = 1024;
    g.row_bytes = 512;
    g.tags = @splat(0);
    @memset(&Recording.counts, 19);
    Recording.used = 0;
    Recording.cleared_before_copy = false;
    Recording.cleared_before_cached = false;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const module: cuda.Module = .{ .d = &d, .handle = null };
    _ = try g.gather(module, stream, 123, 0, 0, 512, 64, 0, 0, 32, 8);
    try t.expectEqualSlices(usize, &.{ 1, 2, 3, 4 }, Recording.calls[0..Recording.used]);
    try t.expect(Recording.cleared_before_copy);
    try t.expect(std.mem.allEqual(u32, &Recording.counts, 0));
    _ = try g.gather(module, stream, 123, 0, 0, 512, 64, 0, 0, 1, 8);
    try t.expect(Recording.cleared_before_cached);
    try t.expectEqualSlices(usize, &.{ 1, 5, 6, 3, 7, 8 }, Recording.calls[4..Recording.used]);
}
