//! Actual State allocation puts the retained prompt row inside each charged sequence arena.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
var allocations: [4]?[]align(256) u8 = @splat(null);
fn alloc(ptr: *u64, n: usize) callconv(.c) cuda.abi.Result {
    const bytes = std.testing.allocator.alignedAlloc(u8, .fromByteUnits(256), n) catch return 2;
    for (&allocations) |*entry| if (entry.* == null) {
        entry.* = bytes;
        ptr.* = @intFromPtr(bytes.ptr);
        return 0;
    };
    std.testing.allocator.free(bytes);
    return 2;
}
fn free(ptr: u64) callconv(.c) cuda.abi.Result {
    for (&allocations) |*entry| if (entry.*) |bytes| if (@intFromPtr(bytes.ptr) == ptr) {
        std.testing.allocator.free(bytes);
        entry.* = null;
        return 0;
    };
    return 1;
}
fn fill(ptr: u64, value: u8, n: usize) callconv(.c) cuda.abi.Result {
    const bytes: [*]u8 = @ptrFromInt(ptr);
    @memset(bytes[0..n], value);
    return 0;
}
test "GLM prompt arena: two actual sequence allocations own disjoint, charged prompt rows" {
    const before = cuda.memory.usage(false).device;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = alloc;
    d.api.cuMemFree_v2 = free;
    d.api.cuMemsetD8_v2 = fill;
    var w: glm.weights.Weights = undefined;
    w.config = .{ .hidden = 16 };
    w.share = try glm.shard.Share.of(&w.config, 0, 1);
    w.layers = &.{};
    var a = try glm.forward.State.init(std.testing.allocator, &d, &w, 32);
    defer a.deinit();
    var b = try glm.forward.State.init(std.testing.allocator, &d, &w, 32);
    defer b.deinit();
    try std.testing.expect(a.prompt_hidden != b.prompt_hidden);
    for ([_]glm.forward.State{ a, b }) |state| {
        try std.testing.expect(state.prompt_hidden >= state.arena.buf.ptr);
        try std.testing.expect(state.prompt_hidden + 32 <= state.arena.buf.ptr + state.arena.buf.len);
    }
    const x: [*]u8 = @ptrFromInt(a.prompt_hidden);
    const y: [*]u8 = @ptrFromInt(b.prompt_hidden);
    @memset(x[0..32], 41);
    @memset(y[0..32], 93);
    for (x[0..32]) |byte| try std.testing.expectEqual(@as(u8, 41), byte);
    try std.testing.expectEqual(before + a.arena.buf.len + b.arena.buf.len, cuda.memory.usage(false).device);
    a.deinit();
    a = try glm.forward.State.init(std.testing.allocator, &d, &w, 32);
    for (y[0..32]) |byte| try std.testing.expectEqual(@as(u8, 93), byte);
}
