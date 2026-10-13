//! Real Seq factories and Ops.fill32 wired to synthetic CUDA driver calls; no device or model is used.
const std = @import("std");
const cuda = @import("cuda");
const st = @import("state.zig");
const shape = @import("shape.zig");
const kern = @import("kernels.zig");
const policy = @import("reset_skip.zig");
const Call = struct { dst: u64, value: c_uint, words: usize };
var calls: [96]Call = undefined;
var call_count: usize = 0;
var allocations: usize = 0;
var frees: usize = 0;
var fail_alloc: bool = false;
var fail_fill: bool = false;

fn allocate(ptr: *u64, bytes: usize) callconv(.c) c_int {
    if (fail_alloc) return cuda.abi.error_not_ready;
    if (bytes == 0) return 1;
    allocations += 1;
    ptr.* = 0x100000000;
    return 0;
}
fn freeDevice(ptr: u64) callconv(.c) c_int {
    if (ptr == 0x100000000) frees += 1;
    return 0;
}
fn fill(dst: u64, value: c_uint, words: usize, stream: cuda.abi.Stream) callconv(.c) c_int {
    _ = stream;
    if (fail_fill) return cuda.abi.error_not_ready;
    if (call_count >= calls.len) return 1;
    calls[call_count] = .{ .dst = dst, .value = value, .words = words };
    call_count += 1;
    return 0;
}
fn errorText(code: c_int, out: *?[*:0]const u8) callconv(.c) c_int {
    _ = code;
    out.* = "synthetic reset failure";
    return 0;
}
fn setup(driver: *cuda.Driver, kernels: *kern.Kernels) kern.Ops {
    call_count = 0;
    allocations = 0;
    frees = 0;
    fail_alloc = false;
    fail_fill = false;
    driver.* = undefined;
    driver.api.cuMemAlloc_v2 = allocate;
    driver.api.cuMemFree_v2 = freeDevice;
    driver.api.cuMemsetD32Async = fill;
    driver.api.cuGetErrorName = errorText;
    driver.api.cuGetErrorString = errorText;
    kernels.* = undefined;
    kernels.d = driver;
    return .{ .k = kernels, .s = .{ .d = driver, .handle = null } };
}
fn checkReset(seq: *const st.Seq) !void {
    const g = shape.the_27b;
    var index: usize = 0;
    var bytes: u64 = 0;
    for (0..g.layers) |layer| if (shape.linear(layer)) {
        try std.testing.expectEqual(seq.rec[layer], calls[index].dst);
        try std.testing.expectEqual(g.deltaBytes() / 4, calls[index].words);
        try std.testing.expectEqual(seq.conv[layer], calls[index + 1].dst);
        try std.testing.expectEqual(g.convBytes() / 4, calls[index + 1].words);
        try std.testing.expectEqual(@as(c_uint, 0), calls[index].value | calls[index + 1].value);
        bytes += (calls[index].words + calls[index + 1].words) * 4;
        index += 2;
    };
    try std.testing.expectEqual(@as(usize, 96), index);
    try std.testing.expectEqual(index, call_count);
    try std.testing.expectEqual(@as(u64, 153944064), bytes);
    const actual = policy.resetWork(g);
    try std.testing.expectEqual(@as(u64, @intCast(index)), actual.calls);
    try std.testing.expectEqual(bytes, actual.bytes);
}

test "public Seq factory calls all original driver fills and owns one arena" {
    var d: cuda.Driver = undefined;
    var k: kern.Kernels = undefined;
    const ops = setup(&d, &k);
    var seq = try st.Seq.init(std.testing.allocator, ops, shape.the_27b, 303, false);
    try checkReset(&seq);
    try std.testing.expectEqual(@as(usize, 303), seq.capacity);
    try std.testing.expectEqual(@as(usize, 0), seq.pos);
    seq.deinit();
    try std.testing.expectEqual(@as(usize, 1), allocations);
    try std.testing.expectEqual(allocations, frees);
}

test "restore destination preserves capacity and layout while omitting only initial fills" {
    var d: cuda.Driver = undefined;
    var k: kern.Kernels = undefined;
    const ops = setup(&d, &k);
    var seq = try st.Seq.initRestoreDestination(std.testing.allocator, ops, shape.the_27b, 303);
    defer seq.deinit();
    try std.testing.expectEqual(@as(usize, 0), call_count);
    try std.testing.expectEqual(@as(usize, 303), seq.capacity);
    try std.testing.expectEqual(st.Seq.bytes(shape.the_27b, 303, false), seq.arena.buf.?.len);
    var layout: st.Arena = .{ .buf = null };
    _ = try layout.take(shape.the_27b.hidden * 2);
    _ = try layout.take(shape.the_27b.hidden / 64 * 4);
    try std.testing.expectEqual(@as(u64, 0x100000000), seq.head_in);
    try std.testing.expectEqual(@as(u64, 0x100000000 + 10240), seq.head_xs);
    for (0..shape.the_27b.layers) |layer| {
        if (shape.linear(layer)) {
            const rec_at = std.mem.alignForward(usize, layout.used, 256);
            _ = try layout.take(shape.the_27b.deltaBytes());
            const conv_at = std.mem.alignForward(usize, layout.used, 256);
            _ = try layout.take(shape.the_27b.convBytes());
            try std.testing.expectEqual(0x100000000 + @as(u64, @intCast(rec_at)), seq.rec[layer]);
            try std.testing.expectEqual(0x100000000 + @as(u64, @intCast(conv_at)), seq.conv[layer]);
        } else {
            _ = try layout.take(303 * shape.the_27b.kvInner() * 2);
            _ = try layout.take(303 * shape.the_27b.kvInner() * 2);
        }
    }
    try seq.reset(ops, shape.the_27b);
    try checkReset(&seq);
}

test "restore factory allocation failure owns no device buffer" {
    var d: cuda.Driver = undefined;
    var k: kern.Kernels = undefined;
    const ops = setup(&d, &k);
    fail_alloc = true;
    try std.testing.expectError(error.NotReady, st.Seq.initRestoreDestination(std.testing.allocator, ops, shape.the_27b, 303));
    try std.testing.expectEqual(@as(usize, 0), allocations + frees + call_count);
}

test "public factory reset error frees the allocated arena and host arrays" {
    var d: cuda.Driver = undefined;
    var k: kern.Kernels = undefined;
    const ops = setup(&d, &k);
    fail_fill = true;
    try std.testing.expectError(error.NotReady, st.Seq.init(std.testing.allocator, ops, shape.the_27b, 303, false));
    try std.testing.expectEqual(@as(usize, 1), allocations);
    try std.testing.expectEqual(allocations, frees);
}

test "public Engine sequence stays initialized and private restore factory omits fills" {
    var d: cuda.Driver = undefined;
    var engine: @import("engine.zig").Engine = undefined;
    const ops = setup(&d, &engine.k);
    engine.gpa = std.testing.allocator;
    engine.s = ops.s;
    engine.w.g = shape.the_27b;
    engine.kv8 = false;
    var safe = try engine.sequence(303);
    try checkReset(&safe);
    safe.deinit();
    call_count = 0;
    var deferred = try engine.sequenceForRestore(303);
    try std.testing.expectEqual(@as(usize, 0), call_count);
    try std.testing.expectEqual(@as(usize, 303), deferred.capacity);
    deferred.deinit();
    try std.testing.expectEqual(@as(usize, 2), allocations);
    try std.testing.expectEqual(allocations, frees);
}

test "FP8 public factory keeps scales and fills; private BF16 factory rejects FP8 before allocation" {
    var d: cuda.Driver = undefined;
    var engine: @import("engine.zig").Engine = undefined;
    const ops = setup(&d, &engine.k);
    engine.gpa = std.testing.allocator;
    engine.s = ops.s;
    engine.w.g = shape.the_27b;
    engine.kv8 = true;
    try std.testing.expectError(error.ExactCacheRequiresBf16, engine.sequenceForRestore(303));
    try std.testing.expectEqual(@as(usize, 0), allocations + call_count);
    var safe = try engine.sequence(303);
    defer safe.deinit();
    try checkReset(&safe);
    try std.testing.expect(safe.kv8);
    try std.testing.expectEqual(st.Seq.bytes(shape.the_27b, 303, true), safe.arena.buf.?.len);
    for (0..shape.the_27b.layers) |layer| {
        if (shape.linear(layer)) continue;
        try std.testing.expect(safe.kscale[layer] != 0 and safe.vscale[layer] != 0);
    }
}
