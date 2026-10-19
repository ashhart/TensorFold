//! Actual Zig binding + replay ABI against Python Triton's unfused RMS/store fixture bytes.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const Fixture = @import("../fixture.zig").Fixture;

fn up(gpu: check.Gpu, fx: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const bytes = try fx.bytes(name);
    defer gpu.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(gpu.d, bytes);
}
fn same(gpu: check.Gpu, fx: Fixture, b: cuda.DeviceBuffer, name: []const u8) !void {
    const want = try fx.bytes(name);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, want.len);
    defer gpu.gpa.free(got);
    try b.download(0, got);
    try check.sameBytes(name, got, want);
}

pub fn run(gpu: check.Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var x = try up(gpu, fx, "x");
    defer x.free();
    var w = try up(gpu, fx, "w");
    defer w.free();
    var cs = try up(gpu, fx, "cs");
    defer cs.free();
    var pos = try up(gpu, fx, "pos");
    defer pos.free();
    var sl = try up(gpu, fx, "sl");
    defer sl.free();
    var v = try up(gpu, fx, "v");
    defer v.free();
    var s = try up(gpu, fx, "s");
    defer s.free();
    var norm = try up(gpu, fx, "norm");
    defer norm.free();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const n = try fx.int("n");
    const rows = try fx.int("rows") == 1;
    const ring = try fx.int("ring");
    const eps = try fx.float("eps");
    try v.fill8(0xa5, null);
    try s.fill8(0xa5, null);
    try norm.fill8(0xa5, null);
    try k.kvStore(stream).normStore(.{
        .x = x.ptr,
        .w = w.ptr,
        .cs = cs.ptr,
        .v = v.ptr,
        .s = s.ptr,
        .pos = pos.ptr,
        .sl = if (rows) sl.ptr else 0,
        .norm_out = norm.ptr,
        .ring = @intCast(ring),
        .rows = @intFromBool(rows),
        .eps = @floatCast(eps),
    }, @intCast(n));
    try stream.synchronize();
    try same(gpu, fx, norm, "norm");
    try same(gpu, fx, v, "v");
    try same(gpu, fx, s, "s");

    // Production uses norm_out=null. Also exercise argument packing through the real replay dispatcher.
    try v.fill8(0xa5, null);
    try s.fill8(0xa5, null);
    const rp = dsv41.replay;
    const args = [_]rp.Arg{
        .{ .tensor = .{ .ptr = x.ptr, .dtype = .bf16, .shape = &.{ n, 512 }, .stride = &.{ 512, 1 } } },
        .{ .tensor = .{ .ptr = w.ptr, .dtype = .f32, .shape = &.{512}, .stride = &.{1} } },
        .{ .tensor = .{ .ptr = cs.ptr, .dtype = .f32, .shape = &.{ 1024, 64 }, .stride = &.{ 64, 1 } } },
        .{ .tensor = .{ .ptr = v.ptr, .dtype = .u8, .shape = &.{ 3 * ring + 1, 576 }, .stride = &.{ 576, 1 } } },
        .{ .tensor = .{ .ptr = s.ptr, .dtype = .u8, .shape = &.{ 3 * ring + 1, 8 }, .stride = &.{ 8, 1 } } },
        .{ .tensor = .{ .ptr = pos.ptr, .dtype = if (rows) .i64 else .i32, .shape = if (rows) &.{n} else &.{1}, .stride = &.{1} } },
        if (rows) .{ .tensor = .{ .ptr = sl.ptr, .dtype = .i64, .shape = &.{n}, .stride = &.{1} } } else .none,
        .{ .int = 512 },
        .{ .int = 64 },
        .{ .int = 576 },
        .{ .int = 8 },
        .{ .float = 1.0 / 512.0 },
        .{ .float = eps },
        .{ .int = ring },
        .{ .boolean = rows },
    };
    var scratch: rp.Scratch = .{ .ptr = 0, .len = 0 };
    if (!try rp.call(k, stream, "tf_dsv41_kv_glue_v1", "norm_store", &args, &.{}, &scratch)) return error.NoBinding;
    try stream.synchronize();
    try same(gpu, fx, v, "v");
    try same(gpu, fx, s, "s");
    check.pass("KV norm/store: debug and replay bindings, norm/cache/guards byte-exact ({d} rows)", .{n});
}
