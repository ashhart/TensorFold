//! Device-free admission and layout checks for the two LightOnOCR Qwen image towers.
const std = @import("std");
const v = @import("qwen_hip.zig");
const hip = @import("hip");
const op = @import("qwen_hip_ops.zig");
const core = @import("core");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;

fn config(big: bool) v.Config {
    return .{
        .depth = if (big) 24 else 12,
        .hidden = if (big) 1024 else 768,
        .heads = if (big) 16 else 12,
        .intermediate = if (big) 4096 else 3072,
        .out_hidden = if (big) 2560 else 1024,
        .patch = 16,
        .temporal = 2,
        .merge = 2,
        .positions = 2304,
        .image_token = 248056,
        .vision_start = 248053,
        .vision_end = 248054,
        .min_pixels = 4096,
        .max_pixels = 16777216,
        .rescale = 1.0 / 255.0,
        .mean = .{ 0.5, 0.5, 0.5 },
        .std = .{ 0.5, 0.5, 0.5 },
    };
}

test "LightOnOCR Qwen towers admit their separate graphs and account for widened storage" {
    const small = config(false);
    const big = config(true);
    try v.check(small);
    try v.check(big);
    // The local official safetensors headers give 153 and 297 tensors, each widened and aligned to 256 bytes.
    try std.testing.expectEqual(@as(usize, 402371840), try v.weightBytes(small));
    try std.testing.expectEqual(@as(usize, 1334057216), try v.weightBytes(big));
    try std.testing.expect(try v.scratchBytes(small, 4) < try v.scratchBytes(big, 4));
    const sc = try v.Scratch.init(big, v.max_patches);
    var previous: usize = 0;
    inline for (@typeInfo(v.Scratch).@"struct".field_names) |name| {
        const offset = @field(sc, name);
        try std.testing.expectEqual(@as(usize, 0), offset % 256);
        try std.testing.expect(offset >= previous);
        previous = offset;
    }
    try std.testing.expectEqual(try v.scratchBytes(big, v.max_patches), sc.bytes);
    var other = small;
    other.hidden = 1152;
    other.heads = 16;
    try std.testing.expectError(error.UnsupportedQwenVision, v.check(other));
    try std.testing.expectError(error.BadImagePatches, v.scratchBytes(small, 0));
    try std.testing.expectError(error.BadImagePatches, v.scratchBytes(small, v.max_patches + 1));
}

test "image admission rejects partial patches and invalid image boundaries" {
    try std.testing.expectEqual(@as(usize, 12), try v.rowsOf(&.{ .{ .h = 2, .w = 4 }, .{ .h = 2, .w = 2 } }, 12 * v.patch_width));
    try std.testing.expectError(error.BadImageGrid, v.rowsOf(&.{.{ .h = 3, .w = 4 }}, 12 * v.patch_width));
    try std.testing.expectError(error.BadImageGrid, v.rowsOf(&.{.{ .h = 0, .w = 4 }}, 0));
    try std.testing.expectError(error.BadImagePatches, v.rowsOf(&.{.{ .h = 2, .w = 4 }}, 8 * v.patch_width - 1));
    try std.testing.expectError(error.BadImagePatches, v.rowsOf(&.{}, 0));
    try std.testing.expectError(error.BadImagePatches, v.rowsOf(&.{.{ .h = 130, .w = 128 }}, 16640 * v.patch_width));
}

test "a narrow vision tower reserves its wide merger output accumulator" {
    var c = config(false);
    c.hidden = 64;
    c.heads = 1;
    c.intermediate = 64;
    c.out_hidden = 8192;
    const sc = try v.Scratch.init(c, 4);
    try std.testing.expect(sc.q - sc.acc >= c.out_hidden * 4);
    const small = try v.Scratch.init(config(false), v.max_patches);
    const big = try v.Scratch.init(config(true), v.max_patches);
    try std.testing.expectEqual(@as(usize, 503971840), small.bytes);
    try std.testing.expectEqual(@as(usize, 654966784), big.bytes);
    try std.testing.expectEqual(@as(usize, 24115968), try v.scratchBytes(config(false), 4 * 196));
    try std.testing.expectEqual(@as(usize, 31341312), try v.scratchBytes(config(true), 4 * 196));
}

test "merge-order positions preserve each image's learned-grid corners" {
    var hw: [24]i32 = undefined;
    var index: [48]i32 = undefined;
    var weights: [48]f32 = undefined;
    v.tables(&.{ .{ .h = 2, .w = 4 }, .{ .h = 2, .w = 2 } }, &hw, &index, &weights);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 0, 0, 0, 1, 1, 0, 1, 1 }, &hw);
    try std.testing.expectEqual(@as(i32, 0), index[0]);
    try std.testing.expectEqual(@as(i32, 2303), index[7 * 4]);
    try std.testing.expectEqual(@as(i32, 0), index[8 * 4]);
    try std.testing.expectEqual(@as(i32, 2303), index[11 * 4]);
    for (0..12) |i| {
        var sum: f32 = 0;
        for (weights[i * 4 ..][0..4]) |w| sum += w;
        try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    }
}

test "all native tower entry points compile without opening a device" {
    _ = &v.Tower.init;
    _ = &v.Tower.encode;
    _ = &v.Tower.deinit;
}

const Mock = struct {
    var pending: bool = false;
    var default_pending: bool = false;
    var default_waits: usize = 0;
    var unsafe_launch: bool = false;
    var launches: usize = 0;
    var fail_launch: usize = 0;
    var fail_wait: bool = false;
    var waits: usize = 0;
    fn upload(_: hip.abi.DevicePtr, _: [*]const u8, _: usize) callconv(.c) hip.abi.Result {
        default_pending = true;
        return 0;
    }
    fn launch(_: hip.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: hip.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) hip.abi.Result {
        launches += 1;
        unsafe_launch = unsafe_launch or default_pending;
        if (launches == fail_launch) return 1;
        pending = true;
        return 0;
    }
    fn sync(s: hip.abi.Stream) callconv(.c) hip.abi.Result {
        if (s == null) {
            default_pending = false;
            default_waits += 1;
            return 0;
        }
        waits += 1;
        if (fail_wait) {
            fail_wait = false;
            return 1;
        }
        pending = false;
        return 0;
    }
};

fn setup(r: *hip.Runtime, t: *v.Tower) void {
    Mock.pending = false;
    Mock.default_pending = false;
    Mock.default_waits = 0;
    Mock.unsafe_launch = false;
    Mock.launches = 0;
    Mock.fail_launch = 0;
    Mock.fail_wait = false;
    Mock.waits = 0;
    r.api.hipMemcpyHtoD = Mock.upload;
    r.api.hipModuleLaunchKernel = Mock.launch;
    r.api.hipStreamSynchronize = Mock.sync;
    t.gpa = std.testing.allocator;
    t.r = r;
    t.s = .{ .r = r, .handle = @ptrFromInt(0x10) };
    t.c = config(false);
    t.c.hidden = 64;
    t.c.heads = 1;
    t.c.depth = 1;
    t.c.intermediate = 64;
    t.c.out_hidden = 64;
    t.capacity = 4;
    t.arena = .{ .r = r, .ptr = @ptrFromInt(0x100000), .len = 1048576 };
    t.scratch = t.arena.base();
    t.patch_w = 1;
    t.patch_b = 2;
    t.blocks = std.mem.zeroes(@TypeOf(t.blocks));
    for (&t.blocks) |*b| {
        b.qkv_w = 3;
        b.proj_w = 4;
        b.fc1_w = 5;
        b.fc2_w = 6;
    }
    t.fc1_w = 7;
    t.fc2_w = 8;
    inline for (@typeInfo(op.Kernels).@"struct".field_names) |name| {
        if (comptime !std.mem.eql(u8, name, "mod")) @field(t.ops, name) = .{ .r = r, .handle = @ptrFromInt(0x20) };
    }
}

fn draining(fail_launch: usize, fail_wait: bool) !void {
    var r: hip.Runtime = undefined;
    var t: v.Tower = undefined;
    setup(&r, &t);
    Mock.fail_launch = fail_launch;
    Mock.fail_wait = fail_wait;
    const px = try std.testing.allocator.alloc(f32, 4 * v.patch_width);
    defer std.testing.allocator.free(px);
    @memset(px, 0.25);
    try std.testing.expectError(error.HipFailed, t.encode(px, &.{.{ .h = 2, .w = 2 }}, 0x400000));
    try std.testing.expect(!Mock.pending);
    try std.testing.expect(!Mock.default_pending);
    try std.testing.expect(!Mock.unsafe_launch);
    try std.testing.expectEqual(@as(usize, 4), Mock.default_waits);
    try std.testing.expectEqual(@as(usize, if (fail_wait) 2 else 1), Mock.waits);
    Mock.fail_launch = 0;
    try t.encode(px, &.{.{ .h = 2, .w = 2 }}, 0x400000);
    try std.testing.expect(!Mock.pending);
}

test "a failed launch drains earlier work before scratch can be reused" {
    try draining(3, false);
}

test "a failed completion wait is retried before scratch can be reused" {
    try draining(0, true);
}

const Stop = struct {
    at: usize,
    polls: usize = 0,
    undrained: bool = false,
    fn check(ctx: *anyopaque) bool {
        const stop: *Stop = @ptrCast(@alignCast(ctx));
        stop.polls += 1;
        stop.undrained = stop.undrained or Mock.pending or Mock.default_pending;
        return Mock.launches >= stop.at;
    }
    fn hook(stop: *Stop) CancelCheck {
        return .{ .ptr = stop, .check = check };
    }
};

fn cancelled(at: usize, expected: usize, polls: usize) !void {
    var r: hip.Runtime = undefined;
    var t: v.Tower = undefined;
    setup(&r, &t);
    t.c.depth = 3;
    var stop: Stop = .{ .at = at };
    const px = try std.testing.allocator.alloc(f32, 4 * v.patch_width);
    defer std.testing.allocator.free(px);
    @memset(px, 0.25);
    try std.testing.expectError(error.Cancelled, t.encodeCancelable(px, &.{.{ .h = 2, .w = 2 }}, 0x400000, stop.hook()));
    try std.testing.expectEqual(expected, Mock.launches);
    try std.testing.expect(!stop.undrained);
    try std.testing.expect(!Mock.pending);
    try std.testing.expect(!Mock.default_pending);
    try std.testing.expectEqual(polls, stop.polls);
    const before = Mock.waits;
    try t.encode(px, &.{.{ .h = 2, .w = 2 }}, 0x400000);
    try std.testing.expectEqual(before + 1, Mock.waits);
    try std.testing.expect(!Mock.pending);
}

test "initial image cancellation launches nothing and leaves scratch reusable" {
    try cancelled(0, 0, 1);
}

test "mid-block image cancellation drains and skips later blocks and projector" {
    try cancelled(3, 13, 3);
}

test "image cancellation after the last encoder block skips the projector" {
    try cancelled(35, 35, 5);
}

test "bounded tower reservations reject excess rows before any upload or launch" {
    var r: hip.Runtime = undefined;
    var t: v.Tower = undefined;
    setup(&r, &t);
    const px = try std.testing.allocator.alloc(f32, 8 * v.patch_width);
    defer std.testing.allocator.free(px);
    @memset(px, 0.25);
    try std.testing.expectError(error.BadImagePatches, t.encode(px, &.{.{ .h = 2, .w = 4 }}, 0x400000));
    try std.testing.expectEqual(@as(usize, 0), Mock.default_waits);
    try std.testing.expectEqual(@as(usize, 0), Mock.launches);
    try t.encode(px[0 .. 4 * v.patch_width], &.{.{ .h = 2, .w = 2 }}, 0x400000);
    try std.testing.expectEqual(@as(usize, 18), Mock.launches);
    try std.testing.expect(!Mock.pending);
}

test "invalid reservation capacities are refused before opening the device or checkpoint" {
    var r: hip.Runtime = undefined;
    const s: hip.Stream = .{ .r = &r, .handle = null };
    const lib: hip.Launcher = undefined;
    try std.testing.expectError(error.BadImagePatches, v.Tower.init(std.testing.allocator, undefined, &r, s, &lib, "", config(false), 0));
    try std.testing.expectError(error.BadImagePatches, v.Tower.init(std.testing.allocator, undefined, &r, s, &lib, "", config(false), v.max_patches + 1));
}
