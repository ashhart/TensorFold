//! A fake Objective-C encoder records the real command wrapper's submissions.
const std = @import("std");
const objc = @import("objc.zig");
const metal4 = @import("metal4.zig");
const Size = @import("types.zig").Size;
const Id = objc.Id;
const sel = objc.sel;

extern "c" fn objc_allocateClassPair(parent: objc.Class, name: [*:0]const u8, extra: usize) ?objc.Class;
extern "c" fn objc_registerClassPair(class: objc.Class) void;
extern "c" fn class_addMethod(class: objc.Class, selector: objc.Sel, imp: *const anyopaque, encoding: [*:0]const u8) bool;

const Action = enum { pipeline, constant, groups, threads, barrier, update, wait };
const Entry = struct {
    action: Action,
    value: usize = 0,
    index: usize = 0,
    grid: Size = .{ .width = 0, .height = 0, .depth = 0 },
    threads: Size = .{ .width = 0, .height = 0, .depth = 0 },
};
var trace: [32]Entry = undefined;
var count: usize = 0;
var expect_metal4 = false;

fn record(entry: Entry) void {
    trace[count] = entry;
    count += 1;
}

fn pipeline(_: Id, _: objc.Sel, id: Id) callconv(.c) void {
    record(.{ .action = .pipeline, .value = @intFromPtr(id) });
}
fn constant3(_: Id, _: objc.Sel, bytes: [*]const u8, len: usize, index: usize) callconv(.c) void {
    std.debug.assert(!expect_metal4 and len == 4);
    record(.{ .action = .constant, .value = std.mem.readInt(u32, bytes[0..4], .little), .index = index });
}
fn constant4(_: Id, _: objc.Sel, address: u64, index: usize) callconv(.c) void {
    std.debug.assert(expect_metal4);
    const bytes: *const [4]u8 = @ptrFromInt(address);
    record(.{ .action = .constant, .value = std.mem.readInt(u32, bytes, .little), .index = index });
}
fn groups(_: Id, _: objc.Sel, grid: Size, size: Size) callconv(.c) void {
    record(.{ .action = .groups, .grid = grid, .threads = size });
}
fn threads(_: Id, _: objc.Sel, grid: Size, size: Size) callconv(.c) void {
    record(.{ .action = .threads, .grid = grid, .threads = size });
}
fn barrier3(_: Id, _: objc.Sel, _: usize) callconv(.c) void {
    std.debug.assert(!expect_metal4);
    record(.{ .action = .barrier });
}
fn barrier4(_: Id, _: objc.Sel, after: usize, before: usize, visibility: usize) callconv(.c) void {
    std.debug.assert(expect_metal4 and after == metal4.stage_dispatch and before == after and visibility == 1);
    record(.{ .action = .barrier });
}
fn update3(_: Id, _: objc.Sel, id: Id) callconv(.c) void {
    std.debug.assert(!expect_metal4);
    record(.{ .action = .update, .value = @intFromPtr(id) });
}
fn wait3(_: Id, _: objc.Sel, id: Id) callconv(.c) void {
    std.debug.assert(!expect_metal4);
    record(.{ .action = .wait, .value = @intFromPtr(id) });
}
fn update4(_: Id, _: objc.Sel, fence: Id, stages: usize) callconv(.c) void {
    std.debug.assert(expect_metal4 and stages == metal4.stage_dispatch);
    record(.{ .action = .update, .value = @intFromPtr(fence) });
}
fn wait4(_: Id, _: objc.Sel, fence: Id, stages: usize) callconv(.c) void {
    std.debug.assert(expect_metal4 and stages == metal4.stage_dispatch);
    record(.{ .action = .wait, .value = @intFromPtr(fence) });
}

fn fakeClass() !objc.Class {
    const class = objc_allocateClassPair(objc.class("NSObject"), "TFSubmissionRecorder", 0) orelse return error.NoTestClass;
    const size = "{MTLSize=QQQ}";
    const methods = .{
        .{ "setComputePipelineState:", &pipeline, "v@:@" },
        .{ "setBytes:length:atIndex:", &constant3, "v@:^vQQ" },
        .{ "setAddress:atIndex:", &constant4, "v@:QQ" },
        .{ "dispatchThreadgroups:threadsPerThreadgroup:", &groups, "v@:" ++ size ++ size },
        .{ "dispatchThreads:threadsPerThreadgroup:", &threads, "v@:" ++ size ++ size },
        .{ "memoryBarrierWithScope:", &barrier3, "v@:Q" },
        .{ "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", &barrier4, "v@:QQQ" },
        .{ "updateFence:", &update3, "v@:@" },
        .{ "waitForFence:", &wait3, "v@:@" },
        .{ "updateFence:afterEncoderStages:", &update4, "v@:@Q" },
        .{ "waitForFence:beforeEncoderStages:", &wait4, "v@:@Q" },
    };
    inline for (methods) |m| {
        if (!class_addMethod(class, sel(m[0]), @ptrCast(m[1]), m[2])) return error.NoTestMethod;
    }
    objc_registerClassPair(class);
    return class;
}

fn encode(encoder: anytype, id: Id, serial: bool) void {
    const size: Size = .{ .width = 32, .height = 1, .depth = 1 };
    encoder.setPipeline(.{ .id = id });
    encoder.setValue(@as(u32, 37), 7);
    encoder.dispatchGroups(.{ .width = 3, .height = 4, .depth = 1 }, size);
    if (!serial) encoder.barrier();
    encoder.updateFence(id);
    encoder.waitForFence(id);
    encoder.setPipeline(.{ .id = id });
    encoder.setValue(@as(u32, 89), 2);
    encoder.dispatchThreads(.{ .width = 119, .height = 1, .depth = 1 }, size);
    if (!serial) encoder.barrier();
}

pub fn check(comptime Encoder: type) !void {
    const id = objc.msg(Id, try fakeClass(), "new", .{});
    defer objc.release(id);
    var state: metal4.State = .{ .device = id, .queue = id, .event = id, .set = id, .table = undefined };
    var slot: metal4.Slot = .{ .q = &state, .cb = id, .allocator = id, .table = id, .block = undefined };
    var page: [128]u8 = undefined;
    slot.pages[0] = .{ .buf = id, .ptr = &page, .addr = @intFromPtr(&page) };
    for ([_]bool{ false, true }) |serial| {
        slot.bump = 0;
        slot.serial = serial;
        count = 0;
        expect_metal4 = false;
        encode(Encoder{ .id = id }, id, serial);
        var expected: [32]Entry = undefined;
        const n = count;
        @memcpy(expected[0..n], trace[0..n]);
        count = 0;
        expect_metal4 = true;
        encode(Encoder{ .id = id, .m4 = &slot }, id, serial);
        var comparable: usize = 0;
        for (trace[0..count]) |entry| {
            if (serial and entry.action == .barrier) continue;
            trace[comparable] = entry;
            comparable += 1;
        }
        try std.testing.expectEqual(@as(usize, if (serial) 2 else 0), count - comparable);
        try std.testing.expectEqualSlices(Entry, expected[0..n], trace[0..comparable]);
    }
}
