//! Host tests for the submit, fence-wait and begin paths of Commands: poisoning and the barrier owed to a submission.
const std = @import("std");
const c = @import("vk");
const Device = @import("device.zig").Device;
const commands = @import("commands.zig");
const Commands = commands.Commands;
const Buffer = @import("memory.zig").Buffer;

test "a failed vkQueueSubmit poisons the stream: a caught error cannot reset a pending buffer" {
    const Calls = struct {
        var ends: usize = 0;
        var begins: usize = 0;
        var queues: usize = 0;
        fn end(_: c.VkCommandBuffer) callconv(.c) c.VkResult {
            ends += 1;
            return c.VK_SUCCESS;
        }
        fn begin(_: c.VkCommandBuffer, _: [*c]const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult {
            begins += 1;
            return c.VK_SUCCESS;
        }
        fn queueSubmit(_: c.VkQueue, _: c_uint, _: [*c]const c.VkSubmitInfo, _: c.VkFence) callconv(.c) c.VkResult {
            queues += 1;
            return c.VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    };
    var api: @import("device.zig").DeviceApi = undefined;
    api.vkEndCommandBuffer = Calls.end;
    api.vkBeginCommandBuffer = Calls.begin;
    api.vkQueueSubmit = Calls.queueSubmit;
    var dev: Device = undefined;
    dev.api = api;
    dev.handle = @ptrFromInt(0x2000);
    dev.queue = @ptrFromInt(0x3000);
    var cmds: Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x1000), .fence = @ptrFromInt(0x4000), .recording = true, .pending = 5 };
    std.testing.log_level = .err; // the poison path logs the submit failure, as it is meant to
    defer std.testing.log_level = .warn;
    // runAll catches the escaped error into a FAIL verdict: the pending records must not be reset by a later begin.
    try std.testing.expectError(error.InFlight, cmds.submit(0));
    try std.testing.expectEqual(@as(usize, 1), Calls.queues);
    try std.testing.expectEqual(false, cmds.recording);
    try std.testing.expectError(error.InFlight, cmds.begin());
    try std.testing.expectEqual(@as(usize, 0), Calls.begins);
    try std.testing.expectError(error.InFlight, cmds.submit(0));
    try std.testing.expectEqual(@as(usize, 1), Calls.queues); // the poisoned stream never queues again
}

test "a failed fence wait poisons the stream: a caught Timeout cannot reset a buffer the GPU still owns" {
    const Calls = struct {
        var ends: usize = 0;
        var begins: usize = 0;
        var queues: usize = 0;
        var waits: usize = 0;
        fn end(_: c.VkCommandBuffer) callconv(.c) c.VkResult {
            ends += 1;
            return c.VK_SUCCESS;
        }
        fn begin(_: c.VkCommandBuffer, _: [*c]const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult {
            begins += 1;
            return c.VK_SUCCESS;
        }
        fn queueSubmit(_: c.VkQueue, _: c_uint, _: [*c]const c.VkSubmitInfo, _: c.VkFence) callconv(.c) c.VkResult {
            queues += 1;
            return c.VK_SUCCESS;
        }
        fn wait(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence, _: c.VkBool32, _: c_ulonglong) callconv(.c) c.VkResult {
            waits += 1;
            return c.VK_TIMEOUT;
        }
    };
    var api: @import("device.zig").DeviceApi = undefined;
    api.vkEndCommandBuffer = Calls.end;
    api.vkBeginCommandBuffer = Calls.begin;
    api.vkQueueSubmit = Calls.queueSubmit;
    api.vkWaitForFences = Calls.wait;
    var dev: Device = undefined;
    dev.api = api;
    dev.handle = @ptrFromInt(0x2000);
    dev.queue = @ptrFromInt(0x3000);
    var cmds: Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x1000), .fence = @ptrFromInt(0x4000), .recording = true, .pending = 5 };
    std.testing.log_level = .err; // the poison path logs the wait failure, as it is meant to
    defer std.testing.log_level = .warn;
    // The work may still run: a timed-out fence wait must not let a begin reset the buffer (GDN's caught Timeout).
    try std.testing.expectError(error.Timeout, cmds.submit(0));
    try std.testing.expectEqual(@as(usize, 1), Calls.queues);
    try std.testing.expectEqual(@as(usize, 1), Calls.waits);
    try std.testing.expectEqual(false, cmds.recording);
    try std.testing.expectError(error.Timeout, cmds.begin());
    try std.testing.expectEqual(@as(usize, 0), Calls.begins);
    try std.testing.expectError(error.Timeout, cmds.submit(0));
    try std.testing.expectEqual(@as(usize, 1), Calls.queues);
    try std.testing.expectEqual(@as(usize, 1), Calls.waits);
}

test "a batch begun after a submission starts with one barrier that fences the earlier batch" {
    const Calls = struct {
        var barriers: usize = 0;
        var begins: usize = 0;
        var queues: usize = 0;
        fn begin(_: c.VkCommandBuffer, _: [*c]const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult {
            begins += 1;
            return c.VK_SUCCESS;
        }
        fn reset(_: c.VkCommandBuffer, _: c.VkCommandBufferResetFlags) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn end(_: c.VkCommandBuffer) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn queueSubmit(_: c.VkQueue, _: c_uint, _: [*c]const c.VkSubmitInfo, _: c.VkFence) callconv(.c) c.VkResult {
            queues += 1;
            return c.VK_SUCCESS;
        }
        fn wait(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence, _: c.VkBool32, _: c_ulonglong) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn resetFences(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn barrier(_: c.VkCommandBuffer, _: c.VkPipelineStageFlags, _: c.VkPipelineStageFlags, _: c.VkDependencyFlags, _: u32, _: [*c]const c.VkMemoryBarrier, _: u32, _: [*c]const c.VkBufferMemoryBarrier, _: u32, _: [*c]const c.VkImageMemoryBarrier) callconv(.c) void {
            barriers += 1;
        }
    };
    var api: @import("device.zig").DeviceApi = undefined;
    api.vkResetCommandBuffer = Calls.reset;
    api.vkBeginCommandBuffer = Calls.begin;
    api.vkEndCommandBuffer = Calls.end;
    api.vkQueueSubmit = Calls.queueSubmit;
    api.vkWaitForFences = Calls.wait;
    api.vkResetFences = Calls.resetFences;
    api.vkCmdPipelineBarrier = Calls.barrier;
    var dev: Device = undefined;
    dev.api = api;
    dev.handle = @ptrFromInt(0x2000);
    dev.queue = @ptrFromInt(0x3000);
    var cmds: Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x1000), .fence = @ptrFromInt(0x4000) };
    // The first batch has no earlier submission, so its begin records no barrier.
    _ = try cmds.begin();
    try std.testing.expectEqual(@as(usize, 0), Calls.barriers);
    try cmds.submit(0);
    // The next batch opens with exactly one barrier that fences the submitted batch.
    _ = try cmds.begin();
    try std.testing.expectEqual(@as(usize, 1), Calls.barriers);
    // A begin on an open batch returns it unchanged and records nothing more.
    _ = try cmds.begin();
    try std.testing.expectEqual(@as(usize, 1), Calls.barriers);
    try cmds.submit(0);
    try std.testing.expectEqual(@as(usize, 2), Calls.queues);
    _ = try cmds.begin();
    try std.testing.expectEqual(@as(usize, 2), Calls.barriers);
    try std.testing.expectEqual(@as(usize, 3), Calls.begins);
}

test "a discarded batch keeps the barrier owed to the submission before it" {
    const Calls = struct {
        var barriers: usize = 0;
        fn reset(_: c.VkCommandBuffer, _: c.VkCommandBufferResetFlags) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn begin(_: c.VkCommandBuffer, _: [*c]const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn end(_: c.VkCommandBuffer) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn queueSubmit(_: c.VkQueue, _: c_uint, _: [*c]const c.VkSubmitInfo, _: c.VkFence) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn wait(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence, _: c.VkBool32, _: c_ulonglong) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn resetFences(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn barrier(_: c.VkCommandBuffer, _: c.VkPipelineStageFlags, _: c.VkPipelineStageFlags, _: c.VkDependencyFlags, _: u32, _: [*c]const c.VkMemoryBarrier, _: u32, _: [*c]const c.VkBufferMemoryBarrier, _: u32, _: [*c]const c.VkImageMemoryBarrier) callconv(.c) void {
            barriers += 1;
        }
    };
    var api: @import("device.zig").DeviceApi = undefined;
    api.vkResetCommandBuffer = Calls.reset;
    api.vkBeginCommandBuffer = Calls.begin;
    api.vkEndCommandBuffer = Calls.end;
    api.vkQueueSubmit = Calls.queueSubmit;
    api.vkWaitForFences = Calls.wait;
    api.vkResetFences = Calls.resetFences;
    api.vkCmdPipelineBarrier = Calls.barrier;
    var dev: Device = undefined;
    dev.api = api;
    dev.handle = @ptrFromInt(0x2000);
    dev.queue = @ptrFromInt(0x3000);
    var cmds: Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x1000), .fence = @ptrFromInt(0x4000) };
    _ = try cmds.begin();
    try cmds.submit(0);
    // Batch B opens after A is submitted, then is discarded unsubmitted: its barrier never runs.
    _ = try cmds.begin();
    cmds.discard();
    // Batch C still owes the fence for A, so it records its own barrier.
    _ = try cmds.begin();
    try std.testing.expectEqual(@as(usize, 2), Calls.barriers);
}

test "fillInBatch, updateInBatch and zeroBuffers batch transfer writes safely" {
    const Calls = struct {
        var barriers: usize = 0;
        var dst_stages: [4]u32 = @splat(0);
        var fills: usize = 0;
        var updates: usize = 0;
        var submits: usize = 0;
        fn okCmd(_: c.VkCommandBuffer) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okReset(_: c.VkCommandBuffer, _: c.VkCommandBufferResetFlags) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okBegin(_: c.VkCommandBuffer, _: [*c]const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okWait(_: c.VkDevice, _: u32, _: [*c]const c.VkFence, _: c.VkBool32, _: u64) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okFence(_: c.VkDevice, _: u32, _: [*c]const c.VkFence) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn pb(_: c.VkCommandBuffer, _: c.VkPipelineStageFlags, dst: c.VkPipelineStageFlags, _: c.VkDependencyFlags, _: u32, _: [*c]const c.VkMemoryBarrier, _: u32, _: [*c]const c.VkBufferMemoryBarrier, _: u32, _: [*c]const c.VkImageMemoryBarrier) callconv(.c) void {
            if (barriers < dst_stages.len) dst_stages[barriers] = dst;
            barriers += 1;
        }
        fn fillBuf(_: c.VkCommandBuffer, _: c.VkBuffer, _: c.VkDeviceSize, _: c.VkDeviceSize, _: u32) callconv(.c) void {
            fills += 1;
        }
        fn updateBuf(_: c.VkCommandBuffer, _: c.VkBuffer, _: c.VkDeviceSize, _: c.VkDeviceSize, _: ?*const anyopaque) callconv(.c) void {
            updates += 1;
        }
        fn submit(_: c.VkQueue, _: u32, _: [*c]const c.VkSubmitInfo, _: c.VkFence) callconv(.c) c.VkResult {
            submits += 1;
            return c.VK_SUCCESS;
        }
    };
    var api: @import("device.zig").DeviceApi = undefined;
    api.vkResetCommandBuffer = Calls.okReset;
    api.vkBeginCommandBuffer = Calls.okBegin;
    api.vkEndCommandBuffer = Calls.okCmd;
    api.vkCmdPipelineBarrier = Calls.pb;
    api.vkCmdFillBuffer = Calls.fillBuf;
    api.vkCmdUpdateBuffer = Calls.updateBuf;
    api.vkQueueSubmit = Calls.submit;
    api.vkWaitForFences = Calls.okWait;
    api.vkResetFences = Calls.okFence;
    var dev: Device = undefined;
    dev.api = api;
    dev.handle = @ptrFromInt(0x2000);
    dev.queue = @ptrFromInt(0x3000);
    var cmds: Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x1000), .fence = @ptrFromInt(0x4000) };
    var a: Buffer = .{ .dev = &dev, .handle = @ptrFromInt(0x10), .memory = null, .len = 256, .kind = .device, .mapped = null, .coherent = false };
    var b: Buffer = .{ .dev = &dev, .handle = @ptrFromInt(0x20), .memory = null, .len = 128, .kind = .device, .mapped = null, .coherent = false };
    var bad: Buffer = .{ .dev = &dev, .handle = @ptrFromInt(0x30), .memory = null, .len = 6, .kind = .device, .mapped = null, .coherent = false };
    _ = try cmds.begin();
    cmds.hazards.add(&.{}, &.{.{ .buffer = 0x20, .start = 0, .end = 64 }});
    try commands.fillInBatch(&cmds, a, 0, 4, 7);
    try std.testing.expectEqual(@as(usize, 1), Calls.fills);
    try std.testing.expectEqual(@as(usize, 1), Calls.barriers);
    try std.testing.expect(!cmds.hazards.empty());
    cmds.hazards.add(&.{.{ .buffer = 0x10, .start = 0, .end = 8 }}, &.{});
    const bytes = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try commands.updateInBatch(&cmds, a, 0, &bytes);
    try std.testing.expectEqual(@as(usize, 1), Calls.updates);
    try std.testing.expectEqual(@as(usize, 3), Calls.barriers);
    try std.testing.expectEqual(@as(u32, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT), Calls.dst_stages[1]);
    try std.testing.expect(cmds.hazards.empty());
    try std.testing.expectEqual(@as(usize, 0), Calls.submits);
    try std.testing.expectError(error.Invalid, commands.updateInBatch(&cmds, a, 0, @as([*]const u8, @ptrFromInt(0x4000))[0 .. commands.max_inline_update + 4]));
    try std.testing.expectError(error.Misaligned, commands.zeroBuffers(&cmds, &.{ &a, &bad }, 0));
    try std.testing.expectEqual(@as(usize, 1), Calls.fills);
    try commands.zeroBuffers(&cmds, &.{ &a, &b }, 0);
    try std.testing.expectEqual(@as(usize, 3), Calls.fills);
    try std.testing.expectEqual(@as(usize, 1), Calls.submits);
}
