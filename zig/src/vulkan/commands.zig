//! One command buffer and fence on the device's queue, and staged uploads and readbacks through it.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");
const Device = @import("device.zig").Device;
const Buffer = memory.Buffer;
const Error = runtime.Error;
const check = runtime.check;

pub const Commands = struct {
    dev: *const Device,
    pool: c.VkCommandPool,
    cmd: c.VkCommandBuffer,
    fence: c.VkFence,

    pub fn init(dev: *const Device) Error!Commands {
        const api = &dev.api;
        const pool_info: c.VkCommandPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = dev.queue_family,
        };
        var pool: c.VkCommandPool = null;
        try check(api.vkCreateCommandPool(dev.handle, &pool_info, null, &pool));
        errdefer api.vkDestroyCommandPool(dev.handle, pool, null);
        const alloc: c.VkCommandBufferAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd: c.VkCommandBuffer = null;
        try check(api.vkAllocateCommandBuffers(dev.handle, &alloc, &cmd));
        const fence_info: c.VkFenceCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        var fence: c.VkFence = null;
        try check(api.vkCreateFence(dev.handle, &fence_info, null, &fence));
        return .{ .dev = dev, .pool = pool, .cmd = cmd, .fence = fence };
    }

    /// Frees the pool, its command buffer and the fence; call only after the last submit returned success.
    pub fn deinit(self: *Commands) void {
        self.dev.api.vkDestroyFence(self.dev.handle, self.fence, null);
        self.dev.api.vkDestroyCommandPool(self.dev.handle, self.pool, null);
        self.* = undefined;
    }

    /// The command buffer, reset and begun for one submission.
    pub fn begin(self: *Commands) Error!c.VkCommandBuffer {
        try check(self.dev.api.vkResetCommandBuffer(self.cmd, 0));
        const info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        try check(self.dev.api.vkBeginCommandBuffer(self.cmd, &info));
        return self.cmd;
    }

    /// Finite fence wait; Timeout/InFlight require orExit to avoid freeing resources the GPU may still use.
    pub fn submit(self: *Commands, timeout_ns: u64) Error!void {
        const api = &self.dev.api;
        try check(api.vkEndCommandBuffer(self.cmd));
        const info: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.cmd,
        };
        const queued = api.vkQueueSubmit(self.dev.queue, 1, &info, self.fence);
        submitted(queued) catch |e| {
            std.log.err("vulkan: vkQueueSubmit returned {d}", .{queued});
            return e;
        };
        const waited_on = api.vkWaitForFences(self.dev.handle, 1, &self.fence, 1, timeout_ns);
        waited(waited_on) catch |e| {
            std.log.err("vulkan: vkWaitForFences returned {d} within {d} ns of a submission", .{ waited_on, timeout_ns });
            return e;
        };
        try check(api.vkResetFences(self.dev.handle, 1, &self.fence));
    }
};

/// Any failed submit may leave work in flight; terminate conservatively instead of freeing its resources.
fn submitted(result: c.VkResult) Error!void {
    return switch (result) {
        c.VK_SUCCESS => {},
        else => error.InFlight,
    };
}

/// After a submit, only a signalled fence proves the GPU is done with the submission's resources.
fn waited(result: c.VkResult) Error!void {
    return switch (result) {
        c.VK_SUCCESS => {},
        c.VK_TIMEOUT => error.Timeout,
        else => error.InFlight,
    };
}

/// Work of unknown completion leaves the GPU owning its resources: exit now (124 timeout, 1 otherwise), free nothing.
pub fn orExit(result: anytype) @TypeOf(result) {
    _ = result catch |e| {
        if (e == error.Timeout) std.process.exit(124);
        if (e == error.InFlight) std.process.exit(1);
        return e;
    };
    return result;
}

/// One global memory dependency: `src_access` by `src_stage` made visible to `dst_access` by `dst_stage`.
pub fn barrier(dev: *const Device, cmd: c.VkCommandBuffer, src_stage: u32, src_access: u32, dst_stage: u32, dst_access: u32) void {
    const b: c.VkMemoryBarrier = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
    };
    dev.api.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &b, 0, null, 0, null);
}

const gpu_stages: u32 = c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT;
const gpu_writes: u32 = c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;
const gpu_access: u32 = gpu_writes | c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_TRANSFER_READ_BIT;
const transfer: u32 = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
const transfer_write: u32 = c.VK_ACCESS_TRANSFER_WRITE_BIT;
const transfer_read: u32 = c.VK_ACCESS_TRANSFER_READ_BIT;

/// Both on `dev`, `buf` a device buffer, `staging` a separate staging buffer, and the range inside `buf`.
pub fn operands(dev: *const Device, buf: Buffer, staging: Buffer, offset: u64, n: u64) Error!void {
    if (buf.dev != dev or staging.dev != dev or buf.kind != .device or staging.kind != .staging) return error.Invalid;
    if (buf.handle == staging.handle or staging.mapped == null or staging.len == 0) return error.Invalid;
    try memory.span(buf.len, offset, n);
}

/// `bytes` into `dst` at `offset` through `staging`, a chunk a submission; later dispatches and copies see them.
pub fn upload(cmds: *Commands, dst: Buffer, offset: u64, staging: Buffer, bytes: []const u8, timeout_ns: u64) Error!void {
    try operands(cmds.dev, dst, staging, offset, bytes.len);
    const host = try staging.bytes();
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(host.len, bytes.len - done);
        @memcpy(host[0..n], bytes[done..][0..n]);
        try staging.flush();
        const cmd = try cmds.begin();
        barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_write);
        const region: c.VkBufferCopy = .{ .srcOffset = 0, .dstOffset = offset + @as(u64, done), .size = n };
        cmds.dev.api.vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
        barrier(cmds.dev, cmd, transfer, transfer_write, gpu_stages, gpu_access);
        try cmds.submit(timeout_ns);
        done += n;
    }
}

/// `out.len` bytes of `src` from `offset` through `staging`, after every earlier dispatch or copy that wrote them.
pub fn download(cmds: *Commands, src: Buffer, offset: u64, staging: Buffer, out: []u8, timeout_ns: u64) Error!void {
    try operands(cmds.dev, src, staging, offset, out.len);
    const host = try staging.bytes();
    var done: usize = 0;
    while (done < out.len) {
        const n = @min(host.len, out.len - done);
        const cmd = try cmds.begin();
        barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_read | transfer_write);
        const region: c.VkBufferCopy = .{ .srcOffset = offset + @as(u64, done), .dstOffset = 0, .size = n };
        cmds.dev.api.vkCmdCopyBuffer(cmd, src.handle, staging.handle, 1, &region);
        barrier(cmds.dev, cmd, transfer, transfer_write, c.VK_PIPELINE_STAGE_HOST_BIT, c.VK_ACCESS_HOST_READ_BIT);
        try cmds.submit(timeout_ns);
        try staging.invalidate();
        @memcpy(out[done..][0..n], host[0..n]);
        done += n;
    }
}

test "only a successful submit and a signalled fence let the caller free a submission's resources" {
    try submitted(c.VK_SUCCESS);
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_OUT_OF_HOST_MEMORY));
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_OUT_OF_DEVICE_MEMORY));
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_DEVICE_LOST));
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_UNKNOWN));
    try waited(c.VK_SUCCESS);
    try std.testing.expectError(error.Timeout, waited(c.VK_TIMEOUT));
    try std.testing.expectError(error.InFlight, waited(c.VK_ERROR_OUT_OF_HOST_MEMORY));
    try std.testing.expectError(error.InFlight, waited(c.VK_ERROR_DEVICE_LOST));
    try std.testing.expectError(error.InFlight, waited(c.VK_NOT_READY));
}

test "staged transfers refuse foreign devices, wrong kinds, aliasing and ranges past the buffer" {
    // operand identities only: nothing here is dereferenced or reaches Vulkan
    const dev: *const Device = @ptrFromInt(0x1000);
    const other: *const Device = @ptrFromInt(0x2000);
    const host: [*]u8 = @ptrFromInt(0x3000);
    const buf: Buffer = .{ .dev = dev, .handle = @ptrFromInt(0x10), .memory = @ptrFromInt(0x20), .len = 4096, .kind = .device, .mapped = null, .coherent = false };
    const staging: Buffer = .{ .dev = dev, .handle = @ptrFromInt(0x30), .memory = @ptrFromInt(0x40), .len = 1027, .kind = .staging, .mapped = host, .coherent = true };
    try operands(dev, buf, staging, 0, 4096);
    try operands(dev, buf, staging, 13, 4083);
    try std.testing.expectError(error.OutOfRange, operands(dev, buf, staging, 13, 4084));
    try std.testing.expectError(error.OutOfRange, operands(dev, buf, staging, 4097, 0));
    try std.testing.expectError(error.Invalid, operands(other, buf, staging, 0, 4));
    var foreign = staging;
    foreign.dev = other;
    try std.testing.expectError(error.Invalid, operands(dev, buf, foreign, 0, 4));
    try std.testing.expectError(error.Invalid, operands(dev, staging, staging, 0, 4));
    try std.testing.expectError(error.Invalid, operands(dev, buf, buf, 0, 4));
    var same = staging;
    same.handle = buf.handle;
    try std.testing.expectError(error.Invalid, operands(dev, buf, same, 0, 4));
    var unmapped = staging;
    unmapped.mapped = null;
    try std.testing.expectError(error.Invalid, operands(dev, buf, unmapped, 0, 4));
    var empty = staging;
    empty.len = 0;
    try std.testing.expectError(error.Invalid, operands(dev, buf, empty, 0, 4));
}
