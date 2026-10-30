//! Owned Vulkan buffers, one dedicated allocation each, and the checked geometry of their ranges and bindings.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Device = @import("device.zig").Device;
const Error = runtime.Error;
const check = runtime.check;

/// device: DEVICE_LOCAL, for kernels; staging: HOST_VISIBLE and mapped for its life, for uploads and readbacks.
pub const Kind = enum { device, staging };

const usage: c.VkBufferUsageFlags = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
const device_local: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
const host_visible: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
const host_coherent: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
const host_cached: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
/// VK_WHOLE_SIZE.
const whole_size: c.VkDeviceSize = std.math.maxInt(u64);

pub const Buffer = struct {
    dev: *const Device,
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,
    /// The VkBuffer's size in bytes.
    len: u64,
    kind: Kind,
    /// The whole allocation's host view, for staging buffers only.
    mapped: ?[*]u8,
    coherent: bool,

    /// A nonzero buffer within maxBufferSize and maxMemoryAllocationSize, bound at offset 0 of its own allocation.
    pub fn init(dev: *const Device, kind: Kind, len: u64) Error!Buffer {
        try fits(len, dev.v13.maxBufferSize, dev.v11.maxMemoryAllocationSize);
        const info: c.VkBufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = len,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        var handle: c.VkBuffer = null;
        try check(dev.api.vkCreateBuffer(dev.handle, &info, null, &handle));
        errdefer dev.api.vkDestroyBuffer(dev.handle, handle, null);
        var reqs: c.VkMemoryRequirements = undefined;
        dev.api.vkGetBufferMemoryRequirements(dev.handle, handle, &reqs);
        if (reqs.size > dev.v11.maxMemoryAllocationSize) return error.TooLarge;
        const required = if (kind == .device) device_local else host_visible;
        const preferred = if (kind == .device) 0 else host_coherent | host_cached;
        // the spec guarantees every buffer a DEVICE_LOCAL and a HOST_VISIBLE | HOST_COHERENT type
        const index = memoryType(&dev.memory, reqs.memoryTypeBits, required, preferred) orelse return error.VulkanFailed;
        const alloc: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = reqs.size,
            .memoryTypeIndex = index,
        };
        var memory: c.VkDeviceMemory = null;
        try check(dev.api.vkAllocateMemory(dev.handle, &alloc, null, &memory));
        errdefer dev.api.vkFreeMemory(dev.handle, memory, null);
        try check(dev.api.vkBindBufferMemory(dev.handle, handle, memory, 0));
        var mapped: ?[*]u8 = null;
        if (kind == .staging) {
            var p: ?*anyopaque = null;
            try check(dev.api.vkMapMemory(dev.handle, memory, 0, whole_size, 0, &p));
            mapped = @as([*]u8, @ptrCast(p orelse return error.VulkanFailed));
        }
        const coherent = dev.memory.memoryTypes[index].propertyFlags & host_coherent != 0;
        return .{ .dev = dev, .handle = handle, .memory = memory, .len = len, .kind = kind, .mapped = mapped, .coherent = coherent };
    }

    /// Frees the buffer; call only when no submission using it can still be pending.
    pub fn deinit(self: *Buffer) void {
        const api = &self.dev.api;
        if (self.mapped != null) api.vkUnmapMemory(self.dev.handle, self.memory);
        api.vkDestroyBuffer(self.dev.handle, self.handle, null);
        api.vkFreeMemory(self.dev.handle, self.memory, null);
        self.* = undefined;
    }

    /// Staging bytes only; refuses device-local buffers and lengths beyond the host's address space.
    pub fn bytes(self: Buffer) Error![]u8 {
        const m = self.mapped orelse return error.NotMapped;
        return m[0 .. std.math.cast(usize, self.len) orelse return error.TooLarge];
    }

    /// Makes host writes visible to the device; a coherent staging buffer needs nothing.
    pub fn flush(self: Buffer) Error!void {
        if (self.mapped == null) return error.NotMapped;
        if (self.coherent) return;
        const range = self.wholeRange();
        try check(self.dev.api.vkFlushMappedMemoryRanges(self.dev.handle, 1, &range));
    }

    /// Makes device writes visible to the host after their fence; a coherent staging buffer needs nothing.
    pub fn invalidate(self: Buffer) Error!void {
        if (self.mapped == null) return error.NotMapped;
        if (self.coherent) return;
        const range = self.wholeRange();
        try check(self.dev.api.vkInvalidateMappedMemoryRanges(self.dev.handle, 1, &range));
    }

    /// The whole mapping, whose end is the allocation's end, so it meets nonCoherentAtomSize as the spec requires.
    fn wholeRange(self: Buffer) c.VkMappedMemoryRange {
        return .{ .sType = c.VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = self.memory, .offset = 0, .size = whole_size };
    }

    /// A storage-buffer descriptor for `len` bytes at `offset`, checked against the buffer and the device's limits.
    pub fn descriptor(self: Buffer, offset: u64, len: u64) Error!c.VkDescriptorBufferInfo {
        const limits = self.dev.properties.limits;
        try binding(self.len, offset, len, limits.minStorageBufferOffsetAlignment, limits.maxStorageBufferRange);
        return .{ .buffer = self.handle, .offset = offset, .range = len };
    }
};

/// `n` bytes at `offset` lie inside `total`, without overflow.
pub fn span(total: u64, offset: u64, n: u64) Error!void {
    if (offset > total or n > total - offset) return error.OutOfRange;
}

/// A storage binding: nonempty, inside the buffer, its offset aligned and its range within the device's maximum.
pub fn binding(total: u64, offset: u64, n: u64, alignment: u64, max_range: u32) Error!void {
    if (n == 0) return error.OutOfRange;
    try span(total, offset, n);
    if (alignment == 0 or offset % alignment != 0) return error.Misaligned;
    if (n > max_range) return error.TooLarge;
}

/// A buffer size the device admits: nonzero, within maxBufferSize and maxMemoryAllocationSize.
pub fn fits(len: u64, max_buffer: u64, max_allocation: u64) Error!void {
    if (len == 0) return error.OutOfRange;
    if (len > max_buffer or len > max_allocation) return error.TooLarge;
}

/// The first type in `bits` with every `required` flag, preferring one that also has every `preferred` flag.
pub fn memoryType(props: *const c.VkPhysicalDeviceMemoryProperties, bits: u32, required: u32, preferred: u32) ?u32 {
    var fallback: ?u32 = null;
    const count = @min(props.memoryTypeCount, props.memoryTypes.len);
    for (props.memoryTypes[0..count], 0..) |t, i| {
        if (bits >> @intCast(i) & 1 == 0 or t.propertyFlags & required != required) continue;
        if (t.propertyFlags & preferred == preferred) return @intCast(i);
        if (fallback == null) fallback = @intCast(i);
    }
    return fallback;
}

test "ranges stay inside their buffer without overflow" {
    try span(64, 0, 64);
    try span(64, 64, 0);
    try span(64, 16, 48);
    try std.testing.expectError(error.OutOfRange, span(64, 16, 49));
    try std.testing.expectError(error.OutOfRange, span(64, 65, 0));
    try std.testing.expectError(error.OutOfRange, span(64, 8, std.math.maxInt(u64)));
    try std.testing.expectError(error.OutOfRange, span(std.math.maxInt(u64), std.math.maxInt(u64), 1));
}

test "storage bindings are nonempty, aligned and within the range limit" {
    try binding(4096, 256, 1024, 256, 1 << 27);
    try binding(4096, 0, 4096, 16, 4096);
    try std.testing.expectError(error.OutOfRange, binding(4096, 0, 0, 16, 4096));
    try std.testing.expectError(error.OutOfRange, binding(4096, 256, 3841, 256, 1 << 27));
    try std.testing.expectError(error.Misaligned, binding(4096, 64, 64, 256, 1 << 27));
    try std.testing.expectError(error.Misaligned, binding(4096, 0, 64, 0, 1 << 27));
    try std.testing.expectError(error.TooLarge, binding(1 << 30, 0, (1 << 27) + 4, 16, 1 << 27));
}

test "buffer sizes are nonzero and within both device maxima" {
    try fits(1, 1 << 30, 1 << 31);
    try fits(1 << 30, 1 << 30, 1 << 31);
    try std.testing.expectError(error.OutOfRange, fits(0, 1 << 30, 1 << 31));
    try std.testing.expectError(error.TooLarge, fits((1 << 30) + 1, 1 << 30, 1 << 31));
    try std.testing.expectError(error.TooLarge, fits((1 << 31) + 1, 1 << 32, 1 << 31));
}

test "memory types honor the buffer's bits, required flags and preferred flags" {
    var props = std.mem.zeroes(c.VkPhysicalDeviceMemoryProperties);
    props.memoryTypeCount = 3;
    props.memoryTypes[0].propertyFlags = host_visible | host_coherent;
    props.memoryTypes[1].propertyFlags = device_local;
    props.memoryTypes[2].propertyFlags = host_visible | host_coherent | host_cached;
    try std.testing.expectEqual(@as(?u32, 1), memoryType(&props, 0b111, device_local, 0));
    try std.testing.expectEqual(@as(?u32, 2), memoryType(&props, 0b111, host_visible, host_coherent | host_cached));
    try std.testing.expectEqual(@as(?u32, 0), memoryType(&props, 0b011, host_visible, host_coherent | host_cached));
    try std.testing.expectEqual(@as(?u32, null), memoryType(&props, 0b010, host_visible, 0));
    try std.testing.expectEqual(@as(?u32, null), memoryType(&props, 0b101, device_local, 0));
    props.memoryTypeCount = 2;
    try std.testing.expectEqual(@as(?u32, 0), memoryType(&props, 0b111, host_visible, host_coherent | host_cached));
    props.memoryTypeCount = 99;
    try std.testing.expectEqual(@as(?u32, 2), memoryType(&props, 0b111, host_visible, host_coherent | host_cached));
}
