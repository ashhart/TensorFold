//! The one GPU a Vulkan run uses: chosen fail-closed, created with a compute queue and exactly the features asked for.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Runtime = runtime.Runtime;
const Error = runtime.Error;
const Fn = runtime.Fn;
const check = runtime.check;

pub const DeviceApi = struct {
    vkDestroyDevice: Fn(c.PFN_vkDestroyDevice),
    vkGetDeviceQueue: Fn(c.PFN_vkGetDeviceQueue),
    vkQueueSubmit: Fn(c.PFN_vkQueueSubmit),
    vkCreateBuffer: Fn(c.PFN_vkCreateBuffer),
    vkDestroyBuffer: Fn(c.PFN_vkDestroyBuffer),
    vkGetBufferMemoryRequirements: Fn(c.PFN_vkGetBufferMemoryRequirements),
    vkAllocateMemory: Fn(c.PFN_vkAllocateMemory),
    vkFreeMemory: Fn(c.PFN_vkFreeMemory),
    vkBindBufferMemory: Fn(c.PFN_vkBindBufferMemory),
    vkMapMemory: Fn(c.PFN_vkMapMemory),
    vkUnmapMemory: Fn(c.PFN_vkUnmapMemory),
    vkFlushMappedMemoryRanges: Fn(c.PFN_vkFlushMappedMemoryRanges),
    vkInvalidateMappedMemoryRanges: Fn(c.PFN_vkInvalidateMappedMemoryRanges),
    vkCreateCommandPool: Fn(c.PFN_vkCreateCommandPool),
    vkDestroyCommandPool: Fn(c.PFN_vkDestroyCommandPool),
    vkAllocateCommandBuffers: Fn(c.PFN_vkAllocateCommandBuffers),
    vkResetCommandBuffer: Fn(c.PFN_vkResetCommandBuffer),
    vkBeginCommandBuffer: Fn(c.PFN_vkBeginCommandBuffer),
    vkEndCommandBuffer: Fn(c.PFN_vkEndCommandBuffer),
    vkCreateFence: Fn(c.PFN_vkCreateFence),
    vkDestroyFence: Fn(c.PFN_vkDestroyFence),
    vkWaitForFences: Fn(c.PFN_vkWaitForFences),
    vkResetFences: Fn(c.PFN_vkResetFences),
    vkCmdPipelineBarrier: Fn(c.PFN_vkCmdPipelineBarrier),
    vkCmdCopyBuffer: Fn(c.PFN_vkCmdCopyBuffer),
    vkCmdBindPipeline: Fn(c.PFN_vkCmdBindPipeline),
    vkCmdBindDescriptorSets: Fn(c.PFN_vkCmdBindDescriptorSets),
    vkCmdPushConstants: Fn(c.PFN_vkCmdPushConstants),
    vkCmdDispatch: Fn(c.PFN_vkCmdDispatch),
    vkCreateShaderModule: Fn(c.PFN_vkCreateShaderModule),
    vkDestroyShaderModule: Fn(c.PFN_vkDestroyShaderModule),
    vkCreateDescriptorSetLayout: Fn(c.PFN_vkCreateDescriptorSetLayout),
    vkDestroyDescriptorSetLayout: Fn(c.PFN_vkDestroyDescriptorSetLayout),
    vkCreatePipelineLayout: Fn(c.PFN_vkCreatePipelineLayout),
    vkDestroyPipelineLayout: Fn(c.PFN_vkDestroyPipelineLayout),
    vkCreateComputePipelines: Fn(c.PFN_vkCreateComputePipelines),
    vkDestroyPipeline: Fn(c.PFN_vkDestroyPipeline),
    vkCreateDescriptorPool: Fn(c.PFN_vkCreateDescriptorPool),
    vkDestroyDescriptorPool: Fn(c.PFN_vkDestroyDescriptorPool),
    vkAllocateDescriptorSets: Fn(c.PFN_vkAllocateDescriptorSets),
    vkUpdateDescriptorSets: Fn(c.PFN_vkUpdateDescriptorSets),
};

/// Optional shader features a caller's kernels use; each is one Vulkan feature bit in `feature_bits`.
pub const Features = struct {
    float64: bool = false,
    int64: bool = false,
    int16: bool = false,
    float16: bool = false,
    int8: bool = false,
    storage16: bool = false,
    storage8: bool = false,
    subgroup_size_control: bool = false,
};

/// Each Features field, the struct of the feature chain holding its bit and the bit's Vulkan name.
const feature_bits = .{
    .{ "float64", "core", "shaderFloat64" },
    .{ "int64", "core", "shaderInt64" },
    .{ "int16", "core", "shaderInt16" },
    .{ "float16", "v12", "shaderFloat16" },
    .{ "int8", "v12", "shaderInt8" },
    .{ "storage16", "v11", "storageBuffer16BitAccess" },
    .{ "storage8", "v12", "storageBuffer8BitAccess" },
    .{ "subgroup_size_control", "v13", "subgroupSizeControl" },
};

comptime {
    std.debug.assert(feature_bits.len == @typeInfo(Features).@"struct".field_names.len);
}

/// The 1.0 to 1.3 feature structs, linked; it must not move once linked.
const Chain = struct {
    f2: c.VkPhysicalDeviceFeatures2,
    v11: c.VkPhysicalDeviceVulkan11Features,
    v12: c.VkPhysicalDeviceVulkan12Features,
    v13: c.VkPhysicalDeviceVulkan13Features,

    fn link(self: *Chain) void {
        self.* = .{
            .f2 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &self.v11 },
            .v11 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &self.v12 },
            .v12 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &self.v13 },
            .v13 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES },
        };
    }

    fn bit(self: *Chain, comptime where: []const u8, comptime name: []const u8) *c.VkBool32 {
        if (comptime std.mem.eql(u8, where, "core")) return &@field(self.f2.features, name);
        return &@field(@field(self, where), name);
    }

    fn available(self: *Chain) Features {
        var have: Features = .{};
        inline for (feature_bits) |f| @field(have, f[0]) = self.bit(f[1], f[2]).* != 0;
        return have;
    }

    /// Every bit off except those `want` asks for.
    fn enable(self: *Chain, want: Features) void {
        self.link();
        inline for (feature_bits) |f| self.bit(f[1], f[2]).* = @intFromBool(@field(want, f[0]));
    }
};

/// The first feature `want` asks for that `have` lacks.
pub fn missing(want: Features, have: Features) ?[]const u8 {
    inline for (@typeInfo(Features).@"struct".field_names) |name| {
        if (@field(want, name) and !@field(have, name)) return name;
    }
    return null;
}

/// A compute family without graphics when there is one (a dedicated compute queue), else the first with compute.
pub fn computeFamily(families: []const c.VkQueueFamilyProperties) ?u32 {
    var first: ?u32 = null;
    for (families, 0..) |f, i| {
        if (f.queueCount == 0 or f.queueFlags & c.VK_QUEUE_COMPUTE_BIT == 0) continue;
        if (f.queueFlags & c.VK_QUEUE_GRAPHICS_BIT == 0) return @intCast(i);
        if (first == null) first = @intCast(i);
    }
    return first;
}

/// Why a device cannot serve: below Vulkan 1.3, a CPU implementation, no compute queue, or a requested feature absent.
pub fn refusal(api_version: u32, kind: c.VkPhysicalDeviceType, family: ?u32, lacking: ?[]const u8) ?Error {
    if (!runtime.atLeast13(api_version)) return error.VulkanVersion;
    if (kind == c.VK_PHYSICAL_DEVICE_TYPE_CPU) return error.SoftwareDevice;
    if (family == null) return error.NoComputeQueue;
    if (lacking != null) return error.MissingFeature;
    return null;
}

/// One enumerated device as selection sees it.
pub const Candidate = struct { kind: c.VkPhysicalDeviceType, family: u32, refusal: ?Error, lacking: ?[]const u8 = null };

/// `index` when that device is eligible, else the best eligible kind: discrete, integrated, virtual, then any other.
pub fn choose(candidates: []const Candidate, index: ?u32) Error!usize {
    if (index) |i| {
        if (i >= candidates.len) return error.NoDevice;
        if (candidates[i].refusal) |e| return e;
        return i;
    }
    var best: ?usize = null;
    for (candidates, 0..) |cand, i| {
        if (cand.refusal != null) continue;
        if (best == null or rank(cand.kind) < rank(candidates[best.?].kind)) best = i;
    }
    return best orelse error.NoDevice;
}

fn rank(kind: c.VkPhysicalDeviceType) u8 {
    return switch (kind) {
        c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => 0,
        c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => 1,
        c.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => 2,
        else => 3,
    };
}

/// A device's 1.0 properties, queue families and, on a 1.3 device, its features, judged against `want`.
fn candidate(rt: *const Runtime, pd: c.VkPhysicalDevice, want: Features, props: *c.VkPhysicalDeviceProperties) Candidate {
    var p2: c.VkPhysicalDeviceProperties2 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    rt.api.vkGetPhysicalDeviceProperties2(pd, &p2);
    props.* = p2.properties;
    // shortcut: families past the first 32 are not considered; read the count first if a driver ever lists more
    var families: [32]c.VkQueueFamilyProperties = undefined;
    var n: u32 = families.len;
    rt.api.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, &families);
    const family = computeFamily(families[0..@min(n, families.len)]);
    var lacking: ?[]const u8 = null;
    // the 1.1 to 1.3 feature structs are valid in the chain only on a 1.3 device
    if (runtime.atLeast13(props.apiVersion)) {
        var chain: Chain = undefined;
        chain.link();
        rt.api.vkGetPhysicalDeviceFeatures2(pd, &chain.f2);
        lacking = missing(want, chain.available());
    }
    return .{ .kind = props.deviceType, .family = family orelse 0, .refusal = refusal(props.apiVersion, props.deviceType, family, lacking), .lacking = lacking };
}

pub const Options = struct { index: ?u32 = null, features: Features = .{} };

pub const Device = struct {
    rt: *const Runtime,
    physical: c.VkPhysicalDevice,
    handle: c.VkDevice,
    api: DeviceApi,
    queue: c.VkQueue,
    queue_family: u32,
    properties: c.VkPhysicalDeviceProperties,
    /// The 1.1 to 1.3 properties, with their pNext cleared.
    v11: c.VkPhysicalDeviceVulkan11Properties,
    v12: c.VkPhysicalDeviceVulkan12Properties,
    v13: c.VkPhysicalDeviceVulkan13Properties,
    memory: c.VkPhysicalDeviceMemoryProperties,
    /// Exactly the optional features enabled on the device.
    features: Features,

    /// The chosen GPU with one compute queue and `options.features` enabled; any refusal is logged per device.
    pub fn open(rt: *const Runtime, options: Options) Error!Device {
        // shortcut: at most 16 devices; more is refused, not truncated; read the count first if a host ever lists more
        var handles: [16]c.VkPhysicalDevice = undefined;
        var n: u32 = handles.len;
        const listed = rt.api.vkEnumeratePhysicalDevices(rt.instance, &n, &handles);
        if (listed == c.VK_INCOMPLETE) {
            std.log.err("vulkan: more than {d} devices listed; refusing to choose among a truncated list", .{handles.len});
            return error.TooLarge;
        }
        try check(listed);
        n = @min(n, handles.len);
        var props: [16]c.VkPhysicalDeviceProperties = undefined;
        var cands: [16]Candidate = undefined;
        for (handles[0..n], cands[0..n], props[0..n]) |pd, *cand, *p| cand.* = candidate(rt, pd, options.features, p);
        const i = choose(cands[0..n], options.index) catch |e| {
            std.log.err("vulkan: no device chosen ({s}) among {d}", .{ @errorName(e), n });
            for (cands[0..n], props[0..n], 0..) |cand, p, k| {
                const why = if (cand.refusal) |r| @errorName(r) else "eligible";
                std.log.err("vulkan device {d} {s}: {s} {s}", .{ k, std.mem.sliceTo(&p.deviceName, 0), why, cand.lacking orelse "" });
            }
            return e;
        };
        var dev: Device = .{
            .rt = rt,
            .physical = handles[i],
            .handle = null,
            .api = undefined,
            .queue = null,
            .queue_family = cands[i].family,
            .properties = undefined,
            .v11 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES },
            .v12 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES },
            .v13 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES },
            .memory = undefined,
            .features = options.features,
        };
        var p2: c.VkPhysicalDeviceProperties2 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dev.v11 };
        dev.v11.pNext = &dev.v12;
        dev.v12.pNext = &dev.v13;
        rt.api.vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.properties = p2.properties;
        dev.v11.pNext = null;
        dev.v12.pNext = null;
        rt.api.vkGetPhysicalDeviceMemoryProperties(dev.physical, &dev.memory);
        var chain: Chain = undefined;
        chain.enable(options.features);
        const priority: f32 = 1;
        const queue_info: c.VkDeviceQueueCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = dev.queue_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        const info: c.VkDeviceCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &chain.f2,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_info,
        };
        try check(rt.api.vkCreateDevice(dev.physical, &info, null, &dev.handle));
        if (dev.handle == null) return error.VulkanFailed;
        dev.api = runtime.resolve(DeviceApi, dev.handle, rt.api.vkGetDeviceProcAddr) catch |e| {
            if (rt.api.vkGetDeviceProcAddr(dev.handle, "vkDestroyDevice")) |p| @as(Fn(c.PFN_vkDestroyDevice), @ptrCast(p))(dev.handle, null);
            return e;
        };
        errdefer dev.api.vkDestroyDevice(dev.handle, null);
        dev.api.vkGetDeviceQueue(dev.handle, dev.queue_family, 0, &dev.queue);
        if (dev.queue == null) return error.VulkanFailed;
        return dev;
    }

    /// Destroys the device; call only when every fence it submitted has signalled and its objects are freed.
    pub fn close(self: *Device) void {
        self.api.vkDestroyDevice(self.handle, null);
        self.* = undefined;
    }
};

fn queues(flags: c.VkQueueFlags, count: u32) c.VkQueueFamilyProperties {
    return .{ .queueFlags = flags, .queueCount = count };
}

test "the compute queue prefers a dedicated family and refuses none" {
    const both = c.VK_QUEUE_GRAPHICS_BIT | c.VK_QUEUE_COMPUTE_BIT;
    try std.testing.expectEqual(@as(?u32, 1), computeFamily(&.{ queues(both, 1), queues(c.VK_QUEUE_COMPUTE_BIT, 2) }));
    try std.testing.expectEqual(@as(?u32, 0), computeFamily(&.{ queues(both, 1), queues(c.VK_QUEUE_TRANSFER_BIT, 1) }));
    try std.testing.expectEqual(@as(?u32, null), computeFamily(&.{ queues(c.VK_QUEUE_GRAPHICS_BIT, 1), queues(c.VK_QUEUE_TRANSFER_BIT, 1) }));
    try std.testing.expectEqual(@as(?u32, null), computeFamily(&.{queues(c.VK_QUEUE_COMPUTE_BIT, 0)}));
    try std.testing.expectEqual(@as(?u32, null), computeFamily(&.{}));
}

test "a device below 1.3, a CPU device, no compute queue or a missing feature is refused" {
    const v12 = (1 << 22) | (2 << 12);
    const gpu = c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    try std.testing.expectEqual(@as(?Error, error.VulkanVersion), refusal(v12, gpu, 0, null));
    try std.testing.expectEqual(@as(?Error, error.SoftwareDevice), refusal(runtime.api_1_3, c.VK_PHYSICAL_DEVICE_TYPE_CPU, 0, null));
    try std.testing.expectEqual(@as(?Error, error.NoComputeQueue), refusal(runtime.api_1_3, gpu, null, null));
    try std.testing.expectEqual(@as(?Error, error.MissingFeature), refusal(runtime.api_1_3, gpu, 0, "float64"));
    try std.testing.expectEqual(@as(?Error, null), refusal(runtime.api_1_3, gpu, 2, null));
}

test "selection takes the best eligible kind and an explicit index only when eligible" {
    const cands = [_]Candidate{
        .{ .kind = c.VK_PHYSICAL_DEVICE_TYPE_CPU, .family = 0, .refusal = error.SoftwareDevice },
        .{ .kind = c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, .family = 0, .refusal = null },
        .{ .kind = c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, .family = 1, .refusal = null },
        .{ .kind = c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, .family = 0, .refusal = error.MissingFeature },
    };
    try std.testing.expectEqual(@as(usize, 2), try choose(&cands, null));
    try std.testing.expectEqual(@as(usize, 1), try choose(&cands, 1));
    try std.testing.expectError(error.SoftwareDevice, choose(&cands, 0));
    try std.testing.expectError(error.MissingFeature, choose(&cands, 3));
    try std.testing.expectError(error.NoDevice, choose(&cands, 4));
    try std.testing.expectEqual(@as(usize, 1), try choose(cands[0..2], null));
    try std.testing.expectError(error.NoDevice, choose(cands[0..1], null));
    try std.testing.expectError(error.NoDevice, choose(&.{}, null));
}

test "a requested feature the device lacks is named, and only requested bits are enabled" {
    try std.testing.expectEqual(@as(?[]const u8, null), missing(.{}, .{}));
    try std.testing.expectEqual(@as(?[]const u8, null), missing(.{ .float16 = true }, .{ .float16 = true, .int8 = true }));
    try std.testing.expectEqualStrings("storage16", missing(.{ .float16 = true, .storage16 = true }, .{ .float16 = true }).?);
    var chain: Chain = undefined;
    chain.enable(.{ .float64 = true, .storage8 = true, .subgroup_size_control = true });
    try std.testing.expectEqual(@as(c.VkBool32, 1), chain.f2.features.shaderFloat64);
    try std.testing.expectEqual(@as(c.VkBool32, 1), chain.v12.storageBuffer8BitAccess);
    try std.testing.expectEqual(@as(c.VkBool32, 1), chain.v13.subgroupSizeControl);
    try std.testing.expectEqual(@as(c.VkBool32, 0), chain.v12.shaderFloat16);
    try std.testing.expectEqual(@as(c.VkBool32, 0), chain.f2.features.shaderInt64);
    try std.testing.expect(chain.f2.pNext == @as(?*anyopaque, &chain.v11));
    try std.testing.expect(chain.v12.pNext == @as(?*anyopaque, &chain.v13));
    const have = chain.available();
    try std.testing.expect(have.float64 and have.storage8 and have.subgroup_size_control and !have.int8);
}
