//! The Vulkan loader and one 1.3 instance; partial function tables never escape.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const c = @import("vk");

pub const Error = error{
    DriverUnavailable,
    MissingSymbol,
    VulkanVersion,
    VulkanFailed,
    OutOfMemory,
    /// A fence did not signal in time: the GPU may still use the submission's resources, so the caller exits at once.
    Timeout,
    /// A submit or fence wait failed with the work's state unknown: the GPU may still use it; the caller exits.
    InFlight,
    DeviceLost,
    NoDevice,
    SoftwareDevice,
    NoComputeQueue,
    MissingFeature,
    OutOfRange,
    Misaligned,
    TooLarge,
    NotMapped,
    Invalid,
};

/// The C function pointer behind a `PFN_vk*` typedef.
pub fn Fn(comptime Pfn: type) type {
    return @typeInfo(Pfn).optional.child;
}

/// What the loader answers for a null instance.
const GlobalApi = struct {
    vkEnumerateInstanceVersion: Fn(c.PFN_vkEnumerateInstanceVersion),
    vkCreateInstance: Fn(c.PFN_vkCreateInstance),
};

pub const InstanceApi = struct {
    vkDestroyInstance: Fn(c.PFN_vkDestroyInstance),
    vkEnumeratePhysicalDevices: Fn(c.PFN_vkEnumeratePhysicalDevices),
    vkGetPhysicalDeviceProperties2: Fn(c.PFN_vkGetPhysicalDeviceProperties2),
    vkGetPhysicalDeviceFeatures2: Fn(c.PFN_vkGetPhysicalDeviceFeatures2),
    vkGetPhysicalDeviceQueueFamilyProperties: Fn(c.PFN_vkGetPhysicalDeviceQueueFamilyProperties),
    vkGetPhysicalDeviceMemoryProperties: Fn(c.PFN_vkGetPhysicalDeviceMemoryProperties),
    vkCreateDevice: Fn(c.PFN_vkCreateDevice),
    vkGetDeviceProcAddr: Fn(c.PFN_vkGetDeviceProcAddr),
};

/// VK_MAKE_API_VERSION(0, 1, 3, 0).
pub const api_1_3: u32 = (1 << 22) | (3 << 12);

/// Whether a packed Vulkan version is 1.3 or later; the variant bits are ignored.
pub fn atLeast13(version: u32) bool {
    const major = (version >> 22) & 0x7f;
    const minor = (version >> 12) & 0x3ff;
    return major > 1 or (major == 1 and minor >= 3);
}

/// Every field of `Api` from `get(handle, name)`, or MissingSymbol when any one is absent.
pub fn resolve(comptime Api: type, handle: anytype, get: anytype) error{MissingSymbol}!Api {
    var api: Api = undefined;
    const info = @typeInfo(Api).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        const p = get(handle, name.ptr) orelse return error.MissingSymbol;
        @field(api, name) = @as(T, @ptrCast(p));
    }
    return api;
}

pub const Runtime = struct {
    lib: platform.DynLib,
    instance: c.VkInstance,
    api: InstanceApi,

    pub fn open() Error!Runtime {
        return openPath(switch (builtin.os.tag) {
            .windows => "vulkan-1.dll",
            .macos => "libvulkan.1.dylib",
            else => "libvulkan.so.1",
        });
    }

    /// The loader at `path` and a Vulkan 1.3 instance with no layers or extensions.
    pub fn openPath(path: []const u8) Error!Runtime {
        var lib = platform.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const gipa = lib.lookup(Fn(c.PFN_vkGetInstanceProcAddr), "vkGetInstanceProcAddr") orelse return error.MissingSymbol;
        const global = try resolve(GlobalApi, @as(c.VkInstance, null), gipa);
        var version: u32 = 0;
        try check(global.vkEnumerateInstanceVersion(&version));
        if (!atLeast13(version)) return error.VulkanVersion;
        const app: c.VkApplicationInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pEngineName = "TensorFold",
            .apiVersion = api_1_3,
        };
        const info: c.VkInstanceCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
        var instance: c.VkInstance = null;
        try check(global.vkCreateInstance(&info, null, &instance));
        if (instance == null) return error.VulkanFailed;
        const api = resolve(InstanceApi, instance, gipa) catch |e| {
            if (gipa(instance, "vkDestroyInstance")) |p| @as(Fn(c.PFN_vkDestroyInstance), @ptrCast(p))(instance, null);
            return e;
        };
        return .{ .lib = lib, .instance = instance, .api = api };
    }

    /// Destroys the instance; every device opened on it is already closed.
    pub fn close(self: *Runtime) void {
        self.api.vkDestroyInstance(self.instance, null);
        self.lib.close();
        self.* = undefined;
    }
};

/// VK_SUCCESS, or the named error a failed call maps to.
pub fn check(result: c.VkResult) Error!void {
    return switch (result) {
        c.VK_SUCCESS => {},
        c.VK_ERROR_OUT_OF_HOST_MEMORY, c.VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfMemory,
        c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
        c.VK_TIMEOUT => error.Timeout,
        else => error.VulkanFailed,
    };
}

test "absent Vulkan loader fails closed" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/tensorfold-test/libvulkan.so.1"));
}

test "a real non-Vulkan library cannot satisfy admission" {
    const path = switch (builtin.os.tag) {
        .macos => "/usr/lib/libSystem.B.dylib",
        .linux => "libc.so.6",
        else => return error.SkipZigTest,
    };
    try std.testing.expectError(error.MissingSymbol, Runtime.openPath(path));
}

test "a function table resolves completely or not at all" {
    const Loader = struct {
        fn entry() callconv(.c) void {}
        fn every(_: ?*anyopaque, _: [*c]const u8) callconv(.c) c.PFN_vkVoidFunction {
            return &entry;
        }
        fn noCreate(_: ?*anyopaque, name: [*c]const u8) callconv(.c) c.PFN_vkVoidFunction {
            return if (std.mem.eql(u8, std.mem.span(name), "vkCreateInstance")) null else &entry;
        }
    };
    const api = try resolve(GlobalApi, null, Loader.every);
    try std.testing.expect(@intFromPtr(api.vkCreateInstance) == @intFromPtr(&Loader.entry));
    try std.testing.expectError(error.MissingSymbol, resolve(GlobalApi, null, Loader.noCreate));
}

test "only Vulkan 1.3 or later is admitted" {
    try std.testing.expect(atLeast13(api_1_3));
    try std.testing.expect(atLeast13(api_1_3 | 239));
    try std.testing.expect(atLeast13((1 << 22) | (4 << 12)));
    try std.testing.expect(atLeast13(2 << 22));
    try std.testing.expect(!atLeast13((1 << 22) | (2 << 12) | 250));
    try std.testing.expect(!atLeast13((1 << 29) | (1 << 22) | (2 << 12)));
    try std.testing.expect(!atLeast13(0));
}

test "failed Vulkan results map to named errors" {
    try check(c.VK_SUCCESS);
    try std.testing.expectError(error.OutOfMemory, check(c.VK_ERROR_OUT_OF_DEVICE_MEMORY));
    try std.testing.expectError(error.OutOfMemory, check(c.VK_ERROR_OUT_OF_HOST_MEMORY));
    try std.testing.expectError(error.DeviceLost, check(c.VK_ERROR_DEVICE_LOST));
    try std.testing.expectError(error.Timeout, check(c.VK_TIMEOUT));
    try std.testing.expectError(error.VulkanFailed, check(c.VK_ERROR_INITIALIZATION_FAILED));
}
