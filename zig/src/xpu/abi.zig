//! Level Zero core types and entry points as the spec documents them, declared by hand.

pub const Result = u32;
pub const success: Result = 0;
pub const error_not_ready: Result = 1;
pub const error_out_of_device_memory: Result = 0x70000002;

pub const DriverHandle = ?*opaque {};
pub const DeviceHandle = ?*opaque {};
pub const ContextHandle = ?*opaque {};
pub const CommandListHandle = ?*opaque {};
pub const ModuleHandle = ?*opaque {};
pub const KernelHandle = ?*opaque {};
pub const EventHandle = ?*opaque {};
pub const BuildLogHandle = ?*opaque {};

pub const structure_type_device_properties: u32 = 0x3;

pub const DeviceType = enum(u32) { gpu = 1, cpu = 2, fpga = 3, mca = 4, vpu = 5, _ };

pub const DeviceProperties = extern struct {
    stype: u32 = structure_type_device_properties,
    next: ?*anyopaque = null,
    type: DeviceType,
    vendor_id: u32,
    device_id: u32,
    flags: u32,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_mem_alloc_size: u64,
    max_hardware_contexts: u32,
    max_command_queue_priority: u32,
    num_threads_per_eu: u32,
    physical_eu_simd_width: u32,
    num_eus_per_subslice: u32,
    num_subslices_per_slice: u32,
    num_slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [16]u8,
    name: [256]u8,
};

pub const ContextDesc = extern struct { stype: u32 = 0xd, next: ?*const anyopaque = null, flags: u32 = 0 };

/// mode 2 = asynchronous; flags bit 1 = in order (immediate command lists only).
pub const CommandQueueDesc = extern struct { stype: u32 = 0xe, next: ?*const anyopaque = null, ordinal: u32 = 0, index: u32 = 0, flags: u32 = 0, mode: u32 = 2, priority: u32 = 0 };
pub const queue_flag_in_order: u32 = 2;

pub const DeviceMemAllocDesc = extern struct { stype: u32 = 0x15, next: ?*const anyopaque = null, flags: u32 = 0, ordinal: u32 = 0 };
pub const HostMemAllocDesc = extern struct { stype: u32 = 0x16, next: ?*const anyopaque = null, flags: u32 = 0 };

pub const module_format_spirv: u32 = 0;
pub const module_format_native: u32 = 1;
pub const ModuleDesc = extern struct { stype: u32 = 0x1b, next: ?*const anyopaque = null, format: u32, input_size: usize, input: [*]const u8, build_flags: ?[*:0]const u8 = null, constants: ?*const anyopaque = null };
pub const KernelDesc = extern struct { stype: u32 = 0x1d, next: ?*const anyopaque = null, flags: u32 = 0, name: [*:0]const u8 };
pub const GroupCount = extern struct { x: u32, y: u32, z: u32 };

pub const Api = struct {
    zeInit: *const fn (flags: u32) callconv(.c) Result,
    zeDriverGet: *const fn (count: *u32, drivers: ?[*]DriverHandle) callconv(.c) Result,
    zeDeviceGet: *const fn (driver: DriverHandle, count: *u32, devices: ?[*]DeviceHandle) callconv(.c) Result,
    zeDeviceGetProperties: *const fn (device: DeviceHandle, props: *DeviceProperties) callconv(.c) Result,
    zeContextCreate: *const fn (driver: DriverHandle, desc: *const ContextDesc, ctx: *ContextHandle) callconv(.c) Result,
    zeContextDestroy: *const fn (ctx: ContextHandle) callconv(.c) Result,
    zeCommandListCreateImmediate: *const fn (ctx: ContextHandle, device: DeviceHandle, desc: *const CommandQueueDesc, list: *CommandListHandle) callconv(.c) Result,
    zeCommandListDestroy: *const fn (list: CommandListHandle) callconv(.c) Result,
    zeCommandListHostSynchronize: *const fn (list: CommandListHandle, timeout_ns: u64) callconv(.c) Result,
    zeCommandListAppendBarrier: *const fn (list: CommandListHandle, signal: EventHandle, n_wait: u32, wait: ?[*]EventHandle) callconv(.c) Result,
    zeCommandListAppendMemoryCopy: *const fn (list: CommandListHandle, dst: ?*anyopaque, src: ?*const anyopaque, size: usize, signal: EventHandle, n_wait: u32, wait: ?[*]EventHandle) callconv(.c) Result,
    zeCommandListAppendMemoryFill: *const fn (list: CommandListHandle, ptr: ?*anyopaque, pattern: *const anyopaque, pattern_size: usize, size: usize, signal: EventHandle, n_wait: u32, wait: ?[*]EventHandle) callconv(.c) Result,
    zeCommandListAppendLaunchKernel: *const fn (list: CommandListHandle, kernel: KernelHandle, groups: *const GroupCount, signal: EventHandle, n_wait: u32, wait: ?[*]EventHandle) callconv(.c) Result,
    zeMemAllocDevice: *const fn (ctx: ContextHandle, desc: *const DeviceMemAllocDesc, size: usize, align_: usize, device: DeviceHandle, ptr: *?*anyopaque) callconv(.c) Result,
    zeMemAllocHost: *const fn (ctx: ContextHandle, desc: *const HostMemAllocDesc, size: usize, align_: usize, ptr: *?*anyopaque) callconv(.c) Result,
    zeMemFree: *const fn (ctx: ContextHandle, ptr: ?*anyopaque) callconv(.c) Result,
    zeModuleCreate: *const fn (ctx: ContextHandle, device: DeviceHandle, desc: *const ModuleDesc, module: *ModuleHandle, log: ?*BuildLogHandle) callconv(.c) Result,
    zeModuleDestroy: *const fn (module: ModuleHandle) callconv(.c) Result,
    zeModuleBuildLogGetString: *const fn (log: BuildLogHandle, size: *usize, text: ?[*]u8) callconv(.c) Result,
    zeModuleBuildLogDestroy: *const fn (log: BuildLogHandle) callconv(.c) Result,
    zeKernelCreate: *const fn (module: ModuleHandle, desc: *const KernelDesc, kernel: *KernelHandle) callconv(.c) Result,
    zeKernelDestroy: *const fn (kernel: KernelHandle) callconv(.c) Result,
    zeKernelSetGroupSize: *const fn (kernel: KernelHandle, x: u32, y: u32, z: u32) callconv(.c) Result,
    zeKernelSetArgumentValue: *const fn (kernel: KernelHandle, index: u32, size: usize, value: ?*const anyopaque) callconv(.c) Result,
};
