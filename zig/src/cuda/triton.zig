//! Triton kernels compiled ahead of time: a cubin plus its metadata, launched the way Triton 3.7's own launcher does.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");

/// The launch facts Triton writes into a kernel's metadata JSON (field names as Triton spells them).
pub const Meta = struct {
    name: []const u8,
    num_warps: u32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch_size: u32 = 0,
    global_scratch_align: u32 = 1,
    profile_scratch_size: u32 = 0,
    profile_scratch_align: u32 = 1,
    launch_cooperative_grid: bool = false,
    launch_pdl: bool = false,
};

pub fn parseMeta(gpa: std.mem.Allocator, json: []const u8) !std.json.Parsed(Meta) {
    return std.json.parseFromSlice(Meta, gpa, json, .{ .ignore_unknown_fields = true });
}

pub const Kernel = struct {
    module: Module,
    function: Function,
    meta: Meta,

    /// Loads the cubin and, as Triton's loader does, opts the kernel into its dynamic shared memory above 48 KiB.
    pub fn load(d: *const Driver, device: abi.Device, cubin: []const u8, meta: Meta, name_z: [:0]const u8) Error!Kernel {
        if (!std.mem.eql(u8, meta.name, name_z)) return error.Invalid;
        var m = try Module.load(d, cubin);
        errdefer m.unload();
        const f = try m.function(name_z);
        if (meta.shared > 49152) {
            var optin: c_int = 0;
            try d.check(d.api.cuDeviceGetAttribute(&optin, .max_shared_memory_per_block_optin, device), "cuDeviceGetAttribute");
            if (optin > 49152) {
                try d.check(d.api.cuFuncSetCacheConfig(f.handle, abi.func_cache_prefer_shared), "cuFuncSetCacheConfig");
                const static = try f.attribute(.shared_size_bytes);
                try f.setAttribute(.max_dynamic_shared_size_bytes, optin - static);
            }
        }
        return .{ .module = m, .function = f, .meta = meta };
    }

    pub fn unload(self: *Kernel) void {
        self.module.unload();
        self.* = undefined;
    }

    /// The launch geometry Triton uses for a grid: x times num_ctas, 32 * num_warps threads, metadata shared bytes.
    pub fn config(self: Kernel, grid: launch.Dim3) launch.Config {
        const m = self.meta;
        return .{
            .grid = .{ .x = grid.x * m.num_ctas, .y = grid.y, .z = grid.z },
            .block = .{ .x = 32 * m.num_warps },
            .shared = m.shared,
            .cluster = if (m.num_ctas != 1) .{ .x = m.num_ctas } else null,
            .cluster_spread = m.num_ctas != 1,
            .pdl = m.launch_pdl,
            .cooperative = m.launch_cooperative_grid,
        };
    }

    /// `args`: non-constexpr arguments in order (scratch pointers appended here); `divisible16` args must be multiples of 16.
    pub fn launchOn(self: Kernel, grid: launch.Dim3, stream: Stream, args: *launch.Args, scratch: Scratch, divisible16: []const u32) Error!void {
        const m = self.meta;
        for (divisible16) |i| {
            const v = args.integer(i) orelse return error.Invalid;
            if (v % 16 != 0) return error.Invalid;
        }
        if (m.global_scratch_size > 0 and scratch.global == 0) return error.Invalid;
        if (m.profile_scratch_size > 0 and scratch.profile == 0) return error.Invalid;
        if (m.num_ctas == 16) try self.function.setAttribute(.non_portable_cluster_size_allowed, 1);
        args.add(@as(abi.DevicePtr, scratch.global));
        args.add(@as(abi.DevicePtr, scratch.profile));
        try launch.launch(self.function, self.config(grid), stream, args);
    }

    /// Bytes of global scratch a launch of `grid` needs (Triton: blocks * num_ctas * size).
    pub fn globalScratchBytes(self: Kernel, grid: launch.Dim3) usize {
        return @as(usize, grid.x) * grid.y * grid.z * self.meta.num_ctas * self.meta.global_scratch_size;
    }
};

pub const Scratch = struct { global: abi.DevicePtr = 0, profile: abi.DevicePtr = 0 };
