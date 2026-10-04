//! Kernel launches: argument values packed where cuLaunchKernel reads them, plus the extended launch attributes.

const std = @import("std");
const abi = @import("abi.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Error = @import("driver.zig").Error;

pub const Dim3 = abi.Dim3;

/// Argument values in kernel order; each must have exactly the C type of its kernel parameter.
pub const Args = struct {
    pub const max_args = 48;
    pub const max_bytes = 2048;

    storage: [max_bytes]u8 align(16) = undefined,
    offsets: [max_args]u16 = undefined,
    sizes: [max_args]u16 = undefined,
    ptrs: [max_args]?*anyopaque = undefined,
    used: usize = 0,
    count: usize = 0,

    /// Copies `value` (a pointer-sized address, scalar, or extern struct passed by value) at its C alignment.
    pub fn add(self: *Args, value: anytype) void {
        const T = @TypeOf(value);
        comptime std.debug.assert(@sizeOf(T) > 0);
        const at = std.mem.alignForward(usize, self.used, @alignOf(T));
        std.debug.assert(self.count < max_args and at + @sizeOf(T) <= max_bytes);
        @memcpy(self.storage[at..][0..@sizeOf(T)], std.mem.asBytes(&value));
        self.offsets[self.count] = @intCast(at);
        self.sizes[self.count] = @sizeOf(T);
        self.count += 1;
        self.used = at + @sizeOf(T);
    }

    /// Argument `i` read back as an unsigned integer (4- and 8-byte values only).
    pub fn integer(self: *const Args, i: usize) ?u64 {
        if (i >= self.count) return null;
        const at = self.storage[self.offsets[i]..];
        return switch (self.sizes[i]) {
            4 => std.mem.readInt(u32, at[0..4], .little),
            8 => std.mem.readInt(u64, at[0..8], .little),
            else => null,
        };
    }

    /// The address array, rebuilt here so a moved Args never hands out stale addresses.
    pub fn pointers(self: *Args) ?[*]?*anyopaque {
        if (self.count == 0) return null;
        for (self.offsets[0..self.count], 0..) |off, i| self.ptrs[i] = &self.storage[off];
        return &self.ptrs;
    }
};

pub const Config = struct {
    grid: Dim3,
    block: Dim3,
    shared: u32 = 0,
    cluster: ?Dim3 = null,
    cluster_spread: bool = false,
    pdl: bool = false,
    cooperative: bool = false,

    fn plain(self: Config) bool {
        return self.cluster == null and !self.pdl and !self.cooperative;
    }

    pub fn validate(self: Config) Error!void {
        const g = self.grid;
        const b = self.block;
        if (g.x == 0 or g.y == 0 or g.z == 0 or b.x == 0 or b.y == 0 or b.z == 0) return error.Invalid;
        if (@as(u64, b.x) * b.y * b.z > 1024) return error.Invalid;
    }
};

/// cuLaunchKernel for plain launches, cuLaunchKernelEx when a cluster, PDL or cooperative attribute is set.
pub fn launch(f: Function, cfg: Config, stream: Stream, args: *Args) Error!void {
    try cfg.validate();
    const d = f.d;
    const g = cfg.grid;
    const b = cfg.block;
    if (cfg.plain()) {
        return d.check(d.api.cuLaunchKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers(), null), "cuLaunchKernel");
    }
    var attrs: [4]abi.LaunchAttribute = undefined;
    const n = fillAttributes(cfg, &attrs);
    const lc: abi.LaunchConfig = .{
        .grid_x = g.x,
        .grid_y = g.y,
        .grid_z = g.z,
        .block_x = b.x,
        .block_y = b.y,
        .block_z = b.z,
        .shared_bytes = cfg.shared,
        .stream = stream.handle,
        .attrs = &attrs,
        .num_attrs = @intCast(n),
    };
    try d.check(d.api.cuLaunchKernelEx(&lc, f.handle, args.pointers(), null), "cuLaunchKernelEx");
}

fn attribute(id: abi.LaunchAttributeId) abi.LaunchAttribute {
    return .{ .id = id, .value = .{ .pad = @splat(0) } };
}

fn fillAttributes(cfg: Config, out: *[4]abi.LaunchAttribute) usize {
    var n: usize = 0;
    if (cfg.pdl) {
        out[n] = attribute(.programmatic_stream_serialization);
        out[n].value.int = 1;
        n += 1;
    }
    if (cfg.cooperative) {
        out[n] = attribute(.cooperative);
        out[n].value.int = 1;
        n += 1;
    }
    if (cfg.cluster) |c| {
        out[n] = attribute(.cluster_dimension);
        out[n].value.cluster_dim = .{ .x = c.x, .y = c.y, .z = c.z };
        n += 1;
        if (cfg.cluster_spread) {
            out[n] = attribute(.cluster_scheduling_policy_preference);
            out[n].value.int = abi.cluster_scheduling_spread;
            n += 1;
        }
    }
    return n;
}

test "args keep C alignment and sizes" {
    var a: Args = .{};
    a.add(@as(u64, 0x1122334455667788));
    a.add(@as(i32, -3));
    a.add(@as(bool, true));
    a.add(@as(f64, 2.5));
    const S = extern struct { p: u64, n: c_int, s: f32 };
    a.add(S{ .p = 7, .n = 8, .s = 9 });
    try std.testing.expectEqualSlices(u16, &.{ 0, 8, 12, 16, 24 }, a.offsets[0..a.count]);
    const ptrs = a.pointers().?;
    try std.testing.expectEqual(@as(i32, -3), @as(*align(1) const i32, @ptrCast(ptrs[1].?)).*);
    try std.testing.expectEqual(@as(f64, 2.5), @as(*align(1) const f64, @ptrCast(ptrs[3].?)).*);
}
