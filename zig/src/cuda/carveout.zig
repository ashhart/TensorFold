//! GB10 display memory as device memory: a DRM dumb buffer, mapped and registered with CUDA, carved into cache spans.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const DriverError = @import("driver.zig").Error;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;

pub const width = 4096; // pixels a framebuffer row
pub const bpp = 32;
pub const row_bytes = width * bpp / 8; // the unit a carveout's size rounds down to
pub const default_bytes: usize = 1792 << 20; // what GB10's display reservation holds with room to spare
pub const default_card = "/dev/dri/card0";
pub const alignment = 256; // cuMemAlloc's own

/// struct drm_mode_create_dumb (drm_mode.h): height, width and bpp in; handle, pitch and size out.
pub const CreateDumb = extern struct { height: u32, width: u32, bpp: u32, flags: u32 = 0, handle: u32 = 0, pitch: u32 = 0, size: u64 = 0 };
/// struct drm_mode_map_dumb: the handle in, the offset to mmap the card at out.
pub const MapDumb = extern struct { handle: u32, pad: u32 = 0, offset: u64 = 0 };
/// struct drm_mode_destroy_dumb.
pub const DestroyDumb = extern struct { handle: u32 };

pub const ioctl_create_dumb = linux.IOCTL.IOWR('d', 0xB2, CreateDumb);
pub const ioctl_map_dumb = linux.IOCTL.IOWR('d', 0xB3, MapDumb);
pub const ioctl_destroy_dumb = linux.IOCTL.IOWR('d', 0xB4, DestroyDumb);

pub const Error = DriverError || error{ CardUnavailable, DumbBufferRefused, MapRefused, RegisterRefused };

/// `bytes` in whole framebuffer rows; zero below one row.
pub fn roundBytes(bytes: usize) usize {
    return bytes / row_bytes * row_bytes;
}

/// The size the options ask for: `flag` or `on` = "1" turns it on, `mib` sizes it; null means off.
pub fn requested(flag: bool, on: ?[]const u8, mib: ?[]const u8) error{ InvalidCharacter, Overflow, CarveoutTooSmall }!?usize {
    const wanted = flag or (if (on) |v| std.mem.eql(u8, v, "1") else false);
    if (!wanted) return null;
    const m = mib orelse return default_bytes;
    if (m.len == 0) return default_bytes;
    const bytes = try std.math.mul(usize, try std.fmt.parseInt(usize, m, 10), 1 << 20);
    if (roundBytes(bytes) == 0) return error.CarveoutTooSmall;
    return roundBytes(bytes);
}

/// First-fit placement of 256-byte aligned spans in `size` bytes; with nothing given back it is a bump allocator.
pub const Spans = struct {
    pub const max = 64;
    pub const Range = struct { at: usize, len: usize };

    size: usize,
    live: [max]Range = undefined, // sorted by offset
    n: usize = 0,

    /// The offset of a new span of `len` bytes, or null when no gap (or no slot) holds it.
    pub fn take(s: *Spans, len: usize) ?usize {
        if (len == 0 or s.n == max) return null;
        var at: usize = 0;
        for (0..s.n + 1) |i| {
            const end = if (i < s.n) s.live[i].at else s.size;
            if (at <= end and end - at >= len) {
                std.mem.copyBackwards(Range, s.live[i + 1 .. s.n + 1], s.live[i..s.n]);
                s.live[i] = .{ .at = at, .len = len };
                s.n += 1;
                return at;
            }
            if (i < s.n) at = std.mem.alignForward(usize, s.live[i].at + s.live[i].len, alignment);
        }
        return null;
    }

    /// Gives back the span at `at`; an offset no span starts at is ignored.
    pub fn release(s: *Spans, at: usize) void {
        for (s.live[0..s.n], 0..) |r, i| if (r.at == at) {
            std.mem.copyForwards(Range, s.live[i .. s.n - 1], s.live[i + 1 .. s.n]);
            s.n -= 1;
            return;
        };
    }

    /// Bytes no live span holds (alignment gaps count as free).
    pub fn free(s: *const Spans) usize {
        var used: usize = 0;
        for (s.live[0..s.n]) |r| used += r.len;
        return s.size - used;
    }
};

/// The registered framebuffer: owns the card's fd, the dumb buffer, its mapping and the CUDA registration.
pub const Carveout = struct {
    d: *const Driver,
    fd: linux.fd_t,
    handle: u32,
    host: []align(std.heap.page_size_min) u8,
    dev: abi.DevicePtr,
    flags: c_uint, // the registration flags the driver took
    spans: Spans,

    /// Maps `want` bytes (whole rows) of `card`'s display memory into the current context.
    pub fn open(d: *const Driver, card: [*:0]const u8, want: usize) Error!Carveout {
        const size = roundBytes(want);
        if (size == 0) return error.Invalid;
        const fd_rc = linux.open(card, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, 0);
        if (linux.errno(fd_rc) != .SUCCESS) return error.CardUnavailable;
        const fd: linux.fd_t = @intCast(fd_rc);
        errdefer _ = linux.close(fd);
        var create: CreateDumb = .{ .height = @intCast(size / row_bytes), .width = width, .bpp = bpp };
        if (linux.errno(linux.ioctl(fd, ioctl_create_dumb, @intFromPtr(&create))) != .SUCCESS) return error.DumbBufferRefused;
        errdefer destroy(fd, create.handle);
        if (create.pitch != row_bytes or create.size < size) return error.DumbBufferRefused;
        var map: MapDumb = .{ .handle = create.handle };
        if (linux.errno(linux.ioctl(fd, ioctl_map_dumb, @intFromPtr(&map))) != .SUCCESS) return error.MapRefused;
        const addr = linux.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, @intCast(map.offset));
        if (linux.errno(addr) != .SUCCESS) return error.MapRefused;
        const base: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(addr);
        errdefer _ = linux.munmap(base, size);
        const flags = try register(d, base, size);
        errdefer _ = d.api.cuMemHostUnregister(base);
        var dev: abi.DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, base, 0), "cuMemHostGetDevicePointer");
        return .{ .d = d, .fd = fd, .handle = create.handle, .host = base[0..size], .dev = dev, .flags = flags, .spans = .{ .size = size } };
    }

    /// DEVICEMAP first; a driver that sees the range as I/O memory wants IOMEMORY said too.
    fn register(d: *const Driver, base: [*]u8, len: usize) Error!c_uint {
        const plain = abi.host_register_devicemap;
        if (d.api.cuMemHostRegister_v2(base, len, plain) == abi.success) return plain;
        const io = plain | abi.host_register_iomemory;
        d.check(d.api.cuMemHostRegister_v2(base, len, io), "cuMemHostRegister") catch return error.RegisterRefused;
        return io;
    }

    fn destroy(fd: linux.fd_t, handle: u32) void {
        var x: DestroyDumb = .{ .handle = handle };
        _ = linux.ioctl(fd, ioctl_destroy_dumb, @intFromPtr(&x));
    }

    /// Releases in reverse order of open; every span taken from it must be out of use.
    pub fn close(c: *Carveout) void {
        _ = c.d.api.cuMemHostUnregister(c.host.ptr);
        _ = linux.munmap(c.host.ptr, c.host.len);
        destroy(c.fd, c.handle);
        _ = linux.close(c.fd);
        c.* = undefined;
    }

    /// A zeroed span of `len` bytes, 256-byte aligned; null when it does not fit (the caller falls back).
    pub fn take(c: *Carveout, len: usize) Error!?DeviceBuffer {
        const at = c.spans.take(len) orelse return null;
        errdefer c.spans.release(at);
        const b: DeviceBuffer = .{ .d = c.d, .ptr = c.dev + at, .len = len, .borrowed = true };
        try b.fill8(0, null);
        return b;
    }

    /// Returns a span `take` gave out.
    pub fn give(c: *Carveout, b: DeviceBuffer) void {
        c.spans.release(b.ptr - c.dev);
    }

    pub fn bytes(c: *const Carveout) usize {
        return c.host.len;
    }

    pub fn freeBytes(c: *const Carveout) usize {
        return c.spans.free();
    }
};

test "DRM ioctl numbers are drm_mode.h's" {
    try std.testing.expectEqual(@as(u32, 0xC02064B2), ioctl_create_dumb);
    try std.testing.expectEqual(@as(u32, 0xC01064B3), ioctl_map_dumb);
    try std.testing.expectEqual(@as(u32, 0xC00464B4), ioctl_destroy_dumb);
}

test "dumb buffer structs have drm_mode.h's layout" {
    try std.testing.expectEqual(32, @sizeOf(CreateDumb));
    try std.testing.expectEqual(16, @offsetOf(CreateDumb, "handle"));
    try std.testing.expectEqual(20, @offsetOf(CreateDumb, "pitch"));
    try std.testing.expectEqual(24, @offsetOf(CreateDumb, "size"));
    try std.testing.expectEqual(16, @sizeOf(MapDumb));
    try std.testing.expectEqual(8, @offsetOf(MapDumb, "offset"));
    try std.testing.expectEqual(4, @sizeOf(DestroyDumb));
}

test "spans are 256-byte aligned and refused once full" {
    var s: Spans = .{ .size = 4096 };
    try std.testing.expectEqual(@as(?usize, 0), s.take(100));
    try std.testing.expectEqual(@as(?usize, 256), s.take(1000));
    try std.testing.expectEqual(@as(?usize, 1280), s.take(2816));
    try std.testing.expectEqual(4096 - 3916, s.free());
    try std.testing.expectEqual(@as(?usize, null), s.take(1));
    try std.testing.expectEqual(@as(?usize, null), s.take(0));
    var big: Spans = .{ .size = 1 << 20 };
    try std.testing.expectEqual(@as(?usize, null), big.take((1 << 20) + 1));
    try std.testing.expectEqual(@as(?usize, 0), big.take(1 << 20));
}

test "a given-back span's room is taken again, first fit" {
    var s: Spans = .{ .size = 4096 };
    _ = s.take(512).?;
    const mid = s.take(1024).?;
    _ = s.take(512).?;
    s.release(mid);
    s.release(12345); // no span starts there
    try std.testing.expectEqual(@as(?usize, 512), s.take(768));
    try std.testing.expectEqual(@as(?usize, 2048), s.take(512)); // 1280..1536 is too small
    try std.testing.expectEqual(@as(?usize, 1280), s.take(256));
    try std.testing.expectEqual(@as(usize, 1536), s.free());
}

test "a span list holds at most max spans" {
    var s: Spans = .{ .size = 1 << 20 };
    for (0..Spans.max) |_| _ = s.take(1).?;
    try std.testing.expectEqual(@as(?usize, null), s.take(1));
}

test "the carveout is opt-in, sized in whole rows" {
    try std.testing.expectEqual(@as(?usize, null), try requested(false, null, null));
    try std.testing.expectEqual(@as(?usize, null), try requested(false, "0", "64"));
    try std.testing.expectEqual(@as(?usize, default_bytes), try requested(true, null, null));
    try std.testing.expectEqual(@as(?usize, default_bytes), try requested(false, "1", ""));
    try std.testing.expectEqual(@as(?usize, 64 << 20), try requested(false, "1", "64"));
    try std.testing.expectError(error.InvalidCharacter, requested(true, null, "lots"));
    try std.testing.expectEqual(0, roundBytes(row_bytes - 1));
    try std.testing.expectEqual(row_bytes, roundBytes(row_bytes + 1));
}

test "the driver-facing code compiles on Linux" {
    if (comptime builtin.os.tag == .linux) std.testing.refAllDecls(Carveout);
}
