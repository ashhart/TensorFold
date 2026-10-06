//! The GPU device: queues, buffers (zero-copy over mmap too), events and residency sets.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const command = @import("command.zig");
const sync = @import("sync.zig");
const residency = @import("residency.zig");

const Id = objc.Id;

extern "c" fn MTLCreateSystemDefaultDevice() ?Id;

pub const Error = error{ NoDevice, NoQueue, NoBuffer, NoEvent, NoResidencySet, Unsupported };

pub const Buffer = struct {
    id: Id,

    pub fn contents(self: Buffer) [*]u8 {
        return objc.msg([*]u8, self.id, "contents", .{});
    }

    pub fn length(self: Buffer) usize {
        return objc.msg(usize, self.id, "length", .{});
    }

    /// The buffer viewed as `len` values of T from its start.
    pub fn slice(self: Buffer, comptime T: type, len: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(self.contents())))[0..len];
    }

    pub fn gpuAddress(self: Buffer) u64 {
        return objc.msg(u64, self.id, "gpuAddress", .{});
    }

    pub fn deinit(self: Buffer) void {
        objc.release(self.id);
    }
};

pub const Device = struct {
    id: Id,

    pub fn init() Error!Device {
        return .{ .id = MTLCreateSystemDefaultDevice() orelse return error.NoDevice };
    }

    pub fn deinit(self: Device) void {
        objc.release(self.id);
    }

    pub fn name(self: Device) [*:0]const u8 {
        return objc.utf8(objc.msg(Id, self.id, "name", .{}));
    }

    /// The N of the GPU's applegpu_gN architecture (M1 is 13, M5 17), 0 when it names none.
    pub fn generation(self: Device) u32 {
        if (!self.responds("architecture")) return 0;
        const arch = objc.msg(?Id, self.id, "architecture", .{}) orelse return 0;
        return generationOf(std.mem.span(objc.utf8(objc.msg(Id, arch, "name", .{}))));
    }

    /// Metal 4 tensor units (applegpu_g17, the M5, and later).
    pub fn tensorUnits(self: Device) bool {
        return self.generation() >= 17;
    }

    pub fn maxWorkingSet(self: Device) u64 {
        return objc.msg(u64, self.id, "recommendedMaxWorkingSetSize", .{});
    }

    /// Bytes of every buffer and texture this device holds now.
    pub fn allocated(self: Device) u64 {
        return objc.msg(u64, self.id, "currentAllocatedSize", .{});
    }

    pub fn responds(self: Device, comptime selector: [:0]const u8) bool {
        return objc.msg(bool, self.id, "respondsToSelector:", .{objc.sel(selector)});
    }

    pub fn queue(self: Device) Error!command.Queue {
        return .{ .id = objc.msg(?Id, self.id, "newCommandQueue", .{}) orelse return error.NoQueue };
    }

    /// A new buffer of `len` bytes; `options` from types.ResourceOptions (shared storage by default).
    pub fn buffer(self: Device, len: usize, options: usize) Error!Buffer {
        const id = objc.msg(?Id, self.id, "newBufferWithLength:options:", .{ len, options });
        return .{ .id = id orelse return error.NoBuffer };
    }

    /// A buffer over caller memory with no copy: `ptr` page-aligned, `len` a whole number of pages.
    pub fn bufferNoCopy(self: Device, ptr: *anyopaque, len: usize, options: usize) Error!Buffer {
        const id = objc.msg(?Id, self.id, "newBufferWithBytesNoCopy:length:options:deallocator:", .{ ptr, len, options, @as(?*anyopaque, null) });
        return .{ .id = id orelse return error.NoBuffer };
    }

    pub fn sharedEvent(self: Device) Error!sync.SharedEvent {
        return .{ .id = objc.msg(?Id, self.id, "newSharedEvent", .{}) orelse return error.NoEvent };
    }

    pub fn event(self: Device) Error!sync.Event {
        return .{ .id = objc.msg(?Id, self.id, "newEvent", .{}) orelse return error.NoEvent };
    }

    /// A residency set (macOS 15+), or error.Unsupported.
    pub fn residencySet(self: Device, capacity: usize) Error!residency.ResidencySet {
        if (!self.responds("newResidencySetWithDescriptor:error:")) return error.Unsupported;
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLResidencySetDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setInitialCapacity:", .{capacity});
        var err: ?Id = null;
        const id = objc.msg(?Id, self.id, "newResidencySetWithDescriptor:error:", .{ desc, &err });
        return .{ .id = id orelse {
            std.log.err("residency set: {s}", .{objc.errorText(err)});
            return error.NoResidencySet;
        } };
    }
};

/// The N of an "applegpu_gN..." architecture name ("applegpu_g15d": 15), 0 for any other.
pub fn generationOf(arch: []const u8) u32 {
    const prefix = "applegpu_g";
    if (!std.mem.startsWith(u8, arch, prefix)) return 0;
    var end: usize = prefix.len;
    while (end < arch.len and std.ascii.isDigit(arch[end])) end += 1;
    return std.fmt.parseInt(u32, arch[prefix.len..end], 10) catch 0;
}

test "GPU generations from architecture names" {
    try std.testing.expectEqual(@as(u32, 15), generationOf("applegpu_g15d"));
    try std.testing.expectEqual(@as(u32, 17), generationOf("applegpu_g17s"));
    try std.testing.expectEqual(@as(u32, 13), generationOf("applegpu_g13g"));
    try std.testing.expectEqual(@as(u32, 0), generationOf("air64_v27"));
    try std.testing.expectEqual(@as(u32, 0), generationOf("applegpu_g"));
}

/// A read-only file mapped at a page boundary, its length rounded up to whole pages (the tail reads zero).
pub const MappedFile = struct {
    bytes: []align(std.heap.page_size_min) u8,
    size: usize,

    pub fn open(path: [*:0]const u8) !MappedFile {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end <= 0) return error.EmptyFile;
        const size: usize = @intCast(end);
        const page = std.heap.pageSize();
        const len = (size + page - 1) / page * page;
        const bytes = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        return .{ .bytes = bytes, .size = size };
    }

    pub fn deinit(self: MappedFile) void {
        std.posix.munmap(self.bytes);
    }
};
