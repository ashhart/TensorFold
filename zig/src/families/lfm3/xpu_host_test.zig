const std = @import("std");

pub const rt = struct {
    pub const Runtime = struct {
        gpa: std.mem.Allocator,

        pub fn alloc(self: *Runtime, bytes: usize) !Buffer {
            const memory = try self.gpa.alloc(u8, bytes);
            return .{ .rt = self, .ptr = memory.ptr, .len = memory.len };
        }

        pub fn copy(_: *Runtime, dest: Buffer, source: Buffer, bytes: usize) !void {
            std.debug.assert(bytes <= dest.len and bytes <= source.len);
            @memcpy(dest.ptr.?[0..bytes], source.ptr.?[0..bytes]);
        }

        pub fn sync(_: *Runtime) !void {}
    };

    pub const Buffer = struct {
        rt: *Runtime,
        ptr: ?[*]u8,
        len: usize,

        pub fn free(self: Buffer) void {
            self.rt.gpa.free(self.ptr.?[0..self.len]);
        }
    };

    pub const Kernel = struct {};
};

pub const loader = struct {
    pub const Loader = struct {};
};

pub const kernels = struct {
    pub const dpas = false;
};
