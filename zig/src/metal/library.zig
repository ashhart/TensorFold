//! Libraries (a build-time metallib, or source compiled at run time) and compute pipelines.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const Device = @import("device.zig").Device;

const Id = objc.Id;

pub const Error = error{ LibraryLoad, NoFunction, PipelineBuild };

extern "c" fn dispatch_data_create(buffer: *const anyopaque, size: usize, queue: ?*anyopaque, destructor: ?*anyopaque) ?*anyopaque;
extern "c" fn dispatch_release(object: *anyopaque) void;

/// MTLCompileOptions for a run-time compile; `mlx()` gives what mx.fast.metal_kernel uses on this macOS.
pub const CompileOptions = struct {
    math_mode: types.MathMode = .safe,
    math_functions: types.MathFunctions = .fast,
    language: usize = types.languageVersion(4, 0),

    /// Safe math, fast fp32 functions (Metal's default), and MLX's language version for this OS.
    pub fn mlx() CompileOptions {
        return .{ .language = mlxLanguage(objc.osVersion().major) };
    }
};

/// MLX's custom-kernel language version on macOS `major`: 4.1 from 27, 4.0 on 26, 3.2 on 15, else 3.1.
pub fn mlxLanguage(major: isize) usize {
    if (major >= 27) return types.languageVersion(4, 1);
    if (major >= 26) return types.languageVersion(4, 0);
    if (major >= 15) return types.languageVersion(3, 2);
    return types.languageVersion(3, 1);
}

test "MLX's language version follows the OS" {
    try std.testing.expectEqual(types.languageVersion(4, 0), mlxLanguage(26));
    try std.testing.expectEqual(types.languageVersion(3, 2), mlxLanguage(15));
    try std.testing.expect(objc.osVersion().major >= 11);
}

pub const Library = struct {
    id: Id,

    /// A metallib file on disk.
    pub fn fromFile(device: Device, path: []const u8) Error!Library {
        const str = objc.string(path);
        defer objc.release(str);
        const url = objc.msg(Id, objc.class("NSURL"), "fileURLWithPath:", .{str});
        var err: ?Id = null;
        const id = objc.msg(?Id, device.id, "newLibraryWithURL:error:", .{ url, &err });
        return .{ .id = id orelse return fail("metallib {s}: {s}", path, err) };
    }

    /// A metallib held in memory (for example embedded in the executable); Metal copies the bytes.
    pub fn fromBytes(device: Device, bytes: []const u8) Error!Library {
        const data = dispatch_data_create(bytes.ptr, bytes.len, null, null) orelse return error.LibraryLoad;
        defer dispatch_release(data);
        var err: ?Id = null;
        const id = objc.msg(?Id, device.id, "newLibraryWithData:error:", .{ data, &err });
        return .{ .id = id orelse return fail("embedded metallib ({d} bytes): {s}", bytes.len, err) };
    }

    /// Metal source compiled now with `options`.
    pub fn fromSource(device: Device, source: []const u8, options: CompileOptions) Error!Library {
        const str = objc.string(source);
        defer objc.release(str);
        const opts = objc.msg(Id, objc.msg(Id, objc.class("MTLCompileOptions"), "alloc", .{}), "init", .{});
        defer objc.release(opts);
        objc.msg(void, opts, "setMathMode:", .{@backingInt(options.math_mode)});
        objc.msg(void, opts, "setMathFloatingPointFunctions:", .{@backingInt(options.math_functions)});
        objc.msg(void, opts, "setLanguageVersion:", .{options.language});
        var err: ?Id = null;
        const id = objc.msg(?Id, device.id, "newLibraryWithSource:options:error:", .{ str, opts, &err });
        return .{ .id = id orelse return fail("source ({d} bytes): {s}", source.len, err) };
    }

    pub fn function(self: Library, name: []const u8) Error!Id {
        const str = objc.string(name);
        defer objc.release(str);
        return objc.msg(?Id, self.id, "newFunctionWithName:", .{str}) orelse {
            std.log.err("no function {s} in the library", .{name});
            return error.NoFunction;
        };
    }

    pub fn deinit(self: Library) void {
        objc.release(self.id);
    }
};

fn fail(comptime fmt: []const u8, what: anytype, err: ?Id) Error {
    std.log.err("library " ++ fmt, .{ what, objc.errorText(err) });
    return error.LibraryLoad;
}

pub const Pipeline = struct {
    id: Id,

    /// The kernel `name` of `library`; `indirect` lets indirect command buffers use it.
    pub fn init(device: Device, library: Library, name: []const u8, indirect: bool) Error!Pipeline {
        const func = try library.function(name);
        defer objc.release(func);
        var err: ?Id = null;
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLComputePipelineDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setComputeFunction:", .{func});
        objc.msg(void, desc, "setSupportIndirectCommandBuffers:", .{indirect});
        const id = objc.msg(?Id, device.id, "newComputePipelineStateWithDescriptor:options:reflection:error:", .{ desc, @as(usize, 0), @as(?*anyopaque, null), &err });
        return .{ .id = id orelse {
            std.log.err("pipeline {s}: {s}", .{ name, objc.errorText(err) });
            return error.PipelineBuild;
        } };
    }

    pub fn maxThreads(self: Pipeline) usize {
        return objc.msg(usize, self.id, "maxTotalThreadsPerThreadgroup", .{});
    }

    pub fn simdWidth(self: Pipeline) usize {
        return objc.msg(usize, self.id, "threadExecutionWidth", .{});
    }

    pub fn deinit(self: Pipeline) void {
        objc.release(self.id);
    }
};
