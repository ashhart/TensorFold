//! libtfvideo (zig/src/video/tf_video.c over FFmpeg 9.0.2, `zig build video`), loaded when the first video arrives:
//! a container's first video stream, its frame count and rate as transformers' PyAV loader reads them, and the RGB
//! frames at given indices.
const std = @import("std");
const builtin = @import("builtin");

pub const Info = extern struct { frames: i64, rate_num: c_int, rate_den: c_int, width: c_int, height: c_int, counted: c_int };
const FrameFn = *const fn (ctx: ?*anyopaque, k: c_int, index: c_int, rgb: [*]const u8, width: c_int, height: c_int, stride: c_int) callconv(.c) c_int;

const Fns = struct {
    open: *const fn ([*]const u8, usize, *?*anyopaque, *Info) callconv(.c) c_int,
    frames: *const fn (*anyopaque, [*]const c_int, c_int, FrameFn, ?*anyopaque) callconv(.c) c_int,
    close: *const fn (*anyopaque) callconv(.c) void,
    version: *const fn () callconv(.c) [*:0]const u8,
};

/// Where the library is looked for: beside the server (zig-out/native/lib, as `zig build video` installs it), then
/// the loader's own search path.
const names: []const [:0]const u8 = if (builtin.os.tag == .macos)
    &.{ "@executable_path/../lib/libtfvideo.dylib", "libtfvideo.dylib" }
else
    &.{"libtfvideo.so"};

var lib: ?Fns = null;
var lib_failed = false;
var lib_mutex: std.atomic.Mutex = .unlocked;

fn load() ?Fns {
    while (!lib_mutex.tryLock()) std.atomic.spinLoopHint();
    defer lib_mutex.unlock();
    if (lib) |l| return l;
    if (lib_failed) return null;
    lib_failed = true;
    var dl = for (names) |name| {
        break std.DynLib.open(name) catch continue;
    } else return null;
    var f: Fns = undefined;
    const info = @typeInfo(Fns).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(f, name) = dl.lookup(T, "tf_video_" ++ name) orelse return null;
    }
    lib_failed = false;
    lib = f;
    return f;
}

/// The FFmpeg release the loaded library was built against, or null when it can't be loaded.
pub fn version() ?[]const u8 {
    const f = load() orelse return null;
    return std.mem.span(f.version());
}

pub const Error = error{ NoLibrary, Invalid, NoStream, NoDecoder, OutOfMemory, Decode, Short, Stopped };

fn err(rc: c_int) Error {
    return switch (rc) {
        -2 => error.NoStream,
        -3 => error.NoDecoder,
        -4 => error.OutOfMemory,
        -5 => error.Decode,
        -6 => error.Short,
        -7 => error.Stopped,
        else => error.Invalid,
    };
}

/// An open container over bytes the caller keeps alive until `close`.
pub const Video = struct {
    handle: *anyopaque,
    info: Info,
    fns: Fns,

    pub fn open(bytes: []const u8) Error!Video {
        const f = load() orelse return error.NoLibrary;
        var h: ?*anyopaque = null;
        var info: Info = undefined;
        const rc = f.open(bytes.ptr, bytes.len, &h, &info);
        if (rc != 0) return err(rc);
        return .{ .handle = h.?, .info = info, .fns = f };
    }

    pub fn close(v: Video) void {
        v.fns.close(v.handle);
    }

    /// The frames at `indices` (ascending, distinct), each to `sink.take(k, rgb, width, height, stride)` in order.
    pub fn frames(v: Video, indices: []const u32, sink: anytype) anyerror!void {
        const S = @TypeOf(sink);
        const Ctx = struct { sink: S, failure: ?anyerror = null };
        var ctx: Ctx = .{ .sink = sink };
        const cb = struct {
            fn take(p: ?*anyopaque, k: c_int, _: c_int, rgb: [*]const u8, w: c_int, h: c_int, stride: c_int) callconv(.c) c_int {
                const c: *Ctx = @ptrCast(@alignCast(p.?));
                const n: usize = @as(usize, @intCast(stride)) * @as(usize, @intCast(h));
                c.sink.take(@intCast(k), rgb[0..n], @intCast(w), @intCast(h), @intCast(stride)) catch |e| {
                    c.failure = e;
                    return 1;
                };
                return 0;
            }
        }.take;
        const idx = try std.heap.page_allocator.alloc(c_int, indices.len);
        defer std.heap.page_allocator.free(idx);
        for (idx, indices) |*d, s| d.* = @intCast(s);
        const rc = v.fns.frames(v.handle, idx.ptr, @intCast(idx.len), cb, &ctx);
        if (ctx.failure) |e| return e;
        if (rc < 0) return err(rc);
        if (rc != indices.len) return error.Short;
    }
};
