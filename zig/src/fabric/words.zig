//! Ordered mailbox words: MCDMA's libmcdma-rpc (ABI 1, rpc/mcdma_rpc.h) by dlopen, or its same rules in Zig.
const std = @import("std");
const builtin = @import("builtin");
const layout = @import("layout.zig");

pub const abi: u32 = 1;

pub const Error = error{ LibraryUnavailable, MissingSymbol, WrongAbi };

/// A release store and an acquire wait on a mailbox word; wait returns the word, or 0 at the timeout.
pub const Words = struct {
    ptr: ?*anyopaque,
    store_fn: *const fn (ptr: ?*anyopaque, w: *u64, value: u64) void,
    wait_fn: *const fn (ptr: ?*anyopaque, w: *const u64, seq: u32, equal: bool, spin_ns: u64, timeout_ns: u64) u64,

    pub fn store(ws: Words, w: *u64, value: u64) void {
        ws.store_fn(ws.ptr, w, value);
    }

    /// Wait until the word's sequence equals `seq` (equal) or is non-zero and differs from it (not equal).
    pub fn wait(ws: Words, w: *const u64, seq: u32, equal: bool, spin_ns: u64, timeout_ns: u64) u64 {
        return ws.wait_fn(ws.ptr, w, seq, equal, spin_ns, timeout_ns);
    }
};

/// An acquire load; the library has none, and applications that can order loads themselves need none.
pub fn load(w: *const u64) u64 {
    return @atomicLoad(u64, w, .acquire);
}

/// CLOCK_MONOTONIC in nanoseconds, the clock libmcdma-rpc waits on.
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Sleep about `ns` nanoseconds (the library's short step between polls).
pub fn pause(ns: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = std.c.nanosleep(&ts, null);
}

pub fn checkedNowNs() error{ClockUnavailable}!u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0 or ts.sec < 0 or ts.nsec < 0 or ts.nsec >= std.time.ns_per_s) return error.ClockUnavailable;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The library's rules in Zig, for in-process fakes and for checking the library against them.
pub const native: Words = .{ .ptr = null, .store_fn = nativeStore, .wait_fn = nativeWait };

fn nativeStore(_: ?*anyopaque, w: *u64, value: u64) void {
    @atomicStore(u64, w, value, .release);
}

fn nativeWait(_: ?*anyopaque, w: *const u64, seq: u32, equal: bool, spin_ns: u64, timeout_ns: u64) u64 {
    const start = nowNs();
    while (true) {
        const value = load(w);
        const current = layout.seqOf(value);
        if (if (equal) current == seq else current != 0 and current != seq) return value;
        const elapsed = nowNs() - start;
        if (elapsed >= timeout_ns) return 0;
        if (elapsed < spin_ns) std.atomic.spinLoopHint() else pause(20_000);
    }
}

const AbiFn = *const fn () callconv(.c) u32;
const WaitFn = *const fn (w: *const volatile u64, seq: u32, want_equal: c_int, spin_ns: u64, timeout_ns: u64) callconv(.c) u64;
const StoreFn = *const fn (w: *volatile u64, value: u64) callconv(.c) void;
const WrapFn = *const fn (memory: ?*anyopaque, length: usize) callconv(.c) ?*anyopaque;
const ContentsFn = *const fn (buffer: ?*anyopaque) callconv(.c) ?*anyopaque;
const ReleaseFn = *const fn (buffer: ?*anyopaque) callconv(.c) void;

/// Where MCDMA installs the helper (rpc/Makefile PREFIX=/usr/local), after MCDMA_RPC_LIBRARY.
const default_paths = if (builtin.os.tag == .macos)
    [_][:0]const u8{"/usr/local/lib/libmcdma-rpc.dylib"}
else
    [_][:0]const u8{ "/usr/local/lib/libmcdma-rpc.so", "/usr/lib/libmcdma-rpc.so" };

/// libmcdma-rpc opened at run time; the three Metal functions are optional, as the header says.
pub const Library = struct {
    lib: std.DynLib,
    abi_fn: AbiFn,
    wait_word: WaitFn,
    store_word: StoreFn,
    metal_wrap: ?WrapFn = null,
    metal_contents: ?ContentsFn = null,
    metal_release: ?ReleaseFn = null,

    /// Open `path`, else MCDMA_RPC_LIBRARY, else the install prefix; refuse any ABI but 1.
    pub fn open(path: ?[:0]const u8) Error!Library {
        if (path) |p| return openPath(p);
        if (std.c.getenv("MCDMA_RPC_LIBRARY")) |env| {
            const p = std.mem.span(env);
            if (p.len > 0) return openPath(p);
        }
        for (default_paths) |p| return openPath(p) catch |err| switch (err) {
            error.LibraryUnavailable => continue,
            else => return err,
        };
        return error.LibraryUnavailable;
    }

    pub fn openPath(path: [:0]const u8) Error!Library {
        var lib = std.DynLib.openZ(path.ptr) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var l: Library = .{
            .lib = lib,
            .abi_fn = lib.lookup(AbiFn, "mcdma_rpc_abi") orelse return error.MissingSymbol,
            .wait_word = lib.lookup(WaitFn, "mcdma_rpc_wait_word") orelse return error.MissingSymbol,
            .store_word = lib.lookup(StoreFn, "mcdma_rpc_store_word") orelse return error.MissingSymbol,
        };
        if (l.abi_fn() != abi) return error.WrongAbi;
        if (builtin.os.tag == .macos) {
            l.metal_wrap = l.lib.lookup(WrapFn, "mcdma_rpc_metal_wrap");
            l.metal_contents = l.lib.lookup(ContentsFn, "mcdma_rpc_metal_contents");
            l.metal_release = l.lib.lookup(ReleaseFn, "mcdma_rpc_metal_release");
        }
        return l;
    }

    pub fn close(l: *Library) void {
        l.lib.close();
    }

    pub fn words(l: *Library) Words {
        return .{ .ptr = l, .store_fn = libStore, .wait_fn = libWait };
    }

    /// A Metal buffer (id<MTLBuffer>) aliasing page-aligned `memory`, so the GPU reads what the NIC wrote; null without one.
    pub fn metalWrap(l: *const Library, memory: []align(std.heap.page_size_min) u8) ?*anyopaque {
        const wrap = l.metal_wrap orelse return null;
        const contents = l.metal_contents orelse return null;
        const buffer = wrap(memory.ptr, memory.len) orelse return null;
        if (contents(buffer) != @as(?*anyopaque, memory.ptr)) {
            l.metalRelease(buffer);
            return null;
        }
        return buffer;
    }

    /// Release a buffer from metalWrap; the memory stays mapped.
    pub fn metalRelease(l: *const Library, buffer: *anyopaque) void {
        if (l.metal_release) |release| release(buffer);
    }

    fn libStore(ptr: ?*anyopaque, w: *u64, value: u64) void {
        const l: *Library = @ptrCast(@alignCast(ptr.?));
        l.store_word(w, value);
    }

    fn libWait(ptr: ?*anyopaque, w: *const u64, seq: u32, equal: bool, spin_ns: u64, timeout_ns: u64) u64 {
        const l: *Library = @ptrCast(@alignCast(ptr.?));
        return l.wait_word(w, seq, @intFromBool(equal), spin_ns, timeout_ns);
    }
};

test "native waits follow the library's sequence rule" {
    var w: u64 = 0;
    try std.testing.expectEqual(@as(u64, 0), native.wait(&w, 7, true, 1000, 2_000_000));
    try std.testing.expectEqual(@as(u64, 0), native.wait(&w, 7, false, 1000, 2_000_000));
    native.store(&w, layout.word(7, 12));
    try std.testing.expectEqual(layout.word(7, 12), native.wait(&w, 7, true, 1000, 2_000_000));
    try std.testing.expectEqual(@as(u64, 0), native.wait(&w, 7, false, 1000, 2_000_000));
    try std.testing.expectEqual(layout.word(7, 12), native.wait(&w, 6, false, 1000, 2_000_000));
    const began = nowNs();
    try std.testing.expectEqual(@as(u64, 0), native.wait(&w, 9, true, 0, 30_000_000));
    try std.testing.expect(nowNs() - began >= 30_000_000);
}

test "a missing library is reported, not guessed" {
    try std.testing.expectError(error.LibraryUnavailable, Library.openPath("/nonexistent/libmcdma-rpc.dylib"));
}
