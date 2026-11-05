//! Opt-in pinned staging for DeviceBuffer.download (#2102): the driver writes a reused pinned bounce, then we copy out.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const memory = @import("memory.zig");
const copy_source = @import("copy_source.zig");

/// The largest bounce; a longer download goes through it in pieces of this size.
pub const chunk: usize = 64 << 20;

/// Idle pinned buffers of one driver and context, each as large as the largest piece it has staged.
const Bounce = struct { next: ?*Bounce = null, d: *const Driver, context: abi.Context, host: memory.HostBuffer };
var mutex: std.atomic.Mutex = .unlocked;
var idle: ?*Bounce = null;

fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}

/// Exactly `1` stages; unset or any other value keeps main's copy straight into the caller's slice.
pub fn enabled() bool {
    const raw = @import("profile").get(.TF_CUDA_PINNED_DOWNLOAD) orelse return false;
    return std.mem.eql(u8, raw, "1");
}

/// Copies `out.len` bytes from `src` through a pinned bounce of this thread's context, then into `out`.
pub fn download(d: *const Driver, src: abi.DevicePtr, out: []u8) Error!void {
    const b = try take(d, @min(out.len, chunk));
    var at: usize = 0;
    while (at < out.len) : (at += chunk) {
        const n = @min(out.len - at, chunk);
        d.check(d.api.cuMemcpyDtoH_v2(b.host.bytes.ptr, src + at, n), "cuMemcpyDtoH") catch |e| {
            copy_source.retainPinned(d, b.host, .{ .d = d, .handle = null }); // freed after a later wait succeeds
            std.heap.page_allocator.destroy(b);
            return e;
        };
        @memcpy(out[at..][0..n], b.host.bytes[0..n]);
    }
    lock();
    defer mutex.unlock();
    b.next = idle;
    idle = b;
}

/// An idle bounce of `d` in this thread's context holding `need` bytes, grown or made when none does.
fn take(d: *const Driver, need: usize) Error!*Bounce {
    const context = copy_source.current();
    lock();
    var best: ?*?*Bounce = null;
    var at = &idle;
    while (at.*) |b| : (at = &b.next) {
        if (b.d != d or b.context != context) continue;
        if (best == null or b.host.bytes.len >= need) best = at;
        if (b.host.bytes.len >= need) break;
    }
    const found: ?*Bounce = if (best) |slot| slot.*.? else null;
    if (best) |slot| slot.* = found.?.next;
    mutex.unlock();
    const b = found orelse try std.heap.page_allocator.create(Bounce);
    if (found) |f| {
        if (f.host.bytes.len >= need) return f;
        f.host.free();
    }
    b.* = .{ .d = d, .context = context, .host = memory.HostBuffer.alloc(d, @max(need, 4096)) catch |e| {
        std.heap.page_allocator.destroy(b);
        return e;
    } };
    return b;
}

/// Frees the idle bounces of `d` in `context`, or in every context when `all`: before the context or driver goes.
pub fn drop(d: *const Driver, context: abi.Context, all: bool) void {
    lock();
    var out: ?*Bounce = null;
    var at = &idle;
    while (at.*) |b| {
        if (b.d != d or (!all and b.context != context)) {
            at = &b.next;
            continue;
        }
        at.* = b.next;
        b.next = out;
        out = b;
    }
    mutex.unlock();
    while (out) |b| {
        out = b.next;
        b.host.free();
        std.heap.page_allocator.destroy(b);
    }
}

/// Idle bounces held for `d`, for tests and receipts.
pub fn idleCount(d: *const Driver) usize {
    lock();
    defer mutex.unlock();
    var n: usize = 0;
    var at = idle;
    while (at) |b| : (at = b.next) n += @intFromBool(b.d == d);
    return n;
}

test {
    _ = @import("pinned_download_test.zig");
}
