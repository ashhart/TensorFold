//! A failed Bolt loader retains its stream, kernels and partial weights until the owning native GPU can recover.
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;
var head: ?*Engine = null;
var mutex: std.atomic.Mutex = .unlocked;
fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}
pub fn keep(e: *Engine) void {
    lock();
    defer mutex.unlock();
    e.load_next = head;
    head = e;
}
fn close(e: *Engine) !void {
    try e.ctx.makeCurrent();
    if (@import("radio").weights.recover(e.ctx.d) != 0) return error.LoadRetained;
    if (e.load_pending) |p| try p.close();
    if (e.load_has_weights) e.w.deinit();
    if (e.tp) |*g| g.close();
    e.k.deinit();
    e.side.s.deinit();
    e.side.fork.deinit();
    e.side.join.deinit();
    e.stream.deinit();
    e.gpa.destroy(e);
}
var recovery: std.atomic.Mutex = .unlocked;
pub fn recover(ctx: ?*const cuda.Context) usize {
    // A detached transaction remains a pending owner at every parent/admission boundary.
    if (!recovery.tryLock()) return 1;
    defer recovery.unlock();
    lock();
    var at = head;
    head = null;
    mutex.unlock();
    if (comptime @import("builtin").is_test) if (Testing.detached) |hook| {
        Testing.detached = null;
        hook();
    };
    while (at) |e| {
        at = e.load_next;
        e.load_next = null;
        if (ctx == null or e.ctx == ctx.?) {
            close(e) catch {
                keep(e);
            };
        } else keep(e);
    }
    lock();
    defer mutex.unlock();
    var n: usize = 0;
    at = head;
    while (at) |e| {
        if (ctx == null or e.ctx == ctx.?) n += 1;
        at = e.load_next;
    }
    return n;
}

pub const Testing = if (@import("builtin").is_test) struct {
    pub fn empty() bool {
        lock();
        defer mutex.unlock();
        return head == null;
    }
    pub var detached: ?*const fn () void = null;
    pub const radio = @import("radio").weights.Testing;
    pub const radioRecover = @import("radio").weights.recover;
} else struct {};

pub fn closeEntry(ctx: *const cuda.Context) !void {
    _ = recover(null);
    try ctx.makeCurrent();
}
