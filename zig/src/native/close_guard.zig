//! A failed completion retains the complete native parent and retries it at a later open boundary.
const std = @import("std");
pub fn Queue(comptime P: type, comptime wait: fn (*P) anyerror!void, comptime finish: fn (*P) void) type {
    return struct {
        var head: ?*P = null;
        var mutex: std.atomic.Mutex = .unlocked;
        fn lock() void {
            while (!mutex.tryLock()) std.Thread.yield() catch {};
        }
        pub fn keep(p: *P) void {
            lock();
            defer mutex.unlock();
            p.retained_next = head;
            head = p;
        }
        pub fn recover() void {
            lock();
            const list = head;
            head = null;
            mutex.unlock();
            var at = list;
            while (at) |p| {
                at = p.retained_next;
                p.retained_next = null;
                wait(p) catch {
                    keep(p);
                    continue;
                };
                finish(p);
            }
        }
    };
}
