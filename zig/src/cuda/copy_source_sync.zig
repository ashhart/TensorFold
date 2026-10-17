//! Blocking custody synchronization, independent of the caller's cancelable I/O or worker pool.
const std = @import("std");
// Only the portable futex operations are used; this backend creates no workers and owns no resources.
var backend: std.Io.Threaded = .init_single_threaded;
pub const Mutex = struct {
    inner: std.Io.Mutex = .init,
    pub fn lock(m: *Mutex) void {
        m.inner.lockUncancelable(backend.io());
    }
    pub fn unlock(m: *Mutex) void {
        m.inner.unlock(backend.io());
    }
};
pub const Condition = struct {
    inner: std.Io.Condition = .init,
    pub fn wait(c: *Condition, m: *Mutex) void {
        c.inner.waitUncancelable(backend.io(), &m.inner);
    }
    pub fn broadcast(c: *Condition) void {
        c.inner.broadcast(backend.io());
    }
};
