//! MTLSharedEvent: a 64-bit counter both the host and the GPU can signal and wait on.
const objc = @import("objc.zig");

const Id = objc.Id;

pub const SharedEvent = struct {
    id: Id,

    pub fn value(self: SharedEvent) u64 {
        return objc.msg(u64, self.id, "signaledValue", .{});
    }

    /// Host signal: releases GPU work waiting for `v` or less.
    pub fn set(self: SharedEvent, v: u64) void {
        objc.msg(void, self.id, "setSignaledValue:", .{v});
    }

    /// Block until the value reaches `v`; false on timeout.
    pub fn wait(self: SharedEvent, v: u64, timeout_ms: u64) bool {
        return objc.msg(bool, self.id, "waitUntilSignaledValue:timeoutMS:", .{ v, timeout_ms });
    }

    pub fn deinit(self: SharedEvent) void {
        objc.release(self.id);
    }
};

/// MTLEvent: a GPU-only counter that orders command buffers, also across queues; no host access, cheaper than shared.
pub const Event = struct {
    id: Id,

    pub fn deinit(self: Event) void {
        objc.release(self.id);
    }
};
