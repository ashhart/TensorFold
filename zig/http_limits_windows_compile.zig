//! Compile the production Winsock overload path without linking or running a server.
const std = @import("std");
const admission = @import("src/server/http_admission.zig");
const platform = @import("sys");

export fn tf_http_reject(fd: platform.posix.socket_t) void {
    admission.reject(fd);
}

export fn tf_http_admit(count: *std.atomic.Value(u32), limit: u32) bool {
    return admission.acquire(count, limit);
}
