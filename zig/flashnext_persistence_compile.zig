//! Compile all Metal disk-store startup consumers without device discovery or shader compilation.
export fn compiledPersistenceConsumers() usize {
    return @intFromPtr(&@import("src/native/glm_host.zig").open) ^
        @intFromPtr(&@import("src/native/flashnext_host.zig").open) ^
        @intFromPtr(&@import("src/native/flashnext_batch_host.zig").openConfigured);
}
