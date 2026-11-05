//! Compile the actual Metal identities and their probe callers; never execute a device or compile shaders.
export fn compiledDiskIdentities() usize {
    return @intFromPtr(&@import("src/native/metal_identity.zig").learned) ^
        @intFromPtr(&@import("src/native/qwen27_disk.zig").identity) ^
        @intFromPtr(&@import("src/native/glm_disk.zig").identity) ^
        @intFromPtr(&@import("src/native/flashnext_disk.zig").identity);
}
