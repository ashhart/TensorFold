//! Analyze every branch of the actual native Metal opener without loading a device or a checkpoint.
export fn compileMetalResidentOwners() usize {
    return @intFromPtr(&@import("host").open);
}
