//! Emit the actual native startup without executing a driver.
export fn compileOpen() usize {
    return @intFromPtr(&@import("native").open);
}
