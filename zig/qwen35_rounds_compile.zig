//! Emit the actual device checker body as an unexecuted host object with empty images.
export fn compileChecker() usize {
    return @intFromPtr(&@import("tool").main);
}
