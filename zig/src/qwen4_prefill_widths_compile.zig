//! Emit actual Qwen4 Metal startup and its lane callbacks, never execute this object.
export fn qwen4PrefillWidthsCompile() usize {
    return @intFromPtr(&@import("host").open);
}
