//! Analyze the actual Inkling 2 Mini native opener without loading a device or checkpoint.
export fn compileInkling2Native() usize {
    return @intFromPtr(&@import("src/inkling2_native.zig").open);
}
