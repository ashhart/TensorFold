//! Compile the actual contract entry without executing its device, shader or file actions.
export fn compileFsrContract() usize {
    return @intFromPtr(&@import("contract_main").main);
}
