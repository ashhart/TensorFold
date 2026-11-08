//! Validate optional DeepSeek settings before opening resources, including settings read only by later paths.
const std = @import("std");
const sys = @import("sys");
const environment = sys.environment;

/// Every TF_DSV41_* row of the shared table, parsed exactly as each later reader parses it.
pub fn startup() !void {
    try environment.checkScope(.dsv41, environment.Process{});
}

/// BIAS_VL has always selected the bias when nonempty, including 0; validation does not reinterpret that value.
pub fn imageBiasEnv() !bool {
    const raw = sys.getenv("TF_DSV41_BIAS_VL");
    try environment.check("TF_DSV41_BIAS_VL", raw);
    return raw != null and std.mem.trim(u8, raw.?, " \t\r\n").len > 0;
}

test {
    _ = @import("env_validate_test.zig");
}
