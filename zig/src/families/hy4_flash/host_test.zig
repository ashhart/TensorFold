//! Host-only Hy4-flash family package: config and layer plan. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("plan.zig");
}
