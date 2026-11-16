//! Host-only Inkling 2 family package: config and its refusals. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
}
