//! Host-only Hy4-flash family package: config. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
}
