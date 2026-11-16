//! Host-only Inkling 2 family package: config and its refusals, safetensors headers. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("header.zig");
    _ = @import("header_test.zig");
}
