//! Host-only Inkling 2 family package: config and its refusals, headers, weight map and layer plan. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("header.zig");
    _ = @import("header_test.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("plan.zig");
}
