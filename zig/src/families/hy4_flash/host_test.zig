//! Host-only Hy4-flash family package: config, layer plan, weight map and refusals. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("plan.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("refusal.zig");
}
