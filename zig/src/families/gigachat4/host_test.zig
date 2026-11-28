//! Host-only GigaChat 4 family package: config, weight map, layer plan, norm sites, memory floors, GatedNorm. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("plan.zig");
    _ = @import("plan_test.zig");
    _ = @import("norms.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("memory.zig");
    _ = @import("memory_test.zig");
    _ = @import("gated_norm.zig");
    _ = @import("gated_norm_test.zig");
}
