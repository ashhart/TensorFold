//! CPU scripted timing and actual Metal measurement wiring without importing a device module.
const std = @import("std");

test {
    _ = @import("src/core/lanes/cost_rule.zig");
    _ = @import("src/core/lanes/cost_cache.zig");
}

