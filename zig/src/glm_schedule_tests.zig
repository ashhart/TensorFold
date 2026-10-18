//! GLM's actual prefill capability and chunk driver behind the real shared lane host, without Metal execution.
test {
    _ = @import("families/glm/prefill_test.zig");
}

test {
    _ = @import("core/lane_host_fill_aging_test.zig");
}
