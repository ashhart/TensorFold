//! The lane host retains both landed and series lifecycle regressions.

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_rounds_test.zig");
    _ = @import("lane_halt_test.zig");
    _ = @import("lane_host_end_test.zig");
    _ = @import("lane_host_score_test.zig");
    _ = @import("lane_review_test.zig");
    _ = @import("lane_init_test.zig");
    _ = @import("lane_failure_review_test.zig");
    _ = @import("lane_host_first_test.zig");
}

test {
    _ = @import("lane_host_score_exit_test.zig");
    _ = @import("lane_host_grammar_test.zig");
}

test {
    _ = @import("lane_host_interior_test.zig");
    _ = @import("lane_host_gpu_seconds_test.zig");
}

test {
    _ = @import("memory_pressure_test.zig");
    _ = @import("lane_host_pressure_test.zig");
}
