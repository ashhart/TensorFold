//! The lane host retains both landed and series lifecycle regressions.

test {
    _ = @import("lane_host_test.zig");
    _ = @import("lane_rounds_test.zig");
    _ = @import("lane_halt_test.zig");
    _ = @import("lane_host_end_test.zig");
    _ = @import("lane_review_test.zig");
    _ = @import("lane_init_test.zig");
    _ = @import("lane_failure_review_test.zig");
    _ = @import("lane_host_first_test.zig");
}
