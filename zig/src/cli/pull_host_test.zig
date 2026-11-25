//! Pull-host integration tests and same-revision repair coverage.
test {
    _ = @import("pull_test.zig");
    _ = @import("pull_repair_test.zig");
    _ = @import("pull_kp50_test.zig");
    _ = @import("sys").credential;
}
