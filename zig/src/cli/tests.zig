//! Test roots for the tensorfold CLI commands.
test {
    _ = @import("hub.zig");
    _ = @import("models.zig");
    _ = @import("catalog.zig");
    _ = @import("info.zig");
    _ = @import("pull.zig");
    _ = @import("plan.zig");
    _ = @import("pull_parts.zig");
    _ = @import("pull_file.zig");
    _ = @import("pull_test.zig");
    _ = @import("pull_runs.zig");
    _ = @import("pull_exlr8.zig");
    _ = @import("pull_exlr8_plan.zig");
    _ = @import("rm.zig");
    _ = @import("rm_test.zig");
    _ = @import("tfq3.zig");
    _ = @import("tfq3_encode.zig");
    _ = @import("tfq3_test.zig");
    _ = @import("tfq3_regression_test.zig");
    _ = @import("cli.zig");
}
