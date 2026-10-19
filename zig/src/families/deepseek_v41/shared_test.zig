const std = @import("std");
test {
    _ = @import("engram_host.zig");
    _ = @import("engram_tables.zig");
    _ = @import("engram_shared_test.zig");
    if (@import("builtin").os.tag == .linux) _ = @import("engram_rows_test.zig");
    std.testing.refAllDecls(@import("engram_rows.zig").Rows);
    std.testing.refAllDecls(@import("phases.zig"));
    _ = @import("engram_boundary_test.zig");
    _ = @import("engram_start_test.zig");
    _ = @import("engram_recovery_test.zig");
    inline for (.{ @import("config.zig"), @import("pack.zig"), @import("plan.zig"), @import("stage.zig"), @import("named.zig"), @import("prepare.zig"), @import("prepared.zig"), @import("load.zig"), @import("model.zig"), @import("host_test.zig"), @import("boundary_test.zig") }) |m| std.testing.refAllDecls(m);
    _ = @import("core").exl3_layout;
    _ = @import("core").exl3_load;
}
