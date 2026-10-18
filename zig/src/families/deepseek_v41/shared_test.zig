const std = @import("std");
test {
    inline for (.{ @import("config.zig"), @import("pack.zig"), @import("plan.zig"), @import("stage.zig"), @import("named.zig"), @import("prepare.zig"), @import("prepared.zig"), @import("load.zig"), @import("model.zig"), @import("host_test.zig"), @import("boundary_test.zig") }) |m| std.testing.refAllDecls(m);
    _ = @import("core").exl3_layout;
    _ = @import("core").exl3_load;
}
