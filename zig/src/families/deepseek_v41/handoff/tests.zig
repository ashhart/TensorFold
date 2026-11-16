const std = @import("std");

test {
    std.testing.refAllDecls(@import("geometry.zig"));
    std.testing.refAllDecls(@import("codec.zig"));
    std.testing.refAllDecls(@import("fp4.zig"));
    std.testing.refAllDecls(@import("plan.zig"));
    std.testing.refAllDecls(@import("export.zig"));
}
test {
    std.testing.refAllDecls(@import("synth.zig"));
}
test {
    _ = @import("codec_test.zig");
}
test {
    _ = @import("fp4_test.zig");
}
test {
    _ = @import("plan_test.zig");
}
