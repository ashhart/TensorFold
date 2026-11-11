const std = @import("std");

test {
    std.testing.refAllDecls(@import("geometry.zig"));
    std.testing.refAllDecls(@import("codec.zig"));
    std.testing.refAllDecls(@import("fp4.zig"));
    std.testing.refAllDecls(@import("plan.zig"));
    std.testing.refAllDecls(@import("export.zig"));
    std.testing.refAllDecls(@import("export_metal.zig"));
    std.testing.refAllDecls(@import("metal.zig"));
    std.testing.refAllDecls(@import("import.zig"));
    std.testing.refAllDecls(@import("p2.zig"));
    std.testing.refAllDecls(@import("blob.zig"));
    std.testing.refAllDecls(@import("landing.zig"));
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
test {
    _ = @import("roundtrip_test.zig");
}
test {
    _ = @import("p2_test.zig");
}
test {
    _ = @import("guard_test.zig");
}
test {
    _ = @import("export_metal_test.zig");
}
test {
    _ = @import("landing_test.zig");
}
test {
    _ = @import("shards_test.zig");
}
