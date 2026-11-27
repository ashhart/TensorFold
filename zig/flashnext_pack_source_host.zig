//! Source binding regressions through the real pack builder, without Metal or a device.
test {
    _ = @import("src/families/flashnext/pack_source_test.zig");
}
