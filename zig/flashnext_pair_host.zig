//! Pair handoffs and their failure paths run against a host recording device.
const relay = @import("flashnext_relay_tests.zig");

test {
    _ = relay;
    _ = @import("flashnext_handoff_shape_tests.zig");
    _ = @import("src/families/flashnext/tp_source_test.zig");
}
