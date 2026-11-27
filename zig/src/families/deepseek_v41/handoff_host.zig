//! Module root of the host-only hand-off checks: the library and its tests.
pub const handoff = @import("handoff/handoff.zig");

test {
    _ = @import("handoff/tests.zig");
}
