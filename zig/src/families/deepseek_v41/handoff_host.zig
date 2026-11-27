//! Module root of the host-only hand-off checks: the library, its protocol-2 adapter and their tests.
pub const handoff = @import("handoff/handoff.zig");
pub const p2 = @import("handoff/p2.zig");

test {
    _ = @import("handoff/tests.zig");
}
