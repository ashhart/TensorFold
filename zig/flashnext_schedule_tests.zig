//! Flash Next's production prompt driver and shared lane host behind recording device boundaries.
test {
    _ = @import("src/families/flashnext/prefill_test.zig");
    _ = @import("src/families/flashnext/prefill.zig");
    _ = @import("src/families/flashnext/prefill_host_test.zig");
}
