//! Module root of the host-only hand-off checks: library, tests, protocol-2 adapter and the CUDA export glue.
pub const handoff = @import("handoff/handoff.zig");
pub const p2 = @import("handoff/p2.zig");
pub const cuda_export = @import("cuda_export.zig");
pub const cuda_export_serve = @import("cuda_export_serve.zig");
pub const file = @import("handoff_file.zig");

test {
    _ = @import("handoff/tests.zig");
    _ = @import("cuda_export_test.zig");
    _ = @import("handoff_file_test.zig");
    _ = @import("handoff_http_test.zig");
    _ = @import("cuda_export_serve_test.zig");
}
