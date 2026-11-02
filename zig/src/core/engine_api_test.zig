//! Host behavior checks for the shared engine API.
test {
    _ = @import("lane_embedding_test.zig");
    _ = @import("speech_guard.zig");
    _ = @import("lane_host.zig");
    _ = @import("load_ledger.zig");
    _ = @import("lane_host_test.zig");
    _ = @import("lane_root_copy_test.zig");
    _ = @import("lane_rounds_test.zig");
    _ = @import("lane_halt_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prompt_radix.zig");
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
    _ = @import("originals_test.zig");
}
