//! FlashNext metadata contracts run on a CPU without Metal or tensor-runtime dependencies.
test {
    _ = @import("src/families/flashnext/host_test.zig");
    _ = @import("src/families/flashnext/pack.zig");
    _ = @import("src/families/flashnext/lane_w.zig");
    _ = @import("src/families/flashnext/rows_w.zig");
    _ = @import("src/families/flashnext/experts4.zig");
    _ = @import("src/families/flashnext/tree.zig");
    _ = @import("src/families/flashnext/engine_widths.zig");
    _ = @import("src/families/flashnext/prompt_w.zig");
    _ = @import("src/families/flashnext/host_sample.zig");
    _ = @import("src/families/flashnext/batch_meta_test.zig");
}
