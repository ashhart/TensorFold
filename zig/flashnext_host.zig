//! FlashNext metadata contracts run on a CPU without Metal or tensor-runtime dependencies.
test {
    _ = @import("src/families/flashnext/host_test.zig");
    _ = @import("src/families/flashnext/pack.zig");
    _ = @import("src/families/flashnext/batch_meta_test.zig");
}
