//! FlashNext metadata contracts run on a CPU without Metal or tensor-runtime dependencies.
test {
    _ = @import("src/families/flashnext/host_test.zig");
}
