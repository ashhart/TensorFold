//! Inkling 2 Mini Metal host tests rooted with the production engine sources.

test {
    _ = @import("families/inkling2/metal/host_test.zig");
    _ = @import("inkling2_native.zig");
}
