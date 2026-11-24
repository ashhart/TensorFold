//! Inkling 2 Mini Metal host tests rooted with the production engine sources.
const serving = if (@import("builtin").os.tag.isDarwin()) @import("native/inkling2_serving_test.zig") else struct {};

test {
    _ = @import("families/inkling2/metal/host_test.zig");
    _ = @import("inkling2_native.zig");
    _ = @import("native/inkling2_admit_test.zig");
    _ = serving;
}
