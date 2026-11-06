//! File-only Nemotron lesson and pocket fixtures; no kernel module or device checks are imported.
test {
    _ = @import("src/families/nemotron/living_test.zig");
    _ = @import("src/families/nemotron/pocket_shards_test.zig");
}
