//! Pure CPU learned-disk planning, paired reservation and checksummed GLM-half admission tests.
test {
    _ = @import("core/learned_plan.zig");
    _ = @import("core/learned_fault_test.zig");
    _ = @import("core/learned_part_test.zig");
    _ = @import("core/learned_index_orphan_test.zig");
    _ = @import("core/prompt_imprint_fault_test.zig");
    _ = @import("lanes").learned_dirs;
    _ = @import("lanes").learned_disk;
    _ = @import("core/prompt_cache.zig");
    _ = @import("families/glm/learned_pair.zig");
    _ = @import("families/glm/snapshot_file.zig");
}
