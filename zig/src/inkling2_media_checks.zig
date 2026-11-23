//! Metal vision and bundled DSpark checks for Inkling 2 Mini.
test {
    _ = @import("families/inkling2/metal_vision_test.zig");
    _ = @import("families/inkling2/metal_vision_image_test.zig");
    _ = @import("families/inkling2/metal_vision_real_test.zig");
    _ = @import("families/inkling2/metal_draft_test.zig");
    _ = @import("families/inkling2/metal_draft_ops_test.zig");
    _ = @import("families/inkling2/metal_draft_snapshot_test.zig");
    _ = @import("families/inkling2/metal_draft_real_test.zig");
    _ = @import("families/inkling2/metal_state.zig");
}
