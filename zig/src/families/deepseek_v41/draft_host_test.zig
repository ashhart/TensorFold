test {
    _ = @import("draft/draft.zig");
    _ = @import("sampling_test.zig");
    _ = @import("dsv41_serve");
    _ = @import("serve_knobs_test.zig");
}

test {
    _ = @import("draft/review_release_test.zig");
}

test {
    _ = @import("serve_lanes.zig");
    _ = @import("serve_lanes_test.zig");
}
