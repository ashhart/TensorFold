//! Living Weights (GLM-5.3-Flash): every host-only file's tests in one binary, run by `zig build test` on every OS.
test {
    _ = @import("families/glm/lw_math.zig");
    _ = @import("families/glm/lw_sites.zig");
    _ = @import("families/glm/lw_learner.zig");
}
