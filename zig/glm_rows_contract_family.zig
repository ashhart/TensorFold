//! Shared GLM owners for recording-only contract tests.
pub const glm = struct {
    pub const wide = @import("src/families/glm/wide.zig");
    pub const affine = @import("src/families/glm/affine.zig");
};
