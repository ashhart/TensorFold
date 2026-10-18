//! Actual modules needed by the shader-free compile check of the GPU gather gate.
pub const glm = struct {
    pub const wide = @import("families/glm/wide.zig");
    pub const affine = @import("families/glm/affine.zig");
};
pub const row_projection = @import("core/row_projection.zig");
pub const affine_mm = @import("core/affine_mm.zig");
pub const quant = @import("core/quant/quant.zig");
