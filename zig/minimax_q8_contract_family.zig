//! The MiniMax contract's existing family types over inert host Metal buffers.
pub const frags = @import("src/core/frags.zig");
pub const row_projection = @import("src/core/row_projection.zig");
pub const minimax_m3 = struct {
    pub const config = @import("src/families/minimax_m3/config.zig");
    pub const state = @import("src/families/minimax_m3/state.zig");
};
