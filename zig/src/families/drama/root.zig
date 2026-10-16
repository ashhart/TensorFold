//! Drama 3's native CUDA backbone and temporal codebook tower.
pub const backbone = @import("backbone.zig");
pub const tower = @import("tower.zig");

test {
    _ = backbone;
    _ = tower;
}
