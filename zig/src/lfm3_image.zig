//! LFM3-VL's image steps (#1502's image.zig over core/picture) as a module rooted where its relative imports reach.
pub const image = @import("families/lfm3/image.zig");
pub const picture = @import("core/picture/picture.zig");
