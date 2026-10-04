//! FlashNext host admission is separate from unqualified device arithmetic and engine entrypoints.
pub const Config = @import("config.zig").Config;
pub const config = @import("config.zig");
pub const affine = @import("affine.zig");
pub const weights = @import("weights.zig");
pub const index = @import("index.zig");
