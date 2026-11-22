//! The main forward owns the scan; host launch recorders use the same implementation.
pub const deltaScan = @import("cuda_forward.zig").deltaScan;
