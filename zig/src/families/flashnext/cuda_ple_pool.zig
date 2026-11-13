//! The CUDA reader's name for the parked host workers; the pool lives in host_pool.zig, which the HIP reader shares.
pub const Pool = @import("host_pool.zig").Pool;
pub const Task = @import("host_pool.zig").Task;
pub const max_parts = @import("host_pool.zig").max_parts;

test {
    _ = @import("host_pool.zig");
}
