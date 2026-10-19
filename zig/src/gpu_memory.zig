//! Shared admission equation for optional CUDA instances and resource qualification.
pub fn fits(static: u32, dynamic: u32, limit: u32) bool {
    return static <= limit and dynamic <= limit - static;
}
