//! Candidate storage boundaries shared by allocation, gather and host readback.
pub const Layout = struct {
    device: usize,
    host: usize,
    read: usize,

    pub fn init(world: usize, rows: usize, width: usize) Layout {
        return .{ .device = world * rows * width, .host = world * rows * width, .read = world * rows * width };
    }
};
