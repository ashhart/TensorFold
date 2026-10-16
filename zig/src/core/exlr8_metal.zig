//! EXLR8 on Metal as one module: the reader and decoder, the trellis projections and the row-stored dense kernels.

pub const ex = @import("exlr8.zig");
pub const projection = @import("exlr8_projection.zig");
pub const dense_gpu = @import("exlr8_dense_gpu.zig");

test {
    _ = projection;
    _ = dense_gpu;
}
