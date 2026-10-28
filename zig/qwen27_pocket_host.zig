//! Shared CUDA-writer/Metal-reader pocket fixtures without any device or shader imports.
pub const dense_pocket = @import("src/core/dense_pocket.zig");

test {
    _ = @import("src/families/qwen27/config.zig");
    _ = @import("src/families/qwen27/pocket_rows.zig");
    _ = @import("src/families/qwen3_5/pocket_shards_test.zig");
    _ = dense_pocket;
}
