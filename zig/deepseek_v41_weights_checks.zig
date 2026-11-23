//! Host checks for V4.1 configuration, Engram table plans and resident weight staging; no kernels compiled.
test {
    _ = @import("src/families/deepseek_v41/config.zig");
    _ = @import("src/families/deepseek_v41/view.zig");
    _ = @import("src/families/deepseek_v41/engram_hash.zig");
    _ = @import("src/families/deepseek_v41/load_metal.zig");
    _ = @import("src/families/deepseek_v41/weights.zig");
}
