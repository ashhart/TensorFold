//! Ling CPU exports: the checked GGUF source reader and its exact Q4_K, Q6_K and Q5_0 projections.
test {
    _ = @import("cpu_gguf_views.zig");
    _ = @import("gguf_names.zig");
}
