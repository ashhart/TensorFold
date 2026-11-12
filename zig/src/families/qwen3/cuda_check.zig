//! The Qwen3 CUDA device checks, rooted here where the family's config, names, checks and score are in reach.
test {
    _ = @import("cuda/forward_test.zig");
}
