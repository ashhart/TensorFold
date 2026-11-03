//! The Qwen-VL tower's shard headers read through core's length and entry checks, without Metal.
test {
    _ = @import("src/core/qwen_vl/shards.zig");
}
