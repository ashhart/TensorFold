//! This opt-in diagnostic keeps raw quantized directional failures visible outside the local mathematical suite.
const family = @import("qwen4").family;
test "Qwen4 raw directional: unchanged rule v2 over all six native BF16 cells" {
    try family.learning_raw_direction.run();
}
