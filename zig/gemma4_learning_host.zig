//! Gemma learner host instruments own a separate source namespace from either native family specialization.
test {
    _ = @import("src/families/gemma4/learning_math_test.zig");
    _ = @import("src/families/gemma4/learning_stress_test.zig");
    _ = @import("src/families/gemma4/learning_chain_test.zig");
    _ = @import("src/families/gemma4/learning_runtime_test.zig");
    _ = @import("src/families/gemma4/learning_attention_test.zig");
    _ = @import("src/families/gemma4/learning_train_test.zig");
    _ = @import("src/families/gemma4/learning_save_test.zig");
}
