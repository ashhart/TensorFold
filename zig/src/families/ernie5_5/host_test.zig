//! Host-only ERNIE 5.5 tests: config, names, prompts, tokenizer, MultiMax, key lists and mHC. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("plan.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("prompt.zig");
    _ = @import("template_test.zig");
    _ = @import("tokenizer_test.zig");
    _ = @import("multimax_test.zig");
    _ = @import("window_test.zig");
    _ = @import("mhc_test.zig");
}
