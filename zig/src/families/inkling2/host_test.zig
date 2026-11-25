//! Host-only Inkling 2 family package: config, weight map, layer plan, chat template and tokenizer. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("exl3_config.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("plan.zig");
    _ = @import("loop.zig");
    _ = @import("loop_model.zig");
    _ = @import("loop_model_test.zig");
    _ = @import("prompt.zig");
    _ = @import("template_test.zig");
    _ = @import("tokenizer_test.zig");
}
