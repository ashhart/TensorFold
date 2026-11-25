//! Host-only ERNIE 5.5 family package: config, weight map, layer plan, chat encoding and tokenizer. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("plan.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("prompt.zig");
    _ = @import("template_test.zig");
    _ = @import("tokenizer_test.zig");
}
