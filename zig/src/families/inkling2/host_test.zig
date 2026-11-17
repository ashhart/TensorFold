//! Host-only Inkling 2 family package: config, weight map, layer plan, refusals, chat template and tokenizer. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("header.zig");
    _ = @import("header_test.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("plan.zig");
    _ = @import("prompt.zig");
    _ = @import("refusal.zig");
    _ = @import("refusal_test.zig");
    _ = @import("template_test.zig");
    _ = @import("tokenizer_test.zig");
}
