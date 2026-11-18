//! Host-only Hy4-flash family package: config, layer plan, weight map, refusals, chat template and tokenizer. No GPU.
test {
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("plan.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("refusal.zig");
    _ = @import("template_test.zig");
    _ = @import("tokenizer_test.zig");
}
