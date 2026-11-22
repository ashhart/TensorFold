//! Host-only Inkling 2 family package: config, names, plan, state bytes, admission, template, tokenizer.
test {
    _ = @import("admit.zig");
    _ = @import("admit_test.zig");
    _ = @import("config.zig");
    _ = @import("config_test.zig");
    _ = @import("names.zig");
    _ = @import("names_test.zig");
    _ = @import("plan.zig");
    _ = @import("prompt.zig");
    _ = @import("state_layout.zig");
    _ = @import("template_test.zig");
    _ = @import("tokenizer_test.zig");
}
