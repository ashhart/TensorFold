//! Import the parser from a root that also admits the common quantization module.
test {
    _ = @import("src/families/gemma4/config.zig");
    _ = @import("src/families/gemma4/plan.zig");
    _ = @import("src/families/gemma4/layer_input.zig");
}
