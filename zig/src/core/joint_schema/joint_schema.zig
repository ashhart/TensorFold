//! Clef's joint schema head on the host: its config, tensors and loader, and the FP64 reference the device checks.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const Problem = config.Problem;
pub const tensors = @import("tensors.zig");
pub const load = @import("load.zig");
pub const reference = @import("reference.zig");
pub const tiny = @import("tiny.zig");

test {
    _ = config;
    _ = tensors;
    _ = load;
    _ = reference;
    _ = @import("reference_attn.zig");
    _ = @import("load_test.zig");
    _ = @import("reference_test.zig");
}
