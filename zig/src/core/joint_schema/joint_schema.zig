//! Clef's joint schema head, backend-neutral: config, tensors, loader, device plan, and the FP64 reference.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const Problem = config.Problem;
pub const tensors = @import("tensors.zig");
pub const load = @import("load.zig");
pub const reference = @import("reference.zig");
pub const tiny = @import("tiny.zig");
pub const plan = @import("plan.zig");
pub const device = @import("device.zig");

test {
    _ = config;
    _ = tensors;
    _ = load;
    _ = reference;
    _ = plan;
    _ = device;
    _ = @import("reference_attn.zig");
    _ = @import("load_test.zig");
    _ = @import("reference_test.zig");
}
