//! LFM3-Omni-2B on Metal, with the module surface generate.zig and serve.zig build against on either backend.
pub const config = @import("config.zig");
pub const mel = @import("mel.zig");
pub const common = @import("common.zig");
pub const spec = @import("spec.zig");
pub const stack = @import("metal/stack.zig");
pub const kernels = @import("metal/kernels.zig");
pub const engine = @import("metal/engine.zig");
pub const prompt = @import("prompt.zig");
pub const serve = @import("serve.zig");
pub const wav = @import("wav");

pub const Engine = engine.Engine;

test {
    _ = mel;
    _ = prompt;
    _ = common;
    _ = kernels;
}
