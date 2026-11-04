//! LFM3 on the Intel GPU (XPU backend): the forward and the config it reads.
pub const model = @import("xpu_model.zig");
pub const engine = @import("xpu_engine.zig");
pub const vision = @import("xpu_vision.zig");
pub const config = @import("config.zig");
