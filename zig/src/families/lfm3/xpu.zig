//! LFM3 on the Intel GPU (XPU backend): the forward and the config it reads.
pub const model = @import("xpu_model.zig");
pub const engine = @import("xpu_engine.zig");
pub const config = @import("config.zig");
/// The config fixtures, for the XPU open's tests.
pub const fixtures = if (@import("builtin").is_test) @import("config_test.zig") else struct {};
