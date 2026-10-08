//! Nemotron-H on an Intel GPU in Zig (Level Zero, OpenCL C): plain decode, 1..16-row windows, prefill GEMMs, MTP.

pub const Config = @import("config.zig").Config;
pub const cfg = @import("xpu_config.zig");
pub const model = @import("xpu_model.zig");
pub const win = @import("xpu_win.zig");
pub const pf = @import("xpu_pf.zig");
pub const mtp = @import("xpu_mtp.zig");
pub const engine = @import("xpu_engine.zig");
pub const Engine = engine.Engine;

test {
    _ = @import("config.zig");
    _ = cfg;
}
