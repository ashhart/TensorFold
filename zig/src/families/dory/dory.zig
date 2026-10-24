//! Nemotron-Labs Dory 1.0 (a looped nGPT stack) on CUDA: config, weight map, sublayer schedule and memory.

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const names = @import("names.zig");
pub const schedule = @import("schedule.zig");
pub const memory = @import("memory.zig");

test {
    _ = @import("config.zig");
    _ = @import("names.zig");
    _ = @import("schedule.zig");
    _ = @import("memory.zig");
}
