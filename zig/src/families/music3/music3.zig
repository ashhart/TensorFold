//! MiniMax Music 3 (MiniMax-Music3) on one CUDA GPU: the AR LM and depth decoder, the DiT and the DAV.
pub const config = @import("config.zig");
pub const prompt = @import("prompt.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const ar = @import("ar.zig");
pub const dit = @import("dit.zig");
pub const dav = @import("dav.zig");
pub const engine = @import("engine.zig");
pub const npy = @import("npy.zig");
pub const wav = @import("wav.zig");
pub const parity = @import("parity.zig");

pub const Engine = engine.Engine;
pub const Options = engine.Options;
pub const Song = engine.Song;

test {
    _ = config;
    _ = prompt;
    _ = wav;
    _ = npy;
}
