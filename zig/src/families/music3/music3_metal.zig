//! MiniMax Music 3 on Metal: music3.zig's family over TensorFold's own Metal runtime, for music3-generate.
pub const backend = "metal";
pub const config = @import("config.zig");
pub const common = @import("common.zig");
pub const prompt = @import("prompt.zig");
pub const weights = @import("weights.zig");
pub const npy = @import("npy.zig");
pub const wav = @import("wav.zig");
pub const parity = @import("parity.zig");
pub const gpu = @import("metal/gpu.zig");
pub const kernels = @import("metal/kernels.zig");
pub const ar = @import("metal/ar.zig");
pub const dit = @import("metal/dit.zig");
pub const dav = @import("metal/dav.zig");
pub const engine = @import("metal/engine.zig");

pub const Engine = engine.Engine;
pub const Options = engine.Options;
pub const Song = engine.Song;

test {
    _ = config;
    _ = prompt;
    _ = wav;
    _ = npy;
    _ = kernels;
}
