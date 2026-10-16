//! Fish Audio Drama 3 TTS on Metal: drama3.zig's family over TensorFold's own Metal runtime, for drama3-generate.
pub const config = @import("config.zig");
pub const prompt = @import("prompt.zig");
pub const wav = @import("wav.zig");
pub const common = @import("common.zig");
pub const shards = @import("shards.zig");
pub const gpu = @import("metal/gpu.zig");
pub const kernels = @import("metal/kernels.zig");
pub const weights = @import("metal/weights.zig");
pub const engine = @import("metal/engine.zig");
pub const codec = @import("metal/codec.zig");
pub const tts = @import("metal/tts.zig");

pub const Engine = engine.Engine;
pub const Sampling = engine.Sampling;
pub const Options = engine.Options;

test {
    _ = config;
    _ = prompt;
    _ = wav;
    _ = kernels;
}
