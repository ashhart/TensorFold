//! Fish Audio Drama 3 TTS (fish_omni: the Qwen3.5-35B-A3B backbone and the codebook tower) on one CUDA GPU.
pub const completion = @import("completion.zig");
pub const speech_decode = @import("speech_decode.zig");
pub const model = @import("model.zig");
pub const kv_state = @import("kv_state.zig");
pub const reference_cache = @import("reference_cache.zig");
pub const codec_load = @import("codec_load.zig");
pub const speed_probe = @import("cuda/speed_probe.zig");
pub const replay = @import("replay.zig").Replay(@This());
pub const config = @import("config.zig");
pub const weights = @import("cuda/weights.zig");
pub const kernels = @import("cuda/kernels.zig");
pub const engine = @import("cuda/engine.zig");
pub const prompt = @import("prompt.zig");
pub const codec = @import("cuda/codec.zig");
pub const wav = @import("wav.zig");
pub const tts = @import("cuda/tts.zig");
pub const common = @import("common.zig");
pub const shards = @import("shards.zig");

pub const Engine = engine.Engine;
pub const Sampling = engine.Sampling;
pub const Options = engine.Options;

test {
    _ = common;
    _ = completion;
    _ = speech_decode;
    _ = @import("replay_test.zig");
    _ = config;
    _ = speed_probe;
    _ = model;
    _ = reference_cache;
    _ = kv_state;
    _ = @import("codec_map.zig");
    _ = @import("codec_schema.zig");
    _ = @import("cuda/stager.zig");
    _ = @import("codec_pth.zig");
    _ = prompt;
    _ = wav;
}

test {
    _ = @import("cuda/owner_test.zig");
}
