//! Fish Audio Drama 3 TTS on Metal: drama3.zig's family over TensorFold's own Metal runtime, for drama3-generate.
pub const completion = @import("completion.zig");
pub const speech_decode = @import("speech_decode.zig");
pub const model = @import("model.zig");
pub const kv_state = @import("kv_state.zig");
pub const reference_cache = @import("reference_cache.zig");
pub const codec_load = @import("codec_load.zig");
pub const replay = @import("replay.zig").Replay(@This());
pub const config = @import("config.zig");
pub const prompt = @import("prompt.zig");
pub const wav = @import("wav.zig");
pub const common = @import("common.zig");
pub const shards = @import("shards.zig");
pub const gpu = @import("metal/gpu.zig");
pub const kernels = @import("metal/kernels.zig");
pub const weights = @import("metal/weights.zig");
pub const load_limit = @import("metal/load_limit.zig");
pub const engine = @import("metal/engine.zig");
pub const guidance = @import("metal/guidance.zig");
pub const quant_check = @import("metal/quant_check.zig");
pub const codec = @import("metal/codec.zig");
pub const tts = @import("metal/tts.zig");

pub const Engine = engine.Engine;
pub const Snapshot = engine.Engine.Snapshot;
pub const Sampling = engine.Sampling;
pub const Options = engine.Options;

test {
    _ = completion;
    _ = speech_decode;
    _ = @import("replay_test.zig");
    _ = config;
    _ = model;
    _ = kv_state;
    _ = reference_cache;
    _ = @import("codec_map.zig");
    _ = @import("codec_schema.zig");
    _ = @import("codec_pth.zig");
    _ = prompt;
    _ = wav;
    _ = kernels;
    _ = load_limit;
    _ = guidance;
    _ = @import("metal/quant_format_test.zig");
}

pub const residency = @import("metal/residency.zig");
