//! The ZONOS3 host's device sizes: what a load, a request at once and a guided request are admitted for.
const z3 = @import("zonos3");
const zdec = @import("zonos_decoders.zig");
const dev = @import("zonos_device.zig");

/// The pinned release's weights as the device holds them: the backbone's bf16 and the keep2_plain decoder's fp32.
pub const weight_bytes: u64 = 1_771_847_424 + zdec.weightBytes(.keep2_plain);
/// The clip encoder's weights, state and workspaces, as loaded (Metal rounds each buffer up to 256 bytes).
pub const clip_bytes: u64 = if (dev.on_metal) 5_643_511_296 else 5_643_510_688;

/// Device bytes a guided request's second stream takes, allocated by the first such request.
pub fn twinBytes(o: z3.Options) u64 {
    return z3.Engine.laneBytes(o);
}

/// Device bytes a load takes with `decoder` as its default: the weights, then each part's own count of its state.
pub fn need(context: u32, frames: u32, decoder: z3.codec.Decoder) u64 {
    const weights = weight_bytes - zdec.weightBytes(.keep2_plain) + zdec.weightBytes(decoder);
    return weights + z3.Engine.stateBytes(.{ .max_ctx = context }) + z3.Codec.stateBytes(.{ .max_frames = frames, .decoder = decoder });
}

/// Device bytes each request at once past the first adds: a backbone lane and a lane of the default decoder.
pub fn laneBytes(context: u32, frames: u32, decoder: z3.codec.Decoder) u64 {
    return z3.Engine.laneBytes(.{ .max_ctx = context }) + z3.Codec.laneBytes(.{ .max_frames = frames, .decoder = decoder });
}
