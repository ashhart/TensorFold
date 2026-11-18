//! What a server needs to turn a speech request into ZONOS3 Mini's prompt rows, with no GPU code behind it.
pub const config = @import("config.zig");
pub const prompt = @import("prompt.zig");
pub const release = @import("release.zig");
pub const voices = @import("voices.zig");
pub const steering = @import("steering.zig");
pub const steer_bundle = @import("steer_bundle.zig");

/// The release's sampling defaults (registries/generation.json).
pub const temperature: f32 = 0.75;
pub const top_p: f32 = 0.69;
pub const top_k: u32 = 80;
/// Silence frames prefilled after the text: the reference servers' default.
pub const silence_frames = 3;
/// The longest reference prefix: 60 s of the codec's 15 frames a second.
pub const reference_frames = 900;

test {
    _ = prompt;
    _ = release;
    _ = voices;
    _ = steering;
    _ = steer_bundle;
}
