//! Nemotron 3.5 Transcribe's audio encoder (the checkpoint's `perception.*`) on CUDA, and the WAV files it reads.
pub const Config = @import("config.zig").Config;
pub const Encoder = @import("encoder.zig").Encoder;
pub const Dump = @import("encoder.zig").Dump;
pub const Kind = @import("encoder.zig").Kind;
pub const wav = @import("wav.zig");
pub const kernels = @import("kernels.zig");

test {
    _ = @import("config.zig");
    _ = @import("wav.zig");
    _ = @import("speaker.zig");
    _ = @import("weights.zig");
    _ = @import("kernels.zig");
    _ = @import("encoder.zig");
}
