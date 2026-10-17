//! DeepSeek-V4.1's host-only serving side: the exact sampler (zig/src/server).
pub const sampling = @import("sampling.zig");
/// The committed goldens the serving tests read.
pub const fixtures = @import("dsv41_serve_fixtures");

test {
    @import("std").testing.refAllDecls(@This());
}
