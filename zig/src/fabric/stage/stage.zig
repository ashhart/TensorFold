//! The pipeline stage link: a TCP ring of ranks carrying each round's plan and hidden rows, on Linux and macOS.
pub const net = @import("net.zig");
pub const frame = @import("frame.zig");
pub const ring = @import("ring.zig");
pub const config = @import("config.zig");
pub const secret = @import("secret.zig");

pub const Ring = ring.Ring;
pub const Options = ring.Options;
pub const Peer = ring.Peer;

test {
    _ = net;
    _ = frame;
    _ = ring;
    _ = config;
    _ = secret;
}
