//! The pipeline stage link: a TCP ring of ranks carrying each round's plan and hidden rows, and its protocol payloads.
pub const net = @import("net.zig");
pub const frame = @import("frame.zig");
pub const ring = @import("ring.zig");
pub const config = @import("config.zig");
pub const secret = @import("secret.zig");
pub const payload = @import("payload.zig");
pub const boot = @import("boot.zig");
pub const plan = @import("plan.zig");
pub const keepalive = @import("keepalive.zig");
pub const fixtures = @import("fixtures.zig");

pub const Ring = ring.Ring;
pub const Options = ring.Options;
pub const Peer = ring.Peer;

test {
    _ = net;
    _ = frame;
    _ = ring;
    _ = config;
    _ = secret;
    _ = payload;
    _ = boot;
    _ = plan;
    _ = keepalive;
    _ = fixtures;
    _ = @import("protocol_test.zig");
}
