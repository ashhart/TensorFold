//! The layer-cancel poll a prompt pass carries: one sync and the check after each layer, where a pass can stop.

const hip = @import("hip");
const lanes = @import("lanes");
const fwd = @import("../forward/forward.zig");
const prefix = @import("prefix.zig");

/// Asked after each layer of a long prompt (the stream is idle then): true ends the pass with `error.Cancelled`.
pub const Cancel = lanes.stream.CancelCheck;

pub const Poll = struct {
    stream: hip.Stream,
    cancel: Cancel,

    fn layer(ctx: *anyopaque, _: usize, _: hip.ops.Tensor, _: usize) anyerror!void {
        const p: *Poll = @ptrCast(@alignCast(ctx));
        try p.stream.synchronize();
        try @import("hip_serve").cancel.poll(p.cancel);
    }
};

/// The pass over `len` rows polling the cancel after each layer; null where it cannot stop (a short prompt, or tp).
pub fn trace(world: usize, stream: hip.Stream, cancel: ?Cancel, len: usize, p: *Poll) ?fwd.Trace {
    const c = cancel orelse return null;
    if (!prefix.pollsLayers(world, len)) return null;
    p.* = .{ .stream = stream, .cancel = c };
    return .{ .ctx = p, .layer = Poll.layer, .reads = false };
}
