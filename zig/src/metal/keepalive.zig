//! The idle keepalive's Metal target: one tiny command buffer on the engine's queue, touching its residency sets.
const std = @import("std");
const command = @import("command.zig");
const residency = @import("residency.zig");
const Buffer = @import("device.zig").Buffer;

/// A queue and the residency sets one commit keeps wired. The engine owns both; this only reads them.
pub const Target = struct {
    queue: command.Queue,
    sets: []const residency.ResidencySet = &.{},
    scratch: ?Buffer = null, // a few bytes each tick copies within, so its command buffer runs on the GPU

    /// The core ticker's tick: one command buffer that touches every set and commits.
    pub fn tick(ctx: *anyopaque) void {
        const t: *Target = @ptrCast(@alignCast(ctx));
        const pool = @import("objc.zig").Pool.push();
        defer pool.pop();
        const cmd = t.queue.commandBuffer();
        for (t.sets) |set| cmd.useResidencySet(set);
        if (t.scratch) |b| { // an empty command buffer need not reach the GPU, and then its sets stay as they were
            const blit = cmd.blit();
            blit.copy(b, 0, b, 32, 16);
            blit.end();
        }
        cmd.commit();
    }
};
