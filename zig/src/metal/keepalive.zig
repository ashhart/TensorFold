//! The idle keepalive's Metal target: one tiny command buffer on the engine's queue, touching its residency sets.
const std = @import("std");
pub const Target = TargetFor(@import("metal.zig"));

/// A queue and the residency sets one commit keeps wired. The engine owns both; this only reads them.
pub fn TargetFor(comptime Metal: type) type {
    return struct {
        const Self = @This();
        queue: Metal.Queue,
        sets: []const Metal.ResidencySet = &.{},
        scratch: ?Metal.Buffer = null, // a few bytes each tick copies within, so its command buffer runs on the GPU
        flush: ?Flush = null, // the sets' waiting changes committed first: while idle, never inside a request

        pub const Flush = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void };

        /// The core ticker's tick: one command buffer that touches every set and commits.
        pub fn tick(ctx: *anyopaque) void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const pool = Metal.objc.Pool.push();
            defer pool.pop();
            if (t.flush) |f| f.run(f.ctx);
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
}
