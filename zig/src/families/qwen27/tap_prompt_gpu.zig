//! The one prompt tap copy, completed before a snapshot or a drafter reads the ring.
const std = @import("std");
const mtl = @import("metal");
const policy = @import("tap_prompt.zig");
const Ring = @import("tap_ring.zig").Ring;
const Taps = @import("taps.zig").Taps;
const Context = struct {
    taps: Taps,
    ring: Ring,
    pipeline: mtl.Pipeline,
    queue: mtl.Queue,
    pub fn write(c: @This(), args: policy.Args) !void {
        const pipe = c.pipeline;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = c.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(c.taps.buffer, 0, 0);
        e.setValue(args, 1);
        e.setBuffer(c.ring.buf, 0, 2);
        e.dispatchThreads(mtl.Size.of(args.width * args.planes, args.rows, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            if (!@import("builtin").is_test) std.log.err("prompt tap copy: {s}", .{why});
            return error.DraftTapGpuFailure;
        }
    }
};
pub fn absorb(pipeline: mtl.Pipeline, queue: mtl.Queue, taps: Taps, ring: Ring, rows: u32) !void {
    try policy.absorb(Context{ .taps = taps, .ring = ring, .pipeline = pipeline, .queue = queue }, rows);
}
