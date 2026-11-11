//! The native verify's tail and row reads, driven by the recording-tested submission order.
const mtl = @import("metal");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const be = lanes.backend;
const lb = @import("qwen27_lanes.zig");

pub const Submission = struct {
    b: *lb.Metal,
    windows: []const be.Window,
    out: []be.Verified,

    pub fn runner(s: *Submission) *tf.qwen27.decode_round.Runner {
        return s.b.runner;
    }
    pub fn tail(s: *Submission, picks: bool) tf.qwen27.forward.Tail {
        return .{ .ptr = s.b, .encode = if (picks) lb.Metal.encodePicksTail else @import("qwen27_keep.zig").encodeTail };
    }
    pub fn verify(s: *Submission, plain: bool) !void {
        const b = s.b;
        const before = b.runner.gpu_seconds;
        if (plain) try b.runner.advance(&b.round.?, .all) else try b.runner.verifyHead(&b.round.?, .all);
        for (s.windows) |w| @import("qwen27_slots.zig").charge(b, w.stream, before);
        b.advanced = plain;
    }
    pub fn prefetch(s: *Submission) !void {
        try @import("qwen27_keep.zig").finishPrefetch(s.b);
    }
    pub fn argmax(s: *Submission) !void {
        try s.b.argmaxRows(0, s.b.rows);
    }
    pub fn read(s: *Submission) !void {
        const b = s.b;
        for (s.windows, s.out, b.round.?.firsts) |w, o, first| {
            const rows = w.rows();
            if (w.stream.sampling != null or w.masks.len > 0) try b.draw(w.stream, first, rows, w.positions, o.sampled, w.masks) else {
                const metric = mtl.pass_metrics.phase(.sampling);
                defer metric.end();
                try b.picked(first, rows, o.sampled);
            }
            @memcpy(o.drafts, b.ids[first + 1 ..][0 .. rows - 1]);
        }
    }
};
