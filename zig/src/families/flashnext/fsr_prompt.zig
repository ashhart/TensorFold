//! Cold storage uses the existing prompt chunk hooks with one explicitly owned command stream per chunk.
const fz = @import("replay.zig");
const segments = @import("../../core/segments.zig");

pub fn run(m: *fz.Model, segs: []fz.Prompt.Seg) !void {
    if (segs.len != 1) return error.FsrStaggeredChunksUnsupported;
    const r = m.r;
    const cb = r.queue.commandBuffer();
    r.begin(cb, if (r.serial) .serial else .concurrent);
    errdefer r.abort();
    var lane: segments.Lane = .{ .k = 0, .cb = cb, .enc = r.enc, .fence = undefined };
    var hooks: fz.Prompt.Hooks = .{ .m = m, .segs = segs };
    try hooks.begin(&lane);
    const s = &segs[0];
    for (0..fz.LAYERS) |i| {
        try fz.Prompt.segPre(s, m, i);
        try fz.Prompt.segMixer(s, m, i);
        try fz.Prompt.segPost(s, m, i);
    }
    lane.enc = r.enc;
    try hooks.finish(&lane);
    try m.finish(cb);
}
