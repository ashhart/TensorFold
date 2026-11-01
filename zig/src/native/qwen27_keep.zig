//! Slot zero's target keep, accepted taps and last logits share one owned submission.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lb = @import("qwen27_lanes.zig");
const contract = tf.tree_round;
const submit = @import("qwen27_keep_submit.zig");

pub fn commit(b: *lb.Metal, d: *lb.Draft, path: []const u32) !void {
    const g = &d.generation;
    std.debug.assert(g.keep == null);
    errdefer b.runner.failed = true;
    if (path.len == 0 or path.len > b.rows or path.len > contract.max_rows) return error.NothingToKeep;
    const result: *contract.Result = @ptrCast(@alignCast(g.matcher.buffers[4].contents()));
    result.* = .{ .status = 0, .stop = 0, .nonfinite = 0, .consumed_count = @intCast(path.len), .emitted_count = @intCast(path.len), .matched_count = 0, .bonus_emitted = 0, .pending_valid = 0, .pending_token = 0, .reserved = 0, .path = @splat(0), .tokens = @splat(0) };
    @memcpy(result.path[0..path.len], path);
    try b.runner.taps.complete();
    var c = Commit{ .b = b, .d = d, .path = path };
    try submit.run(&c, submit.prefix(b.prefetched[0..b.prefetched_count], path));
}
const Commit = struct {
    b: *lb.Metal,
    d: *lb.Draft,
    path: []const u32,
    ready: bool = false,
    pub fn encode(c: *Commit, ready: bool) !mtl.CommandBuffer {
        c.ready = ready;
        const b = c.b;
        if (ready) if (b.prepared_keep) |cb| {
            try b.runner.checkDeviceKeep(&b.round.?);
            return cb;
        };
        const g = &c.d.generation;
        const r = b.runner;
        const ref = tf.tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
        const logits = r.model.frame.get(.logits);
        const vocab: u32 = @intCast(r.model.config.vocab);
        const rows: u32 = @intCast(b.rows);
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        var ended = false;
        errdefer if (!ended) e.end();
        try r.keepDevice(&b.round.?, e, g.commit, ref);
        if (!ready) try g.commit.taps(e, ref, .{ .buf = r.taps.buffer }, .{ .buf = g.taps }, .{ .rows = rows, .width = r.taps.width, .capacity = r.taps.capacity, .planes = r.taps.count });
        if (!ready) try g.commit.head(e, ref, .{ .buf = logits.buffer, .off = logits.offset }, .{ .rows = rows, .width = vocab, .stride = vocab });
        e.end();
        ended = true;
        return cb;
    }
    pub fn check(c: *Commit, cb: mtl.CommandBuffer) !void {
        if (q.decode_round.gpuFailed(cb, "slot zero's keep")) {
            c.b.runner.failed = true;
            return error.KeepGpuFailure;
        }
    }
    pub fn hold(c: *Commit, cb: mtl.CommandBuffer) void {
        c.d.generation.keep = cb.retain();
    }
    pub fn drain(c: *Commit) !void {
        return c.d.generation.settle();
    }
    pub fn publish(c: *Commit) !void {
        if (c.ready) {
            const logits = c.b.runner.model.frame.get(.logits);
            const width = c.b.runner.model.config.vocab;
            const words: [*]u16 = @ptrCast(@alignCast(logits.buffer.contents() + logits.offset));
            try submit.restoreHead(words[0 .. c.b.rows * width], width, c.path[c.path.len - 1]);
        }
        try c.b.runner.completeDeviceKeep(&c.b.round.?, c.path);
        try c.d.generation.draft.absorb(.{ .buf = c.d.generation.taps }, @intCast(c.path.len));
        if (c.d.generation.draft.committed_end != c.b.runner.offsets[0]) return error.BadDraftPosition;
    }
};

/// Called after the target head's barrier, before the verify's last command buffer commits.
pub fn encodeTail(ptr: *anyopaque, e: mtl.ComputeEncoder) !void {
    const b = lb.Metal.self(ptr);
    const g = &b.draft.?.generation;
    const r = b.runner;
    std.debug.assert(g.keep == null);
    const logits = r.model.frame.get(.logits);
    const vocab: u32 = @intCast(r.model.config.vocab);
    const shape = try g.matcher.encode(e, .{ .buf = logits.buffer, .off = logits.offset }, b.round.?.ids, b.round.?.windows[0].parents, vocab, contract.max_rows, &.{});
    const ref = tf.tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
    try r.taps.complete();
    try g.commit.taps(e, ref, .{ .buf = r.taps.buffer }, .{ .buf = g.taps }, .{ .rows = shape.rows, .width = r.taps.width, .capacity = r.taps.capacity, .planes = r.taps.count });
    const metric = mtl.pass_metrics.phase(.keep);
    defer metric.end();
    std.debug.assert(b.prepared_keep == null);
    const cb = r.model.queue.commandBuffer();
    const keep = cb.compute(.concurrent);
    var ended = false;
    errdefer if (!ended) keep.end();
    try r.prepareDeviceKeep(&b.round.?, keep, g.commit, ref);
    keep.end();
    ended = true;
    b.prepared_keep = cb.retain();
    b.prefetch_encoded = true;
}

/// The existing verify wait completed both the target rows and their dependent picks and taps.
pub fn finishPrefetch(b: *lb.Metal) !void {
    errdefer b.runner.failed = true;
    if (!b.prefetch_encoded) return error.MissingVerifyPicks;
    const g = &b.draft.?.generation;
    const shape = contract.Match{ .rows = @intCast(b.rows), .vocab = @intCast(b.runner.model.config.vocab), .budget = contract.max_rows };
    const accepted = try g.matcher.read(shape);
    @memcpy(b.prefetched[0..accepted.path.len], accepted.path);
    b.prefetched_count = accepted.path.len;
    @memcpy(b.picks.contents()[0 .. b.rows * @sizeOf(contract.Pick)], g.matcher.buffers[2].contents()[0 .. b.rows * @sizeOf(contract.Pick)]);
}
