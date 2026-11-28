//! Verification rounds and the active-round fingerprint (decode_round.zig's Runner).
pub const fingerprint = @import("round_fingerprint.zig").fingerprint;
const mtl = @import("metal");
const decode_round = @import("decode_round.zig");
const Runner = decode_round.Runner;
const Round = @import("round_plan.zig").Round;
const forward = @import("forward.zig");
const image_rows = @import("image_rows.zig");
const prompt_tail = @import("prompt_tail.zig");
const stage_planes = @import("stage_planes.zig");

pub fn run(r: *Runner, round: *const Round, head: forward.Head, accepted: bool) !void {
    const metric = mtl.pass_metrics.phase(.verify);
    defer metric.end();
    try r.settleKeep();
    if (r.failed or r.active != null or round.ids.len > r.model.frame.capacity) return error.RoundNotReady;
    mtl.pass_metrics.rows(round.ids.len);
    if (r.tail != null and (if (r.model.config.looped) head == .none else (accepted or head != .all))) return error.BadVerifyTail;
    // A rank's head writes only this rank's vocabulary columns; the tail's matcher reads the whole vocabulary.
    if (r.tail != null and r.reduce != null) return error.TailOnRank;
    for (round.slots, round.windows) |slot, window| if (slot >= r.ready or r.offsets[slot] != window.start) return error.CachePositionDiffers;
    if (stage_planes.on) if (stage_planes.active != null) try stage_planes.admit(round.windows); // refusal is atomic
    for (round.slots, round.windows) |slot, window| try r.cover(slot, window.start + @as(u32, @intCast(window.parents.len)));
    for (round.slots, round.windows) |slot, window| try r.reserve(slot, window.start + @as(u32, @intCast(window.parents.len)));
    const paths = try r.allocator.alloc([]const u32, round.windows.len);
    defer r.allocator.free(paths);
    @memset(paths, &.{});
    var accepted_rows: [@import("gpu_frame.zig").max_rows]u32 = undefined;
    if (accepted) for (round.windows, round.firsts, paths) |window, first, *path| {
        for (window.parents, 0..) |parent, row| {
            if (parent != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
            accepted_rows[first + row] = @intCast(row);
        }
        path.* = accepted_rows[first..][0..window.parents.len];
    };
    r.accepted_chain = accepted;
    defer r.accepted_chain = false;
    var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
    defer recurrent.deinit();
    if (r.gdn.marks) |m| {
        if (!accepted or !r.model.kernels.prompt or r.capture_taps) return error.BadPromptMarks;
        try m.check(recurrent);
    }
    var attention_plan = try r.attentionPlan(round);
    defer attention_plan.deinit();
    try r.gdn.prepare(r.model.device, recurrent, accepted);
    try r.model.frame.setRows(round.ids, round.feed_positions);
    const placed = image_rows.place(&r.model.frame, &r.rope, r.images, r.model.config.mrope_section, round);
    try r.scratch.upload(attention_plan);
    r.current_round = round;
    r.current_gdn = &recurrent;
    r.current_attention = &attention_plan;
    r.current_placed = placed;
    defer {
        r.current_round = null;
        r.current_gdn = null;
        r.current_attention = null;
        r.current_placed = .{};
    }
    if (r.capture_taps) {
        r.taps.begin();
        r.tapped = @splat(.{});
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const spanned = if (r.trace) |t| t.spanned else 0; // a profiled pass's spans commit buffers of their own
    const handed = if (r.model.ane) |share| share.gpu_seconds else 0;
    const tail_facts = stage_planes.begin(prompt_tail.facts(r, head, accepted, placed.images != null or r.images != null));
    r.tail_from = prompt_tail.first(tail_facts, round.ids.len, round.windows.len);
    defer r.tail_from = null;
    if (r.tail_from) |t0| r.model.frame.setTail(t0, @intCast(round.ids.len));
    const keys = prompt_tail.keysOnly(tail_facts);
    var stages = forward.Stages{ .model = r.model, .rows = @intCast(round.ids.len), .hooks = .{ .ptr = r, .gdn = Runner.gdnForward, .attention = Runner.attentionForward, .tap = if (r.capture_taps) Runner.tap else null, .keys = if (keys) prompt_tail.keysForward else null }, .head = head, .tail = r.tail, .images = placed.images, .reduce = r.reduce, .picks = r.picks, .tail_from = r.tail_from };
    r.last_gpu_seconds = @import("core").segments.runCommitting(r.model.device, &.{r.model.queue}, r.model.config.layers, .concurrent, &stages, decode_round.commit_every) catch |err| {
        r.failed = true;
        return err;
    };
    // A seam cannot fail while encoding: the transport reports it out of band, and a broken pair fails the round.
    if (r.reduce) |red| if (red.broken()) {
        r.failed = true;
        return error.PairFailed;
    };
    if (r.model.ane) |share| share.drain() catch |err| {
        r.failed = true;
        return err;
    };
    if (r.trace) |t| {
        r.gpu_seconds += r.last_gpu_seconds + (t.spanned - spanned) + (if (r.model.ane) |share| share.gpu_seconds - handed else 0);
        t.round();
    }
    if (r.capture_taps) {
        try r.taps.complete();
        for (round.slots, round.windows, round.firsts) |slot, w, first| r.tapped[slot] = .{ .start = w.start, .first = first, .rows = @intCast(w.parents.len) };
    }
    if (accepted) {
        for (round.slots, round.windows) |slot, window| r.offsets[slot] = window.start + @as(u32, @intCast(window.parents.len));
    } else r.active = fingerprint(round);
    if (stage_planes.on) if (stage_planes.active) |planes| try planes.end(round.windows[0].start); // refusal is atomic
}
