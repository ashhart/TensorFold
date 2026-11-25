//! Qwen3.8-27B's headless partial prompt pass: two layers' residuals copied out as folded memory's taps.
const std = @import("std");
const mtl = @import("metal");
const decode_round = @import("decode_round.zig");
const Runner = decode_round.Runner;
const Frame = @import("gpu_frame.zig").Frame;
const forward = @import("forward.zig");
const Ref = @import("projection.zig").Ref;
const wts = @import("gpu_weights.zig");
const Round = @import("round_plan.zig").Round;
const image_rows = @import("image_rows.zig");

/// The pass on `round`'s slots, stopping after `layer_sem`; residuals at `layer_lex`/`layer_sem` -> the taps (bf16).
pub fn advancePartialTap(r: *Runner, round: *const Round, layer_lex: usize, layer_sem: usize, tap_lex: mtl.Buffer, tap_sem: mtl.Buffer) !void {
    if (r.failed or r.active != null or round.ids.len > r.model.frame.capacity) return error.RoundNotReady;
    if (layer_lex >= layer_sem or layer_sem + 1 >= r.model.config.layers) return error.InvalidLayerTap;
    for (round.slots, round.windows) |slot, window| if (slot >= r.ready or r.offsets[slot] != window.start) return error.CachePositionDiffers;
    for (round.slots, round.windows) |slot, window| try r.cover(slot, window.start + @as(u32, @intCast(window.parents.len)));
    for (round.slots, round.windows) |slot, window| try r.reserve(slot, window.start + @as(u32, @intCast(window.parents.len)));

    const paths = try r.allocator.alloc([]const u32, round.windows.len);
    defer r.allocator.free(paths);
    var accepted_rows: [@import("gpu_frame.zig").max_rows]u32 = undefined;
    for (round.windows, round.firsts, paths) |window, first, *path| {
        for (window.parents, 0..) |parent, row| {
            if (parent != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
            accepted_rows[first + row] = @intCast(row);
        }
        path.* = accepted_rows[first..][0..window.parents.len];
    }
    r.accepted_chain = true;
    defer r.accepted_chain = false;
    var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
    defer recurrent.deinit();
    var attention_plan = try r.attentionPlan(round);
    defer attention_plan.deinit();
    try r.gdn.prepare(r.model.device, recurrent, true);
    try r.model.frame.setRows(round.ids, round.feed_positions);
    const placed = image_rows.place(&r.model.frame, &r.rope, null, r.model.config.mrope_section, round);
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

    const TapCtx = struct {
        r: *Runner,
        layer_lex: usize,
        layer_sem: usize,
        tap_lex: mtl.Buffer,
        tap_sem: mtl.Buffer,
        fn gdnFn(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Gdn, frame: *Frame, rows: u32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            return Runner.gdnForward(self.r, e, block, weights, frame, rows);
        }
        fn attnFn(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            return Runner.attentionForward(self.r, e, block, weights, f, rows);
        }
        fn tapFn(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, rows: u32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            const w: u32 = @intCast(self.r.model.config.hidden);
            if (layer == self.layer_lex) {
                try self.r.model.glue.unstack(e, hidden, .{ .buffer = self.tap_lex }, .{ .rows = rows, .width = w, .stride = w, .offset = 0 });
                e.barrier();
            } else if (layer == self.layer_sem) {
                try self.r.model.glue.unstack(e, hidden, .{ .buffer = self.tap_sem }, .{ .rows = rows, .width = w, .stride = w, .offset = 0 });
                e.barrier();
            }
        }
    };
    var tap_ctx = TapCtx{
        .r = r,
        .layer_lex = layer_lex,
        .layer_sem = layer_sem,
        .tap_lex = tap_lex,
        .tap_sem = tap_sem,
    };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var stages = forward.Stages{
        .model = r.model,
        .rows = @intCast(round.ids.len),
        .hooks = .{
            .ptr = &tap_ctx,
            .gdn = TapCtx.gdnFn,
            .attention = TapCtx.attnFn,
            .tap = TapCtx.tapFn,
        },
        .head = .none,
        .images = placed.images,
        .reduce = r.reduce,
        .picks = r.picks,
    };
    const n_layers = @min(r.model.config.layers, layer_sem + 2);
    _ = @import("core").segments.runCommitting(r.model.device, &.{r.model.queue}, n_layers, .concurrent, &stages, decode_round.commit_every) catch |err| {
        r.failed = true;
        return err;
    };
    for (round.slots, round.windows) |slot, window| r.offsets[slot] = window.start + @as(u32, @intCast(window.parents.len));
}
