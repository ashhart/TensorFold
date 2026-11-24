const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const cache_fit = @import("cache_fit.zig");
const reserve = @import("qwen27_reserve.zig");

/// Room left beside the weights and an 8 GiB margin.
pub fn room(model: *const q.model.Model) usize {
    const dev = model.device;
    return q.prompt_memory.room(dev.maxWorkingSet(), dev.allocated(), reserve.cap()) catch 0;
}

/// Why --drafter is refused before the target or the drafter loads.
pub fn drafterRefusal(a: std.mem.Allocator, dir: []const u8, drafter: []const u8, err: anyerror) ![]const u8 {
    if (err == error.DraftForAnotherTarget) return std.fmt.allocPrint(a, "--drafter {s} drafts for another target: its width, vocabulary or tapped layers do not fit {s} (DraftForAnotherTarget)", .{ drafter, dir });
    return std.fmt.allocPrint(a, "--drafter {s} is not a drafter the native Qwen engine serves for {s} ({s}): it takes z-lab/Qwen3.8-27B-DFlash2 as released for Qwen3.8-27B, or a DFlash drafter (DFlashDraftModel) made for this target", .{ drafter, dir, @errorName(err) });
}

/// --prompt-cache-gib's bytes (0 or less: none).
pub fn gibBytes(gib: f64) u64 {
    return if (gib > 0) std.math.lossyCast(u64, gib * cache_fit.GiB) else 0;
}
