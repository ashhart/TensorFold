//! A CUDA family's options from the serve flags it takes: its ranks, images and videos, NVFP4 precision and fp8 caches.
const std = @import("std");
const api = @import("engine_api");

/// Sets what `o` asks of `F` in `opts`; false with `problem` set when F cannot take a flag given (--precision).
pub fn set(comptime F: type, o: api.Open, opts: *F.Options, problem: *[]const u8) bool {
    if (@hasField(F.Options, "rank")) {
        opts.tp = @intCast(o.tp);
        opts.rank = @intCast(o.rank);
        opts.master = if (o.master.len > 0) o.master else null;
        opts.master_port = o.master_port;
    }
    if (comptime @hasField(F.Options, "vision")) {
        if (o.vision) |vo| {
            const own: ?u32 = comptime if (@hasDecl(F, "default_image_tokens")) F.default_image_tokens else null;
            opts.vision = .{ .image_tokens = if (vo.image_tokens_set) vo.image_tokens else own orelse vo.image_tokens, .max_images = vo.max_images };
            // a family that reads videos takes the counts its tower holds (Nemotron Omni's only a token budget)
            const L = @typeInfo(@FieldType(F.Options, "vision")).optional.child;
            if (comptime @hasField(L, "video_tokens")) opts.vision.?.video_tokens = vo.video_tokens;
            if (comptime @hasField(L, "max_videos")) opts.vision.?.max_videos = vo.max_videos;
            if (comptime @hasField(L, "max_frames")) opts.vision.?.max_frames = vo.max_frames;
        }
    }
    if (o.precision) |p| {
        if (!@hasField(F.Options, "precision")) {
            problem.* = "--precision: this CUDA family reads no NVFP4 checkpoint; serve it without --precision";
            return false;
        }
        opts.precision = std.meta.stringToEnum(@typeInfo(@TypeOf(opts.precision)).optional.child, @tagName(p)).?;
    }
    if (@hasField(F.Options, "kv8")) opts.kv8 = o.kv8;
    return true;
}
