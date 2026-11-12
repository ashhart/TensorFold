//! What a served model takes as input, decided once at startup: /v1/models' input_types and the refusals read it.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");

/// Images with --vision; videos when the model offers them and this server can decode them (libtfvideo loads).
pub const Inputs = struct {
    image: bool = false,
    video: bool = false,

    pub fn of(vision: ?api.Vision, video_decoder: bool) Inputs {
        const v = vision orelse return .{};
        return .{ .image = true, .video = v.video != null and video_decoder };
    }

    /// `["text"]`, then "image" and "video" when taken, in that order.
    pub fn value(i: Inputs, a: std.mem.Allocator) std.mem.Allocator.Error!json.Value {
        const out = try a.alloc(json.Value, 1 + @as(usize, @intFromBool(i.image)) + @intFromBool(i.video));
        out[0] = .{ .string = "text" };
        if (i.image) out[1] = .{ .string = "image" };
        if (i.video) out[out.len - 1] = .{ .string = "video" };
        return .{ .array = out };
    }
};

/// At startup: whether libtfvideo loads for `vision`'s videos, by the loader each video uses, which keeps its answer.
pub fn decodes(vision: ?api.Vision) bool {
    const v = vision orelse return false;
    if (v.video == null) return false;
    const quiet = @import("sys").optional_libraries.atBoundary(); // the line below says it once, with what follows
    defer quiet.deinit();
    if (api.tfvideo.version() != null) return true;
    const message = @import("sys").optional_libraries.spec(.video).message;
    @import("log.zig").line("--vision: {s}; video_url parts are refused and /v1/models lists no video", .{message});
    return false;
}
