//! Real tokenizer fixtures prove that ordinary vocabulary markers must refuse at startup.
const std = @import("std");
const api = @import("engine_api");
const hf_text = @import("hf_text.zig");
const f = @import("control_tokens_fixture.zig");
const markers = @import("vision_markers.zig");
const Server = @import("server.zig").Server;
const log = @import("log.zig");
const t = std.testing;

const Script = struct {
    vision: api.Vision,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = undefined, .cancel = undefined, .status = undefined, .memory = undefined } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 32768, .vision = s.vision };
    }
};
const Capture = struct {
    bytes: [4096]u8 = undefined,
    used: usize = 0,
    fn take(ctx: *anyopaque, bytes: []const u8) void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        std.debug.assert(bytes.len <= c.bytes.len - c.used);
        @memcpy(c.bytes[c.used..][0..bytes.len], bytes);
        c.used += bytes.len;
    }
};
fn noImage(_: *anyopaque, _: std.mem.Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
    return error.UnexpectedImageDecode;
}
fn noVideo(_: *anyopaque, _: std.mem.Allocator, _: api.VideoSource, _: u32, _: u32) anyerror!api.PreparedVideo {
    return error.UnexpectedVideoDecode;
}
fn vision(a: std.mem.Allocator, ctx: *anyopaque, image: u32, video: u32, open: u32, close: u32) !api.Vision {
    return .{ .ctx = ctx, .prepare = noImage, .image_token = image, .markers = try a.dupe(u32, &.{ open, image, close }), .wrap = .{ open, close }, .image_tokens = 64, .max_images = 2, .video = .{
        .prepare = noVideo,
        .video_token = video,
        .group_open = open,
        .group_close = close,
        .time_first = video,
        .markers = try a.dupe(u32, &.{ open, video, close }),
        .video_tokens = 64,
        .max_videos = 2,
        .max_frames = 16,
    } };
}

test "vision markers: plain vocabulary entries cannot become image or video markers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fixture = try f.Fixture.init(a, 0);
    defer fixture.deinit();
    var table = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures/control_tokens/mini_tokenizer.json"), .{});
    const ordinary = "<unsafe_media>";
    const model = table.object.getPtr("model").?;
    try model.object.getPtr("vocab").?.object.put(a, ordinary, .{ .integer = 800 });
    try model.object.put(a, "ignore_merges", .{ .bool = true });
    for (table.object.getPtr("added_tokens").?.array.items) |*token| {
        if (std.mem.eql(u8, token.object.get("content").?.string, "<|image|>")) try token.object.put(a, "special", .{ .bool = false });
    }
    try fixture.tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = try std.json.Stringify.valueAlloc(a, table, .{}) });
    var problem: []const u8 = "";
    const hf = try hf_text.HfText.load(a, t.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    const text = hf.text();
    const image = text.tokenId("<|image|>").?;
    const video = text.tokenId("<|video|>").?;
    const open = text.tokenId("<|im_start|>").?;
    const close = text.tokenId("<|im_end|>").?;
    const bad = text.tokenId(ordinary).?;
    try t.expect(text.isAdded(image) and !text.isSpecial(image));
    try t.expect(!text.isAdded(bad));
    try t.expectEqualSlices(u32, &.{bad}, try hf.tok.encodeMasked(a, ordinary, &.{.{ 0, ordinary.len }}));
    var dummy: u8 = 0;
    const valid = try vision(a, &dummy, image, video, open, close);
    try t.expectEqual(@as(?u32, null), markers.firstUnsafe(text, valid));
    var script: Script = .{ .vision = valid };
    const srv = try Server.init(a, t.io, script.engine(), text, .{ .served_name = "fixture", .model_ids = &.{"fixture"}, .keep_warm_s = 0 }, null);
    srv.deinit();
    var capture: Capture = .{};
    const saved = log.Test.sink;
    log.Test.sink = .{ .ctx = &capture, .call = Capture.take };
    defer log.Test.sink = saved;
    for (0..8) |field| {
        var invalid = valid;
        switch (field) {
            0 => invalid.image_token = bad,
            1 => invalid.markers = try a.dupe(u32, &.{bad}),
            2 => invalid.wrap = .{ open, bad },
            3 => invalid.video.?.video_token = bad,
            4 => invalid.video.?.group_open = bad,
            5 => invalid.video.?.group_close = bad,
            6 => invalid.video.?.markers = try a.dupe(u32, &.{bad}),
            7 => invalid.video.?.time_first = bad,
            else => unreachable,
        }
        try t.expectEqual(@as(?u32, bad), markers.firstUnsafe(text, invalid));
        script.vision = invalid;
        capture.used = 0;
        try t.expectError(error.InvalidVisionMarkers, Server.init(a, t.io, script.engine(), text, .{ .served_name = "fixture", .model_ids = &.{"fixture"}, .keep_warm_s = 0 }, null));
        const message = capture.bytes[0..capture.used];
        try t.expect(std.mem.indexOf(u8, message, "not an added token") != null);
        try t.expect(std.mem.indexOf(u8, message, "image and video input cannot be served safely") != null);
    }
    var unknown = text.vtable.*;
    unknown.added = null;
    try t.expectEqual(@as(?u32, image), markers.firstUnsafe(.{ .ctx = text.ctx, .vtable = &unknown }, valid));
    std.debug.print("VISION_MARKERS all 8 fields refuse plain vocabulary; unflagged additions pass\n", .{});
}
