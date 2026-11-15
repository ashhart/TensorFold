//! /v1/models' input_types (#3004) through the route, and a video_url part to a server that takes no video: a 400.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const messages = @import("messages.zig");
const images = @import("images.zig");
const t = std.testing;

const Wire = struct {
    out: std.Io.Writer.Allocating,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const self: *Wire = @ptrCast(@alignCast(ctx));
        self.out.writer.writeAll(bytes) catch return error.Closed;
    }
};

fn noImage(_: *anyopaque, _: std.mem.Allocator, _: []const u8, _: u32) anyerror!api.PreparedImage {
    return error.Unused;
}

fn noVideo(_: *anyopaque, _: std.mem.Allocator, _: api.VideoSource, _: u32, _: u32) anyerror!api.PreparedVideo {
    return error.Unused;
}

var dummy: u8 = 0;

/// An image offer, and with `video` the video offer a checkpoint with a video token makes.
fn offer(video: bool) api.Vision {
    var v: api.Vision = .{ .ctx = &dummy, .prepare = noImage, .image_token = 9, .image_tokens = 64, .max_images = 4 };
    if (video) v.video = .{ .prepare = noVideo, .video_token = 8, .group_open = 6, .group_close = 7, .video_tokens = 64, .max_videos = 2, .max_frames = 256 };
    return v;
}

var text_vtable: @import("model_text.zig").Text.VTable = undefined;

fn noTemplate(_: *anyopaque) []const u8 {
    return "";
}

/// A server with only discovery's and media's fields set: no engine, model or tokenizer.
fn server(vision: ?api.Vision, video_decoder: bool) Server {
    var srv: Server = undefined;
    srv.io = t.io;
    srv.info = .{ .vision = vision };
    srv.config = .{ .served_name = "model", .model_ids = &.{ "model", "alias" } };
    srv.idle_manager = null;
    srv.thinking_switch = false;
    srv.effort_levels = &.{};
    srv.think_close = &.{};
    srv.video_decoder = video_decoder;
    text_vtable.template_source = noTemplate;
    srv.text = .{ .ctx = undefined, .vtable = &text_vtable };
    return srv;
}

/// GET /v1/models' `data`, as a client reads it.
fn listed(a: std.mem.Allocator, srv: *Server) ![]json.Value {
    var wire: Wire = .{ .out = .init(a) };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "GET", .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    try @import("status_routes.zig").models(srv, &conn, a);
    const bytes = wire.out.written();
    const split = std.mem.indexOf(u8, bytes, "\r\n\r\n").? + 4;
    return (try json.parse(a, bytes[split..])).ok.get("data").?.array;
}

test "model discovery input types: text only, image, and image and video, last in every entry" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { vision: ?api.Vision, decoder: bool, want: []const []const u8 };
    const cases = [_]Case{
        .{ .vision = null, .decoder = true, .want = &.{"text"} },
        .{ .vision = offer(false), .decoder = true, .want = &.{ "text", "image" } },
        .{ .vision = offer(true), .decoder = true, .want = &.{ "text", "image", "video" } },
        // the checkpoint offers video but libtfvideo does not load here: no video, as the refusal below says
        .{ .vision = offer(true), .decoder = false, .want = &.{ "text", "image" } },
    };
    // every key before it is main's, in main's order (context_window 0 adds no length keys)
    const keys = [_][]const u8{ "id", "object", "created", "owned_by", "capabilities", "input_types" };
    for (cases) |c| {
        var srv = server(c.vision, c.decoder);
        for (try listed(a, &srv)) |model| {
            try t.expectEqual(keys.len, model.object.count());
            for (keys, model.object.keys()) |want, got| try t.expectEqualStrings(want, got);
            const types = model.get("input_types").?.array;
            try t.expectEqual(c.want.len, types.len);
            for (c.want, types) |want, got| try t.expectEqualStrings(want, got.string);
        }
    }
}

test "model discovery input types: a video_url part to a model that takes no video is a named 400" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = (try json.parse(a,
        \\[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,YQ=="}}, {"type": "text", "text": "x"}]}]
    )).ok;
    // an image model, and a video checkpoint on a server without libtfvideo (main lets the second through to a 500)
    for ([_]Server{ server(offer(false), true), server(offer(true), false) }) |srv_value| {
        var srv = srv_value;
        for (try listed(a, &srv)) |model| for (model.get("input_types").?.array) |ty| try t.expect(!std.mem.eql(u8, ty.string, "video"));
        var cx: Cx = .{ .a = a };
        try t.expectError(error.Refused, messages.normalizeWith(&cx, body, "system", false, srv.media()));
        try t.expectEqual(@as(u16, 400), cx.status());
        try t.expectEqualStrings("content parts must be text or image_url; audio and video inputs are unsupported", cx.message);
    }
    // the same request where video is listed passes normalization
    var srv = server(offer(true), true);
    var cx: Cx = .{ .a = a };
    _ = try messages.normalizeWith(&cx, body, "system", false, srv.media());
}

test "model discovery input types: a video the decoder library cannot open is a 400 that names it, not a 500" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // this host's libtfvideo would decode the bytes instead (CI hosts have none): ask the loader the video uses
    const sys = @import("sys");
    const quiet = sys.optional_libraries.atBoundary();
    const loads = api.tfvideo.version() != null;
    quiet.deinit();
    if (loads) return error.SkipZigTest;
    const body = (try json.parse(a,
        \\[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,YQ=="}}]}]
    )).ok;
    var cx: Cx = .{ .a = a };
    try t.expectError(error.Refused, images.expand(offer(true), null, images.MarkText.text(), &cx, body, &[_]u32{8}, 0));
    try t.expectEqual(@as(u16, 400), cx.status());
    try t.expect(std.mem.startsWith(u8, cx.message, "video 1: this server takes no video input: "));
    try t.expect(std.mem.indexOf(u8, cx.message, sys.optional_libraries.spec(.video).names[0]) != null);
}
