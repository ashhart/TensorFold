//! A chat request's image_url and video_url parts, prepared and replaced by the text Python's frontend expands them to.

const std = @import("std");
const media_fetch = @import("media_fetch.zig");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const qi = api.qwen_image;

const vision_start = "<|vision_start|>";
const vision_end = "<|vision_end|>";
const max_video_bytes: usize = 16 * 1024 * 1024;
const max_videos = 2;
const max_bytes_one: usize = 10 * 1024 * 1024;
const max_bytes_all: usize = 20 * 1024 * 1024;
const max_pixels_all: u64 = 32 * 1024 * 1024;
const max_tokens_one: u64 = 4096;

pub const Extracted = struct { messages: Value, images: []api.Image };

/// The image URL of a content part (``image_url``, ``image`` or ``input_image``) and its ``detail``; else null.
const Kind = enum { image, video };

fn mediaPart(part: Value) ?struct { kind: Kind, url: ?[]const u8, detail: []const u8 } {
    if (part != .object) return null;
    const ty = part.get("type");
    const is_video = (ty != null and ty.? == .string and (std.mem.eql(u8, ty.?.string, "video_url") or std.mem.eql(u8, ty.?.string, "video"))) or
        part.get("video_url") != null;
    if (is_video) {
        var vurl: ?[]const u8 = null;
        if (part.get("video_url")) |vu| switch (vu) {
            .string => |x| vurl = x,
            .object => if (vu.get("url")) |u| if (u == .string) {
                vurl = u.string;
            },
            else => {},
        };
        if (vurl == null) if (part.get("video")) |v| if (v == .string) {
            vurl = v.string;
        };
        return .{ .kind = .video, .url = vurl, .detail = "auto" };
    }
    const is_image = (ty != null and ty.? == .string and (std.mem.eql(u8, ty.?.string, "image_url") or std.mem.eql(u8, ty.?.string, "image") or
        std.mem.eql(u8, ty.?.string, "input_image"))) or part.get("image_url") != null or part.get("image") != null;
    if (!is_image) return null;
    var url: ?[]const u8 = null;
    var detail: []const u8 = "auto";
    if (part.get("image_url")) |iu| switch (iu) {
        .string => |s| url = s,
        .object => {
            if (iu.get("url")) |u| if (u == .string) {
                url = u.string;
            };
            if (iu.get("detail")) |d| if (d == .string) {
                detail = d.string;
            };
        },
        else => {},
    };
    if (url == null) if (part.get("image")) |im| if (im == .string) {
        url = im.string;
    };
    if (part.get("detail")) |d| if (d == .string) {
        detail = d.string;
    };
    return .{ .kind = .image, .url = url, .detail = detail };
}

fn hasImages(messages: Value) bool {
    if (messages != .array) return false;
    for (messages.array) |m| {
        if (m != .object) continue;
        const c = m.get("content") orelse continue;
        if (c != .array) continue;
        for (c.array) |p| if (mediaPart(p) != null) return true;
    }
    return false;
}

const image_media = [_][]const u8{ "image/jpeg", "image/png", "image/webp" };
const video_media = [_][]const u8{ "video/mp4", "video/webm", "video/quicktime", "video/x-matroska" };

/// The bytes of a ``data:`` URL, or (--vision-urls) a public HTTPS URL fetched before ``deadline`` (images.py).
fn dataBytes(srv: *Server, cx: *Cx, url: []const u8, limit: usize, kind: Kind, deadline: std.Io.Timestamp) errors.Refused![]u8 {
    if (!std.mem.startsWith(u8, url, "data:")) {
        if (!std.mem.startsWith(u8, url, "https://")) return cx.refuse(if (kind == .image) "images require data URLs or public HTTPS URLs" else "videos require data URLs or public HTTPS URLs");
        if (!srv.config.vision_urls) return cx.refuse("image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls");
        if (url.len > media_fetch.max_url_chars) return cx.refuse("image URL is too long");
        var f: media_fetch.Failure = .{};
        const got = media_fetch.fetch(cx.a, srv.io, url, limit, deadline, if (kind == .image) &image_media else &video_media, &f) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Media => return cx.fail(.request, "{s}", .{f.text}),
        };
        return got.data;
    }
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return cx.refuse("an image data: URL needs a comma before its bytes");
    if (!std.mem.endsWith(u8, url[0..comma], ";base64")) return cx.refuse("an image data: URL must be base64-encoded");
    const b64 = url[comma + 1 ..];
    const dec = std.base64.standard.Decoder;
    const n = dec.calcSizeForSlice(b64) catch return cx.refuse("an image's base64 bytes are invalid");
    if (n > limit) return cx.refuse(if (kind == .image) "an image exceeds the 10 MiB limit (20 MiB for a request's media together)" else "a video exceeds the 16 MiB limit (20 MiB for a request's media together)");
    const out = try cx.a.alloc(u8, n);
    dec.decode(out, b64) catch return cx.refuse("an image's base64 bytes are invalid");
    return out;
}

/// A request's images prepared, its image parts as the template's image text; null without images. Needs --vision.
pub fn extract(srv: *Server, cx: *Cx, messages: ?Value) errors.Refused!?Extracted {
    const list = messages orelse return null;
    if (!hasImages(list)) return null;
    if (!srv.info.vision) return cx.refuse("this server accepts text parts only; image, audio and video inputs are unsupported");
    // every media part, in prompt order, with where its text goes
    const Slot = struct { kind: Kind, url: ?[]const u8, detail: []const u8, msg: usize, part: usize };
    var slots: std.ArrayList(Slot) = .empty;
    for (list.array, 0..) |m, i| {
        if (m != .object) continue;
        const c = m.get("content") orelse continue;
        if (c != .array) continue;
        const role = m.get("role");
        for (c.array, 0..) |p, j| {
            const im = mediaPart(p) orelse continue;
            if (role != null and role.? == .string and (std.mem.eql(u8, role.?.string, "system") or std.mem.eql(u8, role.?.string, "developer")))
                return cx.refuse("a system message cannot contain images or videos");
            try slots.append(cx.a, .{ .kind = im.kind, .url = im.url, .detail = im.detail, .msg = i, .part = j });
        }
    }
    var n_images: usize = 0;
    var n_videos: usize = 0;
    for (slots.items) |sl| switch (sl.kind) {
        .image => n_images += 1,
        .video => n_videos += 1,
    };
    if (n_images > srv.config.vision_max_images) return cx.fail(.request, "a request may carry at most {d} images; this one has {d}", .{ srv.config.vision_max_images, n_images });
    if (n_videos > max_videos) return cx.fail(.request, "a request may carry at most {d} videos; this one has {d}", .{ max_videos, n_videos });
    const budget: u64 = srv.config.vision_image_tokens;
    if (n_images > 0 and budget < n_images) return cx.refuse("the image count exceeds the visual-token budget");
    const per: u64 = if (n_images > 0) @min(budget / n_images, max_tokens_one) else 0;
    var images: std.ArrayList(api.Image) = .empty;
    const texts = try cx.a.alloc([]const u8, slots.items.len);
    var total_bytes: usize = 0;
    var total_pixels: u64 = 0;
    const deadline = std.Io.Clock.awake.now(srv.io).addDuration(.fromSeconds(media_fetch.total_s)); // every URL's download
    for (slots.items, 0..) |sl, k| {
        const url = sl.url orelse return cx.refuse(if (sl.kind == .image) "an image part needs an image_url with a url" else "a video part needs a video_url with a url");
        const bytes = try dataBytes(srv, cx, url, @min(if (sl.kind == .image) max_bytes_one else max_video_bytes, max_bytes_all - total_bytes), sl.kind, deadline);
        total_bytes += bytes.len;
        if (total_bytes > max_bytes_all) return cx.refuse("a request's images and videos exceed the 20 MiB limit");
        var text: std.ArrayList(u8) = .empty;
        switch (sl.kind) {
            .image => {
                var img = qi.decode(cx.a, bytes, .{}) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.ImageTooLarge => return cx.refuse("image dimensions exceed the decoded pixel limit"),
                    else => return cx.refuse("image bytes are invalid or unsupported; use JPEG or PNG"),
                };
                total_pixels += @as(u64, img.w) * img.h;
                if (total_pixels > max_pixels_all) return cx.refuse("a request's images exceed the decoded pixel limit");
                const cap = if (std.mem.eql(u8, sl.detail, "low")) @min(per, 256) else per;
                if (srv.info.vision_glm) { // GLM: its canvas within [min(16, cap), cap] tokens, <|image|> placeholders
                    const p = qi.glm.prepare(cx.a, img, @min(16, cap), cap) catch |e| switch (e) {
                        error.OutOfMemory => return error.OutOfMemory,
                        else => return cx.refuse("the image does not fit the visual-token budget"),
                    };
                    img.deinit(cx.a);
                    try images.append(cx.a, .{ .patches = p.patches, .grid = p.grid });
                    try text.appendSlice(cx.a, "<|begin_of_image|>");
                    for (0..p.tokens()) |_| try text.appendSlice(cx.a, "<|image|>");
                    try text.appendSlice(cx.a, "<|end_of_image|>");
                    texts[k] = text.items;
                    continue;
                }
                const p = qi.prepare(cx.a, img, cap) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return cx.refuse("absolute aspect ratio must be smaller than 200"),
                };
                img.deinit(cx.a);
                try images.append(cx.a, .{ .patches = p.patches, .grid = p.grid });
                try text.appendSlice(cx.a, vision_start);
                for (0..p.tokens()) |_| try text.appendSlice(cx.a, "<|image_pad|>");
                try text.appendSlice(cx.a, vision_end);
            },
            .video => if (srv.info.vision_glm) {
                // GLM: <|begin_of_video|>, image groups with their "t.t seconds", <|end_of_video|> (glm5_next)
                var v = try videoGlm(srv, cx, bytes);
                defer v.deinit(cx.a);
                try text.appendSlice(cx.a, "<|begin_of_video|>");
                for (0..v.groups) |g| {
                    try images.append(cx.a, .{ .patches = try cx.a.dupe(f32, v.groupPatches(g)), .grid = .{ 1, @intCast(v.gh), @intCast(v.gw) } });
                    try text.appendSlice(cx.a, "<|begin_of_image|>");
                    for (0..v.tokens()) |_| try text.appendSlice(cx.a, "<|image|>");
                    try text.appendSlice(cx.a, "<|end_of_image|>");
                    var tb: [32]u8 = undefined;
                    try text.appendSlice(cx.a, qi.formatSeconds(&tb, v.times[g]) catch unreachable);
                    try text.appendSlice(cx.a, " seconds");
                }
                try text.appendSlice(cx.a, "<|end_of_video|>");
            } else {
                var v = try video(srv, cx, bytes);
                defer v.deinit(cx.a);
                for (0..v.groups) |g| {
                    try images.append(cx.a, .{ .patches = try cx.a.dupe(f32, v.groupPatches(g)), .grid = .{ 1, @intCast(v.gh), @intCast(v.gw) } });
                    var tb: [32]u8 = undefined;
                    try text.appendSlice(cx.a, "<");
                    try text.appendSlice(cx.a, qi.formatSeconds(&tb, v.times[g]) catch unreachable);
                    try text.appendSlice(cx.a, " seconds>");
                    try text.appendSlice(cx.a, vision_start);
                    for (0..v.tokens()) |_| try text.appendSlice(cx.a, "<|video_pad|>");
                    try text.appendSlice(cx.a, vision_end);
                }
            },
        }
        texts[k] = text.items;
    }
    // the messages with each media part as its text
    const out = try cx.a.alloc(Value, list.array.len);
    @memcpy(out, list.array);
    var k: usize = 0;
    while (k < slots.items.len) {
        const i = slots.items[k].msg;
        const c = list.array[i].get("content").?;
        const parts = try cx.a.dupe(Value, c.array);
        while (k < slots.items.len and slots.items[k].msg == i) : (k += 1) {
            const t = try json.newObject(cx.a);
            try t.put(cx.a, "type", .{ .string = "text" });
            try t.put(cx.a, "text", .{ .string = texts[k] });
            parts[slots.items[k].part] = .{ .object = t };
        }
        const copy = try json.copyObject(cx.a, list.array[i].object);
        try copy.put(cx.a, "content", .{ .array = parts });
        out[i] = .{ .object = copy };
    }
    return .{ .messages = .{ .array = out }, .images = images.items };
}

// -- videos: libtfvideo (FFmpeg), loaded when the first video arrives ------------------------------------------

const TfVideo = extern struct { rgb: ?[*]u8, frames: c_int, height: c_int, width: c_int, indices: ?[*]c_int, fps: f64 };
const SizeFn = *const fn (ctx: ?*anyopaque, frames: c_int, height: c_int, width: c_int, oh: *c_int, ow: *c_int) callconv(.c) c_int;
const DecodeFn = *const fn (data: [*]const u8, len: usize, rate: f64, min_frames: c_int, max_frames: c_int, max_seconds: f64, max_dim: c_int, size: SizeFn, ctx: ?*anyopaque, out: *TfVideo) callconv(.c) c_int;
const FreeFn = *const fn (v: *TfVideo) callconv(.c) void;

var lib_mutex: std.Io.Mutex = .init;
var lib: ?struct { decode: DecodeFn, free: FreeFn } = null;

fn library(io: std.Io) ?@TypeOf(lib.?) {
    lib_mutex.lockUncancelable(io);
    defer lib_mutex.unlock(io);
    if (lib == null) {
        var dl = std.DynLib.open("libtfvideo.so") catch return null;
        const d = dl.lookup(DecodeFn, "tf_video_decode") orelse return null;
        const f = dl.lookup(FreeFn, "tf_video_free") orelse return null;
        lib = .{ .decode = d, .free = f }; // kept open for the server's life
    }
    return lib;
}

fn sizeFor(ctx: ?*anyopaque, frames: c_int, height: c_int, width: c_int, oh: *c_int, ow: *c_int) callconv(.c) c_int {
    const budget: *const u64 = @ptrCast(@alignCast(ctx.?));
    const s = qi.videoSize(@intCast(frames), @intCast(height), @intCast(width), budget.*) catch return -1;
    oh.* = @intCast(s[0]);
    ow.* = @intCast(s[1]);
    return 0;
}

/// GLM's video canvas: the frames' content size for libtfvideo (geometry within the budget), the canvas kept.
const GlmSize = struct { budget: u64, th: u32 = 0, tw: u32 = 0 };

fn sizeForGlm(ctx: ?*anyopaque, frames: c_int, height: c_int, width: c_int, oh: *c_int, ow: *c_int) callconv(.c) c_int {
    const gs: *GlmSize = @ptrCast(@alignCast(ctx.?));
    const g = qi.glm.geometry(@intCast(frames), @intCast(height), @intCast(width), 16, gs.budget) catch return -1;
    gs.th = g[0];
    gs.tw = g[1];
    oh.* = @intCast(g[2]);
    ow.* = @intCast(g[3]);
    return 0;
}

fn videoGlm(srv: *Server, cx: *Cx, bytes: []const u8) errors.Refused!qi.glm.Video {
    const l = library(srv.io) orelse return cx.refuse("video inputs need libtfvideo (FFmpeg) beside the server");
    var gs: GlmSize = .{ .budget = 16384 };
    if (std.c.getenv("TENSORFOLD_VIDEO_TOKENS")) |v| gs.budget = std.fmt.parseInt(u64, std.mem.span(v), 10) catch gs.budget;
    var v: TfVideo = undefined;
    const rc = l.decode(bytes.ptr, bytes.len, 2.0, 2, 256, 3600.0, 8192, sizeForGlm, @ptrCast(&gs), &v);
    if (rc != 0) return cx.refuse(switch (rc) {
        -2 => "the video has no video stream",
        -3 => "video dimensions are missing or exceed the pixel limit",
        -4 => "the video's frame count or rate is missing",
        -5 => "videos are limited to 60 minutes",
        -6 => "the video has fewer than two decodable frames",
        -8 => "a video needs at least 2 frames and an aspect ratio under 200",
        else => "video bytes are invalid or unsupported; use MP4 or WebM",
    });
    defer l.free(&v);
    const frames: usize = @intCast(v.frames);
    const h: usize = @intCast(v.height);
    const w: usize = @intCast(v.width);
    const idx = try cx.a.alloc(i32, frames);
    for (idx, 0..) |*x, i| x.* = v.indices.?[i];
    return qi.glm.prepareVideo(cx.a, v.rgb.?[0 .. frames * h * w * 3], frames, h, w, gs.th, gs.tw, idx, v.fps) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return cx.refuse("the video's frames do not fit the tower's patch grid"),
    };
}

fn video(srv: *Server, cx: *Cx, bytes: []const u8) errors.Refused!qi.Video {
    const l = library(srv.io) orelse return cx.refuse("video inputs need libtfvideo (FFmpeg) beside the server");
    var budget: u64 = 16384;
    if (std.c.getenv("TENSORFOLD_VIDEO_TOKENS")) |v| budget = std.fmt.parseInt(u64, std.mem.span(v), 10) catch budget;
    var v: TfVideo = undefined;
    const rc = l.decode(bytes.ptr, bytes.len, 2.0, 4, 256, 3600.0, 8192, sizeFor, @ptrCast(&budget), &v);
    if (rc != 0) return cx.refuse(switch (rc) {
        -2 => "the video has no video stream",
        -3 => "video dimensions are missing or exceed the pixel limit",
        -4 => "the video's frame count or rate is missing",
        -5 => "videos are limited to 60 minutes",
        -6 => "the video has fewer than two decodable frames",
        -8 => "a video needs at least 2 frames and an aspect ratio under 200",
        else => "video bytes are invalid or unsupported; use MP4 or WebM",
    });
    defer l.free(&v);
    const frames: usize = @intCast(v.frames);
    const h: usize = @intCast(v.height);
    const w: usize = @intCast(v.width);
    const idx = try cx.a.alloc(i32, frames);
    for (idx, 0..) |*x, i| x.* = v.indices.?[i];
    return qi.prepareVideo(cx.a, v.rgb.?[0 .. frames * h * w * 3], frames, h, w, idx, v.fps) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return cx.refuse("the video's frames do not fit the tower's patch grid"),
    };
}
