//! Nemotron Omni's video input: frames sampled at a rate, sized by the video rule and packed in pairs for the tower.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;
const image = @import("image.zig");
const Allocator = std.mem.Allocator;
const picture = core.picture;

pub const Error = error{ VideoTooShort, VideoTooLarge, BadVideo, OutOfMemory };

/// The sampling rate, 1 above 720 rows (the model card), and the frames one video may be sampled to at that rate.
const fps_small: f64 = 2.0;
const fps_large: f64 = 1.0;
const frames_large: usize = 128;

/// Where decoded frames go: RGB rows of `width` * 3 bytes, `stride` bytes apart.
pub const Sink = struct {
    ctx: *anyopaque,
    take: *const fn (ctx: *anyopaque, k: usize, rgb: []const u8, width: u32, height: u32, stride: usize) anyerror!void,
};

/// A decoded video's facts and its frames on request: `decode` gives the frames at `indices` (ascending) to `sink`.
pub const Source = struct {
    frames: u64,
    rate_num: u64,
    rate_den: u64,
    width: u32,
    height: u32,
    ctx: *anyopaque,
    decode: *const fn (ctx: *anyopaque, indices: []const u32, sink: Sink) anyerror!void,
};

/// A video for the tower: one image a pair of frames, each pair's time, and the words the prompt puts before its rows.
pub const Prepared = struct { groups: []image.Prepared, times: []f64, labels: []const []const u8 };

/// Each sampled frame, resized and cut into patches as it arrives.
const Collect = struct {
    a: Allocator,
    c: *const Config,
    grid: image.Grid,
    patches: [][]f32,
    hashes: []u64,
    got: usize = 0,

    fn take(ctx: *anyopaque, k: usize, rgb: []const u8, w: u32, h: u32, stride: usize) anyerror!void {
        const s: *Collect = @ptrCast(@alignCast(ctx));
        if (k != s.got or k >= s.patches.len or w == 0 or h == 0 or stride < @as(usize, w) * 3) return error.BadVideo;
        const th = s.grid.h * s.c.patch;
        const tw = s.grid.w * s.c.patch;
        const planes = try s.a.alloc(f32, 3 * th * tw);
        defer s.a.free(planes);
        try picture.resample.bicubicPlanes(s.a, .{ .pixels = rgb, .width = w, .height = h, .stride = stride }, tw, th, planes);
        image.normalize(s.c, planes);
        const out = try s.a.alloc(f32, planes.len);
        image.cut(s.c, planes, s.grid, out);
        var id: u64 = 0x7669646f;
        for (0..h) |y| id = std.hash.Wyhash.hash(id, rgb[y * stride ..][0 .. @as(usize, w) * 3]);
        s.patches[k] = out;
        s.hashes[k] = id;
        s.got += 1;
    }
};

/// The frames to decode: a second of video gives `fps` of them, spread over the clip when that is more than `cap`.
fn sample(a: Allocator, src: Source, fps: f64, cap: usize) Error![]u32 {
    const rate = @as(f64, @floatFromInt(src.rate_num)) / @as(f64, @floatFromInt(src.rate_den));
    const length = @as(f64, @floatFromInt(src.frames)) / rate;
    const whole: usize = @max(1, @as(usize, @intFromFloat(@floor(length * fps))));
    const want = @min(whole, cap);
    if (want == 0) return error.VideoTooLarge;
    const step = if (want == whole) 1.0 / fps else length / @as(f64, @floatFromInt(want));
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    for (0..want) |k| {
        var idx: u64 = @intFromFloat(@floor(@as(f64, @floatFromInt(k)) * step * rate + 0.5));
        if (out.items.len > 0 and idx <= out.items[out.items.len - 1]) idx = out.items[out.items.len - 1] + 1;
        if (idx >= src.frames) break;
        try out.append(a, @intCast(idx));
    }
    if (out.items.len == 0) return error.VideoTooShort;
    return out.toOwnedSlice(a);
}

/// A video within `max_tokens` visual tokens and `max_frames` sampled frames: the frames in pairs, each pair one image.
pub fn prepare(a: Allocator, c: *const Config, src: Source, max_tokens: u32, max_frames: u32) !Prepared {
    if (c.video_temporal != 2 or src.width == 0 or src.height == 0) return error.BadVideo;
    const large = src.height > 720;
    const grid = image.gridVideo(c, src.width, src.height);
    const per_group: u32 = grid.h * grid.w / (Config.shuffle * Config.shuffle);
    if (max_tokens < per_group) return error.VideoTooLarge;
    const cap = @min(@as(usize, max_frames), if (large) frames_large else std.math.maxInt(usize), @as(usize, max_tokens / per_group) * 2);
    const indices = try sample(a, src, if (large) fps_large else fps_small, cap);
    const n = indices.len;
    const col = try a.create(Collect);
    col.* = .{ .a = a, .c = c, .grid = grid, .patches = try a.alloc([]f32, n), .hashes = try a.alloc(u64, n) };
    try src.decode(src.ctx, indices, .{ .ctx = col, .take = Collect.take });
    if (col.got != n) return error.BadVideo;
    const groups = (n + 1) / 2;
    const patch_count: usize = grid.h * grid.w;
    const pw = c.patchWidth();
    const out = try a.alloc(image.Prepared, groups);
    const times = try a.alloc(f64, groups);
    const labels = try a.alloc([]const u8, groups);
    const rate = @as(f64, @floatFromInt(src.rate_num)) / @as(f64, @floatFromInt(src.rate_den));
    const frame_ms = @floor(1000.0 / rate); // the processor times a frame as its index in whole milliseconds a frame
    for (out, times, labels, 0..) |*g, *t, *label, gi| {
        const first = gi * 2;
        const second = if (first + 1 < n) first + 1 else first; // a lone last frame pairs with itself
        const px = try a.alloc(f32, patch_count * 2 * pw);
        for (0..patch_count) |p| {
            @memcpy(px[p * 2 * pw ..][0..pw], col.patches[first][p * pw ..][0..pw]);
            @memcpy(px[p * 2 * pw + pw ..][0..pw], col.patches[second][p * pw ..][0..pw]);
        }
        const ids = [2]u64{ col.hashes[first], col.hashes[second] };
        const hash = picture.identity.withGrid(picture.identity.ofBytes(0x7ad1e0, std.mem.asBytes(&ids)), .{ grid.h, grid.w });
        g.* = .{ .pixels = px, .gh = grid.h, .gw = grid.w, .tokens = per_group, .hash = hash };
        t.* = @as(f64, @floatFromInt(indices[first])) * frame_ms / 1000.0;
        var words: std.ArrayList(u8) = .empty;
        if (gi > 0) try words.append(a, '\n');
        for (first..@min(first + 2, n)) |f| {
            if (f > first) try words.appendSlice(a, " and ");
            try words.print(a, "{s} {d} sampled at {d:.2} seconds", .{ if (f == first) "Frame" else "frame", f + 1, @as(f64, @floatFromInt(indices[f])) * frame_ms / 1000.0 });
        }
        try words.appendSlice(a, ": ");
        label.* = words.items;
    }
    return .{ .groups = out, .times = times, .labels = labels };
}

test "a source is sampled at 2 a second, thinned to the cap, and never repeats a frame" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dummy: u8 = 0;
    const src: Source = .{ .frames = 150, .rate_num = 30, .rate_den = 1, .width = 640, .height = 480, .ctx = &dummy, .decode = undefined };
    const all = try sample(a, src, 2.0, 256); // 5 seconds
    try std.testing.expectEqual(@as(usize, 10), all.len);
    try std.testing.expectEqual(@as(u32, 0), all[0]);
    try std.testing.expectEqual(@as(u32, 15), all[1]);
    const few = try sample(a, src, 2.0, 4);
    try std.testing.expectEqual(@as(usize, 4), few.len);
    try std.testing.expect(few[3] > few[2] and few[3] < 150);
    const slow: Source = .{ .frames = 3, .rate_num = 30, .rate_den = 1, .width = 640, .height = 480, .ctx = &dummy, .decode = undefined };
    const one = try sample(a, slow, 2.0, 256); // a tenth of a second: its first frame
    try std.testing.expectEqual(@as(usize, 1), one.len);
}

/// A source whose frame `i` is one flat colour, grey level 10 * i.
const Flat = struct {
    fn decode(_: *anyopaque, indices: []const u32, sink: Sink) anyerror!void {
        var px: [64 * 48 * 3]u8 = undefined;
        for (indices, 0..) |idx, k| {
            @memset(&px, @intCast(10 * idx));
            try sink.take(sink.ctx, k, &px, 64, 48, 64 * 3);
        }
    }
};

test "frames pair into tubelets: patches side by side, a lone last frame twice, labelled with the processor's words" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c: Config = .{ .hidden = 1280, .layers = 32, .heads = 16, .ffn = 5120, .eps = 1e-6, .patch = 16, .prefix = 10, .side = 128, .projector_hidden = 20480, .text_hidden = 4096, .final_norm = false, .mean = .{ 0.5, 0.5, 0.5 }, .std = .{ 0.25, 0.25, 0.25 }, .min_patches = 1024, .max_patches = 13312, .max_model_len = 16384, .image_token = 18, .video_temporal = 2 };
    var dummy: u8 = 0;
    // 2.6 seconds at 10 a second: 5 samples at 2 a second (frames 0, 5, 10, 15, 20), so two pairs and a lone frame
    const src: Source = .{ .frames = 26, .rate_num = 10, .rate_den = 1, .width = 64, .height = 48, .ctx = &dummy, .decode = Flat.decode };
    const p = try prepare(a, &c, src, 4096, 256);
    try std.testing.expectEqual(@as(usize, 3), p.groups.len);
    const g = image.gridVideo(&c, 64, 48);
    try std.testing.expectEqual(g.h * g.w / 4, p.groups[0].tokens);
    const pw = c.patchWidth();
    // frame 0 is black (0), frame 5 grey 50: the first patch's first value of each half
    const first = (0.0 / 255.0 - 0.5) / 0.25;
    const second = (50.0 / 255.0 - 0.5) / 0.25;
    try std.testing.expectApproxEqAbs(@as(f32, first), p.groups[0].pixels[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, second), p.groups[0].pixels[pw], 1e-3);
    // the lone frame 20 (grey 200) fills both halves of the last group
    const last = (200.0 / 255.0 - 0.5) / 0.25;
    try std.testing.expectApproxEqAbs(@as(f32, last), p.groups[2].pixels[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, last), p.groups[2].pixels[pw], 1e-3);
    try std.testing.expect(p.groups[0].hash != p.groups[1].hash);
    try std.testing.expectEqualStrings("Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds: ", p.labels[0]);
    try std.testing.expectEqualStrings("\nFrame 3 sampled at 1.00 seconds and frame 4 sampled at 1.50 seconds: ", p.labels[1]);
    try std.testing.expectEqualStrings("\nFrame 5 sampled at 2.00 seconds: ", p.labels[2]);
    try std.testing.expectEqual(@as(f64, 1.0), p.times[1] - p.times[0]); // a pair is a second at 2 frames a second
    // a token budget under one group's rows refuses the video
    try std.testing.expectError(error.VideoTooLarge, prepare(a, &c, src, p.groups[0].tokens - 1, 256));
}
