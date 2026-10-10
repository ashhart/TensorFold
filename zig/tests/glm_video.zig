//! GLM-5.3-Flash's video preparation on one clip, for tools/zig/glm_video_fixtures.py: libtfvideo decodes it, the
//! family samples, sizes, resizes, normalizes and pairs the frames (families/glm/video.zig).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const api = @import("engine_api");
const video = tf.glm.video;
const vision = tf.glm.vision;

const usage =
    \\tf-glm-video MODEL_DIR CLIP OUT_DIR [MAX_TOKENS [MAX_FRAMES]]
    \\Writes OUT_DIR/native_patches.f32 (every group's patches in order, [groups * gh * gw, 1176]) and
    \\OUT_DIR/native.json: the stream's facts, the decoded indices, the grid [groups, gh, gw] and each group's text
    \\("T.T seconds"). MAX_TOKENS defaults to 16384 tokens, MAX_FRAMES to 256 frames.
;

fn readFile(a: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(a, "{s}", .{path}, 0));
    defer f.deinit();
    return a.dupe(u8, f.bytes[0..f.size]);
}

const Decoder = struct {
    v: api.tfvideo.Video,

    fn decode(ctx: *anyopaque, indices: []const u32, sink: *video.Sink) anyerror!void {
        const d: *Decoder = @ptrCast(@alignCast(ctx));
        try d.v.frames(indices, sink);
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const io = init.io;
    const max_tokens: u32 = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 16384;
    const max_frames: u32 = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else 256;
    const cfg_text = try readFile(a, try std.fmt.allocPrint(a, "{s}/config.json", .{args[1]}));
    const proc_text = readFile(a, try std.fmt.allocPrint(a, "{s}/processor_config.json", .{args[1]})) catch null;
    const c = (try vision.Config.parse(a, cfg_text, proc_text)) orelse return error.NoVisionConfig;
    const bytes = try readFile(a, args[2]);
    var d: Decoder = .{ .v = try api.tfvideo.Video.open(bytes) };
    defer d.v.close();
    const i = d.v.info;
    const src: video.Source = .{ .frames = @intCast(i.frames), .rate_num = @intCast(i.rate_num), .rate_den = @intCast(i.rate_den), .width = @intCast(i.width), .height = @intCast(i.height), .ctx = &d, .decode = Decoder.decode };
    const t0 = std.Io.Clock.awake.now(io);
    const p = try video.plan(a, src, c, max_tokens, max_frames);
    const prep = try video.prepare(a, src, c, max_tokens, max_frames);
    const ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[3], .{});
    defer dir.close(io);
    var all: std.ArrayList(u8) = .empty;
    for (prep.groups) |g| try all.appendSlice(a, std.mem.sliceAsBytes(g.pixels));
    try dir.writeFile(io, .{ .sub_path = "native_patches.f32", .data = all.items });
    var texts: std.ArrayList([]const u8) = .empty;
    for (prep.times) |t| {
        var buf: [64]u8 = undefined;
        try texts.append(a, try a.dupe(u8, try video.seconds(&buf, t)));
    }
    const g0 = prep.groups[0];
    const doc = .{ .frames = i.frames, .counted = i.counted != 0, .rate = .{ i.rate_num, i.rate_den }, .width = i.width, .height = i.height, .indices = p.indices, .grid = .{ prep.groups.len, g0.gh, g0.gw }, .tokens_per_group = g0.tokens, .texts = texts.items, .ms = ms };
    var out: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(doc, .{}, &out.writer);
    try dir.writeFile(io, .{ .sub_path = "native.json", .data = out.written() });
    std.debug.print("{s}: {d} frames decoded, grid {d}x{d}x{d}, {d} tokens a group, {d} ms\n", .{ args[2], p.indices.len, prep.groups.len, g0.gh, g0.gw, g0.tokens, ms });
}
