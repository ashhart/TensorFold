//! Flash Next's images and videos (--vision): the shared Qwen-VL tower's rows for a request's prompt pass, its frame.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const vl = tf.qwen_vl;
const vp = tf.flashnext_replay.vis_prompt;
const Tower = tf.qwen_vl_tower.Tower;
const Allocator = std.mem.Allocator;

/// The interleaved M-RoPE sections fn_attn_prep_mrope is written for, and the width it rotates.
const sections = [3]usize{ 11, 11, 10 };
const rotary = 64;

pub const Vision = struct {
    gpa: Allocator,
    c: vl.Config,
    tower: *Tower,
    prep: vp.Prep, // fn_attn_prep_mrope
    capacity: u32, // image tokens one request holds (--vision-image-tokens)
    max_images: u32, // and images (--vision-max-images)
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text
    video_markers: [3]u32, // vision_start, video_pad, vision_end
    video: api.VisionOpen, // the video limits the server was started with

    /// Buffers to keep resident with the engine's.
    pub fn buffers(v: *const Vision) [2]mtl.Buffer {
        return v.tower.buffers();
    }

    pub fn deinit(v: *Vision) void {
        v.prep.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }
};

/// The checkpoint's tower for the replay engine's width under its rope, for `o.image_tokens` image tokens a request.
pub fn open(gpa: Allocator, io: std.Io, eng: *fx.Engine, dir: []const u8, o: api.VisionOpen, a: Allocator, why: *[]const u8) !*Vision {
    if (eng.r.tp != null) {
        why.* = "--vision: images are not read across a Flash Next pair yet; serve on one Mac, or leave --vision out";
        return error.VisionOnPair;
    }
    const c = try configuration(io, dir, a, why);
    const v = try gpa.create(Vision);
    errdefer gpa.destroy(v);
    const videos = c.video_token != null;
    v.* = .{ .gpa = gpa, .c = c, .tower = undefined, .prep = undefined, .capacity = try std.math.add(u32, o.image_tokens, if (videos) o.video_tokens else 0), .max_images = o.max_images, .markers = .{ c.vision_start, c.image_token, c.vision_end }, .video_markers = .{ c.vision_start, c.video_token orelse 0, c.vision_end }, .video = o };
    const planned = try plan(a, c, o);
    const Factory = struct {
        run: *tf.flashnext_replay.Run,
        pub fn buffer(f: @This(), n: usize, options: usize) !mtl.Buffer {
            return f.run.allocateOptions(n, options);
        }
    };
    const Build = struct {
        gpa: Allocator,
        io: std.Io,
        eng: *fx.Engine,
        dir: []const u8,
        c: vl.Config,
        largest: u32,
        fn load(it: @This()) !*Tower {
            return Tower.loadWith(it.gpa, it.io, it.eng.r.device, it.dir, it.c, it.largest, Factory{ .run = it.eng.r });
        }
    };
    v.tower = try memory.loadTower(*Tower, eng.r, try planned.memory.tower(), Build{
        .gpa = gpa,
        .io = io,
        .eng = eng,
        .dir = dir,
        .c = c,
        .largest = planned.largest,
    }, Build.load);
    errdefer v.tower.deinit();
    v.prep = try vp.Prep.init(eng.r.device);
    std.log.info("flash next: image tower loaded ({d:.2} GiB with scratch for {d} tokens a pass; {d} image and {d} video tokens a request)", .{ @as(f64, @floatFromInt(v.tower.bytes)) / (1 << 30), planned.largest, o.image_tokens, if (videos) o.video_tokens else 0 });
    return v;
}

const memory = tf.flashnext_replay.vision_memory;

fn configuration(io: std.Io, dir: []const u8, a: Allocator, why: *[]const u8) !vl.Config {
    const read = struct {
        fn f(al: Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
            const path = std.fs.path.join(al, &.{ d, name }) catch return null;
            return std.Io.Dir.cwd().readFileAlloc(i, path, al, .limited(16 << 20)) catch null;
        }
    }.f;
    const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
    const c = (vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json")) catch |err| {
        why.* = "--vision: this checkpoint's vision_config or image processor names a tower the native engine does not read";
        return err;
    }) orelse {
        why.* = try std.fmt.allocPrint(a, "--vision: {s} has no image tower (its config.json names no vision_config)", .{dir});
        return error.NoVisionTower;
    };
    const mc = try tf.flashnext_config.parse(a, config);
    if (c.out_hidden != tf.flashnext_replay.D or !mc.mrope_interleaved or !std.mem.eql(usize, &mc.mrope_section, &sections) or mc.rotary_dim != rotary) {
        why.* = "--vision: the image tower's output width or the text's M-RoPE sections do not match Flash Next's";
        return error.VisionShape;
    }
    return c;
}

const Planned = struct { memory: memory.Plan, largest: u32 };

fn plan(a: Allocator, c: vl.Config, o: api.VisionOpen) !Planned {
    const videos = c.video_token != null;
    const per = @as(u64, c.factor()) * c.factor();
    const image_most = @min(@as(u64, o.image_tokens), c.max_pixels / per);
    const pair_most = if (videos) @min(@as(u64, o.video_tokens), c.max_pixels / (vl.video.min_frames * per)) else 0;
    const largest: u32 = @intCast(@max(image_most, pair_most));
    if (largest == 0) return error.ImageTooLarge;
    const tower = try tf.qwen_vl_tower.tower.memoryPlan(a, c, largest);
    const total = try std.math.add(u64, o.image_tokens, if (videos) o.video_tokens else 0);
    const row = tf.flashnext_replay.D * 2;
    return .{ .largest = largest, .memory = .{
        .source = tower.source,
        .weights = tower.weights,
        .scratch = tower.scratch,
        .rows = try std.math.mul(u64, total, 4 * row),
        .temporary = try std.math.mul(u64, largest, row),
    } };
}

pub fn budgetSettings(io: std.Io, dir: []const u8, vision: ?api.VisionOpen, paired: bool, a: Allocator, why: *[]const u8, settings: anytype) !@TypeOf(settings) {
    const o = vision orelse return settings;
    if (paired) {
        why.* = "--vision: images are not read across a Flash Next pair yet; serve on one Mac, or leave --vision out";
        return error.VisionOnPair;
    }
    const c = try configuration(io, dir, a, why);
    const p = try plan(a, c, o);
    return memory.withPlan(settings, p.memory);
}

/// The request's images through the tower, but those ending by `from` (resumed past): the prompt pass's rows and frame.
pub fn build(v: *Vision, eng: *fx.Engine, a: Allocator, images: []const api.Image, prompt_len: usize, from: usize) !*vp.VisPrompt {
    const merge = v.c.merge;
    var total: usize = 0;
    const spans = try a.alloc(vp.Span, images.len);
    for (images, spans, 0..) |img, *s, i| {
        if (img.gh % merge != 0 or img.gw % merge != 0 or img.tokens * merge * merge != img.gh * img.gw) return error.ImageGrid;
        if (@as(usize, img.at) + img.tokens > prompt_len or (i > 0 and img.at < spans[i - 1].at + spans[i - 1].n)) return error.ImageOutOfPrompt;
        s.* = .{ .at = img.at, .n = img.tokens, .mh = img.gh / merge, .mw = img.gw / merge, .row = @intCast(total) };
        total += img.tokens;
    }
    if (total > v.capacity) return error.ImageTooLarge;
    const row = tf.flashnext_replay.D * 2;
    var most: usize = 0;
    for (spans) |s| most = @max(most, s.n);
    const storage = try memory.outputs(mtl.Buffer, eng.r, total, most, row, mtl.ResourceOptions.shared);
    const rows = storage.rows;
    errdefer rows.deinit();
    const out = storage.temporary;
    defer out.deinit();
    const t0 = std.c.mach_absolute_time();
    for (images, spans) |img, s| { // one image a pass: the tower's host-written tables are one image's
        if (s.at + s.n <= from) continue; // a restored state holds its rows
        try v.tower.image(eng.r.queue, img.pixels, img.gh, img.gw, .{ .buf = out });
        const src = out.contents();
        const dst = rows.contents();
        for (0..s.n) |k| for (0..4) |stream| @memcpy(dst[((s.row + k) * 4 + stream) * row ..][0..row], src[k * row ..][0..row]);
    }
    std.log.info("flash next vision: {d} images, {d} tokens through the tower in {d:.1} ms", .{ images.len, total, @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e3 });
    const p = try a.create(vp.VisPrompt);
    const last = spans[spans.len - 1];
    p.* = .{ .rows = rows, .spans = spans, .pos3 = try vp.frame(a, spans), .frame_end = last.at + last.n, .prep = v.prep.prep };
    return p;
}

/// An image request's cache words: the server's image-keyed ones, the frame (vis_prompt.frameKey) in the first.
pub fn key(a: Allocator, r: *const api.Request) ![]const u32 {
    const f = vp.frameKey(r.images);
    if (f == 0) return r.cacheKey();
    const k = try a.dupe(u32, r.cacheKey());
    k[0] = 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(f, std.mem.asBytes(&k[0])) & 0x7fff_ffff));
    return k;
}

/// The engine's prompt pass is over: its image rows go.
pub fn release(eng: *fx.Engine, p: ?*vp.VisPrompt) void {
    eng.m.vis = null;
    if (p) |q| q.rows.deinit();
}

/// Info.vision for a host: images, and videos when the checkpoint names a video token.
pub fn offer(v: *Vision, ctx: *anyopaque, prepare_image: @FieldType(api.Vision, "prepare"), prepare_video: @FieldType(api.VideoOffer, "prepare")) api.Vision {
    var o: api.Vision = .{ .ctx = ctx, .prepare = prepare_image, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.video.image_tokens, .max_images = v.max_images, .explain = vl.geometry.problem };
    if (v.c.video_token) |t| o.video = .{ .prepare = prepare_video, .video_token = t, .group_open = v.c.vision_start, .group_close = v.c.vision_end, .time_first = t, .markers = &v.video_markers, .video_tokens = v.video.video_tokens, .max_videos = v.video.max_videos, .max_frames = v.video.max_frames };
    return o;
}

/// Info.vision's video prepare: a decoded video's frames to the tower's pairs (core/qwen_vl video), on the HTTP thread.
pub fn prepareVideo(v: *const Vision, a: Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
    const Adapter = struct {
        src: api.VideoSource,
        fn decode(c: *anyopaque, indices: []const u32, sink: *vl.video.Sink) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(c));
            const into = struct {
                fn take(s: *anyopaque, k: usize, rgb: []const u8, w: u32, h: u32, stride: usize) anyerror!void {
                    const g: *vl.video.Sink = @ptrCast(@alignCast(s));
                    try g.take(k, rgb, w, h, stride);
                }
            }.take;
            try self.src.decode(self.src.ctx, indices, .{ .ctx = sink, .take = into });
        }
    };
    var ad: Adapter = .{ .src = src };
    const s: vl.video.Source = .{ .frames = src.frames, .rate_num = src.rate_num, .rate_den = src.rate_den, .width = src.width, .height = src.height, .ctx = &ad, .decode = Adapter.decode };
    const p = try vl.video.prepare(a, s, v.c, max_tokens, max_frames);
    const groups = try a.alloc(api.PreparedImage, p.groups.len);
    for (groups, p.groups) |*g, q| g.* = .{ .pixels = q.pixels, .gh = q.gh, .gw = q.gw, .tokens = q.tokens, .hash = q.hash };
    return .{ .groups = groups, .times = p.times };
}

/// Info.vision's prepare: an image's bytes to the tower's patches (core/qwen_vl geometry), on the HTTP thread.
pub fn prepare(v: *const Vision, io: std.Io, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const p = try vl.geometry.prepare(io, a, bytes, v.c, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
